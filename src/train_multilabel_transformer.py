#!/usr/bin/env python3
"""
Multi-label intent classification with a pretrained transformer.

Expected CSV columns:
- conversation_text
- gt_primary_intent
- gt_secondary_intents

Labels used:
- gt_primary_intent + gt_secondary_intents

Usage:
  python src/train_multilabel_transformer.py \
    --csv data/processed/conversation_dataset.csv \
    --output-dir models/multilabel_transformer \
    --model-name distilbert-base-uncased \
    --epochs 3
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_secondary(x) -> List[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(i) for i in x]

    s = str(x).strip()
    if not s:
        return []

    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(i) for i in obj]
    except Exception:
        pass

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(i) for i in obj]
    except Exception:
        pass

    return [p.strip() for p in s.strip("[]").split(",") if p.strip()]


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["conversation_text", "gt_primary_intent"]).copy()
    df["conversation_text"] = df["conversation_text"].astype(str)
    return df


def make_label_sets(df: pd.DataFrame) -> List[List[str]]:
    label_sets = []
    for _, row in df.iterrows():
        primary = str(row["gt_primary_intent"])
        secondary = parse_secondary(row.get("gt_secondary_intents"))
        labels = [primary] + [s for s in secondary if s and s != primary]
        label_sets.append(sorted(set(labels)))
    return label_sets


def split_df(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["gt_primary_intent"],
    )
    rel_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=rel_val,
        random_state=seed,
        stratify=train_df["gt_primary_intent"],
    )
    return train_df, val_df, test_df


def encode_dataset(df: pd.DataFrame, mlb: MultiLabelBinarizer) -> Dataset:
    ds = Dataset.from_pandas(df[["conversation_text"]].copy(), preserve_index=False)
    labels = mlb.transform(make_label_sets(df))
    ds = ds.add_column("labels", [row.astype(float).tolist() for row in labels])
    return ds


def tokenize_dataset(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    def tok(batch):
        return tokenizer(batch["conversation_text"], truncation=True, max_length=max_length)
    return ds.map(tok, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    for i in range(preds.shape[0]):
        if preds[i].sum() == 0:
            preds[i, np.argmax(probs[i])] = 1

    return {
        "micro_precision": precision_score(labels, preds, average="micro", zero_division=0),
        "micro_recall": recall_score(labels, preds, average="micro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "subset_accuracy": float((labels == preds).all(axis=1).mean()),
    }


def predict_and_save(trainer: Trainer, ds: Dataset, mlb: MultiLabelBinarizer, out_dir: Path, split_name: str):
    pred = trainer.predict(ds)
    logits = pred.predictions
    labels = np.array(pred.label_ids)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    for i in range(preds.shape[0]):
        if preds[i].sum() == 0:
            preds[i, np.argmax(probs[i])] = 1

    summary = {
        "micro_precision": precision_score(labels, preds, average="micro", zero_division=0),
        "micro_recall": recall_score(labels, preds, average="micro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "subset_accuracy": float((labels == preds).all(axis=1).mean()),
    }

    with open(out_dir / f"{split_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = []
    for true_row, pred_row, prob_row in zip(labels, preds, probs):
        true_labels = [mlb.classes_[j] for j, v in enumerate(true_row) if v == 1]
        pred_labels = [mlb.classes_[j] for j, v in enumerate(pred_row) if v == 1]
        rows.append({
            "true_labels": json.dumps(true_labels),
            "pred_labels": json.dumps(pred_labels),
            "top5_probs": json.dumps({mlb.classes_[j]: float(prob_row[j]) for j in np.argsort(-prob_row)[:5]}),
        })

    pd.DataFrame(rows).to_csv(out_dir / f"{split_name}_predictions.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train multi-label transformer")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="models/multilabel_transformer")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.csv)
    mlb = MultiLabelBinarizer()
    mlb.fit(make_label_sets(df))

    with open(out_dir / "label_classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": list(mlb.classes_)}, f, indent=2)

    train_df, val_df, test_df = split_df(df, args.test_size, args.val_size, args.seed)

    # ... [logging split counts] ...

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 1. Encode and Tokenize
    train_ds = tokenize_dataset(encode_dataset(train_df, mlb), tokenizer, args.max_length)
    val_ds = tokenize_dataset(encode_dataset(val_df, mlb), tokenizer, args.max_length)
    test_ds = tokenize_dataset(encode_dataset(test_df, mlb), tokenizer, args.max_length)

    # Redundant rename lines removed here (these caused your error)

    # 2. Keep only columns the model expects
    # Labels must be named "labels" for the Trainer to automatically find them
    cols_to_keep = {"input_ids", "attention_mask", "labels"}

    def prepare_for_model(ds):
        remove_cols = [c for c in ds.column_names if c not in cols_to_keep]
        ds = ds.remove_columns(remove_cols)
        ds.set_format(type="torch")
        return ds

    train_ds = prepare_for_model(train_ds)
    val_ds = prepare_for_model(val_ds)
    test_ds = prepare_for_model(test_ds)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(mlb.classes_),
        problem_type="multi_label_classification",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Evaluating validation...")
    val_summary = predict_and_save(trainer, val_ds, mlb, out_dir, "validation")
    print("Evaluating test...")
    test_summary = predict_and_save(trainer, test_ds, mlb, out_dir, "test")

    final_dir = out_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    summary = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "validation": val_summary,
        "test": test_summary,
        "classes": list(mlb.classes_),
    }
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()