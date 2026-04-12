#!/usr/bin/env python3
"""
train_transformer_intent.py

End-to-end transformer baseline for intent classification.

Pipeline:
1) Load CSV dataset
2) Split into train/validation/test
3) Tokenize conversation_text
4) Fine-tune a transformer classifier
5) Evaluate on validation/test
6) Save model artifacts, metrics, label mappings, and predictions

Expected CSV columns:
- conversation_text
- gt_primary_intent

Usage:
  python src/train_transformer_intent.py \
    --csv data/processed/conversation_dataset.csv \
    --output-dir models/transformer_intent \
    --model-name distilbert-base-uncased \
    --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

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


def load_dataset(csv_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)
    if df.empty:
        raise ValueError("Dataset is empty after dropping missing values.")
    return df


def build_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    uniq = sorted(set(labels))
    label2id = {lab: i for i, lab in enumerate(uniq)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def split_dataframe(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )
    val_relative = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_relative,
        random_state=random_state,
        stratify=train_df[label_col],
    )
    return train_df, val_df, test_df


def make_hf_dataset(df: pd.DataFrame, text_col: str, label_col: str, label2id: Dict[str, int]) -> Dataset:
    temp = df[[text_col, label_col]].copy()
    temp["labels"] = temp[label_col].map(label2id).astype(int)
    temp = temp.rename(columns={text_col: "text"})
    return Dataset.from_pandas(temp[["text", "labels"]], preserve_index=False)


def tokenize_dataset(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return ds.map(_tokenize, batched=True, remove_columns=["text"])


def compute_metrics_builder(id2label: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted"),
        }
    return compute_metrics


def evaluate_and_save(trainer: Trainer, eval_dataset: Dataset, out_dir: Path, split_name: str, id2label: Dict[int, str]):
    pred_out = trainer.predict(eval_dataset)
    logits = pred_out.predictions
    labels = pred_out.label_ids
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")

    report = classification_report(
        labels,
        preds,
        target_names=[id2label[i] for i in range(len(id2label))],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds)

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    with open(out_dir / f"{split_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame(
        {
            "label_id": labels,
            "pred_id": preds,
            "label": [id2label[int(x)] for x in labels],
            "pred": [id2label[int(x)] for x in preds],
        }
    )
    pred_df.to_csv(out_dir / f"{split_name}_predictions.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train transformer intent classifier")
    parser.add_argument("--csv", type=str, required=True, help="Path to conversation_dataset.csv")
    parser.add_argument("--output-dir", type=str, default="models/transformer_intent", help="Directory to save artifacts")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased", help="HF model name")
    parser.add_argument("--text-col", type=str, default="conversation_text")
    parser.add_argument("--label-col", type=str, default="gt_primary_intent")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 if CUDA is available")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.csv, args.text_col, args.label_col)
    label2id, id2label = build_label_maps(df[args.label_col].tolist())

    with open(out_dir / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)

    train_df, val_df, test_df = split_dataframe(
        df=df,
        label_col=args.label_col,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )

    with open(out_dir / "dataset_split_counts.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": int(len(train_df)),
                "validation": int(len(val_df)),
                "test": int(len(test_df)),
                "total": int(len(df)),
            },
            f,
            indent=2,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = tokenize_dataset(make_hf_dataset(train_df, args.text_col, args.label_col, label2id), tokenizer, args.max_length)
    val_ds = tokenize_dataset(make_hf_dataset(val_df, args.text_col, args.label_col, label2id), tokenizer, args.max_length)
    test_ds = tokenize_dataset(make_hf_dataset(test_df, args.text_col, args.label_col, label2id), tokenizer, args.max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label={i: l for i, l in id2label.items()},
    )

    # training_args = TrainingArguments(
    #     output_dir=str(out_dir / "checkpoints"),
    #     overwrite_output_dir=args.overwrite_output_dir,
    #     learning_rate=args.learning_rate,
    #     per_device_train_batch_size=args.train_batch_size,
    #     per_device_eval_batch_size=args.eval_batch_size,
    #     num_train_epochs=args.epochs,
    #     weight_decay=args.weight_decay,
    #     warmup_steps=args.warmup_steps,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     logging_strategy="steps",
    #     logging_steps=25,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="macro_f1",
    #     greater_is_better=True,
    #     save_total_limit=2,
    #     report_to="none",
    #     fp16=bool(args.fp16 and torch.cuda.is_available()),
    #     seed=args.seed,
    # )

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=2,
        report_to=[],

        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(id2label),
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on validation split...")
    val_metrics = evaluate_and_save(trainer, val_ds, out_dir, "validation", id2label)

    print("Evaluating on test split...")
    test_metrics = evaluate_and_save(trainer, test_ds, out_dir, "test", id2label)

    final_model_dir = out_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    summary = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "validation": {
            "accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
        },
        "test": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
        },
    }

    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
