#!/usr/bin/env python3
"""
eval_multi_intent.py

Evaluate multi-intent predictions against gold labels.

Expected prediction CSV columns:
- true_labels: JSON string like ["CARD_REPLACEMENT", "ADDRESS_UPDATE"]
- pred_labels: JSON string like ["CARD_REPLACEMENT", "PAYMENT_ISSUE"]

Optional columns:
- top5_probs (ignored)
- text (ignored)

This script can evaluate outputs from:
- TF-IDF baseline
- Transformer multi-label classifier
- LLM extraction pipeline

Usage:
  python src/eval_multi_intent.py \
    --pred-csv models/multi_intent_tfidf/eval_predictions.csv \
    --output-dir results/multi_intent_tfidf
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import List, Set, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    jaccard_score,
)
from sklearn.preprocessing import MultiLabelBinarizer


def parse_labels(x) -> List[str]:
    """
    Parse a label list stored as JSON string, Python literal string, or comma-separated string.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x]

    s = str(x).strip()
    if not s:
        return []

    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(v) for v in obj]
    except Exception:
        pass

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(v) for v in obj]
    except Exception:
        pass

    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    return [p.strip().strip("'").strip('"') for p in s.split(",") if p.strip()]


def load_predictions(pred_csv: str) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    if "true_labels" not in df.columns or "pred_labels" not in df.columns:
        raise ValueError(
            "Prediction CSV must contain columns: true_labels and pred_labels"
        )
    df["true_labels_list"] = df["true_labels"].apply(parse_labels)
    df["pred_labels_list"] = df["pred_labels"].apply(parse_labels)
    return df


def fit_mlb(df: pd.DataFrame) -> MultiLabelBinarizer:
    all_labels = set()
    for labels in df["true_labels_list"]:
        all_labels.update(labels)
    for labels in df["pred_labels_list"]:
        all_labels.update(labels)

    mlb = MultiLabelBinarizer()
    mlb.fit([sorted(all_labels)])
    return mlb


def binarize(df: pd.DataFrame, mlb: MultiLabelBinarizer):
    y_true = mlb.transform(df["true_labels_list"])
    y_pred = mlb.transform(df["pred_labels_list"])
    return y_true, y_pred


def exact_match_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).all(axis=1).mean())


def average_jaccard(df: pd.DataFrame) -> float:
    scores = []
    for gold, pred in zip(df["true_labels_list"], df["pred_labels_list"]):
        g = set(gold)
        p = set(pred)
        if not g and not p:
            scores.append(1.0)
        else:
            scores.append(len(g & p) / len(g | p) if len(g | p) > 0 else 0.0)
    return float(np.mean(scores))


def per_sample_label_count_error(df: pd.DataFrame) -> float:
    errs = []
    for gold, pred in zip(df["true_labels_list"], df["pred_labels_list"]):
        errs.append(abs(len(gold) - len(pred)))
    return float(np.mean(errs))


def build_label_report(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    return report


def evaluate(pred_csv: str) -> Dict[str, Any]:
    df = load_predictions(pred_csv)
    mlb = fit_mlb(df)
    y_true, y_pred = binarize(df, mlb)

    subset_acc = exact_match_rate(y_true, y_pred)
    micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    weighted_p = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_r = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    jacc = average_jaccard(df)
    label_count_mae = per_sample_label_count_error(df)

    # Jaccard with sklearn over multilabel indicator matrix
    jaccard_micro = float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))

    report = build_label_report(y_true, y_pred, list(mlb.classes_))

    results = {
        "n_samples": int(len(df)),
        "n_labels": int(len(mlb.classes_)),
        "classes": list(mlb.classes_),
        "subset_accuracy": float(subset_acc),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "jaccard_samples": float(jaccard_micro),
        "average_jaccard_manual": float(jacc),
        "label_count_mae": float(label_count_mae),
        "classification_report": report,
    }
    return results


def save_outputs(results: Dict[str, Any], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "multi_intent_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # compact comparison table for slides/thesis
    summary_table = pd.DataFrame([{
        "subset_accuracy": results["subset_accuracy"],
        "micro_f1": results["micro_f1"],
        "macro_f1": results["macro_f1"],
        "weighted_f1": results["weighted_f1"],
        "jaccard_samples": results["jaccard_samples"],
        "label_count_mae": results["label_count_mae"],
        "n_samples": results["n_samples"],
        "n_labels": results["n_labels"],
    }])
    summary_table.to_csv(out_dir / "multi_intent_eval_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-intent predictions")
    parser.add_argument("--pred-csv", required=True, help="Path to predictions CSV")
    parser.add_argument("--output-dir", default=None, help="Directory to save summary files")
    args = parser.parse_args()

    results = evaluate(args.pred_csv)

    print(json.dumps({
        k: v for k, v in results.items()
        if k not in {"classification_report", "classes"}
    }, indent=2))

    # print per-class metrics in a compact way
    print("\nPer-class F1:")
    report = results["classification_report"]
    for cls in results["classes"]:
        if cls in report:
            print(f"{cls}: {report[cls]['f1-score']:.4f}")

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.pred_csv).parent
    save_outputs(results, out_dir)
    print(f"\nSaved evaluation files to {out_dir.resolve()}")


if __name__ == "__main__":
    main()