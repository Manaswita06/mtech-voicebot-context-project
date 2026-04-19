#!/usr/bin/env python3
"""
Multi-label intent classification baseline using TF-IDF + OneVsRest Logistic Regression.

Expected CSV columns:
- conversation_text
- gt_primary_intent
- gt_secondary_intents   (JSON string, Python list string, or blank)

Labels used:
- gt_primary_intent + gt_secondary_intents

Usage:
  python src/multi_intent_tfidf_baseline.py \
    --csv data/processed/conversation_dataset.csv \
    --output-dir models/multi_intent_tfidf
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


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


def build_targets(df: pd.DataFrame) -> Tuple[np.ndarray, MultiLabelBinarizer, List[List[str]]]:
    label_lists = []
    for _, row in df.iterrows():
        primary = str(row["gt_primary_intent"])
        secondary = parse_secondary(row.get("gt_secondary_intents"))
        labels = [primary] + [s for s in secondary if s and s != primary]
        label_lists.append(sorted(set(labels)))

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_lists)
    return Y, mlb, label_lists


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["conversation_text", "gt_primary_intent"]).copy()
    df["conversation_text"] = df["conversation_text"].astype(str)
    return df


def train_eval(df: pd.DataFrame, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["gt_primary_intent"],
    )

    X_train = train_df["conversation_text"].values
    X_test = test_df["conversation_text"].values

    Y_train, mlb, _ = build_targets(train_df)
    Y_test = mlb.transform(
        [
            [row["gt_primary_intent"]] + [s for s in parse_secondary(row.get("gt_secondary_intents")) if s != row["gt_primary_intent"]]
            for _, row in test_df.iterrows()
        ]
    )

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
    )

    X_train_t = vectorizer.fit_transform(X_train)
    X_test_t = vectorizer.transform(X_test)

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, solver="liblinear")
    )
    clf.fit(X_train_t, Y_train)

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_test_t)
    else:
        decision = clf.decision_function(X_test_t)
        probs = 1 / (1 + np.exp(-decision))

    threshold = 0.5
    Y_pred = (probs >= threshold).astype(int)

    for i in range(Y_pred.shape[0]):
        if Y_pred[i].sum() == 0:
            Y_pred[i, np.argmax(probs[i])] = 1

    micro_p = precision_score(Y_test, Y_pred, average="micro", zero_division=0)
    micro_r = recall_score(Y_test, Y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(Y_test, Y_pred, average="micro", zero_division=0)

    macro_p = precision_score(Y_test, Y_pred, average="macro", zero_division=0)
    macro_r = recall_score(Y_test, Y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(Y_test, Y_pred, average="macro", zero_division=0)

    subset_acc = (Y_test == Y_pred).all(axis=1).mean()

    print(f"Subset accuracy: {subset_acc:.4f}")
    print(f"Micro P/R/F1: {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}")
    print(f"Macro P/R/F1: {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}")

    report = classification_report(
        Y_test,
        Y_pred,
        target_names=mlb.classes_,
        zero_division=0,
        output_dict=True,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, out / "tfidf_vectorizer.joblib")
    joblib.dump(clf, out / "ovr_logreg.joblib")
    joblib.dump(mlb, out / "multilabel_binarizer.joblib")

    rows = []
    for text, true_row, pred_row, p in zip(X_test, Y_test, Y_pred, probs):
        true_labels = [mlb.classes_[j] for j, v in enumerate(true_row) if v == 1]
        pred_labels = [mlb.classes_[j] for j, v in enumerate(pred_row) if v == 1]
        rows.append({
            "text": text,
            "true_labels": json.dumps(true_labels),
            "pred_labels": json.dumps(pred_labels),
            "top5_probs": json.dumps({mlb.classes_[j]: float(p[j]) for j in np.argsort(-p)[:5]}),
        })

    pd.DataFrame(rows).to_csv(out / "eval_predictions.csv", index=False)

    summary = {
        "subset_accuracy": float(subset_acc),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "classes": list(mlb.classes_),
        "classification_report": report,
    }

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved artifacts to {out.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Multi-label TF-IDF baseline")
    parser.add_argument("--csv", required=True, help="Path to conversation_dataset.csv")
    parser.add_argument("--output-dir", default="models/multi_intent_tfidf")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_dataset(args.csv)
    train_eval(df, args.output_dir, test_size=args.test_size, random_state=args.random_state)


if __name__ == "__main__":
    main()