#!/usr/bin/env python3
"""
intent_tfidf_baseline.py

Train and evaluate a TF-IDF + LogisticRegression baseline for intent classification.

Inputs:
 - CSV with at least columns: conversation_text, gt_primary_intent

Outputs:
 - prints accuracy and classification report
 - saves: tfidf_vectorizer.joblib, tfidf_logreg.joblib
 - saves eval_predictions.csv with columns: text, label, pred, prob_<label>...

Usage (from project root):
  python src/intent_tfidf_baseline.py --csv data/processed/conversation_dataset.csv --output-dir models/tfidf_baseline
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

def load_dataset(csv_path: str, text_col: str = "conversation_text", label_col: str = "gt_primary_intent"):
    df = pd.read_csv(csv_path)
    # drop rows with missing text or label
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)
    return df


def train_eval(df: pd.DataFrame, text_col: str, label_col: str, output_dir: str, test_size: float, random_state: int):
    X = df[text_col].values
    y = df[label_col].values

    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # TF-IDF vectorizer
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), analyzer="word")
    X_train_t = vect.fit_transform(X_train)
    X_test_t = vect.transform(X_test)

    # classifier
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
    clf.fit(X_train_t, y_train)

    # predictions
    y_pred = clf.predict(X_test_t)
    y_proba = clf.predict_proba(X_test_t) if hasattr(clf, "predict_proba") else None

    # --- METRICS CALCULATION ---
    acc = accuracy_score(y_test, y_pred)

    # Get report as a string for printing
    report_str = classification_report(y_test, y_pred)
    # Get report as a dictionary for saving
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Add overall accuracy to the dictionary for the JSON file
    report_dict["accuracy_overall"] = acc

    print(f"\nTest accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(report_str)

    # --- PREPARE EVAL DATA ---
    eval_df = pd.DataFrame({
        "text": X_test,
        "label": y_test,
        "pred": y_pred
    })
    if y_proba is not None:
        labels = clf.classes_
        for i, lab in enumerate(labels):
            eval_df[f"prob_{lab}"] = y_proba[:, i]

    # --- SAVE ARTIFACTS ---
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Save Models
    joblib.dump(vect, out / "tfidf_vectorizer.joblib")
    joblib.dump(clf, out / "tfidf_logreg.joblib")

    # 2. Save Prediction CSV
    eval_df.to_csv(out / "eval_predictions.csv", index=False)

    # 3. Save Metrics as JSON
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)

    # 4. Save Metrics as CSV (Classification Report)
    # Transposing makes it look like the standard report table
    metrics_df = pd.DataFrame(report_dict).transpose()
    metrics_df.to_csv(out / "metrics.csv", index=True)

    print(f"\nSaved artifacts, metrics.json, and metrics.csv to {out.resolve()}")

    return {
        "vectorizer_path": str((out / "tfidf_vectorizer.joblib").resolve()),
        "model_path": str((out / "tfidf_logreg.joblib").resolve()),
        "metrics_json": str((out / "metrics.json").resolve()),
        "metrics_csv": str((out / "metrics.csv").resolve())
    }

def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + LogisticRegression intent baseline")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV dataset (conversation_dataset.csv)")
    parser.add_argument("--text-col", type=str, default="conversation_text", help="Text column to use")
    parser.add_argument("--label-col", type=str, default="gt_primary_intent", help="Label column to use")
    parser.add_argument("--output-dir", type=str, default="models/tfidf_baseline", help="Directory to save artifacts")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = load_dataset(args.csv, args.text_col, args.label_col)
    if df.empty:
        raise SystemExit("ERROR: dataset is empty after loading. Check CSV path and columns.")
    artifacts = train_eval(df, args.text_col, args.label_col, args.output_dir, args.test_size, args.random_state)
    print("\nArtifacts:", artifacts)

if __name__ == "__main__":
    main()