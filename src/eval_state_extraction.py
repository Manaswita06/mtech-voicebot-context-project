#!/usr/bin/env python3
"""eval_state_extraction.py

Evaluate state extraction predictions against gold labels.
Input: directory of *.state.json files produced by state_extraction_pipeline.py
Outputs: summary JSON and rows CSV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_record(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_records(pred_dir: Path):
    for p in sorted(pred_dir.glob("*.state.json")):
        yield p, load_record(p)


def to_set(x) -> set:
    if x is None:
        return set()
    if isinstance(x, list):
        return set(x)
    if isinstance(x, str):
        return {x}
    return set(x)


def mae(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def pearson(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(pred_dir: str) -> Dict[str, Any]:
    pred_path = Path(pred_dir)
    rows = []
    for p, rec in iter_records(pred_path):
        gold = rec.get("gold", {})
        pred = rec.get("prediction", {})
        rows.append({
            "source_file": p.name,
            "provider": pred.get("provider"),
            "model": pred.get("model"),
            "gold_primary_intent": gold.get("gt_primary_intent"),
            "pred_primary_intent": pred.get("primary_intent"),
            "gold_secondary_intents": gold.get("gt_secondary_intents", []),
            "pred_secondary_intents": pred.get("secondary_intents", []),
            "gold_multi_intent": bool(gold.get("gt_multi_intent", False)),
            "pred_multi_intent": bool(pred.get("multi_intent", False)),
            "gold_ambiguity_level": float(gold.get("gt_ambiguity_level", 0.0)),
            "pred_ambiguity_level": float(pred.get("ambiguity_level", 0.0)),
            "gold_tool_failure": bool(gold.get("gt_tool_failure", False)),
            "pred_tool_failure": bool(pred.get("tool_failure", False)),
            "gold_failure_count": int(gold.get("gt_failure_count", 0)),
            "pred_failure_count": int(pred.get("failure_count", len(pred.get("failure_reasons", [])))),
            "gold_sentiment_overall": float(gold.get("gt_sentiment_overall", 0.0)),
            "pred_sentiment_overall": float(pred.get("sentiment_overall", 0.0)),
            "gold_turn_count": int(gold.get("gt_turn_count", 0)),
            "pred_turn_count": int(pred.get("turn_count", 0)),
            "schema_valid": len(pred.get("validation_errors", [])) == 0,
            "validation_errors": pred.get("validation_errors", []),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No .state.json prediction files found in {pred_dir}")

    intent_acc = accuracy_score(df["gold_primary_intent"], df["pred_primary_intent"])
    intent_prf = precision_recall_fscore_support(
        df["gold_primary_intent"], df["pred_primary_intent"], average="macro", zero_division=0
    )

    gold_sec = df["gold_secondary_intents"].apply(to_set)
    pred_sec = df["pred_secondary_intents"].apply(to_set)
    tp = fp = fn = 0
    jaccs = []
    for g, p in zip(gold_sec, pred_sec):
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        union = g | p
        jaccs.append(1.0 if not union else len(g & p) / len(union))
    sec_p = tp / (tp + fp) if (tp + fp) else 0.0
    sec_r = tp / (tp + fn) if (tp + fn) else 0.0
    sec_f1 = (2 * sec_p * sec_r / (sec_p + sec_r)) if (sec_p + sec_r) else 0.0
    sec_exact = float((gold_sec == pred_sec).mean())

    multi_acc = accuracy_score(df["gold_multi_intent"], df["pred_multi_intent"])
    tool_acc = accuracy_score(df["gold_tool_failure"], df["pred_tool_failure"])

    amb_mae = mae(df["gold_ambiguity_level"], df["pred_ambiguity_level"])
    amb_rmse = rmse(df["gold_ambiguity_level"], df["pred_ambiguity_level"])
    amb_corr = pearson(df["gold_ambiguity_level"], df["pred_ambiguity_level"])

    sent_mae = mae(df["gold_sentiment_overall"], df["pred_sentiment_overall"])
    sent_rmse = rmse(df["gold_sentiment_overall"], df["pred_sentiment_overall"])
    sent_corr = pearson(df["gold_sentiment_overall"], df["pred_sentiment_overall"])

    turn_mae = mae(df["gold_turn_count"], df["pred_turn_count"])
    failure_count_mae = mae(df["gold_failure_count"], df["pred_failure_count"])
    schema_valid_rate = float(df["schema_valid"].mean())

    summary = {
        "n_samples": int(len(df)),
        "provider": str(df["provider"].dropna().iloc[0]) if df["provider"].notna().any() else None,
        "model": str(df["model"].dropna().iloc[0]) if df["model"].notna().any() else None,
        "primary_intent": {
            "accuracy": float(intent_acc),
            "macro_precision": float(intent_prf[0]),
            "macro_recall": float(intent_prf[1]),
            "macro_f1": float(intent_prf[2]),
        },
        "secondary_intents": {
            "precision": float(sec_p),
            "recall": float(sec_r),
            "f1": float(sec_f1),
            "exact_match": float(sec_exact),
        },
        "multi_intent_accuracy": float(multi_acc),
        "tool_failure_accuracy": float(tool_acc),
        "ambiguity": {"mae": float(amb_mae), "rmse": float(amb_rmse), "pearson": float(amb_corr)},
        "sentiment_overall": {"mae": float(sent_mae), "rmse": float(sent_rmse), "pearson": float(sent_corr)},
        "turn_count_mae": float(turn_mae),
        "failure_count_mae": float(failure_count_mae),
        "schema_valid_rate": float(schema_valid_rate),
        "average_jaccard_secondary": float(np.mean(jaccs)),
    }
    return summary, df


def main():
    ap = argparse.ArgumentParser(description="Evaluate state extraction outputs")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    summary, df = evaluate(args.pred_dir)
    print(json.dumps(summary, indent=2))

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.pred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "state_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    df.to_csv(out_dir / "state_eval_rows.csv", index=False)
    print(f"Saved summary + rows to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
