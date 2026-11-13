#!/usr/bin/env python3
"""Compute ECE and Brier score for confidence-bearing RECAP-Voice predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


SCALAR_FIELDS = (
    "intent",
    "customer_goal",
    "resolution_state",
    "verification",
    "escalated",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def ece(confidences: np.ndarray, outcomes: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (confidences >= edges[index]) & (
            confidences < edges[index + 1]
            if index < bins - 1
            else confidences <= edges[index + 1]
        )
        if mask.any():
            result += float(mask.mean()) * abs(
                float(outcomes[mask].mean()) - float(confidences[mask].mean())
            )
    return result


def confidence_bin_rows(
    field_name: str,
    confidences: np.ndarray,
    outcomes: np.ndarray,
    bins: int,
) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        mask = (confidences >= lower) & (
            confidences < upper if index < bins - 1 else confidences <= upper
        )
        if not mask.any():
            continue
        bin_confidence = float(confidences[mask].mean())
        bin_accuracy = float(outcomes[mask].mean())
        rows.append(
            {
                "record_type": "confidence_bin",
                "field": field_name,
                "n": int(mask.sum()),
                "accuracy": bin_accuracy,
                "mean_confidence": bin_confidence,
                "ece": "",
                "brier_score": "",
                "bins": bins,
                "bin_index": index,
                "bin_lower": lower,
                "bin_upper": upper,
                "bin_ece_contribution": float(mask.mean())
                * abs(bin_accuracy - bin_confidence),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins must be at least 2")
    if not args.pred_dir.is_dir() or not args.gold_dir.is_dir():
        parser.error("--gold-dir and --pred-dir must be existing directories")

    observations: dict[str, list[tuple[float, bool]]] = {}
    packet_count = 0
    for prediction_path in sorted(args.pred_dir.glob("*.json")):
        gold_path = args.gold_dir / prediction_path.name
        if not gold_path.is_file():
            continue
        prediction = load_json(prediction_path).get("state", {})
        gold = load_json(gold_path).get("state", {})
        packet_count += 1
        for field_name in SCALAR_FIELDS:
            pred_field = prediction.get(field_name)
            gold_field = gold.get(field_name)
            if not isinstance(pred_field, dict) or not isinstance(gold_field, dict):
                continue
            confidence = pred_field.get("confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                continue
            score = float(confidence)
            if not np.isfinite(score):
                continue
            score = float(np.clip(score, 0.0, 1.0))
            observations.setdefault(field_name, []).append(
                (score, pred_field.get("value") == gold_field.get("value"))
            )

    if not observations:
        parser.error("No confidence-bearing scalar predictions matched gold packets")
    rows = []
    for field_name, pairs in sorted(observations.items()):
        confidences = np.asarray([pair[0] for pair in pairs], dtype=float)
        outcomes = np.asarray([pair[1] for pair in pairs], dtype=float)
        rows.append(
            {
                "record_type": "field_summary",
                "field": field_name,
                "n": len(pairs),
                "accuracy": float(outcomes.mean()),
                "mean_confidence": float(confidences.mean()),
                "ece": ece(confidences, outcomes, args.bins),
                "brier_score": float(np.mean(np.square(confidences - outcomes))),
                "bins": args.bins,
                "bin_index": "",
                "bin_lower": "",
                "bin_upper": "",
                "bin_ece_contribution": "",
            }
        )
        rows.extend(confidence_bin_rows(field_name, confidences, outcomes, args.bins))
    all_pairs = [pair for pairs in observations.values() for pair in pairs]
    confidence_array = np.asarray([pair[0] for pair in all_pairs], dtype=float)
    outcome_array = np.asarray([pair[1] for pair in all_pairs], dtype=float)
    rows.append(
        {
            "record_type": "field_summary",
            "field": "__all_scalar_fields__",
            "n": len(all_pairs),
            "accuracy": float(outcome_array.mean()),
            "mean_confidence": float(confidence_array.mean()),
            "ece": ece(confidence_array, outcome_array, args.bins),
            "brier_score": float(np.mean(np.square(confidence_array - outcome_array))),
            "bins": args.bins,
            "bin_index": "",
            "bin_lower": "",
            "bin_upper": "",
            "bin_ece_contribution": "",
        }
    )
    rows.extend(
        confidence_bin_rows(
            "__all_scalar_fields__", confidence_array, outcome_array, args.bins
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "record_type",
                "field",
                "n",
                "accuracy",
                "mean_confidence",
                "ece",
                "brier_score",
                "bins",
                "bin_index",
                "bin_lower",
                "bin_upper",
                "bin_ece_contribution",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Calibration report for {packet_count} paired predictions written to {args.output}")


if __name__ == "__main__":
    main()
