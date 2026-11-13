#!/usr/bin/env python3
"""Explore pre-outcome customer interaction patterns and outcome prediction."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import re
from pathlib import Path
from typing import Any

from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WORD_RE = re.compile(r"[a-z0-9']+")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "do", "for", "from", "have", "i", "in", "is", "it", "me", "my", "of",
    "on", "or", "please", "the", "this", "to", "was", "would", "you",
}
OUTCOME_ACTS = {"resolve", "resolution", "unresolved", "escalation", "escalate"}
FEATURES = [
    "pre_outcome_turn_count",
    "customer_turn_count",
    "customer_turn_share",
    "customer_word_count",
    "mean_customer_words",
    "customer_issue_restatement_count",
    "customer_question_count",
]


def load_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("conversations", raw.get("data"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of conversations in {path}")
    records = [row for row in raw if isinstance(row, dict)]
    if not records:
        raise ValueError(f"No conversations found in {path}")
    return records


def tokens(text: str) -> set[str]:
    return {
        word for word in WORD_RE.findall(text.casefold())
        if word not in STOP_WORDS
    }


def role(turn: dict[str, Any]) -> str:
    return str(turn.get("role", "")).casefold()


def pre_outcome_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, turn in enumerate(turns):
        act = str(turn.get("dialogue_act", "")).casefold()
        if role(turn) == "agent" and act in OUTCOME_ACTS:
            return turns[:index]

    return turns[:-2] if len(turns) > 2 else turns


def extract_features(record: dict[str, Any]) -> dict[str, float]:
    turns = record.get("turns")
    if not isinstance(turns, list):
        raise ValueError(
            f"{record.get('conversation_id', 'record')}: missing turns"
        )

    prefix = pre_outcome_turns(turns)
    customer_turns = [turn for turn in prefix if role(turn) == "customer"]
    customer_token_sets = [
        tokens(str(turn.get("text", ""))) for turn in customer_turns
    ]
    initial_issue = customer_token_sets[0] if customer_token_sets else set()
    restatements = sum(
        1
        for current in customer_token_sets[1:]
        if len(current) >= 3
        and initial_issue
        and len(current & initial_issue) / len(current | initial_issue) >= 0.6
    )
    word_counts = [
        len(WORD_RE.findall(str(turn.get("text", ""))))
        for turn in customer_turns
    ]
    prefix_count = len(prefix)

    return {
        "pre_outcome_turn_count": float(prefix_count),
        "customer_turn_count": float(len(customer_turns)),
        "customer_turn_share": (
            len(customer_turns) / prefix_count if prefix_count else 0.0
        ),
        "customer_word_count": float(sum(word_counts)),
        "mean_customer_words": (
            statistics.mean(word_counts) if word_counts else 0.0
        ),
        "customer_issue_restatement_count": float(restatements),
        "customer_question_count": float(sum(
            "?" in str(turn.get("text", ""))
            or "question" in str(turn.get("dialogue_act", "")).casefold()
            for turn in customer_turns
        )),
    }


def outcome_label(record: dict[str, Any]) -> int | None:
    status = str(record.get("resolution_status", "")).casefold()
    escalated = record.get("escalated") is True
    if status in {"resolved", "closed"} and not escalated:
        return 0
    if status in {
        "unresolved", "pending", "open", "partial", "partially_resolved",
        "escalated",
    } or escalated:
        return 1
    return None


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def compare_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    resolved = [row for row in rows if row["non_resolved"] == 0]
    non_resolved = [row for row in rows if row["non_resolved"] == 1]
    comparisons = []

    for feature in FEATURES + ["total_turn_count"]:
        left = [row[feature] for row in resolved if row.get(feature) is not None]
        right = [
            row[feature] for row in non_resolved
            if row.get(feature) is not None
        ]
        if not left or not right:
            continue
        test = mannwhitneyu(left, right, alternative="two-sided")
        comparisons.append({
            "feature": feature,
            "resolved_n": len(left),
            "resolved_median": statistics.median(left),
            "non_resolved_n": len(right),
            "non_resolved_median": statistics.median(right),
            "mann_whitney_p": float(test.pvalue),
        })

    adjusted = holm_adjust([row["mann_whitney_p"] for row in comparisons])
    for row, p_value in zip(comparisons, adjusted):
        row["holm_adjusted_p"] = p_value
    return comparisons


def grouped_prediction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({row["scenario"] for row in rows})
    if len(groups) < 2:
        return {"status": "Skipped: fewer than two scenarios."}

    n_splits = min(5, len(groups))
    splitter = GroupKFold(n_splits=n_splits)
    x = [[row[name] for name in FEATURES] for row in rows]
    y = [row["non_resolved"] for row in rows]
    scenarios = [row["scenario"] for row in rows]
    predictions: list[int | None] = [None] * len(rows)
    probabilities: list[float | None] = [None] * len(rows)

    for train_indices, test_indices in splitter.split(x, y, groups=scenarios):
        if len({y[index] for index in train_indices}) < 2:
            return {
                "status": (
                    "Skipped: a scenario-held-out training fold has one class."
                )
            }

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        )
        model.fit(
            [x[index] for index in train_indices],
            [y[index] for index in train_indices],
        )
        fold_predictions = model.predict([x[i] for i in test_indices])
        fold_probabilities = model.predict_proba(
            [x[i] for i in test_indices]
        )[:, 1]
        for index, label, probability in zip(
            test_indices, fold_predictions, fold_probabilities
        ):
            predictions[index] = int(label)
            probabilities[index] = float(probability)

    valid = [
        index for index, prediction in enumerate(predictions)
        if prediction is not None
    ]
    actual = [y[index] for index in valid]
    predicted = [predictions[index] for index in valid]
    scores = [probabilities[index] for index in valid]

    result = {
        "status": "ok",
        "validation": f"{n_splits}-fold GroupKFold by scenario",
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(actual, scores))
            if len(set(actual)) == 2 else None
        ),
    }
    for index in valid:
        rows[index]["predicted_non_resolved"] = predictions[index]
        rows[index]["predicted_probability"] = probabilities[index]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/experiments/voice_heldout_dataset.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("customer_behavior_analysis")
    )
    args = parser.parse_args()

    records = load_records(args.dataset)
    rows = []
    for record in records:
        label = outcome_label(record)
        if label is None:
            continue
        turns = record.get("turns")
        if not isinstance(turns, list):
            raise ValueError(
                f"{record.get('conversation_id', 'record')}: missing turns"
            )
        rows.append({
            "conversation_id": str(record.get("conversation_id", "")),
            "scenario": str(record.get("scenario", "UNKNOWN")),
            "outcome": (
                "non_resolved_or_escalated" if label else "resolved"
            ),
            "non_resolved": label,
            "total_turn_count": len(turns),
            **extract_features(record),
            "predicted_non_resolved": None,
            "predicted_probability": None,
        })

    if not rows:
        raise ValueError("No conversations have recognized resolution labels.")

    comparisons = compare_groups(rows)
    model_result = grouped_prediction(rows)
    counts = {
        "resolved": sum(row["non_resolved"] == 0 for row in rows),
        "non_resolved_or_escalated": sum(
            row["non_resolved"] == 1 for row in rows
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)

    with (args.output / "per_conversation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with (args.output / "group_comparisons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        columns = list(comparisons[0]) if comparisons else [
            "feature", "resolved_n", "resolved_median",
            "non_resolved_n", "non_resolved_median",
            "mann_whitney_p", "holm_adjusted_p",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(comparisons)

    summary = {
        "dataset": str(args.dataset),
        "conversation_count": len(rows),
        "outcome_counts": counts,
        "group_comparisons": comparisons,
        "grouped_prediction": model_result,
        "interpretation_note": (
            "Exploratory results from synthetic scripted conversations. "
            "Restatement count is a lexical-overlap heuristic, not semantic "
            "re-ask annotation. Associations and predictive scores do not "
            "establish causality or real-customer behavior."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
