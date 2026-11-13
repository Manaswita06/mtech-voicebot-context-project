#!/usr/bin/env python3
"""Score robustness runs and quantify paired error propagation from clean audio."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


CONDITION_ORDER = (
    "clean",
    "telephone_8khz",
)
METRICS = ("wer", "DER", "state_field_macro_score")
PREFER_LOWER = {"wer", "DER"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = tokens(reference), tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1] / len(ref)


def expected_calibration_error(confidences: list[float], correct: list[bool], bins: int = 10) -> float | None:
    if not confidences:
        return None
    confidence_array = np.asarray(confidences, dtype=float)
    correctness_array = np.asarray(correct, dtype=float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        in_bin = (
            (confidence_array >= boundaries[index])
            & (confidence_array < boundaries[index + 1] if index < bins - 1 else confidence_array <= boundaries[index + 1])
        )
        if np.any(in_bin):
            ece += float(np.mean(in_bin)) * abs(
                float(np.mean(correctness_array[in_bin]))
                - float(np.mean(confidence_array[in_bin]))
            )
    return ece


def collect_run(run_dir: Path, dataset: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for condition_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        condition = condition_dir.name
        asr_dir = condition_dir / "transcripts_whisperx"
        diar_rows = {
            row["conversation_id"]: row
            for row in load_csv(condition_dir / "diarization_evaluation/diarization_evaluation_summary.csv")
            if row.get("conversation_id")
        }
        state_rows = {
            row["conversation_id"]: row
            for row in load_csv(condition_dir / "state_evaluation/per_conversation.csv")
            if row.get("conversation_id")
        }
        prediction_dir = condition_dir / "recap_predictions"
        conversation_ids = sorted(set(diar_rows) & set(state_rows))
        for conversation_id in conversation_ids:
            asr_path = asr_dir / f"{conversation_id}.json"
            if not asr_path.is_file():
                continue
            transcript = load_json(asr_path)
            gold = dataset.get(conversation_id, {})
            reference = " ".join(str(turn.get("text", "")) for turn in gold.get("turns", []))
            prediction = str(transcript.get("full_text", ""))
            rows.append(
                {
                    "run": run_dir.name,
                    "conversation_id": conversation_id,
                    "condition": condition,
                    "wer": word_error_rate(reference, prediction),
                    "DER": float(diar_rows[conversation_id]["DER"]),
                    "state_field_macro_score": float(state_rows[conversation_id]["state_field_macro_score"]),
                }
            )

    # Calibration uses state-level confidence against the gold state and is
    # written as one metric row per condition below.
    for condition_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        condition = condition_dir.name
        confidences: list[float] = []
        correctness: list[bool] = []
        prediction_dir = condition_dir / "recap_predictions"
        for pred_path in sorted(prediction_dir.glob("*.json")):
            conversation_id = pred_path.stem
            gold_path = Path(__file__).resolve().parents[1] / "gold_state" / pred_path.name
            if not gold_path.exists():
                continue
            prediction = load_json(pred_path).get("state", {})
            reference = load_json(gold_path).get("state", {})
            for field in ("intent", "resolution_state", "verification", "escalated"):
                pred_field = prediction.get(field, {})
                gold_field = reference.get(field, {})
                confidence = pred_field.get("confidence")
                if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                    continue
                confidences.append(float(np.clip(confidence, 0.0, 1.0)))
                correctness.append(pred_field.get("value") == gold_field.get("value"))
        ece = expected_calibration_error(confidences, correctness)
        if ece is not None:
            rows.append(
                {
                    "run": run_dir.name,
                    "conversation_id": "__aggregate__",
                    "condition": condition,
                    "ece": ece,
                    "calibration_count": len(confidences),
                }
            )
    return rows


def bootstrap_ci(values: list[float], seed: int, samples: int) -> tuple[float, float] | None:
    if not values:
        return None
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(data, size=(samples, len(data)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def paired_tests(rows: list[dict[str, Any]], bootstrap_seed: int, bootstrap_samples: int) -> list[dict[str, Any]]:
    metric_lookup: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("conversation_id") == "__aggregate__":
            continue
        for metric in METRICS:
            metric_lookup[(str(row["condition"]), str(row["conversation_id"]), metric)][
                str(row["run"])
            ] = float(row[metric])

    conditions = sorted(
        condition for condition, _, _ in metric_lookup if condition != "clean"
    )
    conditions = list(dict.fromkeys(conditions))
    tests = []
    for condition in conditions:
        for metric in METRICS:
            deltas_by_id = []
            ids = sorted({
                conversation_id
                for candidate, conversation_id, candidate_metric in metric_lookup
                if candidate == condition and candidate_metric == metric
                and ("clean", conversation_id, metric) in metric_lookup
            })
            for conversation_id in ids:
                clean = metric_lookup.get(("clean", conversation_id, metric), [])
                changed = metric_lookup.get((condition, conversation_id, metric), [])
                shared_runs = sorted(set(clean) & set(changed))
                if shared_runs:
                    deltas_by_id.append(
                        float(np.mean([changed[run] - clean[run] for run in shared_runs]))
                    )
            # Resample conversations, not repeated augmentation seeds.
            unique_deltas = deltas_by_id
            if unique_deltas:
                try:
                    test = wilcoxon(unique_deltas, alternative="two-sided", zero_method="pratt")
                    p_value = float(test.pvalue)
                    statistic = float(test.statistic)
                except ValueError:
                    p_value = None
                    statistic = None
            else:
                p_value = None
                statistic = None
            ci = bootstrap_ci(unique_deltas, bootstrap_seed, bootstrap_samples)
            tests.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "n_conversations": len(unique_deltas),
                    "mean_delta_vs_clean": float(np.mean(unique_deltas)) if unique_deltas else None,
                    "ci_95_low": ci[0] if ci else None,
                    "ci_95_high": ci[1] if ci else None,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_value": p_value,
                    "direction": "lower_is_better" if metric in PREFER_LOWER else "higher_is_better",
                    "bootstrap_seed": bootstrap_seed,
                }
            )

    # Holm-Bonferroni family-wise adjustment over the paired condition/metric tests.
    valid = sorted(
        ((index, row["wilcoxon_p_value"]) for index, row in enumerate(tests) if row["wilcoxon_p_value"] is not None),
        key=lambda item: item[1],
    )
    running = 0.0
    total = len(valid)
    for rank, (index, p_value) in enumerate(valid):
        adjusted = min(1.0, (total - rank) * p_value)
        running = max(running, adjusted)
        tests[index]["wilcoxon_p_holm"] = running
    for row in tests:
        row.setdefault("wilcoxon_p_holm", None)
    return tests


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_error_propagation(rows: list[dict[str, Any]], output: Path) -> None:
    averages: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("conversation_id") == "__aggregate__":
            continue
        for metric in METRICS:
            averages[(str(row["condition"]), metric)].append(float(row[metric]))

    conditions = [condition for condition in CONDITION_ORDER if any(key[0] == condition for key in averages)]
    if not conditions:
        raise ValueError("No completed condition metrics to plot")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = {"wer": "WER (lower is better)", "DER": "DER (lower is better)", "state_field_macro_score": "State F1 / macro score (higher is better)"}
    for axis, metric in zip(axes, METRICS):
        means = [float(np.mean(averages[(condition, metric)])) for condition in conditions]
        axis.plot(conditions, means, marker="o")
        axis.set_title(labels[metric])
        axis.set_ylabel("Score")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(True, alpha=0.25)
    fig.suptitle("Error propagation across audio conditions")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True, help="Pipeline run directory; repeat for independent augmentation seeds.")
    parser.add_argument("--dataset", type=Path, default=Path("data/emotional_conversations.json"))
    parser.add_argument("--output", type=Path, default=Path("robustness/analysis"))
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be >= 100")

    records = load_json(args.dataset)
    records = records if isinstance(records, list) else records.get("conversations", [])
    dataset = {
        str(record["conversation_id"]): record
        for record in records
        if isinstance(record, dict) and record.get("conversation_id")
    }
    rows: list[dict[str, Any]] = []
    for run_dir in args.run:
        if not run_dir.is_dir():
            parser.error(f"Run directory does not exist: {run_dir}")
        rows.extend(collect_run(run_dir, dataset))
    if not rows:
        parser.error("No completed per-conversation metrics found in the specified run directories")

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_conversation_metrics.csv", rows)
    tests = paired_tests(rows, args.bootstrap_seed, args.bootstrap_samples)
    write_csv(args.output / "paired_tests.csv", tests)
    plot_error_propagation(rows, args.output / "error_propagation.png")
    ece_rows = [row for row in rows if "ece" in row]
    write_csv(args.output / "calibration_ece.csv", ece_rows)
    print(f"Robustness analysis written to {args.output}")


if __name__ == "__main__":
    main()
