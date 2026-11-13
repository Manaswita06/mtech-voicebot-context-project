#!/usr/bin/env python3
"""Compute pairwise Cohen's kappa and conversation-bootstrap intervals from human ratings."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score


def read_annotations(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"conversation_id", "rater_id", "metric", "rating"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"Annotation CSV must contain columns: {', '.join(sorted(required))}"
        )
    return rows


def kappa_for_pairs(pairs: list[tuple[str, str]], weighting: str | None) -> float | None:
    if not pairs:
        return None
    left, right = zip(*pairs)
    try:
        result = cohen_kappa_score(left, right, weights=weighting)
    except (TypeError, ValueError):
        return None
    return float(result) if result == result else None


def bootstrap_interval(
    pairs: list[tuple[str, str]],
    seed: int,
    samples: int,
    weighting: str | None,
) -> tuple[float, float] | None:
    if len(pairs) < 2:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        estimate = kappa_for_pairs(sample, weighting)
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return None
    estimates.sort()
    return estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("human_study/agreement"))
    parser.add_argument("--weighting", choices=("none", "linear", "quadratic"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be >= 100")

    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    for row in read_annotations(args.annotations):
        conversation_id = row["conversation_id"].strip()
        rater_id = row["rater_id"].strip()
        metric = row["metric"].strip()
        rating = row["rating"].strip()
        if not all((conversation_id, rater_id, metric, rating)):
            raise ValueError("Blank conversation_id, rater_id, metric, or rating in annotation CSV")
        if rater_id in grouped[metric][conversation_id]:
            raise ValueError(
                f"Duplicate rating for metric={metric}, conversation={conversation_id}, rater={rater_id}"
            )
        grouped[metric][conversation_id][rater_id] = rating

    output_rows: list[dict[str, Any]] = []
    weighting = None if args.weighting == "none" else args.weighting
    for metric, conversations in sorted(grouped.items()):
        raters = sorted({rater for ratings in conversations.values() for rater in ratings})
        if len(raters) < 2:
            continue
        for rater_a, rater_b in itertools.combinations(raters, 2):
            paired = [
                (conversation_id, ratings[rater_a], ratings[rater_b])
                for conversation_id, ratings in sorted(conversations.items())
                if rater_a in ratings and rater_b in ratings
            ]
            values = [(left, right) for _, left, right in paired]
            kappa = kappa_for_pairs(values, weighting)
            interval = bootstrap_interval(
                values, args.seed, args.bootstrap_samples, weighting
            )
            output_rows.append(
                {
                    "metric": metric,
                    "rater_a": rater_a,
                    "rater_b": rater_b,
                    "n_paired_conversations": len(values),
                    "observed_agreement": (
                        sum(left == right for left, right in values) / len(values)
                        if values else None
                    ),
                    "cohen_kappa": kappa,
                    "bootstrap_95_low": interval[0] if interval else None,
                    "bootstrap_95_high": interval[1] if interval else None,
                    "weighting": args.weighting,
                    "bootstrap_seed": args.seed,
                }
            )

    if not output_rows:
        raise ValueError("No metrics had at least two raters with overlapping conversation ratings")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "cohen_kappa.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    (args.output / "cohen_kappa.json").write_text(
        json.dumps(output_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote pairwise agreement for {len(output_rows)} metric-rater pairs to {args.output}")


if __name__ == "__main__":
    main()
