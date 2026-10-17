#!/usr/bin/env python3
"""
19_extract_research_observations.py

Build a thesis-oriented observation table from RECAP-Voice experiment outputs.

Inputs:
    handover_verification/
    acoustic_features/
    handover_capsules/
    handoff_benchmark/
    ablation_packets/

Outputs:
    research_observations/observations.csv
    research_observations/observations.json
    research_observations/observation_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_row(rows, cid, stage, observation, value, unit="", interpretation=""):
    rows.append({
        "conversation_id": cid,
        "stage": stage,
        "observation": observation,
        "value": value,
        "unit": unit,
        "interpretation": interpretation,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-dir", type=Path, default=Path("handover_verification"))
    parser.add_argument("--acoustic-dir", type=Path, default=Path("acoustic_features"))
    parser.add_argument("--capsule-dir", type=Path, default=Path("handover_capsules"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("handoff_benchmark"))
    parser.add_argument("--ablation-dir", type=Path, default=Path("ablation_packets"))
    parser.add_argument("--output", type=Path, default=Path("research_observations"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    # 13: verifier
    for path in sorted(args.verification_dir.glob("*.json")):
        if path.name == "verification_summary.json":
            continue
        report = load_json(path)
        cid = str(report.get("conversation_id") or path.stem)

        add_row(rows, cid, "13_verifier", "claims_checked",
                report.get("num_claims"), "claims",
                "Number of extracted claims audited.")
        add_row(rows, cid, "13_verifier", "verified_claims",
                report.get("verified_claims"), "claims",
                "Claims whose evidence references passed the structural verifier.")
        add_row(rows, cid, "13_verifier", "review_claims",
                report.get("review_claims"), "claims",
                "Claims requiring review.")
        add_row(rows, cid, "13_verifier", "evidence_support_rate",
                report.get("evidence_support_rate"), "ratio",
                "Structural evidence-reference support; not semantic entailment.")

        review_fields = [
            c.get("field")
            for c in report.get("claims", [])
            if c.get("verification") == "REVIEW"
        ]
        add_row(rows, cid, "13_verifier", "review_fields",
                "; ".join(review_fields), "fields",
                "Fields requiring review.")

    # 14: acoustic context
    for path in sorted(args.acoustic_dir.glob("*.json")):
        if path.name == "acoustic_features.json":
            continue
        record = load_json(path)
        cid = str(record.get("conversation_id") or path.stem)
        f = record.get("features", {})

        metrics = [
            ("duration_seconds", "seconds", "Audio duration."),
            ("silence_ratio", "ratio", "Fraction of frames below the silence threshold."),
            ("turn_count", "turns", "Turns from the selected canonical timing source."),
            ("customer_talk_ratio", "ratio", "Fraction of active turn time attributed to customer."),
            ("mean_inter_turn_gap_seconds", "seconds", "Average positive gap between consecutive turns."),
            ("silence_gaps_over_half_second", "count", "Count of pauses >= 0.5 seconds."),
            ("overlap_count", "count", "Number of overlapping turn transitions."),
            ("overlap_seconds", "seconds", "Total overlap duration."),
        ]
        for key, unit, interpretation in metrics:
            add_row(rows, cid, "14_acoustic", key, f.get(key), unit, interpretation)

        wpm = f.get("mean_words_per_minute", {})
        add_row(rows, cid, "14_acoustic", "customer_words_per_minute",
                wpm.get("customer"), "wpm",
                "Mean customer speech rate from transcript timing.")
        add_row(rows, cid, "14_acoustic", "agent_words_per_minute",
                wpm.get("agent"), "wpm",
                "Mean agent speech rate from transcript timing.")

        pauses = f.get("long_pause_observations", []) or []
        add_row(rows, cid, "14_acoustic", "long_pause_observation_count",
                len(pauses), "events",
                "Explicitly recorded long-pause interaction observations.")

    # 15: handover capsule
    for path in sorted(args.capsule_dir.glob("*.json")):
        cid = path.stem
        capsule = load_json(path)

        entities = capsule.get("entities") or []
        actions = capsule.get("actions_taken") or []
        promises = capsule.get("promises_made") or []
        unresolved = capsule.get("unresolved_items") or []
        flags = (capsule.get("uncertainty") or {}).get("review_flags") or []

        for obs, value, unit, interpretation in [
            ("entity_count", len(entities), "entities",
             "Number of promoted entity claims."),
            ("actions_count", len(actions), "actions",
             "Number of actions transferred to the next agent."),
            ("promise_count", len(promises), "promises",
             "Number of commitments transferred."),
            ("unresolved_item_count", len(unresolved), "items",
             "Number of unresolved issues transferred."),
            ("review_flag_count", len(flags), "flags",
             "Number of capsule fields requiring review."),
            ("has_next_best_action", bool(capsule.get("next_best_action")), "boolean",
             "Whether the composer generated a next step/question."),
            ("capsule_bytes",
             len(json.dumps(capsule, ensure_ascii=False).encode("utf-8")),
             "bytes", "Serialized capsule size."),
        ]:
            add_row(rows, cid, "15_capsule", obs, value, unit, interpretation)

    # 16: benchmark
    benchmark_csv = args.benchmark_dir / "handoff_results.csv"
    if benchmark_csv.exists():
        with benchmark_csv.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cid = r.get("conversation_id", "")
                cond = r.get("condition", "")
                add_row(rows, cid, f"16_benchmark:{cond}",
                        "slot_carryover_f1", float(r["slot_carryover_f1"]),
                        "F1", "Gold-state slot preservation.")
                add_row(rows, cid, f"16_benchmark:{cond}",
                        "unnecessary_reask_rate", float(r["unnecessary_reask_rate"]),
                        "ratio", "Fraction of gold slots not preserved.")
                add_row(rows, cid, f"16_benchmark:{cond}",
                        "packet_bytes", int(r["packet_bytes"]),
                        "bytes", "Serialized context size.")
                add_row(rows, cid, f"16_benchmark:{cond}",
                        "approx_tokens", int(r["approx_tokens"]),
                        "tokens", "Approximate context token count.")

    # 17: ablation/compression packet observations
    for root, stage in [
        (args.ablation_dir / "ablations", "17_ablation"),
        (args.ablation_dir / "compression", "17_compression"),
    ]:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                packet = load_json(path)
            except Exception:
                continue
            cid = path.stem
            label = path.parent.name
            size = len(json.dumps(packet, ensure_ascii=False).encode("utf-8"))
            add_row(rows, cid, f"{stage}:{label}",
                    "packet_bytes", size, "bytes",
                    "Variant packet size; handoff quality must be benchmarked separately.")

    # outputs
    csv_path = args.output / "observations.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    (args.output / "observations.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    grouped = {}
    for r in rows:
        grouped.setdefault((r["stage"], r["observation"]), []).append(r["value"])

    lines = [
        "# RECAP-Voice Research Observation Summary",
        "",
        "Measured observations from stages 13–17.",
        "This report does not convert observations into causal conclusions.",
        "",
    ]

    for (stage, observation), values in sorted(grouped.items()):
        numeric = []
        for value in values:
            try:
                numeric.append(float(value))
            except (TypeError, ValueError):
                pass

        lines.append(f"## {stage} — {observation}")
        if numeric:
            lines.append(
                f"- n = {len(numeric)}; mean = {statistics.mean(numeric):.4f}; "
                f"median = {statistics.median(numeric):.4f}"
            )
        else:
            nonempty = [str(v) for v in values if str(v).strip()]
            lines.append(f"- observed values = {', '.join(nonempty[:20])}")
        lines.append("")

    (args.output / "observation_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Wrote {len(rows)} observations to {args.output}")


if __name__ == "__main__":
    main()
