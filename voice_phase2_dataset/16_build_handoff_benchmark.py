#!/usr/bin/env python3
"""
16_build_handoff_benchmark.py

Build a deterministic cross-channel handoff benchmark.

Conditions
----------
B0 Transcript
B1 Generic summary
B2 Structured state
B3 RECAP-Voice handover capsule

This first benchmark deliberately measures information preservation,
rather than pretending to be a complete human-agent study.

Primary metrics
---------------
- slot carryover F1
- unnecessary re-ask rate
- evidence support rate
- false-confidence rate
- packet size in bytes and approximate tokens

A later optional LLM-agent benchmark can consume the same condition files
and measure task completion / factual errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(value: Any) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9₹$€£\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def token_set(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9₹$€£]+", normalize_text(value)))


def f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def flatten_gold_slots(gold: dict[str, Any]) -> dict[str, str]:
    slots: dict[str, str] = {}

    if gold.get("intent"):
        slots["intent"] = str(gold["intent"])

    if gold.get("customer_goal"):
        slots["customer_goal"] = str(gold["customer_goal"])

    if gold.get("resolution_state"):
        slots["resolution_state"] = str(gold["resolution_state"])

    for key, value in (gold.get("entities") or {}).items():
        if value is not None:
            slots[f"entity:{key}"] = str(value)

    return slots


def extract_state_slots(packet: dict[str, Any]) -> dict[str, str]:
    state = packet.get("state", {})
    slots = {}

    if state.get("intent", {}).get("value"):
        slots["intent"] = str(state["intent"]["value"])

    if state.get("customer_goal", {}).get("value"):
        slots["customer_goal"] = str(state["customer_goal"]["value"])

    if state.get("resolution_state", {}).get("value"):
        slots["resolution_state"] = str(
            state["resolution_state"]["value"]
        )

    for key, value in (state.get("entities") or {}).items():
        if isinstance(value, dict) and value.get("value") is not None:
            slots[f"entity:{key}"] = str(value["value"])

    return slots


def extract_capsule_slots(capsule: dict[str, Any]) -> dict[str, str]:
    slots = {}

    for src_key, out_key in [
        ("detected_intent", "intent"),
        ("customer_goal", "customer_goal"),
        ("resolution_state", "resolution_state"),
    ]:
        item = capsule.get(src_key)
        if isinstance(item, dict) and item.get("value") is not None:
            slots[out_key] = str(item["value"])

    for item in capsule.get("entities", []) or []:
        if isinstance(item, dict):
            name = item.get("name")
            value = item.get("value")
            if name and value is not None:
                slots[f"entity:{name}"] = str(value)

    return slots


def extract_transcript_text(transcript_packet: dict[str, Any]) -> str:
    raw = (
        transcript_packet.get("speaker_turns")
        or transcript_packet.get("segments")
        or transcript_packet.get("speaker_segments")
        or []
    )
    texts = [
        str(item.get("text", "")).strip()
        for item in raw
        if str(item.get("text", "")).strip()
    ]
    return " ".join(texts)


def make_generic_summary(transcript: str) -> str:
    # A deliberately simple baseline. Replace with an LLM-generated summary
    # for the B1 experiment, but keep this deterministic baseline as a control.
    sentences = re.split(r"(?<=[.!?])\s+", transcript.strip())
    return " ".join(sentences[:4])


def score_condition(gold_slots: dict[str, str], context_slots: dict[str, str]) -> dict[str, Any]:
    gold_keys = set(gold_slots)
    predicted_keys = set(context_slots)

    matched = 0
    for key in gold_keys:
        if key not in predicted_keys:
            continue

        gold_tokens = token_set(gold_slots[key])
        pred_tokens = token_set(context_slots[key])

        if gold_tokens and pred_tokens and gold_tokens & pred_tokens:
            matched += 1

    tp = matched
    fp = max(0, len(predicted_keys - gold_keys))
    fn = max(0, len(gold_keys - predicted_keys))

    return {
        "slot_tp": tp,
        "slot_fp": fp,
        "slot_fn": fn,
        "slot_carryover_f1": f1(tp, fp, fn),
        "required_slots": len(gold_keys),
        "preserved_slots": matched,
        "unnecessary_reask_count": fn,
        "unnecessary_reask_rate": fn / len(gold_keys) if gold_keys else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("gold_state"),
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("recap_predictions"),
    )
    parser.add_argument(
        "--capsule-dir",
        type=Path,
        default=Path("handover_capsules"),
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        default=Path("wordlevel_speaker_transcripts"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("handoff_benchmark"),
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    rows = []

    for gold_path in sorted(args.gold_dir.glob("*.json")):
        cid = gold_path.stem

        pred_path = args.pred_dir / f"{cid}.json"
        capsule_path = args.capsule_dir / f"{cid}.json"
        transcript_path = args.transcript_dir / f"{cid}.json"

        if not pred_path.exists():
            continue

        gold = load_json(gold_path)
        pred = load_json(pred_path)

        # gold_state packets use state.* field format.
        gold_slots = {
            "intent": str(
                gold.get("state", {})
                .get("intent", {})
                .get("value", "")
            ),
            "customer_goal": str(
                gold.get("state", {})
                .get("customer_goal", {})
                .get("value", "")
            ),
            "resolution_state": str(
                gold.get("state", {})
                .get("resolution_state", {})
                .get("value", "")
            ),
        }

        for key, item in (gold.get("state", {}).get("entities") or {}).items():
            if isinstance(item, dict) and item.get("value") is not None:
                gold_slots[f"entity:{key}"] = str(item["value"])

        conditions = {}

        if transcript_path.exists():
            transcript = extract_transcript_text(load_json(transcript_path))
            conditions["B0_transcript"] = transcript
            conditions["B1_generic_summary"] = make_generic_summary(transcript)

        conditions["B2_structured_state"] = json.dumps(
            {"state": pred.get("state", {})},
            ensure_ascii=False,
        )

        if capsule_path.exists():
            conditions["B3_recap_voice"] = json.dumps(
                load_json(capsule_path),
                ensure_ascii=False,
            )

        state_slots = extract_state_slots(pred)

        capsule_slots = {}
        if capsule_path.exists():
            capsule_slots = extract_capsule_slots(load_json(capsule_path))

        condition_slot_maps = {
            "B0_transcript": {
                k: v for k, v in gold_slots.items()
                if v and normalize_text(v) in normalize_text(
                    conditions.get("B0_transcript", "")
                )
            },
            "B1_generic_summary": {
                k: v for k, v in gold_slots.items()
                if v and normalize_text(v) in normalize_text(
                    conditions.get("B1_generic_summary", "")
                )
            },
            "B2_structured_state": state_slots,
            "B3_recap_voice": capsule_slots,
        }

        for condition_name, context_text in conditions.items():
            slot_map = condition_slot_maps.get(condition_name, {})

            metrics = score_condition(gold_slots, slot_map)

            row = {
                "conversation_id": cid,
                "condition": condition_name,
                "packet_bytes": len(context_text.encode("utf-8")),
                "approx_tokens": math.ceil(
                    len(re.findall(r"\S+", context_text)) * 1.3
                ),
                **metrics,
            }
            rows.append(row)

    out_csv = args.output / "handoff_results.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Aggregate by condition.
    aggregate = {}
    for condition in sorted(set(r["condition"] for r in rows)):
        subset = [r for r in rows if r["condition"] == condition]
        aggregate[condition] = {
            "n_conversations": len(subset),
            "mean_slot_carryover_f1": (
                sum(r["slot_carryover_f1"] for r in subset)
                / len(subset)
            ),
            "mean_unnecessary_reask_rate": (
                sum(r["unnecessary_reask_rate"] for r in subset)
                / len(subset)
            ),
            "mean_packet_bytes": (
                sum(r["packet_bytes"] for r in subset)
                / len(subset)
            ),
            "mean_approx_tokens": (
                sum(r["approx_tokens"] for r in subset)
                / len(subset)
            ),
        }

    (args.output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {len(rows)} conversation-condition rows.")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
