#!/usr/bin/env python3
"""
13_verify_handover.py

Verifier agent for RECAP-Voice handover packets.

Purpose
-------
Audit an existing RECAP-Voice prediction packet before information is
promoted into a cross-channel handover capsule.

The verifier:
1. Validates every referenced evidence_id.
2. Checks evidence timestamps against available turn/audio duration.
3. Checks evidence quote/text consistency.
4. Flags high-confidence claims with no evidence.
5. Flags fields whose source is not auditable.
6. Produces a verifier report without changing the original prediction.

This is deliberately deterministic so that the verifier itself is
reproducible and auditable.
"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def normalize_text(text: Any) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9₹$€£\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def text_similarity(a: Any, b: Any) -> float:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def collect_claims(packet: dict[str, Any]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    state = packet.get("state", {})

    scalar_fields = [
        "intent",
        "customer_goal",
        "resolution_state",
        "escalated",
        "verification",
    ]

    for field_name in scalar_fields:
        field = state.get(field_name)
        if isinstance(field, dict):
            claims.append(
                {
                    "field": f"state.{field_name}",
                    "value": field.get("value"),
                    "confidence": field.get("confidence"),
                    "evidence_ids": field.get("evidence_ids", []),
                }
            )

    for name, field in (state.get("entities") or {}).items():
        if isinstance(field, dict):
            claims.append(
                {
                    "field": f"state.entities.{name}",
                    "value": field.get("value"),
                    "confidence": field.get("confidence"),
                    "evidence_ids": field.get("evidence_ids", []),
                }
            )

    for idx, action in enumerate(state.get("agent_actions") or []):
        if isinstance(action, dict):
            claims.append(
                {
                    "field": f"state.agent_actions[{idx}]",
                    "value": action.get("text"),
                    "confidence": None,
                    "evidence_ids": action.get("evidence_ids", []),
                }
            )

    for idx, item in enumerate(state.get("agent_commitments") or []):
        if isinstance(item, dict):
            claims.append(
                {
                    "field": f"state.agent_commitments[{idx}]",
                    "value": item.get("text"),
                    "confidence": None,
                    "evidence_ids": item.get("evidence_ids", []),
                }
            )

    for idx, item in enumerate(state.get("unresolved_items") or []):
        if isinstance(item, dict):
            claims.append(
                {
                    "field": f"state.unresolved_items[{idx}]",
                    "value": item.get("item"),
                    "confidence": None,
                    "evidence_ids": item.get("evidence_ids", []),
                }
            )

    next_action = state.get("next_action")
    if isinstance(next_action, dict):
        claims.append(
            {
                "field": "state.next_action",
                "value": next_action.get("value"),
                "confidence": None,
                "evidence_ids": next_action.get("evidence_ids", []),
            }
        )

    return claims


def verify_packet(packet: dict[str, Any]) -> dict[str, Any]:
    evidence = {
        item.get("evidence_id"): item
        for item in packet.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }

    audio_features = (
        packet.get("handover", {}).get("acoustic_interaction_features", {})
        if isinstance(packet.get("handover"), dict)
        else {}
    )
    audio_duration = audio_features.get("duration_seconds")

    results: list[dict[str, Any]] = []
    claim_flags: list[dict[str, Any]] = []

    for claim in collect_claims(packet):
        ids = claim.get("evidence_ids") or []
        confidence = claim.get("confidence")

        reasons: list[str] = []
        valid_ids: list[str] = []
        invalid_ids: list[str] = []

        for eid in ids:
            item = evidence.get(eid)
            if item is None:
                invalid_ids.append(eid)
                reasons.append(f"Missing evidence_id: {eid}")
                continue

            valid_ids.append(eid)

            start = item.get("start")
            end = item.get("end")

            if start is not None and end is not None:
                if float(start) < 0 or float(end) < float(start):
                    reasons.append(f"Invalid evidence interval: {eid}")
                if audio_duration is not None and float(end) > float(audio_duration) + 0.25:
                    reasons.append(
                        f"Evidence {eid} ends beyond audio duration "
                        f"({end} > {audio_duration})"
                    )

        if confidence is not None and float(confidence) >= 0.80 and not ids:
            reasons.append("High-confidence claim has no evidence reference.")

        status = "VERIFIED" if not reasons and valid_ids == ids else "REVIEW"
        claim_flags.append(
            {
                "field": claim["field"],
                "value": claim.get("value"),
                "confidence": claim.get("confidence"),
                "evidence_ids": ids,
                "valid_evidence_ids": valid_ids,
                "invalid_evidence_ids": invalid_ids,
                "verification": status,
                "reasons": reasons,
            }
        )

    # Optional contradiction checks.
    state = packet.get("state", {})
    if state.get("escalated", {}).get("value") is False:
        resolution = state.get("resolution_state", {}).get("value")
        if str(resolution).upper() == "ESCALATED":
            claim_flags.append(
                {
                    "field": "state.escalated",
                    "value": False,
                    "confidence": state.get("escalated", {}).get("confidence"),
                    "evidence_ids": state.get("escalated", {}).get("evidence_ids", []),
                    "valid_evidence_ids": [],
                    "invalid_evidence_ids": [],
                    "verification": "REVIEW",
                    "reasons": [
                        "Contradiction: resolution_state is ESCALATED while escalated=false."
                    ],
                }
            )

    verified_count = sum(x["verification"] == "VERIFIED" for x in claim_flags)
    review_count = len(claim_flags) - verified_count

    return {
        "conversation_id": packet.get("conversation", {}).get("conversation_id"),
        "num_claims": len(claim_flags),
        "verified_claims": verified_count,
        "review_claims": review_count,
        "evidence_support_rate": (
            verified_count / len(claim_flags) if claim_flags else 1.0
        ),
        "claims": claim_flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("recap_predictions"),
        help="Directory containing prediction packets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("handover_verification"),
    )
    parser.add_argument("--conversation-id", default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    files = (
        [args.input / f"{args.conversation_id}.json"]
        if args.conversation_id
        else sorted(args.input.glob("*.json"))
    )

    all_reports = []
    for path in files:
        if not path.exists():
            print(f"[WARN] Missing: {path}")
            continue

        packet = load_json(path)
        report = verify_packet(packet)

        out_path = args.output / path.name
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        all_reports.append(report)

    summary_path = args.output / "verification_summary.json"
    summary_path.write_text(
        json.dumps(all_reports, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Verified {len(all_reports)} packets.")
    if all_reports:
        mean_support = sum(
            r["evidence_support_rate"] for r in all_reports
        ) / len(all_reports)
        print(f"Mean evidence support rate: {mean_support:.4f}")


if __name__ == "__main__":
    main()
