#!/usr/bin/env python3
"""
15_build_handover_capsules.py

Handover composer for RECAP-Voice.

Input:
- recap_predictions/<conversation_id>.json
- handover_verification/<conversation_id>.json
- acoustic_features/<conversation_id>.json (optional)

Output:
- handover_capsules/<conversation_id>.json

Design rule:
----------------
Verified semantic claims and acoustic observations remain separate.
Acoustic observations never become factual claims such as "the customer
was angry" or "the customer was dishonest."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evidence_lookup(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["evidence_id"]: item
        for item in packet.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }


def make_evidence_claim(
    packet: dict[str, Any],
    field_name: str,
    field: dict[str, Any] | None,
    verification_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(field, dict):
        return None

    value = field.get("value")
    if value is None or value == "":
        return None

    evidence_ids = field.get("evidence_ids", [])
    evidence = evidence_lookup(packet)

    spans = []
    invalid_ids = []

    for eid in evidence_ids:
        if eid not in evidence:
            invalid_ids.append(eid)
            continue
        item = evidence[eid]
        spans.append(
            {
                "evidence_id": eid,
                "quote": item.get("text"),
                "start_seconds": item.get("start"),
                "end_seconds": item.get("end"),
                "turn_ids": item.get("turn_ids", []),
            }
        )

    field_report = verification_map.get(field_name, {})
    verification = field_report.get("verification")
    if verification is None:
        verification = "REVIEW" if invalid_ids else "REFERENCES_PRESENT"

    return {
        "field": field_name,
        "value": value,
        "confidence": field.get("confidence"),
        "evidence_ids": evidence_ids,
        "evidence": spans,
        "invalid_evidence_ids": invalid_ids,
        "verification": verification,
    }


def simple_next_best_question(packet: dict[str, Any]) -> str | None:
    state = packet.get("state", {})

    unresolved = state.get("unresolved_items") or []
    if unresolved:
        first = unresolved[0]
        if isinstance(first, dict):
            item = str(first.get("item", "")).strip()
            if item:
                return item.rstrip(".") + "?"

    verification = state.get("verification", {}).get("value")
    if verification == "REQUESTED":
        return "Could you complete the required verification?"

    entities = state.get("entities") or {}
    if not entities:
        return None

    return None


def build_capsule(
    packet: dict[str, Any],
    verifier: dict[str, Any] | None,
    acoustic: dict[str, Any] | None,
) -> dict[str, Any]:

    verification_map = {
        item["field"]: item
        for item in (verifier or {}).get("claims", [])
        if isinstance(item, dict)
    }

    state = packet.get("state", {})

    customer_goal = make_evidence_claim(
        packet,
        "state.customer_goal",
        state.get("customer_goal"),
        verification_map,
    )
    intent = make_evidence_claim(
        packet,
        "state.intent",
        state.get("intent"),
        verification_map,
    )
    resolution = make_evidence_claim(
        packet,
        "state.resolution_state",
        state.get("resolution_state"),
        verification_map,
    )

    entities = []
    for key, field in (state.get("entities") or {}).items():
        if not isinstance(field, dict):
            continue
        entity_field = f"state.entities.{key}"
        verified = verification_map.get(entity_field, {})
        entities.append(
            {
                "name": key,
                "value": field.get("value"),
                "confidence": field.get("confidence"),
                "evidence_ids": field.get("evidence_ids", []),
                "verification": verified.get(
                    "verification", "REFERENCES_PRESENT"
                ),
            }
        )

    actions = []
    for idx, action in enumerate(state.get("agent_actions") or []):
        if not isinstance(action, dict):
            continue
        action_field = f"state.agent_actions[{idx}]"
        verified = verification_map.get(action_field, {})
        actions.append(
            {
                "dialogue_act": action.get("dialogue_act"),
                "text": action.get("text"),
                "evidence_ids": action.get("evidence_ids", []),
                "verification": verified.get("verification", "REFERENCES_PRESENT"),
            }
        )

    promises = []
    for idx, promise in enumerate(state.get("agent_commitments") or []):
        if not isinstance(promise, dict):
            continue
        promises.append(
            {
                "text": promise.get("text"),
                "status": promise.get("status"),
                "evidence_ids": promise.get("evidence_ids", []),
            }
        )

    unresolved = []
    for idx, item in enumerate(state.get("unresolved_items") or []):
        if not isinstance(item, dict):
            continue
        unresolved.append(
            {
                "item": item.get("item"),
                "status": item.get("status"),
                "evidence_ids": item.get("evidence_ids", []),
            }
        )

    acoustic_features = (acoustic or {}).get("features", {})
    acoustic_observations = acoustic_features.get(
        "long_pause_observations", []
    )

    # Preserve only interaction observations, never semantic conclusions.
    interaction_summary = {
        "customer_talk_ratio": acoustic_features.get("customer_talk_ratio"),
        "mean_words_per_minute": acoustic_features.get(
            "mean_words_per_minute"
        ),
        "silence_gaps_over_half_second": acoustic_features.get(
            "silence_gaps_over_half_second"
        ),
        "overlap_count": acoustic_features.get("overlap_count"),
        "overlap_seconds": acoustic_features.get("overlap_seconds"),
        "turn_count": acoustic_features.get("turn_count"),
        "turn_count_source": acoustic_features.get("turn_count_source"),
    }

    review_flags = []
    if verifier:
        for claim in verifier.get("claims", []):
            if claim.get("verification") == "REVIEW":
                review_flags.append(claim["field"])

    return {
        "conversation_id": packet.get("conversation", {}).get("conversation_id"),
        "capsule_version": "0.1.0",
        "customer_goal": customer_goal,
        "detected_intent": intent,
        "resolution_state": resolution,
        "entities": entities,
        "actions_taken": actions,
        "promises_made": promises,
        "unresolved_items": unresolved,
        "next_best_action": simple_next_best_question(packet),
        "interaction_context": interaction_summary,
        "acoustic_observations": acoustic_observations,
        "uncertainty": {
            "overall_status": packet.get("uncertainty", {}).get(
                "overall_status"
            ),
            "speaker_resolution": packet.get("uncertainty", {}).get(
                "speaker_resolution"
            ),
            "review_flags": sorted(set(review_flags)),
        },
        "handover_instruction": (
            "Use verified semantic claims as facts. Use acoustic observations "
            "only as interaction context. Ask a clarification question before "
            "acting on unresolved or review-flagged information."
        ),
        "provenance": {
            "source_prediction": packet.get("provenance", {}),
            "verifier": "13_verify_handover.py",
            "composer": "15_build_handover_capsules.py",
            "acoustic_source": (acoustic or {}).get("sources"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("recap_predictions"),
    )
    parser.add_argument(
        "--verification-dir",
        type=Path,
        default=Path("handover_verification"),
    )
    parser.add_argument(
        "--acoustic-dir",
        type=Path,
        default=Path("acoustic_features"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("handover_capsules"),
    )
    parser.add_argument("--conversation-id", default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    ids = (
        [args.conversation_id]
        if args.conversation_id
        else sorted(p.stem for p in args.pred_dir.glob("*.json"))
    )

    written = 0

    for cid in ids:
        pred_path = args.pred_dir / f"{cid}.json"
        if not pred_path.exists():
            print(f"[WARN] Missing prediction: {pred_path}")
            continue

        packet = load_json(pred_path)

        verification_path = args.verification_dir / f"{cid}.json"
        acoustic_path = args.acoustic_dir / f"{cid}.json"

        verifier = (
            load_json(verification_path)
            if verification_path.exists()
            else None
        )
        acoustic = (
            load_json(acoustic_path)
            if acoustic_path.exists()
            else None
        )

        capsule = build_capsule(packet, verifier, acoustic)

        (args.output / f"{cid}.json").write_text(
            json.dumps(capsule, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written += 1

    print(f"Built {written} handover capsules.")


if __name__ == "__main__":
    main()
