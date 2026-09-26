#!/usr/bin/env python3
"""
10_build_gold_state.py

Build deterministic GOLD conversational state from the synthetic dataset
metadata. This is a reference artifact for evaluating RECAP-Voice; it is
NOT an LLM-generated prediction and it must never be used to "correct"
model output during inference.

Input (current project dataset):
    data/emotional_conversations.json

Outputs:
    gold_state/<conversation_id>.json
    gold_state/gold_states.jsonl
    gold_state/gold_state_summary.csv

Optional filtering:
    --split train|validation|test
    --conversation-id conv_0001
    --max-conversations N

The script uses only Python standard library dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"
DEFAULT_METADATA = Path("data/emotional_conversations.json")
DEFAULT_OUTPUT_DIR = Path("gold_state")


# These are intentionally conservative. If the dataset grows new dialogue
# acts later, add them explicitly rather than guessing that every agent turn
# is a commitment.
COMMITMENT_ACTS = {
    "commitment",
    "promise",
    "action_commitment",
    "next_step",
    "follow_up_commitment",
}

ACTION_LIKE_ACTS = {
    "verify",
    "verification_request",
    "inform",
    "process",
    "action",
    "status_update",
    "escalate",
    "resolution",
    "commitment",
    "promise",
    "next_step",
}

CLOSING_ACTS = {"closing", "farewell", "goodbye"}


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_role(role: Any) -> str | None:
    if role is None:
        return None

    value = str(role).strip().lower()

    if value in {"customer", "client", "user"}:
        return "CUSTOMER"
    if value in {"agent", "associate", "representative"}:
        return "AGENT"

    return value.upper()


def normalize_resolution(value: Any) -> str:
    value = str(value or "unknown").strip().lower()

    mapping = {
        "resolved": "RESOLVED",
        "closed": "RESOLVED",
        "partial": "PARTIALLY_RESOLVED",
        "partially_resolved": "PARTIALLY_RESOLVED",
        "unresolved": "UNRESOLVED",
        "pending": "UNRESOLVED",
        "open": "UNRESOLVED",
    }

    return mapping.get(value, value.upper())


def normalize_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return bool(value)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value.upper() or "UNKNOWN"


def first_customer_turn(turns: list[dict[str, Any]]) -> dict[str, Any] | None:
    for turn in turns:
        if normalize_role(turn.get("role")) == "CUSTOMER":
            return turn
    return None


def last_agent_turn(turns: list[dict[str, Any]]) -> dict[str, Any] | None:
    for turn in reversed(turns):
        if normalize_role(turn.get("role")) == "AGENT":
            return turn
    return None


def agent_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        t for t in turns if normalize_role(t.get("role")) == "AGENT"
    ]


def customer_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        t for t in turns if normalize_role(t.get("role")) == "CUSTOMER"
    ]


def evidence_id(field_name: str, index: int) -> str:
    return f"E-{index:03d}-{slugify(field_name)}"


def turn_evidence_id(turn_id: Any) -> str:
    return f"TURN-{turn_id}"


def make_field(
    value: Any,
    evidence_ids: list[str] | None = None,
    *,
    confidence: float | None = 1.0,
    source_type: str = "synthetic_ground_truth",
    status: str = "GOLD",
) -> dict[str, Any]:
    return {
        "value": value,
        "confidence": confidence,
        "evidence_ids": evidence_ids or [],
        "source_type": source_type,
        "status": status,
    }


def text_contains_value(text: str, value: Any) -> bool:
    if value is None:
        return False

    value_text = str(value).strip().lower()
    text_norm = str(text).strip().lower()

    if not value_text:
        return False

    return value_text in text_norm


# ---------------------------------------------------------------------------
# Evidence generation
# ---------------------------------------------------------------------------


def build_evidence(turns: list[dict[str, Any]], entity_values: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Create auditable turn-level evidence pointers.

    Evidence is intentionally turn-based because the source dataset does not
    contain gold word-level timestamps. Downstream code can enrich these
    references with actual ASR/diarization timestamps later.
    """

    evidence: list[dict[str, Any]] = []
    field_to_evidence: dict[str, list[str]] = {}
    counter = 1

    # Intent / customer goal evidence: first customer utterance.
    first_customer = first_customer_turn(turns)
    if first_customer is not None:
        ev_id = evidence_id("intent", counter)
        counter += 1
        evidence.append(
            {
                "evidence_id": ev_id,
                "field": "state.intent",
                "source_type": "conversation_turn",
                "turn_ids": [first_customer.get("turn_id")],
                "text": first_customer.get("text", ""),
            }
        )
        field_to_evidence.setdefault("state.intent", []).append(ev_id)
        field_to_evidence.setdefault("state.customer_goal", []).append(ev_id)

    # Resolution evidence: last agent turn is the best deterministic proxy
    # available in the synthetic metadata.
    last_agent = last_agent_turn(turns)
    if last_agent is not None:
        ev_id = evidence_id("resolution_state", counter)
        counter += 1
        evidence.append(
            {
                "evidence_id": ev_id,
                "field": "state.resolution_state",
                "source_type": "conversation_turn",
                "turn_ids": [last_agent.get("turn_id")],
                "text": last_agent.get("text", ""),
            }
        )
        field_to_evidence.setdefault("state.resolution_state", []).append(ev_id)

    # Verification evidence.
    verification_turns = [
        t for t in turns
        if str(t.get("dialogue_act", "")).strip().lower()
        in {"verification_request", "verification_response"}
    ]
    if verification_turns:
        ev_id = evidence_id("verification", counter)
        counter += 1
        evidence.append(
            {
                "evidence_id": ev_id,
                "field": "state.verification",
                "source_type": "conversation_turn",
                "turn_ids": [t.get("turn_id") for t in verification_turns],
                "text": " ".join(str(t.get("text", "")) for t in verification_turns),
            }
        )
        field_to_evidence.setdefault("state.verification", []).append(ev_id)

    # Entity evidence. Match the deterministic metadata value against turn
    # text when possible. If not found, retain a metadata-only source.
    for key, value in entity_values.items():
        matching_turns = [
            t for t in turns if text_contains_value(t.get("text", ""), value)
        ]

        if matching_turns:
            ev_id = evidence_id(f"entity_{key}", counter)
            counter += 1
            evidence.append(
                {
                    "evidence_id": ev_id,
                    "field": f"state.entities.{key}",
                    "source_type": "conversation_turn",
                    "turn_ids": [t.get("turn_id") for t in matching_turns],
                    "text": " ".join(str(t.get("text", "")) for t in matching_turns),
                }
            )
            field_to_evidence.setdefault(f"state.entities.{key}", []).append(ev_id)
        else:
            ev_id = evidence_id(f"entity_{key}", counter)
            counter += 1
            evidence.append(
                {
                    "evidence_id": ev_id,
                    "field": f"state.entities.{key}",
                    "source_type": "dataset_metadata",
                    "turn_ids": [],
                    "text": None,
                    "metadata_key": key,
                }
            )
            field_to_evidence.setdefault(f"state.entities.{key}", []).append(ev_id)

    return evidence, field_to_evidence


# ---------------------------------------------------------------------------
# Gold state construction
# ---------------------------------------------------------------------------


def build_reference_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve exact gold role/dialogue-act/text labels for evaluation."""
    result = []
    for turn in turns:
        result.append(
            {
                "turn_id": turn.get("turn_id"),
                "role": normalize_role(turn.get("role")),
                "dialogue_act": turn.get("dialogue_act"),
                "text": turn.get("text", ""),
                "voice_attributes": turn.get("voice_attributes", {}),
            }
        )
    return result


def build_repair_events(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return empty gold repair events initially.

    Repair labels are a separate human-annotation task in the 8-week plan.
    We deliberately do NOT infer them from dialogue acts here because the
    current dataset does not provide gold repair annotations.
    """
    return []


def build_state(conversation: dict[str, Any], evidence_map: dict[str, list[str]]) -> dict[str, Any]:
    turns = conversation.get("turns", [])
    first_customer = first_customer_turn(turns)
    last_agent = last_agent_turn(turns)
    resolution = normalize_resolution(conversation.get("resolution_status"))

    # Customer goal is intentionally the exact first customer request. A
    # later LLM-based summarizer can generate a normalized goal for predictions.
    customer_goal = (
        first_customer.get("text", "")
        if first_customer is not None
        else None
    )

    # Explicit commitments only. Current dataset may have none; do not turn
    # every agent response into a commitment.
    commitments = []
    for turn in agent_turns(turns):
        act = str(turn.get("dialogue_act", "")).strip().lower()
        if act in COMMITMENT_ACTS:
            commitments.append(
                {
                    "text": turn.get("text", ""),
                    "status": "gold_explicit",
                    "evidence_ids": [turn_evidence_id(turn.get("turn_id"))],
                }
            )

    # Agent actions preserve what actually happened, even where it is not a
    # commitment.
    actions = []
    for turn in agent_turns(turns):
        act = str(turn.get("dialogue_act", "")).strip().lower()
        if act in ACTION_LIKE_ACTS:
            actions.append(
                {
                    "dialogue_act": turn.get("dialogue_act"),
                    "text": turn.get("text", ""),
                    "evidence_ids": [turn_evidence_id(turn.get("turn_id"))],
                }
            )

    # Deterministic unresolved item. This is intentionally minimal: the
    # metadata only exposes resolution_status, not a gold list of unresolved
    # sub-issues.
    unresolved_items = []
    if resolution in {"UNRESOLVED", "PARTIALLY_RESOLVED"} and first_customer is not None:
        unresolved_items.append(
            {
                "item": first_customer.get("text", ""),
                "evidence_ids": [turn_evidence_id(first_customer.get("turn_id"))],
                "status": "gold_primary_issue",
            }
        )

    # A next action is only emitted when the last agent turn is not simply a
    # closing/farewell and there is unresolved work.
    next_action = None
    if resolution != "RESOLVED" and last_agent is not None:
        last_act = str(last_agent.get("dialogue_act", "")).strip().lower()
        if last_act not in CLOSING_ACTS:
            next_action = {
                "value": last_agent.get("text", ""),
                "evidence_ids": [turn_evidence_id(last_agent.get("turn_id"))],
                "status": "gold_last_agent_action",
            }

    # Entities are stored as structured fields.
    entities = {}
    for key, value in (conversation.get("entities") or {}).items():
        entities[str(key)] = make_field(
            value,
            evidence_map.get(f"state.entities.{key}", []),
            confidence=1.0,
            source_type="synthetic_ground_truth",
            status="GOLD",
        )

    verification_turns = [
        t for t in turns
        if str(t.get("dialogue_act", "")).strip().lower()
        in {"verification_request", "verification_response"}
    ]
    verification_status = "NOT_MENTIONED"
    if verification_turns:
        acts = {
            str(t.get("dialogue_act", "")).strip().lower()
            for t in verification_turns
        }
        if "verification_request" in acts and "verification_response" in acts:
            verification_status = "COMPLETED"
        elif "verification_request" in acts:
            verification_status = "REQUESTED"

    return {
        "intent": make_field(
            conversation.get("intent") or conversation.get("scenario"),
            evidence_map.get("state.intent", []),
        ),
        "customer_goal": make_field(
            customer_goal,
            evidence_map.get("state.customer_goal", []),
        ),
        "resolution_state": make_field(
            resolution,
            evidence_map.get("state.resolution_state", []),
        ),
        "escalated": make_field(
            normalize_bool(conversation.get("escalated")),
            [],
            source_type="synthetic_ground_truth",
        ),
        "verification": make_field(
            verification_status,
            evidence_map.get("state.verification", []),
        ),
        "entities": entities,
        "agent_actions": actions,
        "agent_commitments": commitments,
        "unresolved_items": unresolved_items,
        "next_action": next_action,
    }


def build_interaction_gold(conversation: dict[str, Any]) -> dict[str, Any]:
    turns = conversation.get("turns", [])
    customer_count = len(customer_turns(turns))
    agent_count = len(agent_turns(turns))

    # No acoustic timing is present in the current JSON metadata, so those
    # fields remain null rather than being fabricated.
    return {
        "turn_count": len(turns),
        "customer_turn_count": customer_count,
        "agent_turn_count": agent_count,
        "customer_talk_ratio": None,
        "mean_response_latency_ms": None,
        "silence_event_count": None,
        "overlap_rate": None,
        "speaking_rate_distribution": None,
        "prosody_summary": None,
    }


def build_uncertainty_gold(conversation: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall_status": "GOLD_REFERENCE",
        "speaker_resolution": {
            "status": "KNOWN_FROM_SYNTHETIC_METADATA",
            "confidence": 1.0,
            "reason": "Gold customer/agent roles are provided by the synthetic dataset metadata.",
        },
        "field_uncertainty": [],
    }


def build_packet(conversation: dict[str, Any]) -> dict[str, Any]:
    turns = conversation.get("turns", [])
    evidence, evidence_map = build_evidence(
        turns,
        conversation.get("entities") or {},
    )

    packet = {
        "schema_name": "RECAP-Voice",
        "schema_version": SCHEMA_VERSION,
        "packet_type": "gold_reference",
        "conversation": {
            "conversation_id": conversation.get("conversation_id"),
            "channel": "voice",
            "scenario": conversation.get("scenario"),
            "scenario_family": conversation.get("scenario_family"),
            "split": conversation.get("split"),
            "num_turns": conversation.get("num_turns", len(turns)),
            "customer_speaker": conversation.get("customer_speaker"),
            "agent_speaker": conversation.get("agent_speaker"),
            "customer_gender": conversation.get("customer_gender"),
            "agent_gender": conversation.get("agent_gender"),
        },
        "state": build_state(conversation, evidence_map),
        "interaction": build_interaction_gold(conversation),
        "repair_events": build_repair_events(turns),
        "evidence": evidence,
        "uncertainty": build_uncertainty_gold(conversation),
        "reference_turns": build_reference_turns(turns),
        "provenance": {
            "source": "data/emotional_conversations.json",
            "mode": "synthetic_ground_truth",
            "generated_by": "10_build_gold_state.py",
        },
    }
    return packet


# ---------------------------------------------------------------------------
# CLI / output
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic RECAP-Voice gold states from the synthetic conversation metadata."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Path to emotional_conversations.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated gold states",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default=None,
        help="Optional dataset split filter",
    )
    parser.add_argument(
        "--conversation-id",
        default=None,
        help="Process one conversation only",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Process at most N conversations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.metadata.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {args.metadata}"
        )

    dataset = load_json(args.metadata)
    if not isinstance(dataset, list):
        raise ValueError(
            "Expected emotional_conversations.json to contain a JSON list."
        )

    filtered = []
    for conversation in dataset:
        if args.split is not None and conversation.get("split") != args.split:
            continue
        if args.conversation_id is not None and conversation.get("conversation_id") != args.conversation_id:
            continue
        filtered.append(conversation)

    if args.max_conversations is not None:
        filtered = filtered[: args.max_conversations]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = args.output_dir / "gold_states.jsonl"
    csv_path = args.output_dir / "gold_state_summary.csv"

    summary_rows = []

    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for conversation in filtered:
            packet = build_packet(conversation)
            conversation_id = packet["conversation"]["conversation_id"]

            per_file = args.output_dir / f"{conversation_id}.json"
            with per_file.open("w", encoding="utf-8") as f:
                json.dump(packet, f, indent=2, ensure_ascii=False)

            jsonl.write(json.dumps(packet, ensure_ascii=False) + "\n")

            summary_rows.append(
                {
                    "conversation_id": conversation_id,
                    "split": packet["conversation"].get("split"),
                    "scenario": packet["conversation"].get("scenario"),
                    "intent": packet["state"]["intent"]["value"],
                    "resolution_state": packet["state"]["resolution_state"]["value"],
                    "escalated": packet["state"]["escalated"]["value"],
                    "entity_count": len(packet["state"]["entities"]),
                    "agent_action_count": len(packet["state"]["agent_actions"]),
                    "commitment_count": len(packet["state"]["agent_commitments"]),
                    "unresolved_item_count": len(packet["state"]["unresolved_items"]),
                    "repair_event_count": len(packet["repair_events"]),
                    "reference_turn_count": len(packet["reference_turns"]),
                }
            )

    if summary_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Processed conversations: {len(filtered)}")
    print(f"Per-conversation gold states: {args.output_dir}")
    print(f"JSONL: {jsonl_path}")
    print(f"Summary CSV: {csv_path}")
    print()
    print("Important: repair_events are intentionally empty because the current")
    print("dataset metadata does not contain gold repair annotations. Add those")
    print("annotations later rather than inferring them silently here.")


if __name__ == "__main__":
    main()
