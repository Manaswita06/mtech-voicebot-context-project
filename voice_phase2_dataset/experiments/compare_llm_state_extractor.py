#!/usr/bin/env python3
"""Compare zero-shot and train-only few-shot OpenAI state extraction with stage 11."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def field_schema(value_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "value": value_schema,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_turn_ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["value", "confidence", "evidence_turn_ids"],
    }


STRING_OR_NULL = {"type": ["string", "null"]}
STATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": field_schema(STRING_OR_NULL),
        "customer_goal": field_schema(STRING_OR_NULL),
        "resolution_state": field_schema(STRING_OR_NULL),
        "escalated": field_schema({"type": ["boolean", "null"]}),
        "verification": field_schema(STRING_OR_NULL),
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "value": STRING_OR_NULL,
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_turn_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["name", "value", "confidence", "evidence_turn_ids"],
            },
        },
        "agent_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "dialogue_act": STRING_OR_NULL,
                    "text": {"type": "string"},
                    "evidence_turn_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["dialogue_act", "text", "evidence_turn_ids"],
            },
        },
        "agent_commitments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "status": {"type": "string"},
                    "evidence_turn_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "status", "evidence_turn_ids"],
            },
        },
        "unresolved_items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "item": {"type": "string"},
                    "status": {"type": "string"},
                    "evidence_turn_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["item", "status", "evidence_turn_ids"],
            },
        },
        "next_action": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "value": {"type": "string"},
                "status": {"type": "string"},
                "evidence_turn_ids": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["value", "status", "evidence_turn_ids"],
        },
    },
    "required": [
        "intent", "customer_goal", "resolution_state", "escalated", "verification",
        "entities", "agent_actions", "agent_commitments", "unresolved_items", "next_action",
    ],
}

INSTRUCTIONS = """
Extract the current customer-support state from the supplied speaker-attributed
transcript. Return only facts supported by transcript turns. Do not infer emotion,
demographics, identity, or an action that was merely promised. Distinguish
resolved, unresolved, partially resolved, and escalated outcomes. For each
non-null value, cite the turn IDs that directly support it; use an empty list
when there is no direct evidence. Confidence is an estimate, not a guarantee.
""".strip()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    raw = load_json(path)
    records = raw if isinstance(raw, list) else raw.get("conversations", raw.get("data", []))
    return {
        str(record["conversation_id"]): record
        for record in records
        if isinstance(record, dict) and record.get("conversation_id")
    }


def transcript_turns(packet: dict[str, Any]) -> list[dict[str, Any]]:
    raw_turns = packet.get("speaker_turns") or packet.get("segments") or []
    result = []
    for index, turn in enumerate(raw_turns, start=1):
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        role = str(turn.get("predicted_role") or turn.get("role") or "UNKNOWN").upper()
        result.append(
            {
                "turn_id": int(turn.get("turn_id", index)),
                "role": role,
                "text": text,
                "start": turn.get("start"),
                "end": turn.get("end"),
            }
        )
    return result


def state_values(packet: dict[str, Any]) -> dict[str, Any]:
    state = packet.get("state", {})
    result = {}
    for field in ("intent", "customer_goal", "resolution_state", "escalated", "verification"):
        item = state.get(field)
        result[field] = item.get("value") if isinstance(item, dict) else item
    result["entities"] = {
        key: item.get("value") if isinstance(item, dict) else item
        for key, item in state.get("entities", {}).items()
    }
    result["agent_actions"] = [
        {"dialogue_act": item.get("dialogue_act"), "text": item.get("text")}
        for item in state.get("agent_actions", [])
    ]
    result["agent_commitments"] = [
        {"text": item.get("text"), "status": item.get("status")}
        for item in state.get("agent_commitments", [])
    ]
    result["unresolved_items"] = [
        {"item": item.get("item"), "status": item.get("status")}
        for item in state.get("unresolved_items", [])
    ]
    next_action = state.get("next_action")
    result["next_action"] = (
        next_action.get("value") if isinstance(next_action, dict) else next_action
    )
    return result


def state_demonstration(packet: dict[str, Any]) -> dict[str, Any]:
    values = state_values(packet)

    def scored(value: Any) -> dict[str, Any]:
        return {"value": value, "confidence": 1.0, "evidence_turn_ids": []}

    return {
        **{
            key: scored(values[key])
            for key in ("intent", "customer_goal", "resolution_state", "escalated", "verification")
        },
        "entities": {key: scored(value) for key, value in values["entities"].items()},
        "agent_actions": [
            {**item, "evidence_turn_ids": []} for item in values["agent_actions"]
        ],
        "agent_commitments": [
            {**item, "evidence_turn_ids": []} for item in values["agent_commitments"]
        ],
        "unresolved_items": [
            {**item, "evidence_turn_ids": []} for item in values["unresolved_items"]
        ],
        "next_action": (
            None
            if values["next_action"] is None
            else {
                "value": values["next_action"],
                "status": "PREDICTED",
                "evidence_turn_ids": [],
            }
        ),
    }


def call_model(
    client: OpenAI,
    model: str,
    turns: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], float]:
    payload = {
        "demonstrations": examples,
        "transcript": [
            {key: turn[key] for key in ("turn_id", "role", "text")}
            for turn in turns
        ],
    }
    started = time.perf_counter()
    response = client.responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=json.dumps(payload, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "recap_voice_state",
                "strict": True,
                "schema": STATE_SCHEMA,
            }
        },
    )
    elapsed = time.perf_counter() - started
    text = (getattr(response, "output_text", "") or "").strip()
    if not text:
        raise ValueError("Empty structured output from OpenAI")
    extracted = json.loads(text)
    if not isinstance(extracted, dict):
        raise ValueError("Structured state output must be a JSON object")
    missing_fields = set(STATE_SCHEMA["required"]) - extracted.keys()
    if missing_fields:
        raise ValueError(
            "Structured state output omitted required fields: "
            + ", ".join(sorted(missing_fields))
        )
    entities = extracted.get("entities")
    if not isinstance(entities, list):
        raise ValueError("Structured state entities must be an array")
    extracted["entities"] = {
        str(item["name"]): {
            "value": item["value"],
            "confidence": item["confidence"],
            "evidence_turn_ids": item["evidence_turn_ids"],
        }
        for item in entities
    }
    return extracted, response, elapsed


def make_evidence_and_field(
    name: str,
    value: dict[str, Any],
    turn_lookup: dict[int, dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_ids = []
    for turn_id in value.get("evidence_turn_ids", []):
        turn = turn_lookup.get(int(turn_id))
        if turn is None:
            continue
        evidence_id = f"LLM-E{len(evidence) + 1:04d}"
        evidence.append(
            {
                "evidence_id": evidence_id,
                "field": f"state.{name}",
                "source_type": "asr",
                "turn_ids": [turn_id],
                "text": turn["text"],
                "start": turn.get("start"),
                "end": turn.get("end"),
            }
        )
        evidence_ids.append(evidence_id)
    return {
        "value": value.get("value"),
        "confidence": max(0.0, min(1.0, float(value.get("confidence", 0.0)))),
        "evidence_ids": evidence_ids,
        "source_type": "llm",
        "status": "PREDICTED",
    }


def make_packet(
    conversation_id: str,
    split: str,
    turns: list[dict[str, Any]],
    extracted: dict[str, Any],
    model: str,
    mode: str,
    usage: Any,
    latency: float,
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    turn_lookup = {int(turn["turn_id"]): turn for turn in turns}
    state = {
        name: make_evidence_and_field(name, extracted[name], turn_lookup, evidence)
        for name in ("intent", "customer_goal", "resolution_state", "escalated", "verification")
    }
    state["entities"] = {
        key: make_evidence_and_field(f"entities.{key}", item, turn_lookup, evidence)
        for key, item in extracted["entities"].items()
    }
    actions = []
    for action in extracted["agent_actions"]:
        action_evidence = make_evidence_and_field(
            "agent_actions", {"value": action["text"], "evidence_turn_ids": action["evidence_turn_ids"]},
            turn_lookup, evidence,
        )
        actions.append(
            {
                "dialogue_act": action.get("dialogue_act"),
                "text": action["text"],
                "evidence_ids": action_evidence["evidence_ids"],
            }
        )
    state["agent_actions"] = actions
    state["agent_commitments"] = []
    state["unresolved_items"] = []
    for commitment in extracted["agent_commitments"]:
        item = make_evidence_and_field(
            "agent_commitments", {"value": commitment["text"], "evidence_turn_ids": commitment["evidence_turn_ids"]},
            turn_lookup, evidence,
        )
        state["agent_commitments"].append(
            {"text": commitment["text"], "status": commitment["status"], "evidence_ids": item["evidence_ids"]}
        )
    for unresolved in extracted["unresolved_items"]:
        item = make_evidence_and_field(
            "unresolved_items", {"value": unresolved["item"], "evidence_turn_ids": unresolved["evidence_turn_ids"]},
            turn_lookup, evidence,
        )
        state["unresolved_items"].append(
            {"item": unresolved["item"], "status": unresolved["status"], "evidence_ids": item["evidence_ids"]}
        )
    next_action = extracted.get("next_action")
    state["next_action"] = (
        None
        if next_action is None
        else {
            "value": next_action["value"],
            "status": next_action["status"],
            "evidence_ids": make_evidence_and_field(
                "next_action",
                {"value": next_action["value"], "evidence_turn_ids": next_action["evidence_turn_ids"]},
                turn_lookup,
                evidence,
            )["evidence_ids"],
        }
    )
    usage_data = {
        key: getattr(usage, key, None)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    } if usage is not None else {}
    return {
        "schema_name": "RECAP-Voice",
        "schema_version": "1.0.0",
        "packet_type": "recap_voice",
        "conversation": {
            "conversation_id": conversation_id,
            "channel": "voice",
            "scenario": None,
            "scenario_family": None,
            "split": split,
            "num_turns": len(turns),
            "customer_speaker": None,
            "agent_speaker": None,
            "customer_gender": None,
            "agent_gender": None,
        },
        "state": state,
        "interaction": {},
        "repair_events": [],
        "evidence": evidence,
        "uncertainty": {
            "overall_status": "MEDIUM",
            "speaker_resolution": {
                "status": "RESOLVED",
                "confidence": None,
                "reason": "Predicted role labels from the input transcript; see role-mapping provenance.",
            },
            "field_uncertainty": [],
        },
        "provenance": {
            "source": "wordlevel_speaker_transcripts",
            "mode": mode,
            "generated_by": "experiments/compare_llm_state_extractor.py",
            "model": model,
            "latency_seconds": round(latency, 4),
            "usage": usage_data,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/emotional_conversations.json")
    parser.add_argument("--wordlevel-dir", type=Path, default=ROOT / "wordlevel_speaker_transcripts")
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "gold_state")
    parser.add_argument("--rule-dir", type=Path, default=ROOT / "recap_predictions")
    parser.add_argument("--output", type=Path, default=ROOT / "llm_state_comparison")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--few-shot-examples", type=int, default=3)
    parser.add_argument("--limit", type=int, default=10, help="API-cost guard; set 0 to process the complete held-out split.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit < 0 or args.few_shot_examples < 0:
        parser.error("--limit and --few-shot-examples must be zero or positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty; use a new --output: {args.output}")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        parser.error("OPENAI_API_KEY is missing; set it in voice_phase2_dataset/.env or the environment")

    conversations = load_dataset(args.dataset)
    candidates = [
        cid for cid, record in sorted(conversations.items())
        if str(record.get("split", "")).casefold() == args.split
        and (args.wordlevel_dir / f"{cid}.json").exists()
        and (args.gold_dir / f"{cid}.json").exists()
    ]
    if args.limit:
        candidates = candidates[: args.limit]
    if not candidates:
        parser.error(f"No held-out records with matching transcripts and gold states for {args.split}")

    train_ids = [
        cid for cid, record in conversations.items()
        if str(record.get("split", "")).casefold() == "train"
        and (args.gold_dir / f"{cid}.json").exists()
        and (args.wordlevel_dir / f"{cid}.json").exists()
    ]
    if not train_ids:
        parser.error("No train-split gold states are available for few-shot examples")
    rng = random.Random(args.seed)
    rng.shuffle(train_ids)
    example_records = []
    for cid in train_ids[: args.few_shot_examples]:
        transcript = transcript_turns(load_json(args.wordlevel_dir / f"{cid}.json"))
        gold = load_json(args.gold_dir / f"{cid}.json")
        example_records.append(
            {
                "transcript": [
                    {key: turn[key] for key in ("turn_id", "role", "text")}
                    for turn in transcript
                ],
                "target_state": state_demonstration(gold),
            }
        )

    client = OpenAI()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for mode, demonstrations in (("zero_shot", []), ("few_shot", example_records)):
        prediction_dir = args.output / mode / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        for index, conversation_id in enumerate(candidates, start=1):
            turns = transcript_turns(load_json(args.wordlevel_dir / f"{conversation_id}.json"))
            extracted, response, latency = call_model(
                client, args.model, turns, demonstrations
            )
            packet = make_packet(
                conversation_id,
                args.split,
                turns,
                extracted,
                args.model,
                mode,
                getattr(response, "usage", None),
                latency,
            )
            (prediction_dir / f"{conversation_id}.json").write_text(
                json.dumps(packet, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            summaries.append(
                {
                    "conversation_id": conversation_id,
                    "mode": mode,
                    "model": args.model,
                    "latency_seconds": latency,
                    "input_tokens": getattr(getattr(response, "usage", None), "input_tokens", None),
                    "output_tokens": getattr(getattr(response, "usage", None), "output_tokens", None),
                }
            )
            print(f"[{index}/{len(candidates)}] {mode}: {conversation_id}")

        evaluation_dir = args.output / mode / "evaluation"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "12_evaluate_state.py"),
                "--gold-dir", str(args.gold_dir.resolve()),
                "--pred-dir", str(prediction_dir.resolve()),
                "--output-dir", str(evaluation_dir.resolve()),
            ],
            cwd=ROOT,
            check=True,
        )
    with (args.output / "model_usage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = __import__("csv").DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Zero/few-shot state extraction comparison saved to {args.output}")
    print("Run 11_extract_state.py separately for the rule baseline on the same input transcripts.")
    print("Do not treat model confidence as calibrated; report ECE on the held-out evaluation split.")


if __name__ == "__main__":
    main()
