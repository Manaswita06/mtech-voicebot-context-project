#!/usr/bin/env python3
"""Create held-out-voice test calls with paraphrases and non-resolved outcomes."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
from pathlib import Path
from typing import Any


GENDER_BY_VOICE = {
    "aiden": "male",
    "dylan": "male",
    "eric": "male",
    "ryan": "male",
    "uncle_fu": "male",
    "ono_anna": "female",
    "serena": "female",
    "sohee": "female",
    "vivian": "female",
}

PARAPHRASES = (
    (re.compile(r"\bI would like to know\b", re.IGNORECASE), "Could you tell me"),
    (re.compile(r"\bI need help\b", re.IGNORECASE), "Could you help me"),
    (re.compile(r"\bCan you help me\b", re.IGNORECASE), "Would you be able to help me"),
    (re.compile(r"\bI am unable to\b", re.IGNORECASE), "I can't seem to"),
    (re.compile(r"\bMy (card|payment|transaction|account)\b", re.IGNORECASE), r"I'm having trouble with my \1"),
    (re.compile(r"\bI received\b", re.IGNORECASE), "I got"),
)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else raw.get("conversations", raw.get("data", []))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty conversation list in {path}")
    return [record for record in records if isinstance(record, dict)]


def paraphrase_first_customer_turn(turns: list[dict[str, Any]], variant: int) -> None:
    customer_turn = next(
        (turn for turn in turns if str(turn.get("role", "")).casefold() == "customer"),
        None,
    )
    if customer_turn is None:
        raise ValueError("Conversation has no customer turn to paraphrase")
    text = str(customer_turn.get("text", ""))
    pattern, replacement = PARAPHRASES[variant % len(PARAPHRASES)]
    rewritten, count = pattern.subn(replacement, text, count=1)
    if count == 0:
        rewritten = f"Could you please help me with this: {text[0].lower() + text[1:]}"
    customer_turn["text"] = rewritten


def rewrite_outcome(conversation: dict[str, Any], outcome: str) -> None:
    turns = conversation.get("turns", [])
    resolution_indices = [
        index for index, turn in enumerate(turns)
        if str(turn.get("role", "")).casefold() == "agent"
        and str(turn.get("dialogue_act", "")).casefold() == "resolve"
    ]
    old_outcome_indices = resolution_indices or [
        index for index, turn in enumerate(turns)
        if str(turn.get("role", "")).casefold() == "agent"
        and str(turn.get("dialogue_act", "")).casefold() == "inform"
    ][-1:]
    replacement = (
        {
            "text": "The request is still under review, so I cannot confirm that it has been completed.",
            "dialogue_act": "unresolved",
        }
        if outcome == "unresolved"
        else {
            "text": "I have escalated the case to the specialist team for further investigation.",
            "dialogue_act": "escalation",
        }
    )
    if old_outcome_indices:
        first = old_outcome_indices[0]
        turns[first]["text"] = replacement["text"]
        turns[first]["dialogue_act"] = replacement["dialogue_act"]
        for index in reversed(old_outcome_indices[1:]):
            del turns[index]
    else:
        closing_index = next(
            (
                index for index, turn in enumerate(turns)
                if str(turn.get("dialogue_act", "")).casefold() in {"closing", "farewell"}
            ),
            len(turns),
        )
        turns.insert(
            closing_index,
            {
                "turn_id": 0,
                "role": "agent",
                **replacement,
                "voice_attributes": {
                    "emotion": "professional",
                    "emotion_intensity": 0.4,
                    "speaking_rate": "normal",
                    "prosody": "controlled",
                    "pause_before": 0,
                    "pause_after": 0,
                },
            },
        )

    customer_response = (
        "Understood. Please keep me updated when there is a decision."
        if outcome == "unresolved"
        else "Thank you. Please let me know when the specialist team responds."
    )
    for turn in turns:
        if str(turn.get("role", "")).casefold() == "customer" and str(
            turn.get("dialogue_act", "")
        ).casefold() in {"confirm", "acknowledge"}:
            turn["text"] = customer_response
            turn["dialogue_act"] = "follow_up"

    for index, turn in enumerate(turns, start=1):
        turn["turn_id"] = index
    conversation["num_turns"] = len(turns)
    conversation["resolution_status"] = outcome
    conversation["escalated"] = outcome == "escalated"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/emotional_conversations.json"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/voice_heldout_dataset.json"))
    parser.add_argument("--audio-dataset", type=Path, default=Path("data/experiments/hard_cases_audio.json"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("data/experiments/voice_heldout_metadata.csv"))
    parser.add_argument("--heldout-voices", default="aiden,ono_anna")
    parser.add_argument("--cases-per-scenario", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cases_per_scenario < 1:
        parser.error("--cases-per-scenario must be positive")
    heldout_voices = {
        voice.strip().casefold() for voice in args.heldout_voices.split(",") if voice.strip()
    }
    if len(heldout_voices) < 2 or not heldout_voices.issubset(GENDER_BY_VOICE):
        parser.error("--heldout-voices must name at least two supported voice IDs")

    source = load_dataset(args.source)
    source_by_scenario: dict[str, dict[str, Any]] = {}
    for record in source:
        scenario = str(record.get("scenario", ""))
        source_by_scenario.setdefault(scenario, record)
    if not source_by_scenario:
        parser.error("No scenarios in source dataset")

    # Make the training partition speaker-disjoint: no example featuring a
    # held-out voice remains eligible for the train-only rule lexicon.
    split_records = copy.deepcopy(source)
    for record in split_records:
        voices = {
            str(record.get("customer_speaker", "")).casefold(),
            str(record.get("agent_speaker", "")).casefold(),
        }
        if voices & heldout_voices:
            record["split"] = "test"
        record["voice_holdout_group"] = (
            "heldout_test" if voices & heldout_voices else "train_eligible"
        )

    rng = random.Random(args.seed)
    hard_cases = []
    for scenario_index, (scenario, base) in enumerate(sorted(source_by_scenario.items())):
        for case_index in range(args.cases_per_scenario):
            case = copy.deepcopy(base)
            serial = scenario_index * args.cases_per_scenario + case_index + 1
            case["conversation_id"] = f"hard_{serial:04d}"
            case["split"] = "test"
            case["voice_holdout_group"] = "heldout_test"
            case["customer_speaker"] = sorted(heldout_voices)[0]
            case["agent_speaker"] = sorted(heldout_voices)[1]
            case["customer_gender"] = GENDER_BY_VOICE[case["customer_speaker"]]
            case["agent_gender"] = GENDER_BY_VOICE[case["agent_speaker"]]
            outcome = ("unresolved", "escalated")[case_index % 2]
            rewrite_outcome(case, outcome)
            paraphrase_first_customer_turn(case["turns"], serial + rng.randrange(len(PARAPHRASES)))
            hard_cases.append(case)

    all_records = split_records + hard_cases
    write_json(args.output, all_records)
    write_jsonl(args.output.with_suffix(".jsonl"), all_records)
    write_json(args.audio_dataset, hard_cases)
    args.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("conversation_id", "scenario", "split", "resolution_status", "escalated", "customer_speaker", "agent_speaker"),
        )
        writer.writeheader()
        for record in all_records:
            writer.writerow({
                key: record.get(key, "")
                for key in writer.fieldnames
            })
    print(
        f"Saved {len(all_records)} training/evaluation records and {len(hard_cases)} "
        f"hard held-out-voice audio records; held-out voices={','.join(sorted(heldout_voices))}"
    )


if __name__ == "__main__":
    main()
