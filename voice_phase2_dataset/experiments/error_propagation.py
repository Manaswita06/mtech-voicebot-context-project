#!/usr/bin/env python3
"""Compare oracle text, ASR with oracle roles, and fully predicted role labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE_EXTRACTOR = ROOT / "11_extract_state.py"
STATE_EVALUATOR = ROOT / "12_evaluate_state.py"
CONDITIONS = ("gold_transcript", "asr_gold_roles", "fully_predicted")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    records = data if isinstance(data, list) else data.get("conversations", data.get("data", []))
    if not isinstance(records, list):
        raise ValueError(f"Unexpected conversation format: {path}")
    return [item for item in records if isinstance(item, dict)]


def group_asr_words(words: list[dict[str, Any]], role_key: str) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for word in sorted(words, key=lambda item: float(item.get("start", 0.0))):
        role = str(word.get(role_key) or "").upper()
        if role not in {"CUSTOMER", "AGENT"}:
            role = None
        text = str(word.get("word") or "").strip()
        if not text:
            continue
        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        if grouped:
            previous = grouped[-1]
            gap = start - float(previous["end"])
            if role and role == previous["role"] and gap <= 1.0:
                previous["text"] += f" {text}"
                previous["end"] = end
                continue
        grouped.append(
            {
                "turn_id": len(grouped) + 1,
                "role": role,
                "predicted_role": role,
                "start": start,
                "end": end,
                "text": text,
            }
        )
    return grouped


def build_packet(conversation_id: str, role_mapping_status: str, turns: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "num_speakers_detected": 2,
        "role_mapping_status": role_mapping_status,
        "segments": turns,
        "experiment_input_provenance": {
            "role_mapping_status": role_mapping_status,
            "uses_gold_role_labels": role_mapping_status == "RESOLVED_TWO_SPEAKERS",
        },
    }


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def normalized_words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalized_words(reference), normalized_words(hypothesis)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/emotional_conversations.json")
    parser.add_argument("--wordlevel-dir", type=Path, default=ROOT / "wordlevel_speaker_transcripts")
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "gold_state")
    parser.add_argument("--output", type=Path, default=ROOT / "error_propagation")
    parser.add_argument("--split", choices=("test", "validation", "train", "all"), default="test")
    parser.add_argument("--conversation-id", action="append")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty; use a new --output: {args.output}")

    conversations = {
        str(record.get("conversation_id")): record
        for record in load_records(args.dataset)
        if record.get("conversation_id")
    }
    ids = sorted(
        cid for cid, record in conversations.items()
        if args.split == "all" or str(record.get("split", "")).casefold() == args.split
    )
    if args.conversation_id:
        requested = set(args.conversation_id)
        absent = requested - conversations.keys()
        if absent:
            parser.error(f"Unknown conversation IDs: {', '.join(sorted(absent))}")
        ids = [cid for cid in ids if cid in requested]
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        parser.error(f"No conversations found for split {args.split!r}")
    if args.split == "train":
        parser.error("Use a held-out split; train examples are used to construct the rule extractor's lexicons")

    args.output.mkdir(parents=True, exist_ok=True)
    input_dirs = {condition: args.output / "inputs" / condition for condition in CONDITIONS}
    for input_dir in input_dirs.values():
        input_dir.mkdir(parents=True, exist_ok=True)
    wer_by_id: dict[str, float] = {}

    for conversation_id in ids:
        conversation = conversations[conversation_id]
        gold_turns = [
            {
                "turn_id": turn.get("turn_id", index),
                "role": str(turn.get("role", "")).upper(),
                "predicted_role": str(turn.get("role", "")).upper(),
                "text": str(turn.get("text", "")).strip(),
            }
            for index, turn in enumerate(conversation.get("turns", []), start=1)
            if str(turn.get("text", "")).strip()
        ]
        save_json(
            input_dirs["gold_transcript"] / f"{conversation_id}.json",
            build_packet(conversation_id, "RESOLVED_TWO_SPEAKERS", gold_turns),
        )

        source_path = args.wordlevel_dir / f"{conversation_id}.json"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing word-level transcript: {source_path}")
        source = load_json(source_path)
        words = source.get("words", [])
        if not words:
            raise ValueError(f"No aligned words in {source_path}")

        asr_gold_turns = group_asr_words(words, "ground_truth_role")
        if not any(turn["role"] for turn in asr_gold_turns):
            raise ValueError(f"Ground-truth speaker roles are unavailable in {source_path}")
        save_json(
            input_dirs["asr_gold_roles"] / f"{conversation_id}.json",
            build_packet(conversation_id, "RESOLVED_TWO_SPEAKERS", asr_gold_turns),
        )

        predicted_turns = source.get("speaker_turns", [])
        predicted_turns = [
            {
                "turn_id": turn.get("turn_id", index),
                "role": turn.get("predicted_role"),
                "predicted_role": turn.get("predicted_role"),
                "speaker": turn.get("speaker"),
                "start": turn.get("start"),
                "end": turn.get("end"),
                "text": turn.get("text", ""),
            }
            for index, turn in enumerate(predicted_turns, start=1)
            if str(turn.get("text", "")).strip()
        ]
        save_json(
            input_dirs["fully_predicted"] / f"{conversation_id}.json",
            build_packet(
                conversation_id,
                str(source.get("role_mapping_status") or "UNRESOLVED"),
                predicted_turns,
            ),
        )

        reference_text = " ".join(turn["text"] for turn in gold_turns)
        hypothesis_text = " ".join(str(word.get("word", "")) for word in words)
        wer_by_id[conversation_id] = word_error_rate(reference_text, hypothesis_text)

    metric_rows = []
    for condition in CONDITIONS:
        prediction_dir = args.output / "predictions" / condition
        evaluation_dir = args.output / "evaluations" / condition
        run(
            [
                sys.executable,
                str(STATE_EXTRACTOR),
                "--input-dir",
                str(input_dirs[condition]),
                "--metadata",
                str(args.dataset.resolve()),
                "--output-dir",
                str(prediction_dir),
            ]
        )
        run(
            [
                sys.executable,
                str(STATE_EVALUATOR),
                "--gold-dir",
                str(args.gold_dir.resolve()),
                "--pred-dir",
                str(prediction_dir),
                "--output-dir",
                str(evaluation_dir),
            ]
        )
        result_rows = list(
            csv.DictReader((evaluation_dir / "per_conversation.csv").open(encoding="utf-8"))
        )
        for row in result_rows:
            conversation_id = row["conversation_id"]
            metric_rows.append(
                {
                    "conversation_id": conversation_id,
                    "condition": condition,
                    "wer": 0.0 if condition == "gold_transcript" else wer_by_id[conversation_id],
                    "state_field_macro_score": row.get("state_field_macro_score", ""),
                    "intent_accuracy": row.get("intent_correct", ""),
                    "customer_goal_f1": row.get("customer_goal_f1", ""),
                    "resolution_accuracy": row.get("resolution_correct", ""),
                    "verification_accuracy": row.get("verification_correct", ""),
                    "escalation_accuracy": row.get("escalated_correct", ""),
                }
            )

    metrics_path = args.output / "error_propagation_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"Error propagation metrics: {metrics_path}")
    print("Note: use only held-out conversations; gold-role labels are an oracle condition, not a deployable role mapper.")


if __name__ == "__main__":
    main()
