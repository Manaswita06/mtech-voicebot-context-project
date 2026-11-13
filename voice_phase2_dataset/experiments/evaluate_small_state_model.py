#!/usr/bin/env python3
"""Evaluate a fine-tuned seq2seq state extractor on a held-out split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def make_prompt(transcript: dict[str, Any]) -> str:
    turns = transcript.get("speaker_turns") or transcript.get("segments") or []
    text = "\n".join(
        f"{turn.get('predicted_role') or turn.get('role') or 'UNKNOWN'}: {turn.get('text', '')}"
        for turn in turns
        if str(turn.get("text", "")).strip()
    )
    return (
        "Extract a JSON customer-support state from this transcript. "
        "Do not add facts that are not stated. Transcript:\n" + text
    )


def state_packet(
    conversation_id: str,
    split: str,
    values: dict[str, Any],
    confidence: float | None,
    model_path: Path,
) -> dict[str, Any]:
    def field(value: Any) -> dict[str, Any]:
        return {
            "value": value,
            "confidence": confidence,
            "evidence_ids": [],
            "source_type": "llm",
            "status": "PREDICTED",
        }

    raw_entities = values.get("entities") or {}
    if not isinstance(raw_entities, dict):
        raw_entities = {}
    raw_actions = values.get("agent_actions") or []
    raw_commitments = values.get("agent_commitments") or []
    raw_unresolved = values.get("unresolved_items") or []
    next_action = values.get("next_action")
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
            "num_turns": 0,
            "customer_speaker": None,
            "agent_speaker": None,
            "customer_gender": None,
            "agent_gender": None,
        },
        "state": {
            "intent": field(values.get("intent")),
            "customer_goal": field(values.get("customer_goal")),
            "resolution_state": field(values.get("resolution_state")),
            "escalated": field(values.get("escalated")),
            "verification": field(values.get("verification")),
            "entities": {key: field(value) for key, value in raw_entities.items()},
            "agent_actions": [
                {
                    "dialogue_act": item.get("dialogue_act"),
                    "text": str(item.get("text", "")),
                    "evidence_ids": [],
                }
                for item in raw_actions
                if isinstance(item, dict)
            ],
            "agent_commitments": [
                {
                    "text": str(item.get("text", "")),
                    "status": str(item.get("status", "PREDICTED")),
                    "evidence_ids": [],
                }
                for item in raw_commitments
                if isinstance(item, dict)
            ],
            "unresolved_items": [
                {
                    "item": str(item.get("item", "")),
                    "status": str(item.get("status", "PREDICTED")),
                    "evidence_ids": [],
                }
                for item in raw_unresolved
                if isinstance(item, dict)
            ],
            "next_action": (
                None
                if next_action in (None, "")
                else {"value": str(next_action), "status": "PREDICTED", "evidence_ids": []}
            ),
        },
        "evidence": [],
        "provenance": {
            "generated_by": "experiments/evaluate_small_state_model.py",
            "model_path": str(model_path),
            "confidence_method": "mean generated-token probability; uncalibrated",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/emotional_conversations.json")
    parser.add_argument("--transcript-dir", type=Path, default=ROOT / "wordlevel_speaker_transcripts")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "gold_state")
    parser.add_argument("--output", type=Path, default=ROOT / "llm_state_comparison/fine_tuned")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty: {args.output}")

    raw = load_json(args.dataset)
    records = raw if isinstance(raw, list) else raw.get("conversations", [])
    selected = [
        record for record in records
        if str(record.get("split", "")).casefold() == args.split
        and (args.transcript_dir / f"{record.get('conversation_id')}.json").exists()
        and (args.gold_dir / f"{record.get('conversation_id')}.json").exists()
    ]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        parser.error(f"No held-out transcript/gold pairs for split {args.split!r}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    prediction_dir = args.output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    usage_rows = []
    malformed_rows = []

    for record in selected:
        conversation_id = str(record["conversation_id"])
        transcript = load_json(args.transcript_dir / f"{conversation_id}.json")
        inputs = tokenizer(
            make_prompt(transcript),
            max_length=args.max_input_tokens,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_output_tokens,
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        inference_latency = time.perf_counter() - inference_started
        output_text = tokenizer.decode(generated.sequences[0], skip_special_tokens=True)
        confidence = None
        if generated.scores:
            transition_scores = model.compute_transition_scores(
                generated.sequences,
                generated.scores,
                normalize_logits=True,
            )
            valid_scores = transition_scores[0]
            valid_scores = valid_scores[torch.isfinite(valid_scores)]
            if valid_scores.numel():
                confidence = float(torch.exp(valid_scores.mean()).clamp(0.0, 1.0).cpu())
        try:
            values = json.loads(output_text)
            if not isinstance(values, dict):
                raise ValueError("Model output is not a JSON object")
        except (json.JSONDecodeError, ValueError) as error:
            malformed_rows.append(
                {
                    "conversation_id": conversation_id,
                    "error": str(error),
                    "output_text": output_text,
                }
            )
            usage_rows.append(
                {
                    "conversation_id": conversation_id,
                    "generated_tokens": len(generated.scores),
                    "confidence_mean_token_probability": confidence,
                    "inference_latency_seconds": round(inference_latency, 4),
                    "status": "malformed_output",
                }
            )
            print(f"Failed to parse model output for {conversation_id}: {error}")
            continue

        packet = state_packet(conversation_id, args.split, values, confidence, args.model)
        packet["conversation"]["num_turns"] = len(
            transcript.get("speaker_turns") or transcript.get("segments") or []
        )
        (prediction_dir / f"{conversation_id}.json").write_text(
            json.dumps(packet, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        usage_rows.append(
            {
                "conversation_id": conversation_id,
                "generated_tokens": len(generated.scores),
                "confidence_mean_token_probability": confidence,
                "inference_latency_seconds": round(inference_latency, 4),
                "status": "ok",
            }
        )
        print(f"Evaluated fine-tuned model on {conversation_id}")

    with (args.output / "model_usage.csv").open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=list(usage_rows[0]))
        writer.writeheader()
        writer.writerows(usage_rows)
    if malformed_rows:
        with (args.output / "malformed_outputs.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in malformed_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    successful = sum(row["status"] == "ok" for row in usage_rows)
    if not successful:
        print(
            f"No valid JSON predictions among {len(usage_rows)} test calls; "
            f"state evaluation skipped. Raw failures: {args.output / 'malformed_outputs.jsonl'}"
        )
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "12_evaluate_state.py"),
            "--gold-dir", str(args.gold_dir.resolve()),
            "--pred-dir", str(prediction_dir.resolve()),
            "--output-dir", str((args.output / "evaluation").resolve()),
        ],
        cwd=ROOT,
        check=True,
    )
    print(f"Fine-tuned model evaluation written to {args.output}")
    print("Generated-token confidence is uncalibrated; report ECE and calibrate on validation data before interpretation.")


if __name__ == "__main__":
    main()
