#!/usr/bin/env python3
"""Fine-tune a small text-to-text model on train-split transcript/state pairs."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def state_target(packet: dict[str, Any]) -> dict[str, Any]:
    state = packet.get("state", {})

    def value(item: Any) -> Any:
        return item.get("value") if isinstance(item, dict) else item

    return {
        "intent": value(state.get("intent")),
        "customer_goal": value(state.get("customer_goal")),
        "resolution_state": value(state.get("resolution_state")),
        "escalated": value(state.get("escalated")),
        "verification": value(state.get("verification")),
        "entities": {
            key: value(item) for key, item in state.get("entities", {}).items()
        },
        "agent_actions": [
            {"dialogue_act": item.get("dialogue_act"), "text": item.get("text")}
            for item in state.get("agent_actions", [])
        ],
        "agent_commitments": [
            {"text": item.get("text"), "status": item.get("status")}
            for item in state.get("agent_commitments", [])
        ],
        "unresolved_items": [
            {"item": item.get("item"), "status": item.get("status")}
            for item in state.get("unresolved_items", [])
        ],
        "next_action": value(state.get("next_action")),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/emotional_conversations.json"))
    parser.add_argument("--transcript-dir", type=Path, default=Path("wordlevel_speaker_transcripts"))
    parser.add_argument("--gold-dir", type=Path, default=Path("gold_state"))
    parser.add_argument("--model", default="google/flan-t5-small")
    parser.add_argument("--output", type=Path, default=Path("models/state_extractor_flan_t5_small"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs and batch-size must be positive, and learning-rate must be > 0")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty: {args.output}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dataset = load_json(args.dataset)
    dataset = dataset if isinstance(dataset, list) else dataset.get("conversations", [])
    examples = []
    for record in dataset:
        if str(record.get("split", "")).casefold() != "train":
            continue
        conversation_id = str(record.get("conversation_id", ""))
        transcript_path = args.transcript_dir / f"{conversation_id}.json"
        gold_path = args.gold_dir / f"{conversation_id}.json"
        if not transcript_path.exists() or not gold_path.exists():
            continue
        examples.append(
            (
                make_prompt(load_json(transcript_path)),
                json.dumps(state_target(load_json(gold_path)), ensure_ascii=False, separators=(",", ":")),
            )
        )
    if not examples:
        parser.error("No matching train-split transcript/gold examples found")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    added_json_tokens = tokenizer.add_tokens(["{", "}"])
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if added_json_tokens:
        model.resize_token_embeddings(len(tokenizer))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)

    training_started = time.perf_counter()
    for epoch in range(args.epochs):
        order = torch.randperm(len(examples), generator=generator).tolist()
        model.train()
        epoch_loss = 0.0
        batches = 0
        for offset in range(0, len(order), args.batch_size):
            batch = [examples[index] for index in order[offset : offset + args.batch_size]]
            prompts, targets = zip(*batch)
            inputs = tokenizer(
                list(prompts),
                max_length=args.max_input_tokens,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            target_encoding = tokenizer(
                list(targets),
                max_length=args.max_target_tokens,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            labels = target_encoding["input_ids"].to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        print(f"epoch={epoch + 1}/{args.epochs} mean_loss={epoch_loss / max(1, batches):.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    training_duration = time.perf_counter() - training_started
    (args.output / "training_metadata.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "training_examples": len(examples),
                "split": "train",
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_input_tokens": args.max_input_tokens,
                "max_target_tokens": args.max_target_tokens,
                "added_json_tokens": added_json_tokens,
                "device": str(device),
                "training_duration_seconds": round(training_duration, 3),
                "validation": "No validation split used; evaluation is reported on the held-out test split.",
                "target_format": "JSON state values only; evidence/confidence are not trained",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved fine-tuned model to {args.output}")


if __name__ == "__main__":
    main()
