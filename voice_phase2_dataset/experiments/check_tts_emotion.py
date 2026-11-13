#!/usr/bin/env python3
"""Audit whether generated turn audio resembles the requested coarse emotion class."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from transformers import pipeline


DEFAULT_MODEL = "superb/wav2vec2-base-superb-er"
LABEL_TO_COARSE = {
    "ang": "angry",
    "anger": "angry",
    "angry": "angry",
    "frustration": "angry",
    "frustrated": "angry",
    "hap": "happy",
    "happy": "happy",
    "happiness": "happy",
    "excited": "happy",
    "sad": "sad",
    "sadness": "sad",
    "neu": "neutral",
    "neutral": "neutral",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else raw.get("conversations", raw.get("data", []))
    if not isinstance(records, list):
        raise ValueError(f"Unexpected dataset format in {path}")
    return [item for item in records if isinstance(item, dict)]


def audio_for_turn(root: Path, conversation_id: str, turn_id: Any, role: str) -> Path | None:
    turn_dir = root / conversation_id
    try:
        number = int(turn_id)
    except (TypeError, ValueError):
        number = None
    patterns = [f"turn_{turn_id}_{role}_*.wav"]
    if number is not None:
        patterns.extend((f"turn_{number:03d}_{role}_*.wav", f"turn_{number:04d}_{role}_*.wav"))
    for pattern in patterns:
        matches = sorted(turn_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_mono_audio(path: Path, target_rate: int = 16000) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1)
    if sample_rate != target_rate:
        divisor = np.gcd(sample_rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, sample_rate // divisor)
    return np.asarray(mono, dtype=np.float32)


def coarse_label(label: str) -> str | None:
    normalized = label.casefold().strip().replace("_", " ").replace("-", " ")
    return LABEL_TO_COARSE.get(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/emotional_conversations.json"))
    parser.add_argument("--turn-audio-dir", type=Path, default=Path("new_generated_turns"))
    parser.add_argument("--output", type=Path, default=Path("emotion_audit"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conversation-id")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", choices=("test", "validation", "train", "all"), default="test")
    parser.add_argument("--device", type=int, default=-1, help="Transformers pipeline device: -1 for CPU, 0 for first GPU.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or positive")

    records = load_records(args.dataset)
    selected = [
        record for record in records
        if args.split == "all" or str(record.get("split", "")).casefold() == args.split
    ]
    if args.conversation_id:
        selected = [
            record for record in selected
            if record.get("conversation_id") == args.conversation_id
        ]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        parser.error(f"No conversations selected for split={args.split!r}")

    classifier = pipeline(
        "audio-classification",
        model=args.model,
        device=args.device,
    )
    rows = []
    for conversation in selected:
        conversation_id = str(conversation.get("conversation_id"))
        for turn in conversation.get("turns", []):
            attributes = turn.get("voice_attributes") or {}
            requested = str(attributes.get("emotion", "")).strip()
            if not requested:
                continue
            role = str(turn.get("role", "")).casefold()
            audio_path = audio_for_turn(
                args.turn_audio_dir, conversation_id, turn.get("turn_id"), role
            )
            if audio_path is None:
                rows.append(
                    {
                        "conversation_id": conversation_id,
                        "turn_id": turn.get("turn_id"),
                        "requested_emotion": requested,
                        "audio_path": "",
                        "status": "missing_audio",
                        "predicted_label": "",
                        "confidence": "",
                        "requested_coarse_class": coarse_label(requested) or "",
                        "predicted_coarse_class": "",
                        "coarse_match": "",
                    }
                )
                continue
            try:
                audio = load_mono_audio(audio_path)
                predictions = classifier(
                    {"raw": audio, "sampling_rate": 16000},
                    top_k=None,
                )
                best = max(predictions, key=lambda item: float(item["score"]))
                predicted = str(best["label"])
                requested_coarse = coarse_label(requested)
                predicted_coarse = coarse_label(predicted)
                comparable = requested_coarse is not None and predicted_coarse is not None
                rows.append(
                    {
                        "conversation_id": conversation_id,
                        "turn_id": turn.get("turn_id"),
                        "requested_emotion": requested,
                        "audio_path": str(audio_path),
                        "status": "ok",
                        "predicted_label": predicted,
                        "confidence": float(best["score"]),
                        "requested_coarse_class": requested_coarse or "",
                        "predicted_coarse_class": predicted_coarse or "",
                        "coarse_match": (
                            int(requested_coarse == predicted_coarse) if comparable else ""
                        ),
                    }
                )
            except (OSError, ValueError, RuntimeError) as error:
                rows.append(
                    {
                        "conversation_id": conversation_id,
                        "turn_id": turn.get("turn_id"),
                        "requested_emotion": requested,
                        "audio_path": str(audio_path),
                        "status": "inference_error",
                        "predicted_label": "",
                        "confidence": "",
                        "requested_coarse_class": coarse_label(requested) or "",
                        "predicted_coarse_class": "",
                        "coarse_match": "",
                        "error": str(error),
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "turn_emotion_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row.get("coarse_match") != ""]
    summary = {
        "model": args.model,
        "requested_split": args.split,
        "turns_scored": len(rows),
        "coarse_comparable_turns": len(valid),
        "coarse_match_rate": (
            sum(int(row["coarse_match"]) for row in valid) / len(valid)
            if valid else None
        ),
        "predicted_label_counts": dict(Counter(
            row["predicted_label"] for row in rows if row.get("predicted_label")
        )),
        "interpretation": (
            "This pretrained classifier's coarse label match is only a screening signal. "
            "It does not validate fine-grained requested styles such as anxious, reassuring, "
            "professional, or emotion intensity."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Emotion audit saved to {args.output}")


if __name__ == "__main__":
    main()
