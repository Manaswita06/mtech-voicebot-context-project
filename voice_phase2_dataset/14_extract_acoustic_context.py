#!/usr/bin/env python3
"""
14_extract_acoustic_context.py

Extract audio-derived interaction context for RECAP-Voice.

The output intentionally separates:
- measurable acoustic / turn-taking signals
from
- semantic facts about the conversation.

It does NOT infer emotion, honesty, intent, or mental state.

Features:
- duration
- sample rate / channels
- RMS / peak amplitude
- clipped sample ratio
- silence ratio
- turn count from the canonical timing file
- customer talk ratio
- words per minute by role
- inter-turn gaps
- overlap count / duration
- long-pause observations

The canonical turn count comes from the timing/transcript file passed to
--turn-dir. This avoids silently mixing two different segmentation schemes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_role(value: Any) -> str | None:
    text = str(value or "").upper()
    if "CUSTOMER" in text or text == "USER":
        return "CUSTOMER"
    if "AGENT" in text:
        return "AGENT"
    return None


def normalize_turns(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = (
        data.get("speaker_turns")
        or data.get("segments")
        or data.get("speaker_segments")
        or []
    )

    turns = []
    for idx, item in enumerate(raw, start=1):
        text = str(item.get("text", "")).strip()
        start = item.get("start")
        end = item.get("end")
        if not text or start is None or end is None:
            continue

        turns.append(
            {
                "turn_id": item.get("turn_id", idx),
                "role": normalize_role(
                    item.get("predicted_role") or item.get("role")
                ),
                "start": float(start),
                "end": float(end),
                "text": text,
            }
        )

    return sorted(turns, key=lambda x: (x["start"], x["end"]))


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def compute_audio_features(wav_path: Path, turns: list[dict[str, Any]]) -> dict[str, Any]:
    audio, sr = sf.read(str(wav_path), always_2d=True)
    audio_mono = np.mean(audio, axis=1)
    duration = len(audio_mono) / float(sr)

    rms = float(np.sqrt(np.mean(np.square(audio_mono)) + 1e-12))
    peak = float(np.max(np.abs(audio_mono))) if len(audio_mono) else 0.0
    clipped_ratio = float(np.mean(np.abs(audio_mono) >= 0.999))

    # Simple amplitude gate. This is intended as an interaction diagnostic,
    # not a speech activity detector benchmark.
    rms_db = 20.0 * math.log10(max(rms, 1e-12))
    silence_threshold_dbfs = -45.0

    frame = max(int(sr * 0.020), 1)
    n_frames = max(len(audio_mono) // frame, 1)
    silent_frames = 0

    for idx in range(n_frames):
        chunk = audio_mono[idx * frame : (idx + 1) * frame]
        if len(chunk) == 0:
            continue
        chunk_rms = np.sqrt(np.mean(np.square(chunk)) + 1e-12)
        chunk_db = 20.0 * math.log10(max(float(chunk_rms), 1e-12))
        silent_frames += int(chunk_db < silence_threshold_dbfs)

    silence_ratio = silent_frames / n_frames

    # Canonical turn statistics.
    total_turn_time = sum(max(0.0, t["end"] - t["start"]) for t in turns)

    customer_time = sum(
        max(0.0, t["end"] - t["start"])
        for t in turns
        if t["role"] == "CUSTOMER"
    )

    customer_talk_ratio = (
        customer_time / total_turn_time if total_turn_time else None
    )

    role_wpm: dict[str, float | None] = {"CUSTOMER": None, "AGENT": None}
    for role in ("CUSTOMER", "AGENT"):
        role_turns = [t for t in turns if t["role"] == role]
        words = sum(word_count(t["text"]) for t in role_turns)
        seconds = sum(max(0.0, t["end"] - t["start"]) for t in role_turns)
        role_wpm[role] = (
            words / (seconds / 60.0) if seconds > 0 else None
        )

    gaps: list[float] = []
    overlaps: list[float] = []
    long_pauses: list[dict[str, Any]] = []

    for prev, cur in zip(turns, turns[1:]):
        gap = float(cur["start"] - prev["end"])
        if gap >= 0:
            gaps.append(gap)
            if gap >= 0.5:
                long_pauses.append(
                    {
                        "type": "long_pause",
                        "start_seconds": round(prev["end"], 3),
                        "end_seconds": round(cur["start"], 3),
                        "duration_seconds": round(gap, 3),
                        "confidence": 1.0,
                        "interpretation": "turn-taking silence",
                    }
                )
        else:
            overlaps.append(abs(gap))

    return {
        "duration_seconds": round(duration, 4),
        "sample_rate_hz": int(sr),
        "channels": int(audio.shape[1]),
        "rms_amplitude": round(rms, 6),
        "peak_amplitude": round(peak, 6),
        "rms_dbfs": round(rms_db, 3),
        "clipped_sample_ratio": round(clipped_ratio, 6),
        "silence_threshold_dbfs": silence_threshold_dbfs,
        "silence_ratio": round(silence_ratio, 4),
        "turn_count": len(turns),
        "turn_count_source": "canonical_turn_timing_file",
        "customer_talk_ratio": (
            round(customer_talk_ratio, 4)
            if customer_talk_ratio is not None
            else None
        ),
        "mean_words_per_minute": {
            "customer": (
                round(role_wpm["CUSTOMER"], 2)
                if role_wpm["CUSTOMER"] is not None
                else None
            ),
            "agent": (
                round(role_wpm["AGENT"], 2)
                if role_wpm["AGENT"] is not None
                else None
            ),
        },
        "mean_inter_turn_gap_seconds": (
            round(float(np.mean(gaps)), 4) if gaps else None
        ),
        "silence_gaps_over_half_second": len(
            [g for g in gaps if g > 0.5]
        ),
        "overlap_count": len(overlaps),
        "overlap_seconds": round(sum(overlaps), 4),
        "long_pause_observations": long_pauses,
        "note": (
            "Acoustic and turn-taking measurements only; these observations "
            "do not prove emotion, intent, honesty, or psychological state."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=Path("conversations_audio"))
    parser.add_argument("--turn-dir", type=Path, default=Path("speaker_transcripts"))
    parser.add_argument("--output", type=Path, default=Path("acoustic_features"))
    parser.add_argument("--conversation-id", default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.conversation_id:
        ids = [args.conversation_id]
    else:
        ids = sorted(
            p.stem for p in args.audio_dir.glob("*.wav")
        )

    all_features = []

    for cid in ids:
        wav_path = args.audio_dir / f"{cid}.wav"
        timing_path = args.turn_dir / f"{cid}.json"

        if not wav_path.exists():
            print(f"[WARN] Missing audio: {wav_path}")
            continue
        if not timing_path.exists():
            print(f"[WARN] Missing timing file: {timing_path}")
            continue

        turns = normalize_turns(load_json(timing_path))
        features = compute_audio_features(wav_path, turns)

        result = {
            "conversation_id": cid,
            "features": features,
            "sources": {
                "audio": str(wav_path),
                "turn_timing": str(timing_path),
            },
        }

        out_path = args.output / f"{cid}.json"
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        all_features.append(result)

    (args.output / "acoustic_features.json").write_text(
        json.dumps(all_features, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Extracted acoustic context for {len(all_features)} conversations.")


if __name__ == "__main__":
    main()
