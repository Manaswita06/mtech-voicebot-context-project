#!/usr/bin/env python3
"""Compare pyannote speaker labels with ECAPA clustering on its speech segments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from speechbrain.inference.speaker import EncoderClassifier


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def reference_role(start: float, end: float, turns: list[dict[str, Any]]) -> str | None:
    overlap_by_role = {"CUSTOMER": 0.0, "AGENT": 0.0}
    for turn in turns:
        role = str(turn.get("role", "")).upper()
        if role not in overlap_by_role:
            continue
        turn_start = float(turn.get("start", 0.0))
        turn_end = float(turn.get("end", turn_start))
        overlap_by_role[role] += max(
            0.0, min(end, turn_end) - max(start, turn_start)
        )
    if not any(overlap_by_role.values()):
        return None
    if overlap_by_role["CUSTOMER"] == overlap_by_role["AGENT"]:
        return None
    return max(overlap_by_role, key=overlap_by_role.get)


def speech_embedding(
    classifier: EncoderClassifier,
    samples: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    device: torch.device,
) -> np.ndarray | None:
    left = max(0, int(start * sample_rate))
    right = min(len(samples), int(end * sample_rate))
    clip = samples[left:right]
    if len(clip) < int(0.3 * sample_rate):
        return None
    if sample_rate != 16000:
        divisor = int(np.gcd(sample_rate, 16000))
        clip = resample_poly(
            clip, 16000 // divisor, sample_rate // divisor
        ).astype(np.float32)
    audio = torch.from_numpy(np.asarray(clip, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.inference_mode():
        embedding = classifier.encode_batch(audio)
    return embedding.squeeze().detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/emotional_conversations.json"))
    parser.add_argument("--audio-dir", type=Path, default=Path("conversations_audio"))
    parser.add_argument("--diarization-dir", type=Path, default=Path("diarization"))
    parser.add_argument("--wordlevel-dir", type=Path, default=Path("wordlevel_speaker_transcripts"))
    parser.add_argument("--output", type=Path, default=Path("experiments/ecapa_comparison.csv"))
    parser.add_argument("--model-cache", type=Path, default=Path("models/ecapa_voxceleb"))
    parser.add_argument("--limit", type=int, default=0, help="Zero evaluates all matching calls.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or positive")
    if args.output.exists():
        parser.error(f"Output file already exists: {args.output}")

    records = load_json(args.dataset)
    records = records if isinstance(records, list) else records.get("conversations", [])
    candidates = [
        record for record in records
        if (args.audio_dir / f"{record.get('conversation_id')}.wav").is_file()
        and (args.diarization_dir / f"{record.get('conversation_id')}.json").is_file()
        and (args.wordlevel_dir / f"{record.get('conversation_id')}.json").is_file()
    ]
    if args.limit:
        candidates = candidates[: args.limit]
    if not candidates:
        parser.error("No matching audio, diarization, and metadata records found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(args.model_cache),
        run_opts={"device": str(device)},
    )
    rows = []
    skipped = 0
    for record in candidates:
        conversation_id = str(record["conversation_id"])
        samples, sample_rate = sf.read(
            args.audio_dir / f"{conversation_id}.wav",
            dtype="float32",
            always_2d=True,
        )
        mono = np.mean(samples, axis=1)
        diarization = load_json(args.diarization_dir / f"{conversation_id}.json")
        segments = diarization.get("segments", [])
        wordlevel = load_json(args.wordlevel_dir / f"{conversation_id}.json")
        gold_turns = [
            {
                "role": word.get("ground_truth_role"),
                "start": word.get("start"),
                "end": word.get("end"),
            }
            for word in wordlevel.get("words", [])
        ]
        vectors, pyannote_labels, role_labels = [], [], []
        for segment in segments:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            role = reference_role(start, end, gold_turns)
            if role is None:
                continue
            embedding = speech_embedding(
                classifier, mono, sample_rate, start, end, device
            )
            if embedding is None:
                continue
            vectors.append(embedding)
            pyannote_labels.append(str(segment.get("speaker", "UNKNOWN")))
            role_labels.append(role)

        if len(vectors) < 2 or len(set(role_labels)) < 2:
            skipped += 1
            continue
        ecapa_labels = AgglomerativeClustering(
            n_clusters=2, metric="cosine", linkage="average"
        ).fit_predict(np.stack(vectors))
        rows.append(
            {
                "conversation_id": conversation_id,
                "n_segments": len(vectors),
                "pyannote_role_ari": adjusted_rand_score(role_labels, pyannote_labels),
                "ecapa_cluster_role_ari": adjusted_rand_score(role_labels, ecapa_labels),
                "ecapa_delta_vs_pyannote_ari": (
                    adjusted_rand_score(role_labels, ecapa_labels)
                    - adjusted_rand_score(role_labels, pyannote_labels)
                ),
                "note": "ECAPA clusters pyannote speech segments; scoring uses oracle word roles, not end-to-end diarization.",
            }
        )

    if not rows:
        parser.error("No calls had enough eligible segments for two-role clustering")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean_delta = float(np.mean([row["ecapa_delta_vs_pyannote_ari"] for row in rows]))
    print(
        f"ECAPA comparison: {len(rows)} calls, {skipped} skipped, "
        f"mean ARI delta={mean_delta:.4f}; wrote {args.output}"
    )


if __name__ == "__main__":
    main()
