#!/usr/bin/env python3
"""Run the existing ASR → diarization → state pipeline for generated audio conditions."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


DATASET_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(DATASET_ROOT / ".env", override=False)

VARIANT_ROOT = DATASET_ROOT / "robustness" / "audio"
PIPELINE_ROOT = DATASET_ROOT / "robustness" / "pipeline"
CONDITIONS = ("clean", "telephone_8khz")
STAGES = {
    "asr": DATASET_ROOT / "05b_run_whisperx_large_v3_turbo.py",
    "diarization": DATASET_ROOT / "06_run_diarization.py",
    "speaker_transcripts": DATASET_ROOT / "07b_build_wordlevel_speaker_transcripts.py",
    "der": DATASET_ROOT / "08_evaluate_diarization.py",
    "state": DATASET_ROOT / "11_extract_state.py",
    "state_evaluation": DATASET_ROOT / "12_evaluate_state.py",
}


def run_stage(command: list[str], environment: dict[str, str], condition: str, stage: str) -> None:
    print(f"\n[{condition}] {stage}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=DATASET_ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        choices=CONDITIONS,
        help="Run the source clean baseline and/or telephone_8khz audio. Defaults to both.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PIPELINE_ROOT / "run",
        help="Must be a new or empty directory; completed results are never overwritten.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N audio files per condition (for a smoke test).",
    )
    parser.add_argument(
        "--split",
        choices=("test", "validation", "all"),
        default="test",
        help="Run a held-out split; defaults to test.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or positive")
    if sys.version_info[:2] != (3, 12):
        parser.error(
            "The speech pipeline requires Python 3.12 and the pinned dependencies "
            "from requirements_speech.txt. Run it with "
            "~/.venvs/voice-phase2-speech/bin/python, not the project .env Python."
        )
    expected_versions = {
        "torch": "2.8.0",
        "torchaudio": "2.8.0",
        "pyannote.audio": "4.0.7",
        "whisperx": "3.8.6",
    }
    mismatched = []
    for package, expected in expected_versions.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "not installed"
        if installed != expected:
            mismatched.append(f"{package} {installed} (expected {expected})")
    if mismatched:
        parser.error(
            "Incompatible speech dependencies: "
            + ", ".join(mismatched)
            + ". Install voice_phase2_dataset/requirements_speech.txt in "
            "~/.venvs/voice-phase2-speech and rerun with that environment's Python."
        )
    if not os.environ.get("HF_TOKEN", "").strip():
        parser.error("HF_TOKEN must be set to run the pyannote diarization stage")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty: {args.output}; choose a new --output path")

    conditions = args.condition or list(CONDITIONS)
    if not conditions:
        parser.error("No audio conditions selected")

    metadata_path = DATASET_ROOT / "data/emotional_conversations.json"
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        parser.error(f"Expected a conversation list in {metadata_path}")
    conversation_ids = sorted(
        str(record["conversation_id"])
        for record in records
        if record.get("conversation_id")
        and (
            args.split == "all"
            or str(record.get("split", "")).casefold() == args.split
        )
    )
    if args.limit:
        conversation_ids = conversation_ids[: args.limit]
    if not conversation_ids:
        parser.error(f"No conversation IDs found for split={args.split!r}")

    args.output.mkdir(parents=True, exist_ok=True)
    base_environment = os.environ.copy()
    base_environment["VOICE_METADATA_FILE"] = str(metadata_path)
    base_environment["VOICE_TURN_AUDIO_DIR"] = str(DATASET_ROOT / "new_generated_turns")
    base_environment["VOICE_CONVERSATION_IDS"] = ",".join(conversation_ids)

    for condition in conditions:
        audio_dir = (
            DATASET_ROOT / "conversations_audio"
            if condition == "clean"
            else VARIANT_ROOT / condition
        )
        audio_files = sorted(audio_dir.glob("*.wav"))
        available_ids = {path.stem for path in audio_files}
        missing_ids = set(conversation_ids) - available_ids
        if missing_ids:
            parser.error(
                f"Missing {condition} audio for selected calls: "
                f"{', '.join(sorted(missing_ids)[:10])}"
            )
        audio_files = [
            audio_file for audio_file in audio_files
            if audio_file.stem in conversation_ids
        ]
        if not audio_files:
            parser.error(f"No WAV audio found for condition {condition}: {audio_dir}")

        run_dir = args.output / condition
        if run_dir.exists() and any(run_dir.iterdir()):
            parser.error(f"Condition output already exists; refusing to overwrite: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        env = base_environment.copy()
        if args.limit:
            env["VOICE_MAX_CONVERSATIONS"] = str(args.limit)
        env["VOICE_AUDIO_DIR"] = str(audio_dir)
        env["VOICE_WHISPERX_DIR"] = str(run_dir / "transcripts_whisperx")
        env["VOICE_WHISPERX_TEXT_DIR"] = str(run_dir / "transcripts_whisperx_text")
        env["VOICE_WHISPERX_MODEL_DIR"] = str(DATASET_ROOT / "models/whisperx")
        env["VOICE_DIARIZATION_DIR"] = str(run_dir / "diarization")
        env["VOICE_DIARIZATION_RTTM_DIR"] = str(run_dir / "diarization_rttm")
        env["VOICE_DIARIZATION_SUMMARY"] = str(run_dir / "diarization_summary.csv")
        env["VOICE_WORDLEVEL_DIR"] = str(run_dir / "wordlevel_speaker_transcripts")
        env["VOICE_WORDLEVEL_TEXT_DIR"] = str(run_dir / "wordlevel_speaker_transcripts_text")
        env["VOICE_WORDLEVEL_SUMMARY"] = str(run_dir / "wordlevel_speaker_summary.csv")
        env["VOICE_DIARIZATION_EVALUATION_DIR"] = str(run_dir / "diarization_evaluation")

        run_stage([sys.executable, str(STAGES["asr"])], env, condition, "WhisperX ASR + alignment")
        run_stage([sys.executable, str(STAGES["diarization"])], env, condition, "two-speaker diarization")
        run_stage([sys.executable, str(STAGES["speaker_transcripts"])], env, condition, "lexical role mapping")
        run_stage([sys.executable, str(STAGES["der"])], env, condition, "DER evaluation")

        prediction_dir = run_dir / "recap_predictions"
        run_stage(
            [
                sys.executable,
                str(STAGES["state"]),
                "--input-dir",
                str(run_dir / "wordlevel_speaker_transcripts"),
                "--metadata",
                str(DATASET_ROOT / "data/emotional_conversations.json"),
                "--output-dir",
                str(prediction_dir),
            ],
            env,
            condition,
            "state extraction",
        )
        run_stage(
            [
                sys.executable,
                str(STAGES["state_evaluation"]),
                "--gold-dir",
                str(DATASET_ROOT / "gold_state"),
                "--pred-dir",
                str(prediction_dir),
                "--output-dir",
                str(run_dir / "state_evaluation"),
            ],
            env,
            condition,
            "state F1 evaluation",
        )

    print(f"\nRobustness pipeline outputs: {args.output}")


if __name__ == "__main__":
    main()
