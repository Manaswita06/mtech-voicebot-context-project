import gc
import json
import time
from pathlib import Path

import torch
import whisperx


# ============================================================
# Configuration
# ============================================================

AUDIO_DIR = Path("conversations_audio")

OUTPUT_DIR = Path("transcripts_whisperx")
OUTPUT_TEXT_DIR = Path("transcripts_whisperx_text")

MODEL_DIR = Path("models/whisperx")

MODEL_NAME = "large-v3-turbo"

LANGUAGE = "en"

# Conservative settings for your 12 GB Mac.
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# Keep this small initially.
BATCH_SIZE = 1
CHUNK_SIZE = 30

# Initial validation run.
# Change to None after validating the first 3-5 conversations.
MAX_CONVERSATIONS = None


# ============================================================
# Utility
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize whitespace in transcript text."""
    return " ".join(text.strip().split())


def save_json(path: Path, data: dict) -> None:
    """Save JSON with UTF-8 encoding."""
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_TEXT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("05B - WHISPER LARGE-V3-TURBO + WORD ALIGNMENT")
    print("=" * 80)

    print(f"Model:          {MODEL_NAME}")
    print(f"Device:         {DEVICE}")
    print(f"Compute type:   {COMPUTE_TYPE}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Chunk size:     {CHUNK_SIZE}")
    print()

    # --------------------------------------------------------
    # Find audio files
    # --------------------------------------------------------

    audio_files = sorted(
        AUDIO_DIR.glob("*.wav")
    )

    if MAX_CONVERSATIONS is not None:
        audio_files = (
            audio_files[
                :MAX_CONVERSATIONS
            ]
        )

    print(
        f"Conversations to process: "
        f"{len(audio_files)}"
    )

    if not audio_files:
        raise FileNotFoundError(
            f"No WAV files found in {AUDIO_DIR}"
        )

    # --------------------------------------------------------
    # Load WhisperX ASR model
    # --------------------------------------------------------

    print()
    print("Loading WhisperX ASR model...")
    print(
        "The first run may take time because "
        "model weights must be downloaded."
    )

    model = whisperx.load_model(
        MODEL_NAME,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language=LANGUAGE,
        download_root=str(MODEL_DIR),
        vad_method="silero",
    )

    print("ASR model loaded.")

    # --------------------------------------------------------
    # Process conversations
    # --------------------------------------------------------

    processed = 0
    failed = 0

    for audio_file in audio_files:

        conversation_id = audio_file.stem

        print()
        print("-" * 80)
        print(
            f"Processing: {conversation_id}"
        )
        print(
            f"Audio: {audio_file}"
        )

        start_time = time.time()

        try:

            # ------------------------------------------------
            # Load audio
            # ------------------------------------------------

            audio = whisperx.load_audio(
                str(audio_file)
            )

            # ------------------------------------------------
            # Stage 1: ASR
            # ------------------------------------------------

            print(
                "  [1/2] Running "
                "large-v3-turbo ASR..."
            )

            result = model.transcribe(
                audio,
                batch_size=BATCH_SIZE,
                chunk_size=CHUNK_SIZE,
                print_progress=True,
            )

            detected_language = result.get(
                "language",
                LANGUAGE,
            )

            segments_before_alignment = (
                result.get(
                    "segments",
                    [],
                )
            )

            print(
                f"  ASR segments: "
                f"{len(segments_before_alignment)}"
            )

            print(
                f"  Detected language: "
                f"{detected_language}"
            )

            # ------------------------------------------------
            # Stage 2: Word-level alignment
            # ------------------------------------------------

            print(
                "  [2/2] Performing "
                "word-level forced alignment..."
            )

            align_model = None

            align_model, align_metadata = (
                whisperx.load_align_model(
                    language_code=detected_language,
                    device=DEVICE,
                    model_dir=str(MODEL_DIR),
                )
            )

            aligned_result = whisperx.align(
                segments_before_alignment,
                align_model,
                align_metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
            )

            # ------------------------------------------------
            # Count aligned words
            # ------------------------------------------------

            total_words = 0
            aligned_words = 0

            for segment in aligned_result.get(
                "segments",
                [],
            ):

                words = segment.get(
                    "words",
                    [],
                )

                for word in words:

                    total_words += 1

                    if (
                        word.get("start")
                        is not None
                        and word.get("end")
                        is not None
                    ):
                        aligned_words += 1

            alignment_rate = (
                aligned_words / total_words
                if total_words > 0
                else 0.0
            )

            # ------------------------------------------------
            # Build full text
            # ------------------------------------------------

            full_text = normalize_text(
                " ".join(
                    segment.get(
                        "text",
                        "",
                    )
                    for segment
                    in aligned_result.get(
                        "segments",
                        [],
                    )
                )
            )

            elapsed = (
                time.time()
                - start_time
            )

            # ------------------------------------------------
            # Save JSON
            # ------------------------------------------------

            output = {
                "conversation_id": (
                    conversation_id
                ),
                "audio_file": str(
                    audio_file
                ),
                "model": MODEL_NAME,
                "language": detected_language,
                "device": DEVICE,
                "compute_type": COMPUTE_TYPE,
                "full_text": full_text,
                "num_segments": len(
                    aligned_result.get(
                        "segments",
                        [],
                    )
                ),
                "total_words": total_words,
                "aligned_words": aligned_words,
                "word_alignment_rate": (
                    alignment_rate
                ),
                "processing_time_seconds": (
                    round(
                        elapsed,
                        3,
                    )
                ),
                "segments": (
                    aligned_result.get(
                        "segments",
                        [],
                    )
                ),
            }

            output_json = (
                OUTPUT_DIR
                / f"{conversation_id}.json"
            )

            save_json(
                output_json,
                output,
            )

            # ------------------------------------------------
            # Save readable text
            # ------------------------------------------------

            output_text = (
                OUTPUT_TEXT_DIR
                / f"{conversation_id}.txt"
            )

            with open(
                output_text,
                "w",
                encoding="utf-8",
            ) as f:

                f.write(
                    f"Conversation: "
                    f"{conversation_id}\n"
                )

                f.write(
                    f"Model: "
                    f"{MODEL_NAME}\n"
                )

                f.write(
                    f"Word alignment rate: "
                    f"{alignment_rate:.2%}\n\n"
                )

                for segment in aligned_result.get(
                    "segments",
                    [],
                ):

                    start = segment.get(
                        "start",
                        0.0,
                    )

                    end = segment.get(
                        "end",
                        0.0,
                    )

                    text = normalize_text(
                        segment.get(
                            "text",
                            "",
                        )
                    )

                    f.write(
                        f"[{start:.3f}s - "
                        f"{end:.3f}s] "
                        f"{text}\n"
                    )

                    for word in segment.get(
                        "words",
                        [],
                    ):

                        word_text = word.get(
                            "word",
                            "",
                        )

                        word_start = word.get(
                            "start"
                        )

                        word_end = word.get(
                            "end"
                        )

                        if (
                            word_start
                            is not None
                            and word_end
                            is not None
                        ):

                            f.write(
                                f"    "
                                f"{word_start:.3f} - "
                                f"{word_end:.3f}: "
                                f"{word_text}\n"
                            )

            print()
            print(
                f"  Word alignment rate: "
                f"{alignment_rate:.2%}"
            )

            print(
                f"  Total words: "
                f"{total_words}"
            )

            print(
                f"  Aligned words: "
                f"{aligned_words}"
            )

            print(
                f"  Processing time: "
                f"{elapsed:.1f}s"
            )

            print(
                f"  [PASS] Saved:"
            )

            print(
                f"         {output_json}"
            )

            processed += 1

            # ------------------------------------------------
            # Free alignment model
            # ------------------------------------------------

            del align_model
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:

            print(
                f"  [FAIL] "
                f"{type(e).__name__}: {e}"
            )

            failed += 1

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del model
    gc.collect()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Processed: {processed}")
    print(f"Failed:    {failed}")
    print(
        f"JSON output: {OUTPUT_DIR}"
    )
    print(
        f"Text output: {OUTPUT_TEXT_DIR}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()