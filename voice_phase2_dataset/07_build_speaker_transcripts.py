import csv
import json
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# Configuration
# ============================================================

TRANSCRIPTS_DIR = Path("transcripts")
DIARIZATION_DIR = Path("diarization")
DATASET_FILE = Path("data/emotional_conversations.json")

# Original individual turn audio files.
# These are required to reconstruct the ground-truth timing
# of each conversation turn.
TURN_AUDIO_DIR = Path("new_generated_turns")

OUTPUT_DIR = Path("speaker_transcripts")
OUTPUT_TEXT_DIR = Path("speaker_transcripts_text")

SUMMARY_FILE = Path("speaker_transcript_summary.csv")

# Initial test.
# Set to None after validating the first few conversations.
MAX_CONVERSATIONS = None


# ============================================================
# JSON / file utilities
# ============================================================

def load_json(path: Path):
    """Load JSON from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_wav_duration_seconds(path: Path) -> float:
    """
    Return WAV duration in seconds using the standard library.
    """
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()

        if sample_rate <= 0:
            raise ValueError(
                f"Invalid sample rate in WAV file: {path}"
            )

        return frames / sample_rate


def normalize_text(text: str) -> str:
    """Normalize transcript text."""
    return " ".join(text.strip().split())


# ============================================================
# Temporal overlap utilities
# ============================================================

def calculate_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    """
    Calculate the temporal overlap between two intervals.
    """
    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)

    if overlap_end <= overlap_start:
        return 0.0

    return overlap_end - overlap_start


def find_best_diarization_speaker(
    start: float,
    end: float,
    diarization_segments: List[dict],
) -> Tuple[Optional[str], float]:
    """
    Find the diarization speaker with the greatest temporal
    overlap with an ASR segment.
    """

    overlap_by_speaker: Dict[str, float] = {}

    for segment in diarization_segments:
        diar_start = float(segment["start"])
        diar_end = float(segment["end"])
        speaker = segment["speaker"]

        overlap = calculate_overlap(
            start,
            end,
            diar_start,
            diar_end,
        )

        overlap_by_speaker[speaker] = (
            overlap_by_speaker.get(speaker, 0.0)
            + overlap
        )

    if not overlap_by_speaker:
        return None, 0.0

    best_speaker = max(
        overlap_by_speaker,
        key=overlap_by_speaker.get,
    )

    return (
        best_speaker,
        overlap_by_speaker[best_speaker],
    )


def find_best_ground_truth_turn(
    start: float,
    end: float,
    ground_truth_turns: List[dict],
) -> Tuple[Optional[dict], float]:
    """
    Find the ground-truth turn with the largest temporal overlap
    with an ASR segment.
    """

    best_turn = None
    best_overlap = 0.0

    for turn in ground_truth_turns:
        overlap = calculate_overlap(
            start,
            end,
            float(turn["start"]),
            float(turn["end"]),
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_turn = turn

    return best_turn, best_overlap


# ============================================================
# Conversation metadata
# ============================================================

def load_conversations() -> Dict[str, dict]:
    """
    Load all generated conversations and index them by
    conversation_id.
    """

    conversations = load_json(DATASET_FILE)

    result = {}

    for conversation in conversations:
        conversation_id = conversation["conversation_id"]
        result[conversation_id] = conversation

    return result


# ============================================================
# Ground-truth turn timing
# ============================================================

def find_turn_audio_file(
    conversation_id: str,
    turn_id: int,
    role: str,
) -> Optional[Path]:
    """
    Find the generated WAV file for a particular conversation turn.

    Expected naming pattern:

        turn_<turn_id>_<role>_<speaker>.wav
    """

    conversation_dir = (
        TURN_AUDIO_DIR / conversation_id
    )

    if not conversation_dir.exists():
        return None

    role = role.lower()

    patterns = [
        f"turn_{turn_id}_{role}_*.wav",
        f"turn_{turn_id}_{role}_*.WAV",
    ]

    matches = []

    for pattern in patterns:
        matches.extend(
            conversation_dir.glob(pattern)
        )

    matches = sorted(set(matches))

    if not matches:
        return None

    if len(matches) > 1:
        print(
            f"  [WARNING] Multiple audio files found for "
            f"{conversation_id}, turn {turn_id}:"
        )

        for match in matches:
            print(f"             {match}")

        print(
            "             Using the first matching file."
        )

    return matches[0]


def build_ground_truth_turn_timing(
    conversation: dict,
) -> List[dict]:
    """
    Reconstruct the exact sequential timing of the original
    generated conversation turns using the individual turn WAVs.

    The merged conversation was created by concatenating these
    turn WAVs, so cumulative WAV durations provide the expected
    turn boundaries.
    """

    conversation_id = conversation["conversation_id"]

    turns = conversation["turns"]

    ground_truth_turns = []

    current_time = 0.0

    for turn in turns:
        turn_id = int(turn["turn_id"])
        role = turn["role"].lower()

        audio_file = find_turn_audio_file(
            conversation_id,
            turn_id,
            role,
        )

        if audio_file is None:
            raise FileNotFoundError(
                f"Could not find turn audio for "
                f"{conversation_id}, turn {turn_id}, "
                f"role={role}"
            )

        duration = get_wav_duration_seconds(
            audio_file
        )

        start = current_time
        end = current_time + duration

        ground_truth_turns.append(
            {
                "turn_id": turn_id,
                "role": role.upper(),
                "dialogue_act": turn.get(
                    "dialogue_act"
                ),
                "text": turn.get("text", ""),
                "voice": (
                    conversation["customer_speaker"]
                    if role == "customer"
                    else conversation["agent_speaker"]
                ),
                "start": start,
                "end": end,
                "duration": duration,
                "audio_file": str(audio_file),
            }
        )

        current_time = end

    return ground_truth_turns


# ============================================================
# Speaker-role mapping
# ============================================================

def build_speaker_role_mapping(
    diarization_segments: List[dict],
    ground_truth_turns: List[dict],
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Map pyannote speaker labels to CUSTOMER / AGENT.

    The mapping is determined by total temporal overlap between
    diarization speakers and the known ground-truth customer/agent
    turn intervals.

    For two detected speakers, both possible assignments are
    evaluated and the assignment with the larger total overlap
    is selected.

    For one detected speaker, no role is fabricated because one
    speaker cannot reliably represent both customer and agent.
    """

    speakers = sorted(
        set(
            segment["speaker"]
            for segment in diarization_segments
        )
    )

    overlap_matrix: Dict[str, Dict[str, float]] = {}

    for speaker in speakers:

        overlap_matrix[speaker] = {
            "CUSTOMER": 0.0,
            "AGENT": 0.0,
        }

        for diar_segment in diarization_segments:

            if diar_segment["speaker"] != speaker:
                continue

            diar_start = float(
                diar_segment["start"]
            )
            diar_end = float(
                diar_segment["end"]
            )

            for turn in ground_truth_turns:

                overlap = calculate_overlap(
                    diar_start,
                    diar_end,
                    float(turn["start"]),
                    float(turn["end"]),
                )

                if turn["role"] in (
                    "CUSTOMER",
                    "AGENT",
                ):
                    overlap_matrix[speaker][
                        turn["role"]
                    ] += overlap

    # --------------------------------------------------------
    # No speakers
    # --------------------------------------------------------

    if len(speakers) == 0:
        return {}, {}

    # --------------------------------------------------------
    # One speaker
    # --------------------------------------------------------

    if len(speakers) == 1:

        speaker = speakers[0]

        customer_overlap = overlap_matrix[
            speaker
        ]["CUSTOMER"]

        agent_overlap = overlap_matrix[
            speaker
        ]["AGENT"]

        # Do not claim this is CUSTOMER or AGENT.
        # The speaker cluster covers both ground-truth roles.
        mapping = {
            speaker: "UNRESOLVED"
        }

        diagnostics = {
            "customer_overlap": customer_overlap,
            "agent_overlap": agent_overlap,
            "total_overlap": (
                customer_overlap
                + agent_overlap
            ),
        }

        return mapping, diagnostics

    # --------------------------------------------------------
    # Two speakers
    # --------------------------------------------------------

    if len(speakers) == 2:

        speaker_a = speakers[0]
        speaker_b = speakers[1]

        # Mapping A:
        #
        # speaker_a -> CUSTOMER
        # speaker_b -> AGENT

        score_a = (
            overlap_matrix[speaker_a]["CUSTOMER"]
            + overlap_matrix[speaker_b]["AGENT"]
        )

        # Mapping B:
        #
        # speaker_a -> AGENT
        # speaker_b -> CUSTOMER

        score_b = (
            overlap_matrix[speaker_a]["AGENT"]
            + overlap_matrix[speaker_b]["CUSTOMER"]
        )

        if score_a >= score_b:

            mapping = {
                speaker_a: "CUSTOMER",
                speaker_b: "AGENT",
            }

            selected_score = score_a
            alternative_score = score_b

        else:

            mapping = {
                speaker_a: "AGENT",
                speaker_b: "CUSTOMER",
            }

            selected_score = score_b
            alternative_score = score_a

        diagnostics = {
            "selected_score": selected_score,
            "alternative_score": alternative_score,
            "confidence_margin": (
                selected_score
                - alternative_score
            ),
        }

        return mapping, diagnostics

    # --------------------------------------------------------
    # More than two speakers
    # --------------------------------------------------------

    mapping = {
        speaker: "UNRESOLVED"
        for speaker in speakers
    }

    diagnostics = {
        "reason": "More than two speakers detected",
        "num_speakers": len(speakers),
    }

    return mapping, diagnostics


# ============================================================
# Build speaker-attributed transcript
# ============================================================

def build_speaker_transcript(
    conversation: dict,
    asr_data: dict,
    diarization_data: dict,
) -> dict:
    """
    Combine:

        1. Whisper ASR
        2. Pyannote diarization
        3. Ground-truth turn timing

    into a speaker-attributed transcript.
    """

    conversation_id = conversation[
        "conversation_id"
    ]

    ground_truth_turns = (
        build_ground_truth_turn_timing(
            conversation
        )
    )

    diarization_segments = (
        diarization_data.get("segments", [])
    )

    speaker_ids = sorted(
        set(
            segment["speaker"]
            for segment in diarization_segments
        )
    )

    speaker_mapping, mapping_diagnostics = (
        build_speaker_role_mapping(
            diarization_segments,
            ground_truth_turns,
        )
    )

    output_segments = []

    asr_segments = asr_data.get(
        "segments",
        [],
    )

    for segment in asr_segments:

        start = float(segment["start"])
        end = float(segment["end"])
        text = normalize_text(
            segment.get("text", "")
        )

        if not text:
            continue

        # ----------------------------------------------------
        # Predicted diarization speaker
        # ----------------------------------------------------

        speaker_id, speaker_overlap = (
            find_best_diarization_speaker(
                start,
                end,
                diarization_segments,
            )
        )

        if speaker_id is None:
            predicted_role = "UNRESOLVED"
        else:
            predicted_role = speaker_mapping.get(
                speaker_id,
                "UNRESOLVED",
            )

        # ----------------------------------------------------
        # Ground-truth role
        # ----------------------------------------------------

        ground_truth_turn, gt_overlap = (
            find_best_ground_truth_turn(
                start,
                end,
                ground_truth_turns,
            )
        )

        if ground_truth_turn is None:
            ground_truth_role = "UNKNOWN"
            ground_truth_turn_id = None
            ground_truth_dialogue_act = None
        else:
            ground_truth_role = (
                ground_truth_turn["role"]
            )

            ground_truth_turn_id = (
                ground_truth_turn["turn_id"]
            )

            ground_truth_dialogue_act = (
                ground_truth_turn[
                    "dialogue_act"
                ]
            )

        output_segments.append(
            {
                "start": start,
                "end": end,
                "duration": round(
                    end - start,
                    3,
                ),
                "text": text,
                "speaker_id": speaker_id,
                "predicted_role": predicted_role,
                "ground_truth_role": ground_truth_role,
                "ground_truth_turn_id": (
                    ground_truth_turn_id
                ),
                "ground_truth_dialogue_act": (
                    ground_truth_dialogue_act
                ),
                "speaker_overlap": round(
                    speaker_overlap,
                    3,
                ),
                "ground_truth_overlap": round(
                    gt_overlap,
                    3,
                ),
            }
        )

    return {
        "conversation_id": conversation_id,
        "audio_file": asr_data.get(
            "audio_file"
        ),
        "diarization_status": (
            "two_speakers"
            if len(speaker_ids) == 2
            else (
                "one_speaker"
                if len(speaker_ids) == 1
                else (
                    "no_speakers"
                    if len(speaker_ids) == 0
                    else "more_than_two_speakers"
                )
            )
        ),
        "num_speakers_detected": len(
            speaker_ids
        ),
        "speakers_detected": speaker_ids,
        "speaker_mapping": speaker_mapping,
        "mapping_diagnostics": (
            mapping_diagnostics
        ),
        "ground_truth_turns": (
            ground_truth_turns
        ),
        "segments": output_segments,
        "full_text": " ".join(
            segment["text"]
            for segment in output_segments
        ),
    }


# ============================================================
# Save outputs
# ============================================================

def save_transcript(
    result: dict,
) -> None:
    """
    Save the machine-readable JSON and readable TXT transcript.
    """

    conversation_id = result[
        "conversation_id"
    ]

    json_path = (
        OUTPUT_DIR
        / f"{conversation_id}.json"
    )

    text_path = (
        OUTPUT_TEXT_DIR
        / f"{conversation_id}.txt"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        text_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"Conversation: {conversation_id}\n"
        )

        f.write(
            f"Diarization status: "
            f"{result['diarization_status']}\n"
        )

        f.write(
            f"Speakers detected: "
            f"{result['speakers_detected']}\n"
        )

        f.write(
            f"Speaker mapping: "
            f"{result['speaker_mapping']}\n"
        )

        f.write("\n")
        f.write("=" * 80)
        f.write("\n")
        f.write("ASR + DIARIZATION TRANSCRIPT\n")
        f.write("=" * 80)
        f.write("\n\n")

        for segment in result["segments"]:

            start = segment["start"]
            end = segment["end"]
            predicted_role = segment[
                "predicted_role"
            ]
            speaker_id = segment[
                "speaker_id"
            ]
            ground_truth_role = segment[
                "ground_truth_role"
            ]
            text = segment["text"]

            f.write(
                f"[{start:.2f}s - {end:.2f}s] "
                f"{predicted_role} "
                f"({speaker_id}) | "
                f"GT={ground_truth_role}: "
                f"{text}\n"
            )


# ============================================================
# Summary row
# ============================================================

def build_summary_row(
    result: dict,
) -> dict:
    """
    Create one row for the summary CSV.
    """

    segments = result["segments"]

    evaluated_segments = [
        segment
        for segment in segments
        if segment["ground_truth_role"]
        in ("CUSTOMER", "AGENT")
    ]

    correct_role_segments = [
        segment
        for segment in evaluated_segments
        if segment["predicted_role"]
        == segment["ground_truth_role"]
    ]

    if evaluated_segments:
        speaker_role_accuracy = (
            len(correct_role_segments)
            / len(evaluated_segments)
        )
    else:
        speaker_role_accuracy = None

    return {
        "conversation_id": result[
            "conversation_id"
        ],
        "diarization_status": result[
            "diarization_status"
        ],
        "num_speakers_detected": result[
            "num_speakers_detected"
        ],
        "speaker_ids": "|".join(
            result["speakers_detected"]
        ),
        "speaker_mapping": json.dumps(
            result["speaker_mapping"]
        ),
        "num_asr_segments": len(
            segments
        ),
        "evaluated_segments": len(
            evaluated_segments
        ),
        "correct_role_segments": len(
            correct_role_segments
        ),
        "speaker_role_accuracy": (
            round(
                speaker_role_accuracy,
                4,
            )
            if speaker_role_accuracy
            is not None
            else ""
        ),
    }


def save_summary(
    rows: List[dict],
) -> None:
    """Save the summary CSV."""

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


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

    print("=" * 80)
    print(
        "07 - BUILD SPEAKER-ATTRIBUTED TRANSCRIPTS"
    )
    print("=" * 80)

    print(
        f"Transcript directory: "
        f"{TRANSCRIPTS_DIR}"
    )

    print(
        f"Diarization directory: "
        f"{DIARIZATION_DIR}"
    )

    print(
        f"Turn audio directory: "
        f"{TURN_AUDIO_DIR}"
    )

    print()

    conversations = load_conversations()

    transcript_files = sorted(
        TRANSCRIPTS_DIR.glob("*.json")
    )

    if MAX_CONVERSATIONS is not None:
        transcript_files = (
            transcript_files[
                :MAX_CONVERSATIONS
            ]
        )

    print(
        f"Found {len(transcript_files)} "
        f"transcript files to process."
    )

    processed = 0
    failed = 0

    summary_rows = []

    for transcript_file in transcript_files:

        conversation_id = (
            transcript_file.stem
        )

        print()
        print("-" * 80)
        print(
            f"Processing: {conversation_id}"
        )

        diarization_file = (
            DIARIZATION_DIR
            / f"{conversation_id}.json"
        )

        # ----------------------------------------------------
        # Check required files
        # ----------------------------------------------------

        if not diarization_file.exists():

            print(
                f"  [FAIL] Missing diarization file:"
            )

            print(
                f"         {diarization_file}"
            )

            failed += 1
            continue

        if conversation_id not in conversations:

            print(
                "  [FAIL] Conversation metadata "
                "not found."
            )

            failed += 1
            continue

        try:

            asr_data = load_json(
                transcript_file
            )

            diarization_data = load_json(
                diarization_file
            )

            conversation = conversations[
                conversation_id
            ]

            # ------------------------------------------------
            # Build transcript
            # ------------------------------------------------

            result = build_speaker_transcript(
                conversation=conversation,
                asr_data=asr_data,
                diarization_data=diarization_data,
            )

            # ------------------------------------------------
            # Save outputs
            # ------------------------------------------------

            save_transcript(result)

            summary_row = build_summary_row(
                result
            )

            summary_rows.append(
                summary_row
            )

            # ------------------------------------------------
            # Print diagnostic information
            # ------------------------------------------------

            print(
                f"  Diarization status: "
                f"{result['diarization_status']}"
            )

            print(
                f"  Speakers detected: "
                f"{result['speakers_detected']}"
            )

            print(
                f"  Speaker mapping: "
                f"{result['speaker_mapping']}"
            )

            print(
                f"  ASR segments: "
                f"{len(result['segments'])}"
            )

            if (
                result[
                    "speaker_mapping"
                ]
            ):
                print(
                    f"  Mapping diagnostics: "
                    f"{result['mapping_diagnostics']}"
                )

            # ------------------------------------------------
            # Preview
            # ------------------------------------------------

            print()
            print("  Transcript preview:")

            for segment in result[
                "segments"
            ][:8]:

                print(
                    f"    "
                    f"[{segment['start']:.2f}s - "
                    f"{segment['end']:.2f}s] "
                    f"{segment['predicted_role']} "
                    f"({segment['speaker_id']}) | "
                    f"GT={segment['ground_truth_role']}: "
                    f"{segment['text']}"
                )

            processed += 1

            print(
                "  [PASS] Speaker transcript created."
            )

        except Exception as e:

            print(
                f"  [FAIL] "
                f"{type(e).__name__}: {e}"
            )

            failed += 1

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    save_summary(summary_rows)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Processed: {processed}"
    )

    print(
        f"Failed:    {failed}"
    )

    print(
        f"JSON output: {OUTPUT_DIR}"
    )

    print(
        f"Text output: {OUTPUT_TEXT_DIR}"
    )

    print(
        f"Summary CSV: {SUMMARY_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()