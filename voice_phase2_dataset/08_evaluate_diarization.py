import csv
import json
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# Configuration
# ============================================================

DATASET_FILE = Path("data/emotional_conversations.json")

TURN_AUDIO_DIR = Path("new_generated_turns")
DIARIZATION_DIR = Path("diarization")

OUTPUT_DIR = Path("diarization_evaluation")

SUMMARY_FILE = (
    OUTPUT_DIR / "diarization_evaluation_summary.csv"
)

DATASET_SUMMARY_FILE = (
    OUTPUT_DIR / "diarization_dataset_summary.json"
)

# Initial test.
# Set to None after validating the first few conversations.
MAX_CONVERSATIONS = None

# Evaluation time resolution in seconds.
#
# 0.01 = 10 ms
# 0.02 = 20 ms
# 0.05 = 50 ms
#
# 10 ms gives finer scoring but requires more computation.
TIME_STEP = 0.01


# ============================================================
# File utilities
# ============================================================

def load_json(path: Path):
    """Load JSON from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_wav_duration_seconds(path: Path) -> float:
    """Return WAV duration in seconds."""
    with wave.open(str(path), "rb") as wav_file:

        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()

        if sample_rate <= 0:
            raise ValueError(
                f"Invalid sample rate in {path}"
            )

        return frames / sample_rate


# ============================================================
# Conversation loading
# ============================================================

def load_conversations() -> Dict[str, dict]:
    """
    Load all generated conversations and index by
    conversation_id.
    """

    conversations = load_json(DATASET_FILE)

    return {
        conversation["conversation_id"]: conversation
        for conversation in conversations
    }


# ============================================================
# Turn audio lookup
# ============================================================

def find_turn_audio(
    conversation_id: str,
    turn_id: int,
    role: str,
) -> Optional[Path]:
    """
    Find the generated WAV corresponding to a turn.

    Expected pattern:

        turn_<id>_<role>_<speaker>.wav
    """

    conversation_dir = (
        TURN_AUDIO_DIR / conversation_id
    )

    if not conversation_dir.exists():
        return None

    pattern = (
        f"turn_{turn_id}_{role.lower()}_*.wav"
    )

    matches = sorted(
        conversation_dir.glob(pattern)
    )

    if not matches:
        return None

    return matches[0]


# ============================================================
# Build ground-truth timeline
# ============================================================

def build_ground_truth(
    conversation: dict,
) -> List[dict]:
    """
    Reconstruct the ground-truth timeline using the individual
    turn WAV durations.

    Each turn is assigned its known CUSTOMER / AGENT role.
    """

    conversation_id = (
        conversation["conversation_id"]
    )

    turns = []

    current_time = 0.0

    for turn in conversation["turns"]:

        turn_id = int(turn["turn_id"])
        role = turn["role"].upper()

        audio_file = find_turn_audio(
            conversation_id,
            turn_id,
            turn["role"],
        )

        if audio_file is None:
            raise FileNotFoundError(
                f"Missing audio for "
                f"{conversation_id}, "
                f"turn {turn_id}"
            )

        duration = get_wav_duration_seconds(
            audio_file
        )

        start = current_time
        end = current_time + duration

        turns.append(
            {
                "turn_id": turn_id,
                "role": role,
                "start": start,
                "end": end,
                "duration": duration,
                "audio_file": str(audio_file),
            }
        )

        current_time = end

    return turns


# ============================================================
# Interval utilities
# ============================================================

def interval_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    """Return overlap duration between two intervals."""

    start = max(start_a, start_b)
    end = min(end_a, end_b)

    if end <= start:
        return 0.0

    return end - start


def total_reference_duration(
    ground_truth_turns: List[dict],
) -> float:
    """Return total reference speech duration."""

    return sum(
        turn["duration"]
        for turn in ground_truth_turns
    )


# ============================================================
# Diarization speaker collection
# ============================================================

def collect_diarization_speakers(
    diarization_segments: List[dict],
) -> List[str]:
    """Return sorted unique diarization speakers."""

    return sorted(
        set(
            segment["speaker"]
            for segment in diarization_segments
        )
    )


# ============================================================
# Calculate overlap between diarization speakers and roles
# ============================================================

def build_overlap_matrix(
    diarization_segments: List[dict],
    ground_truth_turns: List[dict],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate how much diarization time overlaps each
    ground-truth role.
    """

    speakers = collect_diarization_speakers(
        diarization_segments
    )

    matrix = {
        speaker: {
            "CUSTOMER": 0.0,
            "AGENT": 0.0,
        }
        for speaker in speakers
    }

    for diar_segment in diarization_segments:

        speaker = diar_segment["speaker"]

        diar_start = float(
            diar_segment["start"]
        )

        diar_end = float(
            diar_segment["end"]
        )

        for turn in ground_truth_turns:

            overlap = interval_overlap(
                diar_start,
                diar_end,
                turn["start"],
                turn["end"],
            )

            matrix[speaker][turn["role"]] += (
                overlap
            )

    return matrix


# ============================================================
# Speaker-label mapping
# ============================================================

def determine_speaker_mapping(
    overlap_matrix: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Find the mapping between pyannote speaker IDs and
    CUSTOMER / AGENT that maximizes total overlap.

    For two speakers, both possible assignments are compared.
    """

    speakers = sorted(
        overlap_matrix.keys()
    )

    # --------------------------------------------------------
    # No speakers
    # --------------------------------------------------------

    if len(speakers) == 0:

        return {}, {
            "selected_score": 0.0,
            "alternative_score": 0.0,
            "confidence_margin": 0.0,
        }

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

        # We deliberately do not pretend that this one cluster
        # represents both participants correctly.
        #
        # Assign it to whichever role has more overlap, but mark
        # the conversation as incomplete later.

        if customer_overlap >= agent_overlap:
            role = "CUSTOMER"
        else:
            role = "AGENT"

        return {
            speaker: role
        }, {
            "selected_score": max(
                customer_overlap,
                agent_overlap,
            ),
            "alternative_score": min(
                customer_overlap,
                agent_overlap,
            ),
            "confidence_margin": abs(
                customer_overlap
                - agent_overlap
            ),
        }

    # --------------------------------------------------------
    # Two speakers
    # --------------------------------------------------------

    if len(speakers) == 2:

        speaker_a = speakers[0]
        speaker_b = speakers[1]

        # Assignment A:
        #
        # A -> CUSTOMER
        # B -> AGENT

        score_a = (
            overlap_matrix[speaker_a]["CUSTOMER"]
            + overlap_matrix[speaker_b]["AGENT"]
        )

        # Assignment B:
        #
        # A -> AGENT
        # B -> CUSTOMER

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

        return mapping, {
            "selected_score": selected_score,
            "alternative_score": alternative_score,
            "confidence_margin": (
                selected_score
                - alternative_score
            ),
        }

    # --------------------------------------------------------
    # More than two speakers
    # --------------------------------------------------------

    mapping = {
        speaker: "UNRESOLVED"
        for speaker in speakers
    }

    return mapping, {
        "selected_score": 0.0,
        "alternative_score": 0.0,
        "confidence_margin": 0.0,
    }


# ============================================================
# Ground-truth role at a time point
# ============================================================

def reference_role_at_time(
    time: float,
    ground_truth_turns: List[dict],
) -> Optional[str]:
    """Return ground-truth role at a given time."""

    for turn in ground_truth_turns:

        if (
            turn["start"]
            <= time
            < turn["end"]
        ):
            return turn["role"]

    return None


# ============================================================
# Predicted speaker at a time point
# ============================================================

def predicted_speaker_at_time(
    time: float,
    diarization_segments: List[dict],
) -> Optional[str]:
    """Return pyannote speaker at a given time."""

    for segment in diarization_segments:

        if (
            segment["start"]
            <= time
            < segment["end"]
        ):
            return segment["speaker"]

    return None


# ============================================================
# Frame-level DER calculation
# ============================================================

def calculate_der(
    ground_truth_turns: List[dict],
    diarization_segments: List[dict],
    speaker_mapping: Dict[str, str],
    duration: float,
) -> dict:
    """
    Calculate frame-level diarization metrics.

    At each TIME_STEP:

        Reference speech:
            CUSTOMER / AGENT

        Prediction:
            CUSTOMER / AGENT / NONE

    Errors are categorized as:

        Missed Speech
        False Alarm
        Speaker Confusion
    """

    missed_speech = 0.0
    false_alarm = 0.0
    speaker_confusion = 0.0

    reference_speech = 0.0

    current_time = 0.0

    while current_time < duration:

        next_time = min(
            current_time + TIME_STEP,
            duration,
        )

        frame_duration = (
            next_time - current_time
        )

        midpoint = (
            current_time
            + next_time
        ) / 2.0

        reference_role = (
            reference_role_at_time(
                midpoint,
                ground_truth_turns,
            )
        )

        predicted_speaker = (
            predicted_speaker_at_time(
                midpoint,
                diarization_segments,
            )
        )

        if reference_role is not None:

            reference_speech += (
                frame_duration
            )

            if predicted_speaker is None:

                missed_speech += (
                    frame_duration
                )

            else:

                predicted_role = (
                    speaker_mapping.get(
                        predicted_speaker
                    )
                )

                if (
                    predicted_role
                    != reference_role
                ):
                    speaker_confusion += (
                        frame_duration
                    )

        else:

            if predicted_speaker is not None:
                false_alarm += (
                    frame_duration
                )

        current_time = next_time

    if reference_speech > 0:

        der = (
            missed_speech
            + false_alarm
            + speaker_confusion
        ) / reference_speech

        miss_rate = (
            missed_speech
            / reference_speech
        )

        confusion_rate = (
            speaker_confusion
            / reference_speech
        )

    else:

        der = 0.0
        miss_rate = 0.0
        confusion_rate = 0.0

    return {
        "reference_speech_duration": (
            reference_speech
        ),
        "missed_speech_duration": (
            missed_speech
        ),
        "false_alarm_duration": (
            false_alarm
        ),
        "speaker_confusion_duration": (
            speaker_confusion
        ),
        "DER": der,
        "miss_rate": miss_rate,
        "confusion_rate": confusion_rate,
    }


# ============================================================
# Turn-level evaluation
# ============================================================

def evaluate_turns(
    ground_truth_turns: List[dict],
    diarization_segments: List[dict],
    speaker_mapping: Dict[str, str],
) -> dict:
    """
    Evaluate whether each ground-truth turn is dominated
    by the correct diarized speaker.
    """

    correct_turns = 0
    total_turns = len(
        ground_truth_turns
    )

    turn_results = []

    for turn in ground_truth_turns:

        overlap_by_speaker = {}

        for segment in diarization_segments:

            overlap = interval_overlap(
                turn["start"],
                turn["end"],
                float(segment["start"]),
                float(segment["end"]),
            )

            if overlap <= 0:
                continue

            speaker = segment["speaker"]

            overlap_by_speaker[speaker] = (
                overlap_by_speaker.get(
                    speaker,
                    0.0,
                )
                + overlap
            )

        if not overlap_by_speaker:

            predicted_role = "MISSED"

            dominant_speaker = None
            dominant_overlap = 0.0

        else:

            dominant_speaker = max(
                overlap_by_speaker,
                key=overlap_by_speaker.get,
            )

            dominant_overlap = (
                overlap_by_speaker[
                    dominant_speaker
                ]
            )

            predicted_role = (
                speaker_mapping.get(
                    dominant_speaker,
                    "UNRESOLVED",
                )
            )

        correct = (
            predicted_role
            == turn["role"]
        )

        if correct:
            correct_turns += 1

        turn_results.append(
            {
                "turn_id": turn["turn_id"],
                "ground_truth_role": turn["role"],
                "predicted_role": predicted_role,
                "dominant_speaker": dominant_speaker,
                "dominant_overlap": (
                    dominant_overlap
                ),
                "turn_duration": turn["duration"],
                "correct": correct,
            }
        )

    if total_turns > 0:

        accuracy = (
            correct_turns
            / total_turns
        )

    else:
        accuracy = 0.0

    return {
        "total_turns": total_turns,
        "correct_turns": correct_turns,
        "turn_accuracy": accuracy,
        "turn_results": turn_results,
    }


# ============================================================
# Evaluate one conversation
# ============================================================

def evaluate_conversation(
    conversation: dict,
    diarization_data: dict,
) -> dict:
    """
    Evaluate one conversation.
    """

    ground_truth_turns = (
        build_ground_truth(conversation)
    )

    diarization_segments = (
        diarization_data.get(
            "segments",
            [],
        )
    )

    speakers = (
        collect_diarization_speakers(
            diarization_segments
        )
    )

    overlap_matrix = (
        build_overlap_matrix(
            diarization_segments,
            ground_truth_turns,
        )
    )

    speaker_mapping, mapping_scores = (
        determine_speaker_mapping(
            overlap_matrix
        )
    )

    reference_duration = (
        total_reference_duration(
            ground_truth_turns
        )
    )

    # Approximate conversation duration from
    # the end of the last reference turn.
    duration = reference_duration

    der_metrics = calculate_der(
        ground_truth_turns,
        diarization_segments,
        speaker_mapping,
        duration,
    )

    turn_metrics = evaluate_turns(
        ground_truth_turns,
        diarization_segments,
        speaker_mapping,
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    detected_reference_duration = (
        reference_duration
        - der_metrics[
            "missed_speech_duration"
        ]
    )

    if reference_duration > 0:

        speech_coverage = (
            detected_reference_duration
            / reference_duration
        )

    else:

        speech_coverage = 0.0

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if len(speakers) == 2:
        diarization_status = (
            "two_speakers"
        )

    elif len(speakers) == 1:
        diarization_status = (
            "one_speaker"
        )

    elif len(speakers) == 0:
        diarization_status = (
            "no_speakers"
        )

    else:
        diarization_status = (
            "more_than_two_speakers"
        )

    return {
        "conversation_id": conversation[
            "conversation_id"
        ],
        "scenario": conversation.get(
            "scenario"
        ),
        "diarization_status": (
            diarization_status
        ),
        "num_speakers_detected": len(
            speakers
        ),
        "speakers_detected": speakers,
        "speaker_mapping": speaker_mapping,
        "mapping_scores": mapping_scores,

        "reference_duration": (
            reference_duration
        ),
        "speech_coverage": (
            speech_coverage
        ),

        **der_metrics,

        "total_turns": (
            turn_metrics["total_turns"]
        ),
        "correct_turns": (
            turn_metrics["correct_turns"]
        ),
        "turn_accuracy": (
            turn_metrics["turn_accuracy"]
        ),

        "turn_results": (
            turn_metrics["turn_results"]
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("08 - DIARIZATION QUALITY EVALUATION")
    print("=" * 80)

    conversations = load_conversations()

    conversation_ids = sorted(
        conversations.keys()
    )

    if MAX_CONVERSATIONS is not None:

        conversation_ids = (
            conversation_ids[
                :MAX_CONVERSATIONS
            ]
        )

    print(
        f"Conversations to evaluate: "
        f"{len(conversation_ids)}"
    )

    results = []
    failed = 0

    for conversation_id in conversation_ids:

        print()
        print("-" * 80)

        print(
            f"Evaluating: "
            f"{conversation_id}"
        )

        diarization_file = (
            DIARIZATION_DIR
            / f"{conversation_id}.json"
        )

        if not diarization_file.exists():

            print(
                "  [SKIP] Diarization file "
                "not found."
            )

            failed += 1
            continue

        try:

            conversation = conversations[
                conversation_id
            ]

            diarization_data = load_json(
                diarization_file
            )

            result = evaluate_conversation(
                conversation,
                diarization_data,
            )

            results.append(result)

            print(
                f"  Status: "
                f"{result['diarization_status']}"
            )

            print(
                f"  Speakers detected: "
                f"{result['speakers_detected']}"
            )

            print(
                f"  Mapping: "
                f"{result['speaker_mapping']}"
            )

            print(
                f"  Speech coverage: "
                f"{result['speech_coverage']:.2%}"
            )

            print(
                f"  Turn accuracy: "
                f"{result['turn_accuracy']:.2%}"
            )

            print(
                f"  Miss rate: "
                f"{result['miss_rate']:.2%}"
            )

            print(
                f"  Confusion rate: "
                f"{result['confusion_rate']:.2%}"
            )

            print(
                f"  DER: "
                f"{result['DER']:.2%}"
            )

        except Exception as e:

            print(
                f"  [FAIL] "
                f"{type(e).__name__}: {e}"
            )

            failed += 1

    # ========================================================
    # Save conversation-level CSV
    # ========================================================

    if results:

        csv_file = SUMMARY_FILE

        fieldnames = [
            "conversation_id",
            "scenario",
            "diarization_status",
            "num_speakers_detected",
            "speakers_detected",
            "speaker_mapping",
            "reference_duration",
            "speech_coverage",
            "missed_speech_duration",
            "false_alarm_duration",
            "speaker_confusion_duration",
            "DER",
            "miss_rate",
            "confusion_rate",
            "total_turns",
            "correct_turns",
            "turn_accuracy",
        ]

        with open(
            csv_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for result in results:

                row = {
                    field: result.get(
                        field,
                        "",
                    )
                    for field in fieldnames
                }

                writer.writerow(row)

        print()
        print(
            f"Conversation-level results saved to:"
        )

        print(
            csv_file
        )

    # ========================================================
    # Dataset-level aggregation
    # ========================================================

    if results:

        total_reference_duration = sum(
            result[
                "reference_speech_duration"
            ]
            for result in results
        )

        total_missed = sum(
            result[
                "missed_speech_duration"
            ]
            for result in results
        )

        total_false_alarm = sum(
            result[
                "false_alarm_duration"
            ]
            for result in results
        )

        total_confusion = sum(
            result[
                "speaker_confusion_duration"
            ]
            for result in results
        )

        total_turns = sum(
            result["total_turns"]
            for result in results
        )

        total_correct_turns = sum(
            result["correct_turns"]
            for result in results
        )

        successful_two_speaker = sum(
            1
            for result in results
            if result[
                "num_speakers_detected"
            ]
            == 2
        )

        one_speaker_cases = sum(
            1
            for result in results
            if result[
                "num_speakers_detected"
            ]
            == 1
        )

        no_speaker_cases = sum(
            1
            for result in results
            if result[
                "num_speakers_detected"
            ]
            == 0
        )

        if total_reference_duration > 0:

            dataset_der = (
                total_missed
                + total_false_alarm
                + total_confusion
            ) / total_reference_duration

            dataset_miss_rate = (
                total_missed
                / total_reference_duration
            )

            dataset_confusion_rate = (
                total_confusion
                / total_reference_duration
            )

            dataset_coverage = (
                1
                - dataset_miss_rate
            )

        else:

            dataset_der = 0.0
            dataset_miss_rate = 0.0
            dataset_confusion_rate = 0.0
            dataset_coverage = 0.0

        if total_turns > 0:

            dataset_turn_accuracy = (
                total_correct_turns
                / total_turns
            )

        else:

            dataset_turn_accuracy = 0.0

        dataset_summary = {
            "conversations_evaluated": len(
                results
            ),
            "conversations_failed": failed,

            "two_speaker_cases": (
                successful_two_speaker
            ),

            "one_speaker_cases": (
                one_speaker_cases
            ),

            "no_speaker_cases": (
                no_speaker_cases
            ),

            "two_speaker_detection_rate": (
                successful_two_speaker
                / len(results)
            ),

            "total_reference_speech_seconds": (
                total_reference_duration
            ),

            "speech_coverage": (
                dataset_coverage
            ),

            "missed_speech_seconds": (
                total_missed
            ),

            "false_alarm_seconds": (
                total_false_alarm
            ),

            "speaker_confusion_seconds": (
                total_confusion
            ),

            "DER": dataset_der,

            "miss_rate": (
                dataset_miss_rate
            ),

            "speaker_confusion_rate": (
                dataset_confusion_rate
            ),

            "total_turns": total_turns,

            "correct_turns": (
                total_correct_turns
            ),

            "turn_accuracy": (
                dataset_turn_accuracy
            ),
        }

        with open(
            DATASET_SUMMARY_FILE,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                dataset_summary,
                f,
                indent=2,
            )

        # ====================================================
        # Print dataset summary
        # ====================================================

        print()
        print("=" * 80)
        print("DATASET-LEVEL DIARIZATION RESULTS")
        print("=" * 80)

        print(
            f"Conversations evaluated: "
            f"{dataset_summary['conversations_evaluated']}"
        )

        print(
            f"Two-speaker detection rate: "
            f"{dataset_summary['two_speaker_detection_rate']:.2%}"
        )

        print(
            f"Speech coverage: "
            f"{dataset_summary['speech_coverage']:.2%}"
        )

        print(
            f"Turn accuracy: "
            f"{dataset_summary['turn_accuracy']:.2%}"
        )

        print(
            f"Miss rate: "
            f"{dataset_summary['miss_rate']:.2%}"
        )

        print(
            f"Speaker confusion rate: "
            f"{dataset_summary['speaker_confusion_rate']:.2%}"
        )

        print(
            f"DER: "
            f"{dataset_summary['DER']:.2%}"
        )

        print()
        print(
            f"Dataset summary saved to:"
        )

        print(
            DATASET_SUMMARY_FILE
        )

        print("=" * 80)


if __name__ == "__main__":
    main()