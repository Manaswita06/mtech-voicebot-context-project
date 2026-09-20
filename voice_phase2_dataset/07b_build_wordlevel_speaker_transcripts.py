#!/usr/bin/env python3

"""
07b_build_wordlevel_speaker_transcripts.py

Combines:
    1. WhisperX large-v3-turbo word-level timestamps
    2. Existing Pyannote Community-1 diarization output
    3. Ground-truth turn timing from the synthetic dataset

Important:
- Pyannote is NOT rerun here.
- WhisperX is NOT used for diarization.
- If Pyannote detects only one speaker, speaker-role mapping is marked
  UNRESOLVED. We do NOT fabricate CUSTOMER/AGENT labels.
- All WhisperX words are preserved even when speaker attribution is unresolved.
- Ground-truth role mapping is used ONLY for evaluation on this synthetic
  dataset; it is not used to "fix" diarization predictions.

Inputs:
    transcripts_whisperx/<conversation_id>.json
    diarization/<conversation_id>.json
    data/emotional_conversations.json
    new_generated_turns/<conversation_id>/*.wav

Outputs:
    wordlevel_speaker_transcripts/<conversation_id>.json
    wordlevel_speaker_transcripts_text/<conversation_id>.txt
    wordlevel_speaker_summary.csv
"""

import csv
import glob
import json
import os
import wave
from itertools import permutations


# ============================================================
# CONFIGURATION
# ============================================================

WHISPERX_DIR = "transcripts_whisperx"
DIARIZATION_DIR = "diarization"
METADATA_FILE = "data/emotional_conversations.json"
TURN_AUDIO_DIR = "new_generated_turns"

OUTPUT_JSON_DIR = "wordlevel_speaker_transcripts"
OUTPUT_TEXT_DIR = "wordlevel_speaker_transcripts_text"
SUMMARY_FILE = "wordlevel_speaker_summary.csv"

# Set to None to process all conversations
MAX_CONVERSATIONS = None

# When grouping consecutive words into speaker turns,
# a larger gap starts a new turn.
MAX_WORD_GAP_FOR_SAME_TURN = 1.0


# ============================================================
# FILE / DIRECTORY HELPERS
# ============================================================

def ensure_directories():
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def wav_duration_seconds(path):
    """Get WAV duration without requiring librosa/soundfile."""
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()

    if rate <= 0:
        raise ValueError(f"Invalid sample rate in {path}")

    return frames / float(rate)


def find_turn_wav(conversation_id, turn_id):
    """
    Find the individual turn WAV.

    Supports several possible turn_id formatting styles.
    """
    conv_dir = os.path.join(TURN_AUDIO_DIR, conversation_id)

    candidates = [
        os.path.join(conv_dir, f"turn_{turn_id}_*.wav"),
    ]

    try:
        tid_int = int(turn_id)
        candidates.extend([
            os.path.join(conv_dir, f"turn_{tid_int:03d}_*.wav"),
            os.path.join(conv_dir, f"turn_{tid_int:04d}_*.wav"),
        ])
    except (ValueError, TypeError):
        pass

    for pattern in candidates:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]

    return None


# ============================================================
# DATA LOADING
# ============================================================

def load_conversation_metadata():
    data = load_json(METADATA_FILE)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "conversations" in data:
            return data["conversations"]

        # Handle possible dictionary keyed by conversation_id
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())

    raise ValueError(
        f"Unexpected format in {METADATA_FILE}"
    )


def build_conversation_lookup(conversations):
    lookup = {}

    for conv in conversations:
        conversation_id = conv.get("conversation_id")

        if conversation_id:
            lookup[conversation_id] = conv

    return lookup


# ============================================================
# REFERENCE TIMELINE
# ============================================================

def build_reference_turns(conversation):
    """
    Reconstruct the ground-truth conversation timeline using
    individual generated turn WAV durations.

    Role values are normalized to CUSTOMER / AGENT so that
    capitalization differences in the metadata do not break
    the mapping calculation.
    """

    turns = conversation.get("turns", [])

    if not turns:
        raise ValueError(
            f"No turns found for conversation "
            f"{conversation.get('conversation_id')}"
        )

    reference_turns = []

    current_time = 0.0

    for turn in turns:

        turn_id = turn.get("turn_id")

        # Normalize role
        raw_role = str(
            turn.get("role", "")
        ).strip().upper()

        if raw_role in {"CUSTOMER", "CLIENT", "USER"}:
            role = "CUSTOMER"
        elif raw_role in {"AGENT", "ASSOCIATE", "REPRESENTATIVE"}:
            role = "AGENT"
        else:
            role = raw_role

        text = turn.get("text", "")

        pause_before = float(
            turn.get("pause_before", 0) or 0
        )

        pause_after = float(
            turn.get("pause_after", 0) or 0
        )

        current_time += pause_before / 1000.0

        wav_path = find_turn_wav(
            conversation["conversation_id"],
            turn_id
        )

        if wav_path is None:
            raise FileNotFoundError(
                f"Could not find audio for "
                f"{conversation['conversation_id']} "
                f"turn {turn_id}"
            )

        duration = wav_duration_seconds(wav_path)

        start = current_time
        end = current_time + duration

        reference_turns.append({
            "turn_id": turn_id,
            "role": role,
            "raw_role": turn.get("role"),
            "text": text,
            "start": start,
            "end": end,
            "duration": duration,
        })

        current_time = end

        current_time += pause_after / 1000.0

    return reference_turns


# ============================================================
# WHISPERX WORD EXTRACTION
# ============================================================

def extract_whisperx_words(whisperx_data):
    """
    Extract word-level timestamps from the WhisperX JSON.

    Expected structure:

        segments[
            {
                "start": ...,
                "end": ...,
                "text": ...,
                "words": [
                    {
                        "word": "...",
                        "start": ...,
                        "end": ...
                    }
                ]
            }
        ]

    Also supports a top-level 'words' list if present.
    """

    extracted = []

    # Case 1: top-level words
    if isinstance(whisperx_data.get("words"), list):
        for word in whisperx_data["words"]:
            if "start" in word and "end" in word:
                extracted.append({
                    "word": word.get("word", "").strip(),
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                    "score": word.get("score"),
                })

        return extracted

    # Case 2: words nested inside segments
    for segment in whisperx_data.get("segments", []):

        words = segment.get("words", [])

        if not words:
            continue

        for word in words:

            if "start" not in word or "end" not in word:
                continue

            text = word.get("word", "").strip()

            if not text:
                continue

            extracted.append({
                "word": text,
                "start": float(word["start"]),
                "end": float(word["end"]),
                "score": word.get("score"),
            })

    return extracted


# ============================================================
# PYANNOTE SEGMENT LOADING
# ============================================================

def extract_diarization_segments(diarization_data):

    segments = []

    for segment in diarization_data.get("segments", []):

        if "start" not in segment:
            continue

        if "end" not in segment:
            continue

        speaker = segment.get("speaker")

        if speaker is None:
            continue

        segments.append({
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "speaker": speaker,
        })

    segments.sort(key=lambda x: x["start"])

    return segments


# ============================================================
# OVERLAP CALCULATION
# ============================================================

def overlap_seconds(start1, end1, start2, end2):
    return max(
        0.0,
        min(end1, end2) - max(start1, start2)
    )


# ============================================================
# SPEAKER -> ROLE MAPPING
# ============================================================

def calculate_speaker_role_mapping(
    diarization_segments,
    reference_turns
):
    """
    Determine which Pyannote speaker corresponds to CUSTOMER
    and AGENT.

    This is ONLY an evaluation-time mapping because the synthetic
    dataset provides the ground-truth role timeline.

    IMPORTANT:
    If Pyannote detects one speaker, this function deliberately
    returns unresolved mapping.
    """

    speakers = sorted(
        set(seg["speaker"] for seg in diarization_segments)
    )

    num_speakers = len(speakers)

    if num_speakers == 0:
        return {
            "status": "UNRESOLVED_NO_SPEAKERS",
            "mapping": {},
            "best_score": 0.0,
            "mapping_margin": 0.0,
        }

    if num_speakers == 1:
        return {
            "status": "UNRESOLVED_ONE_SPEAKER",
            "mapping": {
                speakers[0]: None
            },
            "best_score": 0.0,
            "mapping_margin": 0.0,
        }

    if num_speakers != 2:
        return {
            "status": "UNRESOLVED_UNSUPPORTED_SPEAKER_COUNT",
            "mapping": {
                speaker: None
                for speaker in speakers
            },
            "best_score": 0.0,
            "mapping_margin": 0.0,
        }

    roles = ["CUSTOMER", "AGENT"]

    # Calculate how much diarization speech overlaps each
    # ground-truth role.
    overlap_matrix = {}

    for speaker in speakers:

        overlap_matrix[speaker] = {}

        for role in roles:

            total_overlap = 0.0

            for diar_seg in diarization_segments:

                if diar_seg["speaker"] != speaker:
                    continue

                for ref_turn in reference_turns:

                    if ref_turn["role"] != role:
                        continue

                    overlap = overlap_seconds(
                        diar_seg["start"],
                        diar_seg["end"],
                        ref_turn["start"],
                        ref_turn["end"],
                    )

                    total_overlap += overlap

            overlap_matrix[speaker][role] = total_overlap

    print("\nSpeaker-role overlap matrix:")

    for speaker, role_scores in overlap_matrix.items():
        print(
            f"  {speaker}: "
            f"CUSTOMER={role_scores.get('CUSTOMER', 0.0):.3f}s, "
            f"AGENT={role_scores.get('AGENT', 0.0):.3f}s"
        )

    # Evaluate both possible 2-speaker mappings.
    candidates = []

    for role_assignment in permutations(roles):

        mapping = {
            speakers[0]: role_assignment[0],
            speakers[1]: role_assignment[1],
        }

        score = 0.0

        for speaker, role in mapping.items():
            score += overlap_matrix[speaker][role]

        candidates.append({
            "mapping": mapping,
            "score": score,
        })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = candidates[0]
    second = candidates[1]

    margin = best["score"] - second["score"]

    # Completely ambiguous mapping.
    if best["score"] <= 0:
        return {
            "status": "UNRESOLVED_NO_ROLE_OVERLAP",
            "mapping": {
                speaker: None
                for speaker in speakers
            },
            "best_score": best["score"],
            "mapping_margin": margin,
        }

    if abs(margin) < 1e-6:
        return {
            "status": "UNRESOLVED_AMBIGUOUS_MAPPING",
            "mapping": {
                speaker: None
                for speaker in speakers
            },
            "best_score": best["score"],
            "mapping_margin": margin,
        }

    return {
        "status": "RESOLVED_TWO_SPEAKERS",
        "mapping": best["mapping"],
        "best_score": best["score"],
        "mapping_margin": margin,
    }


# ============================================================
# WORD -> PYANNOTE SPEAKER ASSIGNMENT
# ============================================================

def assign_word_speakers(
    words,
    diarization_segments,
    speaker_to_role
):
    """
    Assign every aligned word to the Pyannote segment with the
    greatest temporal overlap.

    IMPORTANT:
    If a word does not overlap any diarization segment, its speaker
    is left as None rather than being fabricated.
    """

    assigned_words = []

    for word in words:

        word_start = word["start"]
        word_end = word["end"]

        best_speaker = None
        best_overlap = 0.0

        for diar_seg in diarization_segments:

            overlap = overlap_seconds(
                word_start,
                word_end,
                diar_seg["start"],
                diar_seg["end"],
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg["speaker"]

        predicted_role = None

        if best_speaker is not None:
            predicted_role = speaker_to_role.get(
                best_speaker
            )

        assigned_words.append({
            "word": word["word"],
            "start": word_start,
            "end": word_end,
            "duration": word_end - word_start,
            "asr_score": word.get("score"),
            "speaker": best_speaker,
            "predicted_role": predicted_role,
            "speaker_overlap_seconds": best_overlap,
            "speaker_attribution_status": (
                "ASSIGNED"
                if best_speaker is not None
                else "UNASSIGNED"
            ),
        })

    return assigned_words


# ============================================================
# GROUND-TRUTH ROLE FOR EACH WORD
# ============================================================

def get_ground_truth_role(word, reference_turns):

    best_role = None
    best_overlap = 0.0

    for turn in reference_turns:

        overlap = overlap_seconds(
            word["start"],
            word["end"],
            turn["start"],
            turn["end"],
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_role = turn["role"]

    return best_role


# ============================================================
# GROUP WORDS INTO SPEAKER TURNS
# ============================================================

def group_words_into_turns(words):

    if not words:
        return []

    grouped = []

    current = None

    for word in words:

        speaker = word["speaker"]
        role = word["predicted_role"]

        if current is None:

            current = {
                "start": word["start"],
                "end": word["end"],
                "speaker": speaker,
                "predicted_role": role,
                "words": [word["word"]],
            }

            continue

        gap = word["start"] - current["end"]

        same_speaker = (
            speaker == current["speaker"]
            and role == current["predicted_role"]
        )

        if same_speaker and gap <= MAX_WORD_GAP_FOR_SAME_TURN:

            current["end"] = word["end"]
            current["words"].append(word["word"])

        else:

            grouped.append(current)

            current = {
                "start": word["start"],
                "end": word["end"],
                "speaker": speaker,
                "predicted_role": role,
                "words": [word["word"]],
            }

    if current is not None:
        grouped.append(current)

    # Convert word list to text
    for turn in grouped:
        turn["text"] = " ".join(turn.pop("words"))

    return grouped


# ============================================================
# TEXT OUTPUT
# ============================================================

def write_text_output(
    output_path,
    conversation_id,
    whisperx_data,
    diarization_data,
    mapping_info,
    words,
    grouped_turns,
):
    alignment_rate = whisperx_data.get(
        "alignment_rate",
        whisperx_data.get("word_alignment_rate")
    )

    with open(output_path, "w", encoding="utf-8") as f:

        f.write(f"Conversation: {conversation_id}\n")
        f.write(
            f"Model: "
            f"{whisperx_data.get('model', 'large-v3-turbo')}\n"
        )

        if alignment_rate is not None:
            f.write(
                f"Word alignment rate: "
                f"{float(alignment_rate) * 100:.2f}%\n"
                if float(alignment_rate) <= 1
                else
                f"Word alignment rate: "
                f"{float(alignment_rate):.2f}%\n"
            )

        f.write(
            f"Pyannote speakers detected: "
            f"{diarization_data.get('num_speakers_detected', 'N/A')}\n"
        )

        f.write(
            f"Role mapping status: "
            f"{mapping_info['status']}\n"
        )

        f.write(
            f"Speaker-role mapping: "
            f"{mapping_info['mapping']}\n"
        )

        f.write("\n")
        f.write("=" * 70)
        f.write("\nWORD-LEVEL TRANSCRIPT\n")
        f.write("=" * 70)
        f.write("\n\n")

        for word in words:

            speaker = word["speaker"] or "UNRESOLVED"
            role = word["predicted_role"] or "UNRESOLVED"

            f.write(
                f"{word['start']:.3f} - "
                f"{word['end']:.3f}: "
                f"{word['word']} "
                f"[{speaker} / {role}]\n"
            )

        f.write("\n")
        f.write("=" * 70)
        f.write("\nGROUPED SPEAKER TURNS\n")
        f.write("=" * 70)
        f.write("\n\n")

        for turn in grouped_turns:

            speaker = turn["speaker"] or "UNRESOLVED"
            role = turn["predicted_role"] or "UNRESOLVED"

            f.write(
                f"[{turn['start']:.3f}s - "
                f"{turn['end']:.3f}s] "
                f"{speaker} / {role}\n"
            )

            f.write(
                f"{turn['text']}\n\n"
            )


# ============================================================
# SUMMARY METRICS
# ============================================================

def calculate_summary(
    conversation_id,
    whisperx_data,
    diarization_data,
    mapping_info,
    words,
):
    total_words = len(words)

    assigned_words = sum(
        1
        for w in words
        if w["speaker"] is not None
    )

    unresolved_words = total_words - assigned_words

    role_resolved_words = sum(
        1
        for w in words
        if w["predicted_role"] is not None
    )

    correct_role_words = sum(
        1
        for w in words
        if (
            w.get("predicted_role") is not None
            and w.get("ground_truth_role") is not None
            and w["predicted_role"]
            == w["ground_truth_role"]
        )
    )

    role_accuracy = None

    if role_resolved_words > 0:
        role_accuracy = (
            correct_role_words
            / role_resolved_words
        )

    return {
        "conversation_id": conversation_id,

        "num_speakers_detected":
            diarization_data.get(
                "num_speakers_detected",
                len(
                    diarization_data.get(
                        "speakers_detected",
                        []
                    )
                ),
            ),

        "speakers_detected":
            ",".join(
                diarization_data.get(
                    "speakers_detected",
                    []
                )
            ),

        "diarization_status":
            (
                "TWO_SPEAKERS"
                if len(
                    diarization_data.get(
                        "speakers_detected",
                        []
                    )
                ) == 2
                else
                "ONE_SPEAKER"
                if len(
                    diarization_data.get(
                        "speakers_detected",
                        []
                    )
                ) == 1
                else
                "OTHER"
            ),

        "role_mapping_status":
            mapping_info["status"],

        "total_words":
            total_words,

        "speaker_assigned_words":
            assigned_words,

        "unresolved_words":
            unresolved_words,

        "unresolved_word_rate":
            (
                unresolved_words / total_words
                if total_words > 0
                else 0.0
            ),

        "role_resolved_words":
            role_resolved_words,

        "role_accuracy":
            role_accuracy,

        "mapping_score_seconds":
            mapping_info["best_score"],

        "mapping_margin_seconds":
            mapping_info["mapping_margin"],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_directories()

    print("Loading conversation metadata...")
    conversations = load_conversation_metadata()
    conversation_lookup = build_conversation_lookup(
        conversations
    )

    whisperx_files = sorted(
        glob.glob(
            os.path.join(
                WHISPERX_DIR,
                "*.json"
            )
        )
    )

    if MAX_CONVERSATIONS is not None:
        whisperx_files = whisperx_files[
            :MAX_CONVERSATIONS
        ]

    print(
        f"Found {len(whisperx_files)} WhisperX transcripts."
    )

    summary_rows = []

    for index, whisperx_path in enumerate(
        whisperx_files,
        start=1
    ):

        conversation_id = os.path.splitext(
            os.path.basename(whisperx_path)
        )[0]

        print()
        print("=" * 70)
        print(
            f"[{index}/{len(whisperx_files)}] "
            f"{conversation_id}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Load WhisperX
        # ----------------------------------------------------

        whisperx_data = load_json(
            whisperx_path
        )

        # ----------------------------------------------------
        # Load Pyannote
        # ----------------------------------------------------

        diarization_path = os.path.join(
            DIARIZATION_DIR,
            f"{conversation_id}.json"
        )

        if not os.path.exists(diarization_path):

            print(
                "WARNING: No diarization file found. "
                "Skipping."
            )

            continue

        diarization_data = load_json(
            diarization_path
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        conversation = conversation_lookup.get(
            conversation_id
        )

        if conversation is None:

            print(
                "WARNING: Conversation metadata not found. "
                "Skipping."
            )

            continue

        # ----------------------------------------------------
        # Reference timeline
        # ----------------------------------------------------

        reference_turns = build_reference_turns(
            conversation
        )

        # ----------------------------------------------------
        # WhisperX words
        # ----------------------------------------------------

        words = extract_whisperx_words(
            whisperx_data
        )

        print(
            f"WhisperX words: {len(words)}"
        )

        # ----------------------------------------------------
        # Pyannote segments
        # ----------------------------------------------------

        diarization_segments = (
            extract_diarization_segments(
                diarization_data
            )
        )

        speakers = sorted(
            set(
                s["speaker"]
                for s in diarization_segments
            )
        )

        # If JSON explicitly contains speakers_detected
        # but segments are empty, preserve that information.
        if not speakers:
            speakers = sorted(
                diarization_data.get(
                    "speakers_detected",
                    []
                )
            )

        print(
            f"Pyannote speakers detected: "
            f"{len(speakers)}"
        )

        # ----------------------------------------------------
        # Speaker -> role mapping
        # ----------------------------------------------------

        mapping_info = (
            calculate_speaker_role_mapping(
                diarization_segments,
                reference_turns
            )
        )

        print(
            f"Role mapping status: "
            f"{mapping_info['status']}"
        )

        print(
            f"Mapping: "
            f"{mapping_info['mapping']}"
        )

        # ----------------------------------------------------
        # Word -> speaker assignment
        # ----------------------------------------------------

        assigned_words = assign_word_speakers(
            words,
            diarization_segments,
            mapping_info["mapping"],
        )

        # ----------------------------------------------------
        # Ground truth role
        # ----------------------------------------------------

        for word in assigned_words:

            word["ground_truth_role"] = (
                get_ground_truth_role(
                    word,
                    reference_turns
                )
            )

        # ----------------------------------------------------
        # Group into speaker turns
        # ----------------------------------------------------

        grouped_turns = group_words_into_turns(
            assigned_words
        )

        # ----------------------------------------------------
        # Build final JSON
        # ----------------------------------------------------

        result = {
            "conversation_id":
                conversation_id,

            "audio_file":
                diarization_data.get(
                    "audio_file"
                ),

            "model":
                whisperx_data.get(
                    "model",
                    "large-v3-turbo"
                ),

            "word_alignment_rate":
                whisperx_data.get(
                    "word_alignment_rate",
                    whisperx_data.get(
                        "alignment_rate"
                    )
                ),

            "num_speakers_detected":
                len(speakers),

            "speakers_detected":
                speakers,

            "diarization_status":
                (
                    "TWO_SPEAKERS"
                    if len(speakers) == 2
                    else
                    "ONE_SPEAKER"
                    if len(speakers) == 1
                    else
                    "OTHER"
                ),

            "role_mapping_status":
                mapping_info["status"],

            "speaker_to_role":
                mapping_info["mapping"],

            "mapping_score_seconds":
                mapping_info["best_score"],

            "mapping_margin_seconds":
                mapping_info["mapping_margin"],

            "words":
                assigned_words,

            "speaker_turns":
                grouped_turns,
        }

        # ----------------------------------------------------
        # Save JSON
        # ----------------------------------------------------

        output_json = os.path.join(
            OUTPUT_JSON_DIR,
            f"{conversation_id}.json"
        )

        with open(
            output_json,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False
            )

        # ----------------------------------------------------
        # Save readable TXT
        # ----------------------------------------------------

        output_txt = os.path.join(
            OUTPUT_TEXT_DIR,
            f"{conversation_id}.txt"
        )

        write_text_output(
            output_txt,
            conversation_id,
            whisperx_data,
            diarization_data,
            mapping_info,
            assigned_words,
            grouped_turns,
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary = calculate_summary(
            conversation_id,
            whisperx_data,
            diarization_data,
            mapping_info,
            assigned_words,
        )

        summary_rows.append(summary)

        print(
            f"Saved: {output_json}"
        )

        print(
            f"Saved: {output_txt}"
        )

        print(
            f"Words assigned to speaker: "
            f"{summary['speaker_assigned_words']}/"
            f"{summary['total_words']}"
        )

        print(
            f"Unresolved words: "
            f"{summary['unresolved_words']}"
        )

        if mapping_info["status"] == (
            "UNRESOLVED_ONE_SPEAKER"
        ):

            print(
                "⚠ Pyannote detected only ONE speaker."
            )

            print(
                "⚠ All word timestamps are preserved, "
                "but CUSTOMER/AGENT role is unresolved."
            )

    # ========================================================
    # WRITE CSV SUMMARY
    # ========================================================

    if summary_rows:

        fieldnames = list(
            summary_rows[0].keys()
        )

        with open(
            SUMMARY_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(summary_rows)

        print()
        print("=" * 70)
        print("DONE")
        print("=" * 70)

        print(
            f"Summary saved to: {SUMMARY_FILE}"
        )

    else:

        print(
            "No conversations were processed."
        )


if __name__ == "__main__":
    main()