#!/usr/bin/env python3

# ============================================================
# 06_run_diarization.py
#
# Speaker diarization for VoiceSupport-50.
#
# Input:
#
#     conversations_audio/
#         conv_0001.wav
#         conv_0002.wav
#         ...
#
# Output:
#
#     diarization/
#         conv_0001.json
#         conv_0002.json
#         ...
#
#     diarization_rttm/
#         conv_0001.rttm
#         conv_0002.rttm
#         ...
#
#     diarization_summary.csv
#
# IMPORTANT:
#
# This stage identifies speaker segments.
#
# It does NOT yet decide which speaker is the customer
# and which speaker is the agent.
#
# ============================================================

import csv
import json
import os
from pathlib import Path

from pyannote.audio import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_DIR = Path(
    "conversations_audio"
)

DIARIZATION_DIR = Path(
    "diarization"
)

RTTM_DIR = Path(
    "diarization_rttm"
)

SUMMARY_FILE = Path(
    "diarization_summary.csv"
)

MODEL_NAME = (
    "pyannote/speaker-diarization-community-1"
)

# Keep this at 3 while testing.
#
# Change to:
#
# MAX_CONVERSATIONS = None
#
# when you are ready to process all conversations.
MAX_CONVERSATIONS = None


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

DIARIZATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RTTM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD HUGGING FACE TOKEN
# ============================================================

HF_TOKEN = os.environ.get(
    "HF_TOKEN"
)

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set.\n"
        "Run:\n\n"
        "export HF_TOKEN=\"hf_your_token_here\""
    )


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "VoiceSupport-50 Speaker Diarization"
)

print(
    "=" * 80
)

print(
    f"Model: {MODEL_NAME}"
)

print(
    f"Audio directory: {AUDIO_DIR}"
)

print(
    "=" * 80
)


# ============================================================
# LOAD DIARIZATION PIPELINE
# ============================================================

print(
    "\nLoading pyannote diarization pipeline..."
)

pipeline = Pipeline.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN
)

print(
    "Diarization pipeline loaded successfully."
)


# ============================================================
# FIND AUDIO FILES
# ============================================================

audio_files = sorted(
    AUDIO_DIR.glob("*.wav")
)

if not audio_files:
    raise FileNotFoundError(
        f"No WAV files found in {AUDIO_DIR}"
    )


# ============================================================
# LIMIT NUMBER OF CONVERSATIONS FOR TESTING
# ============================================================

if MAX_CONVERSATIONS is not None:
    audio_files = audio_files[
        :MAX_CONVERSATIONS
    ]


print(
    f"\nConversations to process: "
    f"{len(audio_files)}"
)


# ============================================================
# DIARIZE ONE AUDIO FILE
# ============================================================

def diarize_audio(
    audio_path
):
    """
    Run speaker diarization for one WAV file.

    The dataset contains exactly two speakers per
    conversation, so num_speakers=2 is supplied explicitly.
    """

    print(
        "\n" + "-" * 80
    )

    print(
        f"Diarizing: {audio_path.name}"
    )

    print(
        "-" * 80
    )

    output = pipeline(
        str(audio_path),
        num_speakers=2
    )

    return output


# ============================================================
# EXTRACT EXCLUSIVE DIARIZATION SEGMENTS
# ============================================================

def extract_diarization_segments(
    diarize_output
):
    """
    Extract speaker segments from the exclusive diarization
    output.

    Community-1 returns a DiarizeOutput object. The actual
    pyannote Annotation is available as:

        diarize_output.exclusive_speaker_diarization
    """

    annotation = (
        diarize_output.exclusive_speaker_diarization
    )

    segments = []

    for turn, speaker in annotation:

        start = float(
            turn.start
        )

        end = float(
            turn.end
        )

        segments.append(
            {
                "start": round(
                    start,
                    3
                ),
                "end": round(
                    end,
                    3
                ),
                "duration": round(
                    end - start,
                    3
                ),
                "speaker": str(
                    speaker
                )
            }
        )

    return segments


# ============================================================
# SAVE DIARIZATION JSON
# ============================================================

def save_diarization_json(
    conversation_id,
    audio_path,
    segments
):

    speaker_names = sorted(
        {
            segment["speaker"]
            for segment in segments
        }
    )

    result = {
        "conversation_id": (
            conversation_id
        ),
        "audio_file": str(
            audio_path
        ),
        "num_speakers_detected": len(
            speaker_names
        ),
        "speakers_detected": speaker_names,
        "segments": segments
    }

    output_file = (
        DIARIZATION_DIR
        / f"{conversation_id}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    return output_file


# ============================================================
# SAVE RTTM
# ============================================================

def save_rttm(
    conversation_id,
    diarize_output
):
    """
    Save the underlying pyannote Annotation as RTTM.

    IMPORTANT:
    DiarizeOutput itself does not provide write_rttm().
    The method belongs to the Annotation object.
    """

    output_file = (
        RTTM_DIR
        / f"{conversation_id}.rttm"
    )

    annotation = (
        diarize_output.exclusive_speaker_diarization
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as rttm_file:

        annotation.write_rttm(
            rttm_file
        )

    return output_file


# ============================================================
# PROCESS DATASET
# ============================================================

summary_rows = []


for index, audio_path in enumerate(
    audio_files,
    start=1
):

    conversation_id = (
        audio_path.stem
    )

    print(
        f"\nProcessing "
        f"{index}/{len(audio_files)}"
    )

    print(
        f"Conversation ID: "
        f"{conversation_id}"
    )

    json_output = (
        DIARIZATION_DIR
        / f"{conversation_id}.json"
    )

    rttm_output = (
        RTTM_DIR
        / f"{conversation_id}.rttm"
    )


    # --------------------------------------------------------
    # RESUME SUPPORT
    # --------------------------------------------------------

    if (
        json_output.exists()
        and rttm_output.exists()
    ):

        print(
            "Diarization output already exists. "
            "Skipping."
        )

        try:

            with open(
                json_output,
                "r",
                encoding="utf-8"
            ) as file:

                existing = json.load(
                    file
                )

            total_speech_duration = sum(
                segment.get(
                    "duration",
                    0.0
                )
                for segment in existing.get(
                    "segments",
                    []
                )
            )

            summary_rows.append(
                {
                    "conversation_id": (
                        conversation_id
                    ),
                    "audio_file": str(
                        audio_path
                    ),
                    "status": "SKIPPED",
                    "num_speakers": (
                        existing.get(
                            "num_speakers_detected",
                            0
                        )
                    ),
                    "num_segments": len(
                        existing.get(
                            "segments",
                            []
                        )
                    ),
                    "total_speech_duration": (
                        round(
                            total_speech_duration,
                            3
                        )
                    ),
                    "error": ""
                }
            )

        except Exception as error:

            summary_rows.append(
                {
                    "conversation_id": (
                        conversation_id
                    ),
                    "audio_file": str(
                        audio_path
                    ),
                    "status": "ERROR",
                    "num_speakers": 0,
                    "num_segments": 0,
                    "total_speech_duration": 0.0,
                    "error": str(
                        error
                    )
                }
            )

        continue


    # --------------------------------------------------------
    # RUN DIARIZATION
    # --------------------------------------------------------

    try:

        diarize_output = diarize_audio(
            audio_path
        )


        # ----------------------------------------------------
        # EXTRACT EXCLUSIVE SEGMENTS
        # ----------------------------------------------------

        segments = (
            extract_diarization_segments(
                diarize_output
            )
        )


        # ----------------------------------------------------
        # SAVE JSON
        # ----------------------------------------------------

        save_diarization_json(
            conversation_id,
            audio_path,
            segments
        )


        # ----------------------------------------------------
        # SAVE RTTM
        # ----------------------------------------------------

        save_rttm(
            conversation_id,
            diarize_output
        )


        # ----------------------------------------------------
        # COMPUTE STATISTICS
        # ----------------------------------------------------

        unique_speakers = sorted(
            {
                segment["speaker"]
                for segment in segments
            }
        )

        total_speech_duration = sum(
            segment["duration"]
            for segment in segments
        )


        # ----------------------------------------------------
        # SAVE SUMMARY
        # ----------------------------------------------------

        summary_rows.append(
            {
                "conversation_id": (
                    conversation_id
                ),
                "audio_file": str(
                    audio_path
                ),
                "status": "SUCCESS",
                "num_speakers": len(
                    unique_speakers
                ),
                "num_segments": len(
                    segments
                ),
                "total_speech_duration": (
                    round(
                        total_speech_duration,
                        3
                    )
                ),
                "error": ""
            }
        )


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        print(
            "\nSpeakers detected:"
        )

        for speaker in unique_speakers:

            print(
                f"  {speaker}"
            )


        print(
            "\nSegments:"
        )

        for segment in segments:

            print(
                f"  "
                f"{segment['start']:.3f}s - "
                f"{segment['end']:.3f}s "
                f"-> {segment['speaker']}"
            )


    except Exception as error:

        print(
            "\nERROR during diarization:"
        )

        print(
            error
        )

        summary_rows.append(
            {
                "conversation_id": (
                    conversation_id
                ),
                "audio_file": str(
                    audio_path
                ),
                "status": "ERROR",
                "num_speakers": 0,
                "num_segments": 0,
                "total_speech_duration": 0.0,
                "error": str(
                    error
                )
            }
        )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

fieldnames = [
    "conversation_id",
    "audio_file",
    "status",
    "num_speakers",
    "num_segments",
    "total_speech_duration",
    "error"
]


with open(
    SUMMARY_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

successful = sum(
    1
    for row in summary_rows
    if row["status"] in {
        "SUCCESS",
        "SKIPPED"
    }
)

failed = sum(
    1
    for row in summary_rows
    if row["status"] == "ERROR"
)


print(
    "\n" + "=" * 80
)

print(
    "DIARIZATION COMPLETED"
)

print(
    "=" * 80
)

print(
    f"Processed          : {len(audio_files)}"
)

print(
    f"Successful/Skipped : {successful}"
)

print(
    f"Failed             : {failed}"
)

print(
    "\nJSON output:"
)

print(
    DIARIZATION_DIR
)

print(
    "\nRTTM output:"
)

print(
    RTTM_DIR
)

print(
    "\nSummary:"
)

print(
    SUMMARY_FILE
)

print(
    "=" * 80
)