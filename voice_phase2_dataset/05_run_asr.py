#!/usr/bin/env python3

# ============================================================
# 05_run_asr.py
#
# Automatic Speech Recognition (ASR) stage for VoiceSupport-50.
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
#     transcripts/
#         conv_0001.json
#         conv_0002.json
#         ...
#
#     transcripts_text/
#         conv_0001.txt
#         conv_0002.txt
#         ...
#
#     asr_summary.csv
#
# The generated JSON contains:
#
# - conversation_id
# - audio_file
# - language
# - full_text
# - timestamped segments
#
# IMPORTANT:
# This stage performs ASR only.
# Speaker diarization is handled separately in the next stage.
#
# ============================================================

import csv
import json
from pathlib import Path

import torch
import whisper


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_DIR = Path(
    "conversations_audio"
)

TRANSCRIPT_DIR = Path(
    "transcripts"
)

TEXT_DIR = Path(
    "transcripts_text"
)

SUMMARY_FILE = Path(
    "asr_summary.csv"
)


# ============================================================
# ASR MODEL CONFIGURATION
# ============================================================

# Recommended starting point for your 12 GB Mac:
#
# base.en
#
# Once the pipeline is working, you can experiment with:
#
# small.en
#
# Other available Whisper models include:
# tiny.en
# base.en
# small.en
# medium.en
#
# For the first complete run, base.en provides a good
# balance between speed and transcription quality.
# ============================================================

MODEL_NAME = "base.en"


# ============================================================
# PROCESSING CONFIGURATION
# ============================================================

# Set to None to process all conversations.
#
# For initial testing, you can set:
#
# MAX_CONVERSATIONS = 3
#
# After validation, change it to:
#
MAX_CONVERSATIONS = None
# ============================================================


# ============================================================
# DEVICE SELECTION
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = "mps"

elif torch.cuda.is_available():

    DEVICE = "cuda"

else:

    DEVICE = "cpu"


print(
    "\n" + "=" * 80
)

print(
    "VoiceSupport-50 ASR"
)

print(
    "=" * 80
)

print(
    f"Model  : {MODEL_NAME}"
)

print(
    f"Device : {DEVICE}"
)

print(
    "=" * 80
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

TRANSCRIPT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEXT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK INPUT DIRECTORY
# ============================================================

if not AUDIO_DIR.exists():

    raise FileNotFoundError(
        f"Audio directory not found: "
        f"{AUDIO_DIR}"
    )


# ============================================================
# LOAD WHISPER MODEL
# ============================================================

print(
    "\nLoading Whisper model..."
)

model = whisper.load_model(
    MODEL_NAME,
    device=DEVICE
)

print(
    "Whisper model loaded successfully."
)


# ============================================================
# FIND AUDIO FILES
# ============================================================

audio_files = sorted(
    AUDIO_DIR.glob("*.wav")
)


if not audio_files:

    raise FileNotFoundError(
        f"No WAV files found in "
        f"{AUDIO_DIR}"
    )


# ============================================================
# APPLY TEST LIMIT
# ============================================================

if MAX_CONVERSATIONS is not None:

    audio_files = audio_files[
        :MAX_CONVERSATIONS
    ]


print(
    f"\nAudio files to process: "
    f"{len(audio_files)}"
)


# ============================================================
# ASR FUNCTION
# ============================================================

def transcribe_audio(
    audio_path
):
    """
    Transcribe one conversation WAV file.

    Returns the complete Whisper result dictionary.
    """

    print(
        "\n" + "-" * 80
    )

    print(
        f"Transcribing: {audio_path.name}"
    )

    print(
        "-" * 80
    )


    # --------------------------------------------------------
    # WHISPER TRANSCRIPTION
    #
    # fp16=False is deliberately used for compatibility
    # across CPU/MPS environments.
    # --------------------------------------------------------

    result = model.transcribe(

        str(audio_path),

        language="en",

        task="transcribe",

        fp16=False,

        verbose=False
    )


    return result


# ============================================================
# SAVE JSON TRANSCRIPT
# ============================================================

def save_json_transcript(
    conversation_id,
    audio_path,
    result
):

    segments = []

    for segment in result.get(
        "segments",
        []
    ):

        segments.append(
            {
                "segment_id": segment.get(
                    "id"
                ),

                "start": round(
                    float(
                        segment.get(
                            "start",
                            0.0
                        )
                    ),
                    3
                ),

                "end": round(
                    float(
                        segment.get(
                            "end",
                            0.0
                        )
                    ),
                    3
                ),

                "text": segment.get(
                    "text",
                    ""
                ).strip()
            }
        )


    transcript = {

        "conversation_id": (
            conversation_id
        ),

        "audio_file": str(
            audio_path
        ),

        "language": result.get(
            "language",
            "en"
        ),

        "full_text": result.get(
            "text",
            ""
        ).strip(),

        "segments": segments
    }


    output_file = (
        TRANSCRIPT_DIR
        / f"{conversation_id}.json"
    )


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            transcript,
            file,
            indent=2,
            ensure_ascii=False
        )


    return output_file


# ============================================================
# SAVE PLAIN-TEXT TRANSCRIPT
# ============================================================

def save_text_transcript(
    conversation_id,
    result
):

    output_file = (
        TEXT_DIR
        / f"{conversation_id}.txt"
    )


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            result.get(
                "text",
                ""
            ).strip()
        )


    return output_file


# ============================================================
# PROCESS AUDIO DATASET
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
        TRANSCRIPT_DIR
        / f"{conversation_id}.json"
    )

    text_output = (
        TEXT_DIR
        / f"{conversation_id}.txt"
    )


    # --------------------------------------------------------
    # RESUME SUPPORT
    #
    # If both output files already exist, skip transcription.
    # --------------------------------------------------------

    if (
        json_output.exists()
        and text_output.exists()
    ):

        print(
            "Transcript already exists. "
            "Skipping."
        )

        try:

            with open(
                json_output,
                "r",
                encoding="utf-8"
            ) as file:

                existing_transcript = json.load(
                    file
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

                    "segment_count": len(
                        existing_transcript.get(
                            "segments",
                            []
                        )
                    ),

                    "transcript_length": len(
                        existing_transcript.get(
                            "full_text",
                            ""
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

                    "segment_count": 0,

                    "transcript_length": 0,

                    "error": str(
                        error
                    )
                }
            )

        continue


    # --------------------------------------------------------
    # TRANSCRIBE
    # --------------------------------------------------------

    try:

        result = transcribe_audio(
            audio_path
        )


        # ----------------------------------------------------
        # SAVE JSON
        # ----------------------------------------------------

        save_json_transcript(
            conversation_id,
            audio_path,
            result
        )


        # ----------------------------------------------------
        # SAVE TEXT
        # ----------------------------------------------------

        save_text_transcript(
            conversation_id,
            result
        )


        # ----------------------------------------------------
        # COLLECT SUMMARY INFORMATION
        # ----------------------------------------------------

        segment_count = len(
            result.get(
                "segments",
                []
            )
        )

        transcript_text = (
            result.get(
                "text",
                ""
            ).strip()
        )


        summary_rows.append(
            {
                "conversation_id": (
                    conversation_id
                ),

                "audio_file": str(
                    audio_path
                ),

                "status": "SUCCESS",

                "segment_count": (
                    segment_count
                ),

                "transcript_length": (
                    len(
                        transcript_text
                    )
                ),

                "error": ""
            }
        )


        print(
            "\nTranscript:"
        )

        print(
            transcript_text
        )


        print(
            "\nSegments:"
        )

        print(
            segment_count
        )


    except Exception as error:

        print(
            "\nERROR:"
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

                "segment_count": 0,

                "transcript_length": 0,

                "error": str(
                    error
                )
            }
        )


# ============================================================
# SAVE ASR SUMMARY CSV
# ============================================================

fieldnames = [

    "conversation_id",

    "audio_file",

    "status",

    "segment_count",

    "transcript_length",

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
    "ASR COMPLETED"
)

print(
    "=" * 80
)

print(
    f"Processed : {len(audio_files)}"
)

print(
    f"Successful/Skipped : {successful}"
)

print(
    f"Failed    : {failed}"
)

print(
    f"\nJSON transcripts:"
)

print(
    TRANSCRIPT_DIR
)

print(
    f"\nText transcripts:"
)

print(
    TEXT_DIR
)

print(
    f"\nSummary:"
)

print(
    SUMMARY_FILE
)

print(
    "=" * 80
)