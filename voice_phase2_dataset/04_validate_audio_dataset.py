#!/usr/bin/env python3

# ============================================================
# 04_validate_audio_dataset.py
#
# Validate the generated VoiceSupport-50 audio dataset before
# running ASR, speaker diarization, and downstream state
# extraction.
#
# The script checks:
#
# 1. Conversation metadata exists.
# 2. Conversation-level WAV files exist.
# 3. Conversation-level WAV files can be opened.
# 4. Audio files are non-empty.
# 5. Sample rate is consistent.
# 6. Number of channels is consistent.
# 7. Conversation duration is valid.
# 8. Customer/agent speaker metadata exists.
# 9. Expected turn-level WAV files exist.
# 10. Turn-level WAV files are readable and non-empty.
#
# Outputs:
#
#     validation_results.csv
#     validation_summary.json
#
# ============================================================

import csv
import json
import wave
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/emotional_conversations.json"
)

TURNS_DIR = Path(
    "new_generated_turns"
)

CONVERSATIONS_AUDIO_DIR = Path(
    "conversations_audio"
)

VALIDATION_CSV = Path(
    "validation_results.csv"
)

VALIDATION_SUMMARY = Path(
    "validation_summary.json"
)


# ============================================================
# EXPECTED AUDIO CONFIGURATION
# ============================================================

# Qwen3-TTS commonly generates mono speech audio.
#
# We do not force the sample rate to one specific value here.
# Instead, the script checks consistency across the dataset.
# This prevents false failures if your generated sample rate
# differs from what was expected.
#
# Set to None to disable duration checks.
MIN_DURATION_SECONDS = 0.1

# Warn when a conversation becomes unusually long.
# This does not automatically mark the file as invalid.
MAX_DURATION_SECONDS = 600.0


# ============================================================
# SPEAKER METADATA VALIDATION
# ============================================================

VALID_GENDERS = {
    "male",
    "female"
}

VALID_ROLES = {
    "customer",
    "agent"
}


# ============================================================
# HELPER: READ WAV INFORMATION
# ============================================================

def read_wav_info(
    wav_path
):
    """
    Read basic WAV metadata using Python's built-in wave module.

    Returns:
        dict with:
        - sample_rate
        - channels
        - sample_width
        - frames
        - duration_seconds
    """

    with wave.open(
        str(wav_path),
        "rb"
    ) as wav_file:

        sample_rate = (
            wav_file.getframerate()
        )

        channels = (
            wav_file.getnchannels()
        )

        sample_width = (
            wav_file.getsampwidth()
        )

        frames = (
            wav_file.getnframes()
        )

        duration_seconds = (
            frames / sample_rate
            if sample_rate > 0
            else 0.0
        )

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "frames": frames,
        "duration_seconds": duration_seconds
    }


# ============================================================
# HELPER: VALIDATE ONE WAV FILE
# ============================================================

def validate_wav_file(
    wav_path
):
    """
    Validate a single WAV file.

    Returns:
        status:
            PASS / FAIL / WARNING

        info:
            dictionary containing audio metadata.

        errors:
            list of validation errors.

        warnings:
            list of warnings.
    """

    errors = []

    warnings = []

    info = {
        "sample_rate": None,
        "channels": None,
        "sample_width": None,
        "frames": None,
        "duration_seconds": None
    }


    # --------------------------------------------------------
    # CHECK FILE EXISTS
    # --------------------------------------------------------

    if not wav_path.exists():

        errors.append(
            "File does not exist"
        )

        return (
            "FAIL",
            info,
            errors,
            warnings
        )


    # --------------------------------------------------------
    # CHECK FILE SIZE
    # --------------------------------------------------------

    file_size = (
        wav_path.stat().st_size
    )

    if file_size == 0:

        errors.append(
            "File is empty"
        )

        return (
            "FAIL",
            info,
            errors,
            warnings
        )


    # --------------------------------------------------------
    # TRY TO READ WAV
    # --------------------------------------------------------

    try:

        wav_info = read_wav_info(
            wav_path
        )

        info.update(
            wav_info
        )

    except Exception as error:

        errors.append(
            f"Unable to read WAV file: {error}"
        )

        return (
            "FAIL",
            info,
            errors,
            warnings
        )


    # --------------------------------------------------------
    # SAMPLE RATE
    # --------------------------------------------------------

    if info["sample_rate"] <= 0:

        errors.append(
            "Invalid sample rate"
        )


    # --------------------------------------------------------
    # CHANNEL COUNT
    # --------------------------------------------------------

    if info["channels"] <= 0:

        errors.append(
            "Invalid channel count"
        )


    # --------------------------------------------------------
    # AUDIO FRAMES
    # --------------------------------------------------------

    if info["frames"] <= 0:

        errors.append(
            "Audio contains no frames"
        )


    # --------------------------------------------------------
    # DURATION CHECK
    # --------------------------------------------------------

    duration = (
        info["duration_seconds"]
    )

    if duration is None:

        errors.append(
            "Could not determine audio duration"
        )

    elif duration < MIN_DURATION_SECONDS:

        errors.append(
            f"Audio duration too short: "
            f"{duration:.3f} seconds"
        )

    elif duration > MAX_DURATION_SECONDS:

        warnings.append(
            f"Unusually long audio file: "
            f"{duration:.2f} seconds"
        )


    # --------------------------------------------------------
    # DETERMINE STATUS
    # --------------------------------------------------------

    if errors:

        status = "FAIL"

    elif warnings:

        status = "WARNING"

    else:

        status = "PASS"


    return (
        status,
        info,
        errors,
        warnings
    )


# ============================================================
# VALIDATE CONVERSATION METADATA
# ============================================================

def validate_conversation_metadata(
    conversation
):
    """
    Validate metadata stored in emotional_conversations.json.

    Returns:
        errors
        warnings
    """

    errors = []

    warnings = []

    conversation_id = (
        conversation.get(
            "conversation_id"
        )
    )

    # --------------------------------------------------------
    # CONVERSATION ID
    # --------------------------------------------------------

    if not conversation_id:

        errors.append(
            "Missing conversation_id"
        )


    # --------------------------------------------------------
    # SCENARIO
    # --------------------------------------------------------

    if not conversation.get(
        "scenario"
    ):

        errors.append(
            "Missing scenario"
        )


    # --------------------------------------------------------
    # SPEAKER METADATA
    # --------------------------------------------------------

    customer_speaker = (
        conversation.get(
            "customer_speaker"
        )
    )

    customer_gender = (
        conversation.get(
            "customer_gender"
        )
    )

    agent_speaker = (
        conversation.get(
            "agent_speaker"
        )
    )

    agent_gender = (
        conversation.get(
            "agent_gender"
        )
    )


    if not customer_speaker:

        errors.append(
            "Missing customer_speaker"
        )


    if not agent_speaker:

        errors.append(
            "Missing agent_speaker"
        )


    if customer_gender not in VALID_GENDERS:

        errors.append(
            f"Invalid customer_gender: "
            f"{customer_gender}"
        )


    if agent_gender not in VALID_GENDERS:

        errors.append(
            f"Invalid agent_gender: "
            f"{agent_gender}"
        )


    # --------------------------------------------------------
    # CUSTOMER AND AGENT SHOULD NOT SHARE A VOICE
    # --------------------------------------------------------

    if (
        customer_speaker
        and agent_speaker
        and customer_speaker == agent_speaker
    ):

        errors.append(
            "Customer and agent use the same speaker"
        )


    # --------------------------------------------------------
    # TURNS
    # --------------------------------------------------------

    turns = conversation.get(
        "turns",
        []
    )

    if not turns:

        errors.append(
            "Conversation has no turns"
        )

        return (
            errors,
            warnings
        )


    # --------------------------------------------------------
    # VALIDATE TURN ROLES
    # --------------------------------------------------------

    for turn in turns:

        role = (
            turn.get(
                "role"
            )
        )

        turn_id = (
            turn.get(
                "turn_id"
            )
        )

        if role not in VALID_ROLES:

            errors.append(
                f"Turn {turn_id}: "
                f"invalid role '{role}'"
            )


    return (
        errors,
        warnings
    )


# ============================================================
# VALIDATE ONE CONVERSATION
# ============================================================

def validate_conversation(
    conversation
):
    """
    Validate all files associated with one conversation.

    Returns:
        validation record dictionary
    """

    conversation_id = (
        conversation[
            "conversation_id"
        ]
    )

    scenario = (
        conversation.get(
            "scenario",
            ""
        )
    )

    customer_speaker = (
        conversation.get(
            "customer_speaker"
        )
    )

    agent_speaker = (
        conversation.get(
            "agent_speaker"
        )
    )


    all_errors = []

    all_warnings = []


    # ========================================================
    # VALIDATE CONVERSATION METADATA
    # ========================================================

    metadata_errors, metadata_warnings = (
        validate_conversation_metadata(
            conversation
        )
    )

    all_errors.extend(
        metadata_errors
    )

    all_warnings.extend(
        metadata_warnings
    )


    # ========================================================
    # VALIDATE MERGED CONVERSATION WAV
    # ========================================================

    merged_audio_path = (
        CONVERSATIONS_AUDIO_DIR
        / f"{conversation_id}.wav"
    )

    (
        merged_status,
        merged_info,
        merged_errors,
        merged_warnings
    ) = validate_wav_file(
        merged_audio_path
    )

    all_errors.extend(
        [
            f"Conversation WAV: {error}"
            for error in merged_errors
        ]
    )

    all_warnings.extend(
        [
            f"Conversation WAV: {warning}"
            for warning in merged_warnings
        ]
    )


    # ========================================================
    # VALIDATE TURN AUDIO
    # ========================================================

    turns = conversation.get(
        "turns",
        []
    )

    expected_turns = len(
        turns
    )

    found_turns = 0

    missing_turns = 0

    invalid_turns = 0

    turn_sample_rates = set()

    turn_channels = set()

    total_turn_duration = 0.0


    for turn in turns:

        turn_id = (
            turn["turn_id"]
        )

        role = (
            turn["role"].lower()
        )


        # ----------------------------------------------------
        # DETERMINE EXPECTED SPEAKER
        # ----------------------------------------------------

        if role == "customer":

            selected_speaker = (
                customer_speaker
            )

        elif role == "agent":

            selected_speaker = (
                agent_speaker
            )

        else:

            all_errors.append(
                f"Turn {turn_id}: "
                f"unknown role '{role}'"
            )

            invalid_turns += 1

            continue


        # ----------------------------------------------------
        # EXPECTED TURN FILE
        # ----------------------------------------------------

        turn_path = (
            TURNS_DIR
            / conversation_id
            / (
                f"turn_{turn_id}_"
                f"{role}_"
                f"{selected_speaker}.wav"
            )
        )


        # ----------------------------------------------------
        # CHECK TURN AUDIO
        # ----------------------------------------------------

        if not turn_path.exists():

            missing_turns += 1

            all_errors.append(
                f"Missing turn audio: "
                f"{turn_path}"
            )

            continue


        found_turns += 1


        (
            turn_status,
            turn_info,
            turn_errors,
            turn_warnings
        ) = validate_wav_file(
            turn_path
        )


        if turn_status == "FAIL":

            invalid_turns += 1


        all_errors.extend(
            [
                f"Turn {turn_id}: {error}"
                for error in turn_errors
            ]
        )


        all_warnings.extend(
            [
                f"Turn {turn_id}: {warning}"
                for warning in turn_warnings
            ]
        )


        # ----------------------------------------------------
        # COLLECT AUDIO STATISTICS
        # ----------------------------------------------------

        if turn_info[
            "sample_rate"
        ] is not None:

            turn_sample_rates.add(
                turn_info[
                    "sample_rate"
                ]
            )


        if turn_info[
            "channels"
        ] is not None:

            turn_channels.add(
                turn_info[
                    "channels"
                ]
            )


        if turn_info[
            "duration_seconds"
        ] is not None:

            total_turn_duration += (
                turn_info[
                    "duration_seconds"
                ]
            )


    # ========================================================
    # CHECK TURN AUDIO CONSISTENCY
    # ========================================================

    if len(
        turn_sample_rates
    ) > 1:

        all_warnings.append(
            "Turn-level files have different sample rates"
        )


    if len(
        turn_channels
    ) > 1:

        all_warnings.append(
            "Turn-level files have different channel counts"
        )


    # ========================================================
    # CHECK MERGED AUDIO VS TURN AUDIO
    # ========================================================

    merged_duration = (
        merged_info[
            "duration_seconds"
        ]
    )

    # ========================================================
    # CHECK MERGED AUDIO VS TURN AUDIO
    # ========================================================

    merged_duration = (
        merged_info["duration_seconds"]
    )

    duration_difference = None

    if (
            merged_duration is not None
            and total_turn_duration > 0
    ):

        duration_difference = (
                merged_duration - total_turn_duration
        )

        # Allow a small tolerance for WAV metadata,
        # resampling, and floating-point rounding.
        duration_tolerance = 0.25

        if duration_difference < -duration_tolerance:
            all_warnings.append(
                "Merged conversation duration is shorter "
                f"than total turn duration by "
                f"{abs(duration_difference):.3f} seconds"
            )


    # ========================================================
    # DETERMINE CONVERSATION STATUS
    # ========================================================

    if all_errors:

        overall_status = "FAIL"

    elif all_warnings:

        overall_status = "WARNING"

    else:

        overall_status = "PASS"


    # ========================================================
    # BUILD VALIDATION RECORD
    # ========================================================

    return {

        "conversation_id": conversation_id,

        "scenario": scenario,

        "customer_speaker": customer_speaker,

        "agent_speaker": agent_speaker,

        "expected_turns": expected_turns,

        "found_turns": found_turns,

        "missing_turns": missing_turns,

        "invalid_turns": invalid_turns,

        "merged_audio_exists": (
            merged_audio_path.exists()
        ),

        "merged_audio_duration_seconds": (
            merged_duration
            if merged_duration is not None
            else 0.0
        ),

        "turn_audio_duration_seconds": (
            round(
                total_turn_duration,
                3
            )
        ),

        "sample_rates": (
            ", ".join(
                str(rate)
                for rate in sorted(
                    turn_sample_rates
                )
            )
        ),

        "channel_counts": (
            ", ".join(
                str(channels)
                for channels in sorted(
                    turn_channels
                )
            )
        ),

        "status": overall_status,

        "errors": " | ".join(
            all_errors
        ),

        "warnings": " | ".join(
            all_warnings
        )
    }


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print(
        "\n" + "=" * 80
    )

    print(
        "VoiceSupport-50 Audio Dataset Validation"
    )

    print(
        "=" * 80
    )


    # --------------------------------------------------------
    # CHECK INPUT FILE
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Metadata file not found: "
            f"{INPUT_FILE}"
        )


    # --------------------------------------------------------
    # LOAD DATASET METADATA
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        conversations = json.load(
            file
        )


    total_conversations = len(
        conversations
    )


    print(
        f"\nConversations found in metadata: "
        f"{total_conversations}"
    )


    # --------------------------------------------------------
    # VALIDATE ALL CONVERSATIONS
    # --------------------------------------------------------

    validation_records = []


    for index, conversation in enumerate(
        conversations,
        start=1
    ):

        conversation_id = (
            conversation.get(
                "conversation_id",
                f"unknown_{index}"
            )
        )


        print(
            f"\n[{index}/{total_conversations}] "
            f"Validating {conversation_id}..."
        )


        record = validate_conversation(
            conversation
        )


        validation_records.append(
            record
        )


        print(
            f"Status: {record['status']}"
        )


        if record["errors"]:

            print(
                "Errors:"
            )

            for error in record[
                "errors"
            ].split(" | "):

                if error:

                    print(
                        f"  - {error}"
                    )


        if record["warnings"]:

            print(
                "Warnings:"
            )

            for warning in record[
                "warnings"
            ].split(" | "):

                if warning:

                    print(
                        f"  - {warning}"
                    )


    # ========================================================
    # CALCULATE SUMMARY
    # ========================================================

    passed = sum(
        1
        for record in validation_records
        if record["status"] == "PASS"
    )


    warnings = sum(
        1
        for record in validation_records
        if record["status"] == "WARNING"
    )


    failed = sum(
        1
        for record in validation_records
        if record["status"] == "FAIL"
    )


    total_expected_turns = sum(
        record["expected_turns"]
        for record in validation_records
    )


    total_found_turns = sum(
        record["found_turns"]
        for record in validation_records
    )


    total_missing_turns = sum(
        record["missing_turns"]
        for record in validation_records
    )


    total_invalid_turns = sum(
        record["invalid_turns"]
        for record in validation_records
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = [

        "conversation_id",

        "scenario",

        "customer_speaker",

        "agent_speaker",

        "expected_turns",

        "found_turns",

        "missing_turns",

        "invalid_turns",

        "merged_audio_exists",

        "merged_audio_duration_seconds",

        "turn_audio_duration_seconds",

        "sample_rates",

        "channel_counts",

        "status",

        "errors",

        "warnings"
    ]


    with open(
        VALIDATION_CSV,
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
            validation_records
        )


    # ========================================================
    # SAVE JSON SUMMARY
    # ========================================================

    summary = {

        "total_conversations": (
            total_conversations
        ),

        "passed": passed,

        "warnings": warnings,

        "failed": failed,

        "total_expected_turns": (
            total_expected_turns
        ),

        "total_found_turns": (
            total_found_turns
        ),

        "total_missing_turns": (
            total_missing_turns
        ),

        "total_invalid_turns": (
            total_invalid_turns
        ),

        "conversation_audio_directory": (
            str(
                CONVERSATIONS_AUDIO_DIR
            )
        ),

        "turn_audio_directory": (
            str(
                TURNS_DIR
            )
        ),

        "metadata_file": (
            str(
                INPUT_FILE
            )
        )
    }


    with open(
        VALIDATION_SUMMARY,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "VALIDATION SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"Total conversations : "
        f"{total_conversations}"
    )

    print(
        f"PASS                : "
        f"{passed}"
    )

    print(
        f"WARNING             : "
        f"{warnings}"
    )

    print(
        f"FAIL                : "
        f"{failed}"
    )

    print(
        f"Expected turns      : "
        f"{total_expected_turns}"
    )

    print(
        f"Found turns         : "
        f"{total_found_turns}"
    )

    print(
        f"Missing turns       : "
        f"{total_missing_turns}"
    )

    print(
        f"Invalid turns       : "
        f"{total_invalid_turns}"
    )

    print(
        "\nValidation CSV:"
    )

    print(
        VALIDATION_CSV
    )

    print(
        "\nValidation summary:"
    )

    print(
        VALIDATION_SUMMARY
    )

    print(
        "=" * 80
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()