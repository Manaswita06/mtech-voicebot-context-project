#!/usr/bin/env python3

# ============================================================
# merge_conversations.py
#
# Merge individual generated turn-level audio files into
# complete conversation-level WAV files.
#
# Speaker information is read directly from:
#
#     customer_speaker
#     customer_gender
#     agent_speaker
#     agent_gender
#
# stored in:
#
#     data/emotional_conversations.json
#
# Turn audio files are expected in:
#
#     new_generated_turns/
#         conv_0001/
#             turn_1_customer_sohee.wav
#             turn_2_agent_aiden.wav
#             ...
#
# Output:
#
#     conversations_audio/
#         conv_0001.wav
#         conv_0002.wav
#         ...
# ============================================================

import json
from pathlib import Path

from pydub import AudioSegment


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/emotional_conversations.json"
)

TURNS_DIR = Path(
    "new_generated_turns"
)

OUTPUT_DIR = Path(
    "conversations_audio"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD CONVERSATIONS
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    conversations = json.load(
        file
    )


print(
    f"Loaded {len(conversations)} conversations."
)


# ============================================================
# MERGE CONVERSATIONS
# ============================================================

for conversation_index, conversation in enumerate(
    conversations,
    start=1
):

    # --------------------------------------------------------
    # CONVERSATION INFORMATION
    # --------------------------------------------------------

    conversation_id = conversation[
        "conversation_id"
    ]

    customer_speaker = conversation[
        "customer_speaker"
    ]

    customer_gender = conversation[
        "customer_gender"
    ]

    agent_speaker = conversation[
        "agent_speaker"
    ]

    agent_gender = conversation[
        "agent_gender"
    ]


    # --------------------------------------------------------
    # DISPLAY CONVERSATION INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 80)

    print(
        f"Merging conversation "
        f"{conversation_index}/{len(conversations)}"
    )

    print(
        f"Conversation ID: "
        f"{conversation_id}"
    )

    print(
        f"Customer Speaker: "
        f"{customer_speaker} "
        f"({customer_gender})"
    )

    print(
        f"Agent Speaker: "
        f"{agent_speaker} "
        f"({agent_gender})"
    )

    print("=" * 80)


    # --------------------------------------------------------
    # CREATE EMPTY AUDIO
    # --------------------------------------------------------

    combined_audio = AudioSegment.silent(
        duration=0
    )


    # --------------------------------------------------------
    # PROCESS EACH TURN
    # --------------------------------------------------------

    for turn in conversation[
        "turns"
    ]:

        turn_id = turn[
            "turn_id"
        ]

        role = turn[
            "role"
        ].lower()


        # ----------------------------------------------------
        # SELECT SPEAKER
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

            print(
                f"WARNING: Unknown role "
                f"'{role}' in "
                f"{conversation_id}, "
                f"turn {turn_id}"
            )

            continue


        # ----------------------------------------------------
        # GET VOICE ATTRIBUTES
        # ----------------------------------------------------

        voice_attributes = turn.get(
            "voice_attributes",
            {}
        )


        # ----------------------------------------------------
        # GET PAUSE VALUES
        #
        # Defaults to 0 seconds if pause_before
        # or pause_after are not present.
        # ----------------------------------------------------

        pause_before = int(
            voice_attributes.get(
                "pause_before",
                0
            ) * 1000
        )

        pause_after = int(
            voice_attributes.get(
                "pause_after",
                0
            ) * 1000
        )


        # ----------------------------------------------------
        # BUILD TURN AUDIO FILE PATH
        #
        # This must match the naming format used by
        # generate_emotional_audio.py:
        #
        # turn_{turn_id}_{role}_{selected_speaker}.wav
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
        # CHECK IF FILE EXISTS
        # ----------------------------------------------------

        if not turn_path.exists():

            print(
                f"WARNING: Missing audio file:"
            )

            print(
                f"{turn_path}"
            )

            continue


        # ----------------------------------------------------
        # LOAD TURN AUDIO
        # ----------------------------------------------------

        audio = AudioSegment.from_wav(
            turn_path
        )


        # ----------------------------------------------------
        # ADD PAUSE BEFORE TURN
        # ----------------------------------------------------

        if pause_before > 0:

            combined_audio += (
                AudioSegment.silent(
                    duration=pause_before
                )
            )


        # ----------------------------------------------------
        # ADD TURN AUDIO
        # ----------------------------------------------------

        combined_audio += audio


        # ----------------------------------------------------
        # ADD PAUSE AFTER TURN
        # ----------------------------------------------------

        if pause_after > 0:

            combined_audio += (
                AudioSegment.silent(
                    duration=pause_after
                )
            )


        print(
            f"Added turn {turn_id}: "
            f"{role} -> "
            f"{selected_speaker}"
        )


    # --------------------------------------------------------
    # OUTPUT FILE
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"{conversation_id}.wav"
    )


    # --------------------------------------------------------
    # EXPORT COMPLETE CONVERSATION
    # --------------------------------------------------------

    combined_audio.export(
        str(output_path),
        format="wav"
    )


    print(
        f"\nCreated conversation audio:"
    )

    print(
        f"{output_path}"
    )


# ============================================================
# COMPLETED
# ============================================================

print(
    "\nAll conversations merged successfully."
)

print(
    f"Output directory: {OUTPUT_DIR}"
)