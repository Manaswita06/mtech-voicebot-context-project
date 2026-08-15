import json
from pathlib import Path

from pydub import AudioSegment


INPUT_FILE = Path(
    "data/emotional_conversations.json"
)

SPEAKER_FILE = Path(
    "data/speaker_assignments.json"
)

TURNS_DIR = Path(
    "generated_turns"
)

OUTPUT_DIR = Path(
    "conversations_audio"
)

OUTPUT_DIR.mkdir(exist_ok=True)


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    conversations = json.load(f)


with open(
    SPEAKER_FILE,
    "r",
    encoding="utf-8"
) as f:

    speaker_assignments = json.load(f)


for conversation in conversations:

    conversation_id = (
        conversation["conversation_id"]
    )

    assignment = speaker_assignments.get(
        conversation_id,
        {}
    )

    customer_speaker = assignment.get(
        "customer_speaker"
    )

    agent_speaker = assignment.get(
        "agent_speaker"
    )

    combined_audio = AudioSegment.silent(
        duration=0
    )

    for turn in conversation["turns"]:

        turn_id = turn["turn_id"]

        role = turn["role"]

        attributes = turn[
            "voice_attributes"
        ]

        pause_before = int(
            attributes["pause_before"] * 1000
        )

        pause_after = int(
            attributes["pause_after"] * 1000
        )

        if role == "customer":

            selected_speaker = customer_speaker

        else:

            selected_speaker = agent_speaker

        turn_path = (
            TURNS_DIR
            / conversation_id
            / f"turn_{turn_id}_{role}_{selected_speaker}.wav"
        )

        if not turn_path.exists():

            print(
                f"Missing: {turn_path}"
            )

            continue

        audio = AudioSegment.from_wav(
            turn_path
        )

        combined_audio += AudioSegment.silent(
            duration=pause_before
        )

        combined_audio += audio

        combined_audio += AudioSegment.silent(
            duration=pause_after
        )

    output_path = (
        OUTPUT_DIR
        / f"{conversation_id}.wav"
    )

    combined_audio.export(
        output_path,
        format="wav"
    )

    print(
        f"Created {output_path}"
    )


print("All conversations merged.")