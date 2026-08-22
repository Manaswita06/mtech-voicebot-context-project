#!/usr/bin/env python3

# ============================================================
# generate_emotional_audio.py
#
# Generate emotional WAV files from conversations created by
# generate_dataset.py.
#
# This script uses the speaker assignments already stored in:
#
#     customer_speaker
#     agent_speaker
#
# For now, only the first 3 conversations are generated.
# ============================================================

import json
from pathlib import Path

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/emotional_conversations.json")

OUTPUT_DIR = Path("new_generated_turns")

# Keep this as 3 for now.
NUM_CONVERSATIONS_TO_GENERATE = 3


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD QWEN3-TTS MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

print("Loading Qwen3-TTS model...")

tts_model = Qwen3TTSModel.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

print("Qwen3-TTS model loaded successfully.")


# ============================================================
# BUILD VOICE INSTRUCTION
# ============================================================

def build_voice_instruction(
    role,
    emotion,
    intensity,
    speaking_rate,
    prosody
):
    base = (
        f"Speak as a {role} in an enterprise customer "
        f"support phone conversation. "
    )

    emotion_instruction = (
        f"The emotional style is {emotion}. "
        f"Use approximately {intensity:.2f} emotional intensity. "
    )

    if speaking_rate == "fast":
        rate_instruction = (
            "Speak noticeably faster than normal, "
            "but remain understandable. "
        )

    elif speaking_rate == "slow":
        rate_instruction = (
            "Speak slowly with natural hesitation "
            "and clear pauses. "
        )

    else:
        rate_instruction = (
            "Speak at a natural conversational pace. "
        )

    prosody_instruction = (
        f"Use {prosody} prosody with natural variation "
        f"in pitch, rhythm, and emphasis. "
    )

    return (
        base
        + emotion_instruction
        + rate_instruction
        + prosody_instruction
        + "Do not sound robotic or monotone."
    )


# ============================================================
# GENERATE ONE TURN
# ============================================================

def generate_turn_audio(
    text,
    role,
    speaker,
    voice_attributes,
    output_path
):
    instruction = build_voice_instruction(
        role=role,
        emotion=voice_attributes["emotion"],
        intensity=voice_attributes["emotion_intensity"],
        speaking_rate=voice_attributes["speaking_rate"],
        prosody=voice_attributes["prosody"]
    )

    print("\n" + "=" * 70)
    print(f"Generating {role} audio")
    print(f"Speaker: {speaker}")
    print(f"Emotion: {voice_attributes['emotion']}")
    print(
        f"Intensity: "
        f"{voice_attributes['emotion_intensity']}"
    )
    print(
        f"Speaking Rate: "
        f"{voice_attributes['speaking_rate']}"
    )
    print(
        f"Prosody: "
        f"{voice_attributes['prosody']}"
    )

    print("\nText:")
    print(text)

    print("\nInstruction:")
    print(instruction)

    print("=" * 70)

    # --------------------------------------------------------
    # GENERATE AUDIO
    # --------------------------------------------------------

    wavs, sample_rate = tts_model.generate_custom_voice(
        text=text,
        language="English",
        speaker=speaker,
        instruct=instruction,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05
    )

    # --------------------------------------------------------
    # ENSURE OUTPUT DIRECTORY EXISTS
    # --------------------------------------------------------

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAVE WAV FILE
    # --------------------------------------------------------

    sf.write(
        str(output_path),
        wavs[0],
        sample_rate
    )

    print(f"\nSaved: {output_path}")


# ============================================================
# LOAD CONVERSATIONS
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    conversations = json.load(f)


print(
    f"\nLoaded {len(conversations)} conversations."
)


# ============================================================
# GENERATE ONLY FIRST 3 CONVERSATIONS
# ============================================================

# conversations_to_generate = conversations[
#     :NUM_CONVERSATIONS_TO_GENERATE
# ]
conversations_to_generate = conversations


print(
    f"\nGenerating audio for "
    f"{len(conversations_to_generate)} conversations."
)


# ============================================================
# GENERATE ALL CONVERSATIONS
# ============================================================

for conversation_index, conversation in enumerate(
    conversations_to_generate,
    start=1
):

    conversation_id = conversation["conversation_id"]

    scenario = conversation["scenario"]

    # ========================================================
    # USE SPEAKERS ALREADY ASSIGNED BY generate_dataset.py
    # ========================================================

    customer_speaker = conversation["customer_speaker"]

    agent_speaker = conversation["agent_speaker"]

    customer_gender = conversation.get(
        "customer_gender",
        "unknown"
    )

    agent_gender = conversation.get(
        "agent_gender",
        "unknown"
    )

    # ========================================================
    # DISPLAY CONVERSATION INFORMATION
    # ========================================================

    print("\n" + "#" * 80)

    print(
        f"Conversation "
        f"{conversation_index}/"
        f"{len(conversations_to_generate)}"
    )

    print(f"ID: {conversation_id}")

    print(f"Scenario: {scenario}")

    print(
        f"Customer Speaker: {customer_speaker}"
    )

    print(
        f"Customer Gender: {customer_gender}"
    )

    print(
        f"Agent Speaker: {agent_speaker}"
    )

    print(
        f"Agent Gender: {agent_gender}"
    )

    print("#" * 80)

    # ========================================================
    # CREATE CONVERSATION DIRECTORY
    # ========================================================

    conversation_dir = (
        OUTPUT_DIR / conversation_id
    )

    conversation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # GENERATE EACH TURN
    # ========================================================

    for turn in conversation["turns"]:

        turn_id = turn["turn_id"]

        role = turn["role"]

        text = turn["text"]

        voice_attributes = turn["voice_attributes"]

        # ----------------------------------------------------
        # SELECT THE CORRECT SPEAKER
        # ----------------------------------------------------

        if role.lower() == "customer":
            selected_speaker = customer_speaker

        elif role.lower() == "agent":
            selected_speaker = agent_speaker

        else:
            raise ValueError(
                f"Unknown role '{role}' "
                f"in {conversation_id}, "
                f"turn {turn_id}"
            )

        # ----------------------------------------------------
        # CREATE OUTPUT FILE NAME
        # ----------------------------------------------------

        output_file = (
            conversation_dir
            / f"turn_{turn_id}_{role}_{selected_speaker}.wav"
        )

        # ----------------------------------------------------
        # SKIP EXISTING FILES
        # ----------------------------------------------------

        if output_file.exists():

            print(
                f"Skipping existing file: "
                f"{output_file}"
            )

            continue

        # ----------------------------------------------------
        # GENERATE AUDIO
        # ----------------------------------------------------

        try:

            generate_turn_audio(
                text=text,
                role=role,
                speaker=selected_speaker,
                voice_attributes=voice_attributes,
                output_path=output_file
            )

        except Exception as e:

            print(
                f"\nERROR generating "
                f"{conversation_id}, "
                f"turn {turn_id}"
            )

            print(f"Role: {role}")

            print(
                f"Speaker: {selected_speaker}"
            )

            print(f"Error: {e}")


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 80)

print("All audio generation completed.")

print(f"Output directory: {OUTPUT_DIR}")

print("=" * 80)