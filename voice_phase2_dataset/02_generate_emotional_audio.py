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
# By default, only the first 3 conversations are generated. Set
# VOICE_MAX_CONVERSATIONS=0 to render all; set VOICE_OVERWRITE_TURNS=1 to
# replace turn audio after rewriting the conversation text.
# ============================================================

import json
import os
from pathlib import Path

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    os.environ.get("VOICE_DATASET_FILE", "data/emotional_conversations.json")
)

OUTPUT_DIR = Path(
    os.environ.get("VOICE_TURN_AUDIO_DIR", "new_generated_turns")
)

NUM_CONVERSATIONS_TO_GENERATE = int(
    os.environ.get("VOICE_MAX_CONVERSATIONS", "3")
)
OVERWRITE_EXISTING_TURNS = os.environ.get(
    "VOICE_OVERWRITE_TURNS", "0"
).strip().casefold() in {"1", "true", "yes"}
TTS_BATCH_SIZE = int(os.environ.get("VOICE_TTS_BATCH_SIZE", "4"))


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD QWEN3-TTS MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Qwen3-TTS model on {MODEL_DEVICE}...")

tts_model = Qwen3TTSModel.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)
tts_model.model.to(MODEL_DEVICE)

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
# GENERATE A TURN BATCH
# ============================================================

def generate_turn_audio_batch(turns):
    instructions = [
        build_voice_instruction(
            role=turn["role"],
            emotion=turn["voice_attributes"]["emotion"],
            intensity=turn["voice_attributes"]["emotion_intensity"],
            speaking_rate=turn["voice_attributes"]["speaking_rate"],
            prosody=turn["voice_attributes"]["prosody"],
        )
        for turn in turns
    ]
    wavs, sample_rate = tts_model.generate_custom_voice(
        text=[turn["text"] for turn in turns],
        language=["English"] * len(turns),
        speaker=[turn["speaker"] for turn in turns],
        instruct=instructions,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
    )
    if len(wavs) != len(turns):
        raise RuntimeError(f"TTS returned {len(wavs)} clips for {len(turns)} turns")
    for turn, wav in zip(turns, wavs):
        output_path = Path(turn["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), wav, sample_rate)
        print(f"Saved: {output_path}")


# ============================================================
# LOAD CONVERSATIONS
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    conversations = json.load(f)


if NUM_CONVERSATIONS_TO_GENERATE < 0:
    raise ValueError("VOICE_MAX_CONVERSATIONS must be zero or positive.")
if TTS_BATCH_SIZE < 1:
    raise ValueError("VOICE_TTS_BATCH_SIZE must be positive.")

print(
    f"\nLoaded {len(conversations)} conversations."
)

# ============================================================
# SELECT CONVERSATIONS TO GENERATE
# ============================================================

conversations_to_generate = (
    conversations[:NUM_CONVERSATIONS_TO_GENERATE]
    if NUM_CONVERSATIONS_TO_GENERATE
    else conversations
)
generation_failures = []


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
    # PREPARE TURNS FOR BATCHED SYNTHESIS
    # ========================================================

    pending_turns = []
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

        if output_file.exists() and not OVERWRITE_EXISTING_TURNS:

            print(
                f"Skipping existing file: "
                f"{output_file}"
            )

            continue

        pending_turns.append(
            {
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "text": text,
                "role": role,
                "speaker": selected_speaker,
                "voice_attributes": voice_attributes,
                "output_path": output_file,
            }
        )

    for offset in range(0, len(pending_turns), TTS_BATCH_SIZE):
        batch = pending_turns[offset : offset + TTS_BATCH_SIZE]
        try:
            generate_turn_audio_batch(batch)
        except Exception as error:
            turn_labels = ", ".join(
                f"{item['conversation_id']} turn {item['turn_id']}" for item in batch
            )
            print(f"\nERROR generating batch ({turn_labels}): {error}")
            generation_failures.extend(
                f"{item['conversation_id']} turn {item['turn_id']}: {error}"
                for item in batch
            )


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 80)

print("All audio generation completed.")

print(f"Output directory: {OUTPUT_DIR}")

print("=" * 80)

if generation_failures:
    raise SystemExit(
        f"Audio generation failed for {len(generation_failures)} turns; "
        "see errors above and rerun to resume."
    )