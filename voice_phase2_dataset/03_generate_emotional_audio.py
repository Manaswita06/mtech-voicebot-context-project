import json
import random
from pathlib import Path

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/emotional_conversations.json")

OUTPUT_DIR = Path("generated_turns")

# Separate JSON file to store speaker assignments
SPEAKER_INFO_FILE = Path(
    "data/speaker_assignments.json"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SPEAKER_INFO_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

# Makes speaker assignments reproducible
random.seed(42)


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
# AVAILABLE SPEAKERS
# ============================================================

# All speakers can be assigned to either role
CUSTOMER_SPEAKERS = ["aiden", "serena", "eric", "sohee"]
AGENT_SPEAKERS = ["dylan", "ono_anna", "ryan", "uncle_fu", "vivian"]


print("\nAvailable speakers:")

for speaker in CUSTOMER_SPEAKERS+AGENT_SPEAKERS:
    print(f" - {speaker}")


# ============================================================
# LOAD EXISTING SPEAKER ASSIGNMENTS
# ============================================================

if SPEAKER_INFO_FILE.exists():

    with open(
        SPEAKER_INFO_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        speaker_assignments = json.load(f)

    print(
        f"\nLoaded existing speaker assignments for "
        f"{len(speaker_assignments)} conversations."
    )

else:

    speaker_assignments = {}

    print(
        "\nNo existing speaker assignment file found."
    )


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
        "in pitch, rhythm, and emphasis. "
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

    print(
        f"\nSaved: {output_path}"
    )


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
# GENERATE ALL CONVERSATIONS
# ============================================================

for conversation_index, conversation in enumerate(
    conversations[:2],
    start=1
):

    conversation_id = conversation["conversation_id"]


    # ========================================================
    # SELECT OR REUSE SPEAKERS
    # ========================================================

    if conversation_id in speaker_assignments:

        customer_speaker = (
            speaker_assignments[conversation_id]
            ["customer_speaker"]
        )

        agent_speaker = (
            speaker_assignments[conversation_id]
            ["agent_speaker"]
        )

        print(
            f"\nReusing existing speaker assignments "
            f"for {conversation_id}"
        )

    else:

        # Select customer speaker
        # customer_speaker = random.choice(
        #     CUSTOMER_SPEAKERS
        # )
        customer_speaker = conversation["customer_name"]

        agent_speaker = random.choice(
            AGENT_SPEAKERS
        )


        # ====================================================
        # STORE SPEAKER INFORMATION
        # ====================================================

        speaker_assignments[conversation_id] = {

            "scenario": conversation["scenario"],

            "customer_speaker": customer_speaker,

            "agent_speaker": agent_speaker

        }


        # Save immediately so progress is preserved
        with open(
            SPEAKER_INFO_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                speaker_assignments,
                f,
                indent=4,
                ensure_ascii=False
            )


    # ========================================================
    # DISPLAY CONVERSATION INFORMATION
    # ========================================================

    print("\n" + "#" * 80)

    print(
        f"Conversation "
        f"{conversation_index}/{len(conversations)}"
    )

    print(f"ID: {conversation_id}")

    print(
        f"Scenario: {conversation['scenario']}"
    )

    print(
        f"Customer Speaker: {customer_speaker}"
    )

    print(
        f"Agent Speaker: {agent_speaker}"
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


        # ----------------------------------------------------
        # SELECT CONSISTENT SPEAKER
        # ----------------------------------------------------

        if role.lower() == "customer":

            selected_speaker = customer_speaker

        else:

            selected_speaker = agent_speaker


        # ----------------------------------------------------
        # CREATE OUTPUT FILE NAME
        # ----------------------------------------------------

        output_file = (
            conversation_dir /
            f"turn_{turn_id}_{role}_{selected_speaker}.wav"
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
                text=turn["text"],
                role=role,
                speaker=selected_speaker,
                voice_attributes=turn["voice_attributes"],
                output_path=output_file
            )

        except Exception as e:

            print(
                f"\nERROR generating "
                f"{conversation_id}, "
                f"turn {turn_id}"
            )

            print(e)


# ============================================================
# FINAL SAVE OF SPEAKER ASSIGNMENTS
# ============================================================

with open(
    SPEAKER_INFO_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        speaker_assignments,
        f,
        indent=4,
        ensure_ascii=False
    )


print(
    "\nAll audio generation completed."
)

print(
    f"Speaker information saved to: "
    f"{SPEAKER_INFO_FILE}"
)