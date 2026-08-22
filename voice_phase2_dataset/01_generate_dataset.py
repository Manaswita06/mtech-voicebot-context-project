#!/usr/bin/env python3

# ============================================================
# generate_dataset.py
#
# Generate coherent synthetic enterprise-support conversations.
#
# The generator:
#
# 1. Selects one scenario.
# 2. Uses only that scenario's entities.
# 3. Follows only that scenario's dialogue flow.
# 4. Uses only that scenario's templates.
# 5. Ensures customer/agent role consistency.
# 6. Maps persona gender to the correct Qwen TTS voice gender.
#
# Output:
#
# data/emotional_conversations.json
# data/emotional_conversations.jsonl
# data/metadata.csv
# ============================================================

import csv
import json
import random
from pathlib import Path

from data.lexicon import SCENARIOS


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CONVERSATIONS = 500

OUTPUT_DIR = Path("data")

JSON_FILE = OUTPUT_DIR / "emotional_conversations.json"

JSONL_FILE = OUTPUT_DIR / "emotional_conversations.jsonl"

METADATA_FILE = OUTPUT_DIR / "metadata.csv"


# Set a seed for reproducibility.
random.seed(42)


# ============================================================
# PERSONAS WITH GENDER
# ============================================================

CUSTOMER_PERSONAS = {

    "Aarav": "male",

    "Fatima": "female",

    "Rahul": "male",

    "Priya": "female",

    "Ananya": "female",

    "Vikram": "male",

    "Neha": "female",

    "Arjun": "male",

    "Meera": "female",

    "Sana": "female",

    "Rohan": "male",

    "Kavya": "female",

    "Ishaan": "male",

    "Aditi": "female",

    "Nikhil": "male"
}


AGENT_PERSONAS = {

    "Aisha": "female",

    "Meera": "female",

    "Riya": "female",

    "Karan": "male",

    "Anita": "female",

    "Rahul": "male",

    "Sneha": "female",

    "Vivek": "male",

    "Pooja": "female",

    "Arjun": "male"
}


# ============================================================
# VOICE / SPEAKER METADATA
# ============================================================

SPEAKER_INFO = {

    "aiden": {
        "gender": "male"
    },

    "dylan": {
        "gender": "male"
    },

    "eric": {
        "gender": "male"
    },

    "ryan": {
        "gender": "male"
    },

    "uncle_fu": {
        "gender": "male"
    },

    "ono_anna": {
        "gender": "female"
    },

    "serena": {
        "gender": "female"
    },

    "sohee": {
        "gender": "female"
    },

    "vivian": {
        "gender": "female"
    }
}


MALE_SPEAKERS = [

    speaker

    for speaker, info in SPEAKER_INFO.items()

    if info["gender"] == "male"
]


FEMALE_SPEAKERS = [

    speaker

    for speaker, info in SPEAKER_INFO.items()

    if info["gender"] == "female"
]


ALL_SPEAKERS = list(
    SPEAKER_INFO.keys()
)


# ============================================================
# EMOTION AND VOICE ATTRIBUTES
# ============================================================

CUSTOMER_EMOTIONS = [

    "neutral",

    "concerned",

    "frustrated",

    "worried",

    "relieved",

    "confused",

    "polite"
]


AGENT_EMOTIONS = [

    "neutral",

    "calm",

    "empathetic",

    "reassuring",

    "professional",

    "helpful"
]


SPEAKING_RATES = [

    "slow",

    "normal",

    "fast"
]


PROSODY_OPTIONS = [

    "calm",

    "natural",

    "expressive",

    "slightly emphatic",

    "steady"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def choose_split(index, total):

    train_end = int(
        total * 0.70
    )

    validation_end = int(
        total * 0.85
    )

    if index < train_end:

        return "train"

    if index < validation_end:

        return "validation"

    return "test"


# ============================================================
# SELECT SPEAKER BASED ON GENDER
# ============================================================

def select_speaker_by_gender(gender):

    if gender == "male":

        speaker = random.choice(
            MALE_SPEAKERS
        )

    elif gender == "female":

        speaker = random.choice(
            FEMALE_SPEAKERS
        )

    else:

        raise ValueError(
            f"Unknown gender: {gender}"
        )

    return speaker


# ============================================================
# SELECT AGENT SPEAKER
#
# Ensures:
#
# 1. Agent voice matches agent persona gender.
# 2. Agent does not use the same voice as customer.
# ============================================================

def select_agent_speaker(
    agent_gender,
    customer_speaker
):

    if agent_gender == "male":

        available_speakers = [

            speaker

            for speaker in MALE_SPEAKERS

            if speaker != customer_speaker
        ]

    elif agent_gender == "female":

        available_speakers = [

            speaker

            for speaker in FEMALE_SPEAKERS

            if speaker != customer_speaker
        ]

    else:

        raise ValueError(
            f"Unknown gender: {agent_gender}"
        )

    if not available_speakers:

        raise ValueError(
            "No available speaker found "
            "for the agent."
        )

    speaker = random.choice(
        available_speakers
    )

    return speaker


# ============================================================
# GENERATE VOICE ATTRIBUTES
# ============================================================

def generate_voice_attributes(role):

    if role.lower() == "customer":

        emotion = random.choice(
            CUSTOMER_EMOTIONS
        )

    else:

        emotion = random.choice(
            AGENT_EMOTIONS
        )

    emotion_intensity = round(
        random.uniform(
            0.30,
            0.90
        ),
        2
    )

    speaking_rate = random.choice(
        SPEAKING_RATES
    )

    prosody = random.choice(
        PROSODY_OPTIONS
    )

    return {

        "emotion": emotion,

        "emotion_intensity": emotion_intensity,

        "speaking_rate": speaking_rate,

        "prosody": prosody
    }


# ============================================================
# SAMPLE SCENARIO ENTITIES
# ============================================================

def sample_entities(
    entity_pools
):

    entities = {}

    for entity_name, values in entity_pools.items():

        entities[
            entity_name
        ] = random.choice(
            values
        )

    return entities


# ============================================================
# SELECT TEMPLATE
# ============================================================

def select_template(
    scenario_data,
    dialogue_act,
    role
):

    templates = scenario_data[
        "templates"
    ]

    act_templates = templates.get(
        dialogue_act,
        {}
    )

    role_templates = act_templates.get(
        role.lower(),
        []
    )

    if not role_templates:

        raise ValueError(
            f"No template found for "
            f"dialogue_act='{dialogue_act}', "
            f"role='{role}'"
        )

    return random.choice(
        role_templates
    )


# ============================================================
# RENDER TEMPLATE
# ============================================================

def render_template(
    template,
    entities
):

    try:

        return template.format(
            **entities
        )

    except KeyError as error:

        missing_entity = str(
            error
        )

        raise ValueError(
            f"Template requires missing entity "
            f"{missing_entity}: {template}"
        )


# ============================================================
# DETERMINE ROLE
# ============================================================

def determine_role(
    scenario_data,
    dialogue_act
):

    act_templates = scenario_data[
        "templates"
    ].get(
        dialogue_act,
        {}
    )

    available_roles = list(
        act_templates.keys()
    )

    if not available_roles:

        raise ValueError(
            f"No role mapping for "
            f"{dialogue_act}"
        )

    if len(available_roles) == 1:

        return available_roles[0]

    return random.choice(
        available_roles
    )


# ============================================================
# GENERATE ONE CONVERSATION
# ============================================================

def generate_conversation(
    conversation_index,
    total_conversations
):

    # --------------------------------------------------------
    # SELECT SCENARIO
    # --------------------------------------------------------

    scenario = random.choice(
        list(
            SCENARIOS.keys()
        )
    )

    scenario_data = SCENARIOS[
        scenario
    ]

    scenario_family = scenario_data[
        "family"
    ]


    # --------------------------------------------------------
    # SCENARIO-SPECIFIC ENTITIES
    # --------------------------------------------------------

    entities = sample_entities(
        scenario_data[
            "entities"
        ]
    )


    # --------------------------------------------------------
    # SELECT CUSTOMER PERSONA
    # --------------------------------------------------------

    customer_persona = random.choice(
        list(
            CUSTOMER_PERSONAS.keys()
        )
    )

    customer_gender = CUSTOMER_PERSONAS[
        customer_persona
    ]


    # --------------------------------------------------------
    # SELECT AGENT PERSONA
    # --------------------------------------------------------

    agent_persona = random.choice(
        list(
            AGENT_PERSONAS.keys()
        )
    )

    agent_gender = AGENT_PERSONAS[
        agent_persona
    ]


    # --------------------------------------------------------
    # SPEAKER ASSIGNMENT
    #
    # Customer voice matches customer persona gender.
    #
    # Agent voice matches agent persona gender.
    #
    # Customer and agent cannot have the same voice.
    # --------------------------------------------------------

    customer_speaker = (
        select_speaker_by_gender(
            customer_gender
        )
    )

    agent_speaker = (
        select_agent_speaker(
            agent_gender=agent_gender,
            customer_speaker=customer_speaker
        )
    )


    # --------------------------------------------------------
    # CONVERSATION FLOW
    # --------------------------------------------------------

    flow = scenario_data[
        "flow"
    ]

    turns = []


    for turn_index, dialogue_act in enumerate(
        flow,
        start=1
    ):

        role = determine_role(
            scenario_data,
            dialogue_act
        )

        template = select_template(
            scenario_data=scenario_data,
            dialogue_act=dialogue_act,
            role=role
        )

        text = render_template(
            template=template,
            entities=entities
        )

        voice_attributes = (
            generate_voice_attributes(
                role
            )
        )

        turn = {

            "turn_id": turn_index,

            "role": role,

            "dialogue_act": dialogue_act,

            "text": text,

            "voice_attributes": voice_attributes
        }

        turns.append(
            turn
        )


    # --------------------------------------------------------
    # RESOLUTION STATUS
    # --------------------------------------------------------

    resolution_status = "resolved"


    # --------------------------------------------------------
    # BUILD CONVERSATION
    # --------------------------------------------------------

    conversation = {

        "conversation_id": (
            f"conv_{conversation_index:04d}"
        ),

        "scenario": scenario,

        "scenario_family": scenario_family,

        "intent": scenario,

        "customer_speaker": customer_speaker,

        "customer_gender": customer_gender,

        "agent_speaker": agent_speaker,

        "agent_gender": agent_gender,

        "customer_persona": customer_persona,

        "agent_persona": agent_persona,

        "num_turns": len(
            turns
        ),

        "resolution_status": resolution_status,

        "escalated": False,

        "split": choose_split(
            conversation_index - 1,
            total_conversations
        ),

        "entities": entities,

        "turns": turns
    }

    return conversation


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_conversation(
    conversation
):

    scenario = conversation[
        "scenario"
    ]

    if scenario not in SCENARIOS:

        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    scenario_data = SCENARIOS[
        scenario
    ]

    valid_flow = scenario_data[
        "flow"
    ]

    turns = conversation[
        "turns"
    ]


    # --------------------------------------------------------
    # CHECK TURN COUNT
    # --------------------------------------------------------

    if len(
        turns
    ) != len(
        valid_flow
    ):

        raise ValueError(
            f"Incorrect number of turns for "
            f"{conversation['conversation_id']}"
        )


    # --------------------------------------------------------
    # CHECK DIALOGUE FLOW
    # --------------------------------------------------------

    for expected_act, turn in zip(
        valid_flow,
        turns
    ):

        actual_act = turn[
            "dialogue_act"
        ]

        if actual_act != expected_act:

            raise ValueError(
                f"Invalid dialogue act in "
                f"{conversation['conversation_id']}: "
                f"expected {expected_act}, "
                f"got {actual_act}"
            )


    # --------------------------------------------------------
    # CHECK ROLE IS VALID FOR ACT
    # --------------------------------------------------------

    for turn in turns:

        dialogue_act = turn[
            "dialogue_act"
        ]

        role = turn[
            "role"
        ]

        allowed_roles = scenario_data[
            "templates"
        ][
            dialogue_act
        ].keys()

        if role not in allowed_roles:

            raise ValueError(
                f"Invalid role '{role}' "
                f"for dialogue act "
                f"'{dialogue_act}'"
            )


    # --------------------------------------------------------
    # CHECK CUSTOMER VOICE GENDER
    # --------------------------------------------------------

    customer_speaker = conversation[
        "customer_speaker"
    ]

    customer_gender = conversation[
        "customer_gender"
    ]

    if SPEAKER_INFO[
        customer_speaker
    ]["gender"] != customer_gender:

        raise ValueError(
            f"Customer voice gender mismatch in "
            f"{conversation['conversation_id']}"
        )


    # --------------------------------------------------------
    # CHECK AGENT VOICE GENDER
    # --------------------------------------------------------

    agent_speaker = conversation[
        "agent_speaker"
    ]

    agent_gender = conversation[
        "agent_gender"
    ]

    if SPEAKER_INFO[
        agent_speaker
    ]["gender"] != agent_gender:

        raise ValueError(
            f"Agent voice gender mismatch in "
            f"{conversation['conversation_id']}"
        )


    # --------------------------------------------------------
    # CHECK CUSTOMER AND AGENT VOICES ARE DIFFERENT
    # --------------------------------------------------------

    if customer_speaker == agent_speaker:

        raise ValueError(
            f"Customer and agent have the same "
            f"speaker in "
            f"{conversation['conversation_id']}"
        )


    return True


# ============================================================
# GENERATE DATASET
# ============================================================

def generate_dataset():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    conversations = []

    print(
        f"Generating "
        f"{NUM_CONVERSATIONS} conversations..."
    )

    for index in range(
        1,
        NUM_CONVERSATIONS + 1
    ):

        conversation = (
            generate_conversation(
                conversation_index=index,
                total_conversations=NUM_CONVERSATIONS
            )
        )

        validate_conversation(
            conversation
        )

        conversations.append(
            conversation
        )

        if index % 50 == 0:

            print(
                f"Generated "
                f"{index}/"
                f"{NUM_CONVERSATIONS} "
                f"conversations."
            )

    return conversations


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    conversations
):

    with open(
        JSON_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            conversations,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(
    conversations
):

    with open(
        JSONL_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for conversation in conversations:

            file.write(
                json.dumps(
                    conversation,
                    ensure_ascii=False
                )
            )

            file.write(
                "\n"
            )


# ============================================================
# SAVE METADATA CSV
# ============================================================

def save_metadata(
    conversations
):

    fieldnames = [

        "conversation_id",

        "scenario",

        "scenario_family",

        "intent",

        "customer_speaker",

        "customer_gender",

        "agent_speaker",

        "agent_gender",

        "customer_persona",

        "agent_persona",

        "num_turns",

        "resolution_status",

        "escalated",

        "split"
    ]

    with open(
        METADATA_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for conversation in conversations:

            row = {

                field: conversation[
                    field
                ]

                for field in fieldnames
            }

            writer.writerow(
                row
            )


# ============================================================
# MAIN
# ============================================================

def main():

    conversations = (
        generate_dataset()
    )

    save_json(
        conversations
    )

    save_jsonl(
        conversations
    )

    save_metadata(
        conversations
    )

    print(
        "\nDataset generation completed."
    )

    print(
        f"\nJSON file:"
        f"\n{JSON_FILE}"
    )

    print(
        f"\nJSONL file:"
        f"\n{JSONL_FILE}"
    )

    print(
        f"\nMetadata file:"
        f"\n{METADATA_FILE}"
    )

    print(
        f"\nTotal conversations: "
        f"{len(conversations)}"
    )


if __name__ == "__main__":

    main()