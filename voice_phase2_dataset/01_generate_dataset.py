import json
import random
import csv
from pathlib import Path

from data.scenarios import SCENARIOS


OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)


CUSTOMER_NAMES = ["aiden", "serena", "eric", "sohee"]


TEMPLATES = {

    "CARD_REPLACEMENT": [
        "My card is damaged and I need a replacement.",
        "I need to replace my card as soon as possible."
    ],

    "PAYMENT_FAILED": [
        "My payment keeps getting declined.",
        "I have tried several times but my payment is failing."
    ],

    "PAYMENT_PENDING": [
        "My payment has been pending for a long time.",
        "Can you tell me why my transaction is still pending?"
    ],

    "DUPLICATE_CHARGE": [
        "I think I was charged twice for the same transaction.",
        "There are two identical charges on my account."
    ],

    "UNRECOGNIZED_TRANSACTION": [
        "There is a transaction that I do not recognize.",
        "I see a charge on my account that I didn't make."
    ],

    "REFUND_REQUEST": [
        "I would like to request a refund.",
        "Can you help me get a refund for this purchase?"
    ],

    "REFUND_PENDING": [
        "My refund has not arrived yet.",
        "I was told my refund was processed, but I haven't received it."
    ],

    "FRAUD_DISPUTE": [
        "I believe someone has used my account without permission.",
        "There are fraudulent transactions on my account."
    ],

    "LOGIN_ISSUE": [
        "I cannot log in to my account.",
        "The application won't let me sign in."
    ],

    "ACCOUNT_LOCKED": [
        "My account has been locked.",
        "I entered my password incorrectly and now I can't access my account."
    ]
}


DEFAULT_TEMPLATES = [
    "I need some help with {scenario}.",
    "I'm contacting support regarding {scenario}.",
    "Can you help me resolve an issue related to {scenario}?"
]


AGENT_OPENINGS = [
    "I understand. Let me help you with that.",
    "I'm sorry you're experiencing this issue. I'll look into it.",
    "Certainly. Let me check the details for you."
]


AGENT_CLOSINGS = [
    "The issue has now been resolved.",
    "I have completed the requested action.",
    "Is there anything else I can help you with today?"
]


def create_conversation(
    conversation_id,
    scenario,
    family
):

    templates = TEMPLATES.get(
        scenario,
        DEFAULT_TEMPLATES
    )

    customer_message = random.choice(
        templates
    )

    customer_message = customer_message.replace(
        "{scenario}",
        scenario.replace("_", " ").lower()
    )

    turns = [

        {
            "turn_id": 1,
            "role": "customer",
            "text": customer_message
        },

        {
            "turn_id": 2,
            "role": "agent",
            "text": random.choice(AGENT_OPENINGS)
        },

        {
            "turn_id": 3,
            "role": "customer",
            "text": (
                "Yes, please check it for me. "
                "I would appreciate your help."
            )
        },

        {
            "turn_id": 4,
            "role": "agent",
            "text": (
                "I have reviewed the information. "
                + random.choice(AGENT_CLOSINGS)
            )
        }

    ]

    return {

        "conversation_id": conversation_id,

        "scenario": scenario,

        "scenario_family": family,

        "customer_name": random.choice(
            CUSTOMER_NAMES
        ),

        "turns": turns
    }


conversations = []

conversation_number = 1


for scenario, details in SCENARIOS.items():

    for _ in range(10):

        conversation_id = (
            f"conv_{conversation_number:04d}"
        )

        conversation = create_conversation(
            conversation_id,
            scenario,
            details["family"]
        )

        conversations.append(conversation)

        conversation_number += 1


with open(
    OUTPUT_DIR / "conversations.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        conversations,
        f,
        indent=2,
        ensure_ascii=False
    )


with open(
    OUTPUT_DIR / "conversations.jsonl",
    "w",
    encoding="utf-8"
) as f:

    for conversation in conversations:

        f.write(
            json.dumps(
                conversation,
                ensure_ascii=False
            )
            + "\n"
        )


with open(
    OUTPUT_DIR / "metadata.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "conversation_id",
        "scenario",
        "scenario_family",
        "num_turns"
    ])

    for conversation in conversations:

        writer.writerow([
            conversation["conversation_id"],
            conversation["scenario"],
            conversation["scenario_family"],
            len(conversation["turns"])
        ])


print(
    f"Created {len(conversations)} conversations"
)