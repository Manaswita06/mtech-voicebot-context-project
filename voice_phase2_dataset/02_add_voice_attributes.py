import json
import random
from pathlib import Path


INPUT_FILE = Path(
    "data/conversations.json"
)

OUTPUT_FILE = Path(
    "data/emotional_conversations.json"
)


CUSTOMER_EMOTIONS = {

    "neutral": {
        "intensity": (0.2, 0.4),
        "speaking_rate": "normal",
        "prosody": "stable"
    },

    "frustrated": {
        "intensity": (0.6, 0.9),
        "speaking_rate": "fast",
        "prosody": "high_variation"
    },

    "angry": {
        "intensity": (0.75, 1.0),
        "speaking_rate": "fast",
        "prosody": "high_energy"
    },

    "anxious": {
        "intensity": (0.6, 0.9),
        "speaking_rate": "fast",
        "prosody": "unstable"
    },

    "confused": {
        "intensity": (0.4, 0.7),
        "speaking_rate": "slow",
        "prosody": "hesitant"
    },

    "worried": {
        "intensity": (0.5, 0.8),
        "speaking_rate": "medium",
        "prosody": "variable"
    },

    "impatient": {
        "intensity": (0.6, 0.9),
        "speaking_rate": "fast",
        "prosody": "sharp"
    },

    "relieved": {
        "intensity": (0.4, 0.7),
        "speaking_rate": "medium",
        "prosody": "relaxed"
    },

    "happy": {
        "intensity": (0.5, 0.8),
        "speaking_rate": "medium",
        "prosody": "positive"
    },

    "satisfied": {
        "intensity": (0.4, 0.7),
        "speaking_rate": "medium",
        "prosody": "positive"
    }
}


AGENT_STYLES = {

    "calm": {
        "intensity": (0.2, 0.4),
        "speaking_rate": "medium"
    },

    "empathetic": {
        "intensity": (0.4, 0.7),
        "speaking_rate": "medium"
    },

    "reassuring": {
        "intensity": (0.4, 0.7),
        "speaking_rate": "slow"
    },

    "apologetic": {
        "intensity": (0.4, 0.6),
        "speaking_rate": "medium"
    },

    "professional": {
        "intensity": (0.2, 0.5),
        "speaking_rate": "medium"
    }
}


SCENARIO_EMOTIONS = {

    "FRAUD_DISPUTE":
        ["anxious", "angry", "worried"],

    "UNRECOGNIZED_TRANSACTION":
        ["anxious", "worried", "confused"],

    "PAYMENT_FAILED":
        ["frustrated", "impatient", "angry"],

    "PAYMENT_PENDING":
        ["frustrated", "worried"],

    "REFUND_PENDING":
        ["frustrated", "impatient"],

    "ACCOUNT_LOCKED":
        ["frustrated", "anxious"],

    "LOGIN_ISSUE":
        ["frustrated", "confused"],

    "CARD_NOT_RECEIVED":
        ["worried", "frustrated"],

    "DUPLICATE_CHARGE":
        ["angry", "frustrated"],

    "ACCOUNT_BALANCE_QUERY":
        ["neutral", "satisfied"],

    "REWARD_POINTS_QUERY":
        ["neutral", "happy"]
}


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    conversations = json.load(f)


for conversation in conversations:

    scenario = conversation["scenario"]

    allowed_emotions = (
        SCENARIO_EMOTIONS.get(
            scenario,
            list(CUSTOMER_EMOTIONS.keys())
        )
    )

    for turn in conversation["turns"]:

        if turn["role"] == "customer":

            emotion = random.choice(
                allowed_emotions
            )

            profile = CUSTOMER_EMOTIONS[
                emotion
            ]

            turn["voice_attributes"] = {

                "emotion": emotion,

                "emotion_intensity": round(
                    random.uniform(
                        *profile["intensity"]
                    ),
                    2
                ),

                "speaking_rate":
                    profile["speaking_rate"],

                "prosody":
                    profile["prosody"],

                "pause_before": round(
                    random.uniform(0.2, 1.2),
                    2
                ),

                "pause_after": round(
                    random.uniform(0.4, 1.5),
                    2
                )
            }

        else:

            style = random.choice(
                list(AGENT_STYLES.keys())
            )

            profile = AGENT_STYLES[
                style
            ]

            turn["voice_attributes"] = {

                "emotion": style,

                "emotion_intensity": round(
                    random.uniform(
                        *profile["intensity"]
                    ),
                    2
                ),

                "speaking_rate":
                    profile["speaking_rate"],

                "prosody":
                    "controlled",

                "pause_before": round(
                    random.uniform(0.2, 0.8),
                    2
                ),

                "pause_after": round(
                    random.uniform(0.3, 1.0),
                    2
                )
            }


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        conversations,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    "Emotionally annotated dataset created."
)