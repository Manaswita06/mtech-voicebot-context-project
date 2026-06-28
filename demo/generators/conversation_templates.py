import random

SCENARIOS = {
    "PAYMENT_ISSUE": {
        "opening": [
            "My payment failed yesterday.",
            "The payment hasn't been processed.",
            "My payment is still pending."
        ],
        "clarification": [
            "When did you make the payment?",
            "Can you tell me the payment amount?",
            "Which card did you use?"
        ],
        "customer_answer": [
            "Yesterday evening.",
            "Around ₹4500.",
            "I used my Platinum card."
        ]
    },

    "CARD_REPLACEMENT": {
        "opening": [
            "I lost my card.",
            "My wallet was stolen."
        ],
        "clarification": [
            "Was the card lost or stolen?",
            "When did you notice it?"
        ],
        "customer_answer": [
            "It was stolen yesterday.",
            "I noticed this morning."
        ]
    }
}


def generate_scenario():

    intent = random.choice(list(SCENARIOS.keys()))

    scenario = SCENARIOS[intent]

    return (

        intent,

        random.choice(scenario["opening"]),

        random.choice(scenario["clarification"]),

        random.choice(scenario["customer_answer"])

    )