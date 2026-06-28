import random

SCENARIOS = {

    "PAYMENT_ISSUE": {

        "opening": [
            "My payment failed yesterday.",
            "The payment hasn't been processed.",
            "My payment is still pending.",
            "I made a payment but it isn't showing.",
            "The amount was deducted but the payment isn't reflected."
        ],

        "clarification": [
            "When did you make the payment?",
            "Can you tell me the payment amount?",
            "Which card did you use?",
            "Was it an online payment or an offline payment?",
            "Did you receive any confirmation message?"
        ],

        "customer_answer": [
            "Yesterday evening.",
            "Around ₹4500.",
            "I used my Platinum card.",
            "It was an online payment.",
            "No, I didn't receive any confirmation."
        ]
    },

    "CARD_REPLACEMENT": {

        "opening": [
            "I lost my card.",
            "My wallet was stolen.",
            "My card is damaged.",
            "I need a replacement card.",
            "Someone stole my credit card."
        ],

        "clarification": [
            "Was the card lost or stolen?",
            "When did you notice it?",
            "Have you already blocked the card?",
            "Which card needs replacement?",
            "Do you still have the damaged card?"
        ],

        "customer_answer": [
            "It was stolen yesterday.",
            "I noticed this morning.",
            "Yes, I already blocked it.",
            "It's my Platinum card.",
            "The card is physically damaged."
        ]
    },

    "TRANSACTION_DISPUTE": {

        "opening": [
            "I found an unknown transaction.",
            "There is an unauthorized charge on my card.",
            "I don't recognize this transaction.",
            "Someone charged my card without permission.",
            "I'd like to dispute a transaction."
        ],

        "clarification": [
            "Which transaction are you referring to?",
            "What was the transaction amount?",
            "When did it occur?",
            "Have you contacted the merchant?",
            "Is this the first time you've noticed it?"
        ],

        "customer_answer": [
            "It happened yesterday.",
            "The charge was ₹3200.",
            "No, I haven't contacted the merchant.",
            "Yes, this is the first time.",
            "I don't recognize the merchant."
        ]
    },

    "ADDRESS_UPDATE": {

        "opening": [
            "I need to change my address.",
            "I've moved to a new house.",
            "Please update my mailing address.",
            "My registered address has changed.",
            "I'd like to modify my contact address."
        ],

        "clarification": [
            "What is your new address?",
            "Is this your permanent address?",
            "Would you like to update the mailing address as well?",
            "When did you move?",
            "Can you confirm the postal code?"
        ],

        "customer_answer": [
            "I moved last week.",
            "Yes, please update both addresses.",
            "The postal code is 560001.",
            "It's my permanent address.",
            "I'll provide the complete address."
        ]
    },

    "STATEMENT_REQUEST": {

        "opening": [
            "I need my account statement.",
            "Can you send me the latest statement?",
            "I want a PDF copy of my statement.",
            "I haven't received this month's statement.",
            "Please email me my account statement."
        ],

        "clarification": [
            "Which month's statement do you need?",
            "Would you like it by email?",
            "Is this for your credit card account?",
            "Do you need the latest statement?",
            "Would you prefer a PDF version?"
        ],

        "customer_answer": [
            "I need last month's statement.",
            "Yes, please send it by email.",
            "It's for my credit card.",
            "Yes, the latest one.",
            "A PDF would be fine."
        ]
    },

    "GENERAL_QUERY": {

        "opening": [
            "I have a question about my account.",
            "I need some help with my account.",
            "Can someone assist me?",
            "Something doesn't look right.",
            "I have a general enquiry."
        ],

        "clarification": [
            "Could you explain the issue?",
            "What would you like help with?",
            "Can you provide more details?",
            "When did you notice this?",
            "Is there a specific problem you're facing?"
        ],

        "customer_answer": [
            "I'm not completely sure.",
            "I just wanted some clarification.",
            "It started today.",
            "I'm trying to understand my account.",
            "That's all I wanted to know."
        ]
    }

}


def generate_intent():
    return random.choice(list(SCENARIOS.keys()))


def generate_opening(intent):
    return random.choice(SCENARIOS[intent]["opening"])


def generate_clarification(intent):
    return random.choice(SCENARIOS[intent]["clarification"])


def generate_customer_answer(intent):
    return random.choice(SCENARIOS[intent]["customer_answer"])


def generate_scenario():
    """
    Backward compatible function.
    """
    intent = generate_intent()

    return (
        intent,
        generate_opening(intent),
        generate_clarification(intent),
        generate_customer_answer(intent)
    )