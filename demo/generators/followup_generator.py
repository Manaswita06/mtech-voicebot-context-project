import random

FOLLOWUPS=[

"Any update on this?",

"Can you check again?",

"I called yesterday.",

"The issue still exists.",

"Is there any progress?",

"Can someone help me now?"

]


def generate_followup():

    return random.choice(FOLLOWUPS)