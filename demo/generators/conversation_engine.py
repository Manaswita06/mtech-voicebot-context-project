"""
conversation_engine.py

Generates complete multi-turn conversations.

Supports:
- Single intent
- Multi intent
- Controlled tool failures
- Follow-up conversations
- Synthetic dataset generation
"""

from __future__ import annotations

import random
import uuid

from demo.generators.transcript_builder import TranscriptBuilder
from demo.generators.tool_simulator import ToolSimulator


class ConversationEngine:

    BOT_ACK = [
        "Certainly. Let me check that for you.",
        "I'll help you with that.",
        "Sure. Let me verify the details.",
        "Let me look into it."
    ]

    THANKS = [
        "Thank you.",
        "Thanks.",
        "Appreciate your help.",
        "Okay thanks."
    ]

    SECONDARY_TRANSITIONS = [
        "I also have another question.",
        "Before we finish, I need help with something else.",
        "Can I ask one more thing?",
        "There's another issue I'd like to discuss."
    ]

    FOLLOWUP_MESSAGES = [
        "Can I check the status tomorrow?",
        "Will I receive an update?",
        "How can I track this request?",
        "When should I contact support again?"
    ]

    TOOL_MAPPING = {

        "PAYMENT_ISSUE": "PAYMENT_LOOKUP",

        "CARD_REPLACEMENT": "CARD_LOOKUP",

        "ADDRESS_UPDATE": "ADDRESS_UPDATE",

        "STATEMENT_REQUEST": "STATEMENT_FETCH",

        "TRANSACTION_DISPUTE": "DISPUTE_LOOKUP"

    }

    def __init__(self):

        self.tool = ToolSimulator()

    # ---------------------------------------------------------

    def generate(

        self,

        customer,

        intent,

        opening,

        clarification,

        customer_answer,

        secondary_intents=None,

        ambiguity_level=0.0,

        sentiment_score=0.0,

        tool_failure=None,

        failure_reason=None,

        followup_required=False

    ):

        secondary_intents = secondary_intents or []

        conversation_id = str(uuid.uuid4())

        builder = TranscriptBuilder(

            conversation_id,

            customer.customer_id

        )

        # -----------------------------------------------------
        # Opening
        # -----------------------------------------------------

        builder.add_customer(opening)

        builder.add_bot(

            random.choice(self.BOT_ACK)

        )

        builder.add_bot(

            clarification

        )

        builder.add_customer(

            customer_answer

        )

        # -----------------------------------------------------
        # Tool Call
        # -----------------------------------------------------

        tool = self.TOOL_MAPPING.get(intent)

        result = None

        if tool is not None:

            builder.add_tool_request(tool)

            result = self.tool.execute(

                tool_name=tool,

                tool_failure=tool_failure,

                failure_reason=failure_reason

            )

            builder.add_tool_response(result)

            if result["status"] == "success":

                builder.add_bot(

                    "The requested information has been retrieved successfully."

                )

            else:

                builder.add_bot(

                    f"I'm sorry. "

                    f"The backend service returned "

                    f"{result['error_code']}. "

                    f"Please try again later."

                )

        # -----------------------------------------------------
        # Secondary Intent
        # -----------------------------------------------------

        if secondary_intents:

            builder.add_customer(

                random.choice(

                    self.SECONDARY_TRANSITIONS

                )

            )

            builder.add_bot(

                "Sure. Please tell me your other request."

            )

            builder.add_customer(

                f"I also need help regarding "

                f"{secondary_intents[0].replace('_',' ').lower()}."

            )

        # -----------------------------------------------------
        # Follow-up
        # -----------------------------------------------------

        if followup_required:

            builder.add_customer(

                random.choice(

                    self.FOLLOWUP_MESSAGES

                )

            )

            builder.add_bot(

                "Yes. The details will remain available for future interactions."

            )

        # -----------------------------------------------------
        # Closing
        # -----------------------------------------------------

        builder.add_customer(

            random.choice(self.THANKS)

        )

        transcript = builder.build()

        # -----------------------------------------------------
        # Ground Truth Labels
        # -----------------------------------------------------

        transcript["gt_primary_intent"] = intent

        transcript["gt_secondary_intents"] = secondary_intents

        transcript["gt_multi_intent"] = (

            len(secondary_intents) > 0

        )

        transcript["gt_turn_count"] = len(

            transcript["events"]

        )

        transcript["gt_tool_failure"] = (

            result["status"] == "fail"

            if result

            else False

        )

        transcript["gt_failure_count"] = (

            1

            if transcript["gt_tool_failure"]

            else 0

        )

        transcript["gt_failure_reasons"] = (

            [result["error_code"]]

            if transcript["gt_tool_failure"]

            else []

        )

        transcript["gt_sentiment_overall"] = (

            sentiment_score

        )

        transcript["gt_ambiguity_level"] = (

            ambiguity_level

        )

        transcript["gt_followup_required"] = (

            followup_required

        )

        transcript["gt_scenario_family"] = intent

        return transcript