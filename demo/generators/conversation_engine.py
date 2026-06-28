"""
conversation_engine.py

Generates complete multi-turn conversations.
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

    TOOL_MAPPING = {

        "PAYMENT_ISSUE": "PAYMENT_LOOKUP",

        "CARD_REPLACEMENT": "CARD_LOOKUP",

        "ADDRESS_UPDATE": "ADDRESS_UPDATE",

        "STATEMENT_REQUEST": "STATEMENT_FETCH",

        "TRANSACTION_DISPUTE": "DISPUTE_LOOKUP"

    }

    def __init__(self):

        self.tool = ToolSimulator()

    def generate(self,

                 customer,

                 intent,

                 opening,

                 clarification,

                 customer_answer):

        conversation_id = str(uuid.uuid4())

        builder = TranscriptBuilder(

            conversation_id,

            customer.customer_id

        )

        builder.add_customer(opening)

        builder.add_bot(random.choice(self.BOT_ACK))

        builder.add_bot(clarification)

        builder.add_customer(customer_answer)

        tool = self.TOOL_MAPPING.get(intent)

        if tool is not None:

            builder.add_tool_request(tool)

            result = self.tool.execute(tool)

            builder.add_tool_response(result)

            if result["status"] == "success":

                builder.add_bot(

                    "The requested information has been retrieved successfully."

                )

            else:

                builder.add_bot(

                    f"I'm sorry. The system returned "

                    f"{result['error_code']}. "

                    f"Please try again later."

                )

        builder.add_customer(

            random.choice(self.THANKS)

        )

        transcript = builder.build()

        transcript["gt_primary_intent"] = intent

        transcript["gt_turn_count"] = len(transcript["events"])

        transcript["gt_tool_failure"] = (

            result["status"] == "fail"

            if tool else False

        )

        transcript["gt_failure_count"] = (

            1 if transcript["gt_tool_failure"] else 0

        )

        transcript["gt_failure_reasons"] = (

            [result["error_code"]]

            if transcript["gt_tool_failure"]

            else []

        )

        return transcript