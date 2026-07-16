"""
tool_simulator.py

Simulates backend API/tool execution for synthetic conversations.
"""

from __future__ import annotations

import random
import time
from typing import Dict


class ToolSimulator:

    SUCCESS_RATE = {
        "PAYMENT_LOOKUP": 0.90,
        "CARD_LOOKUP": 0.95,
        "ADDRESS_UPDATE": 0.96,
        "STATEMENT_FETCH": 0.92,
        "DISPUTE_LOOKUP": 0.88,
    }

    FAILURE_CODES = [
        "TIMEOUT",
        "SERVICE_UNAVAILABLE",
        "INVALID_REQUEST",
        "DOWNSTREAM_ERROR",
        "AUTHORIZATION_FAILED",
    ]

    def execute(

            self,

            tool_name,

            tool_failure=None,

            failure_reason=None

    ):

        latency = round(random.uniform(0.2, 2.0), 3)

        time.sleep(random.uniform(0.05, 0.2))

        success_probability = self.SUCCESS_RATE.get(tool_name, 0.9)

        success = random.random() <= success_probability

        if success:

            return {
                "tool_name": tool_name,
                "status": "success",
                "latency": latency,
                "error_code": None,
            }

        return {
            "tool_name": tool_name,
            "status": "fail",
            "latency": latency,
            "error_code": random.choice(self.FAILURE_CODES),
            "tool_failure": tool_failure,
            "failure_reason": failure_reason,
        }