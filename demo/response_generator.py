"""
response_generator.py

Generates a context-aware response using the
previous conversation context and the current
customer transcript.

This module DOES NOT perform state extraction.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Dict

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:7b-instruct"
)


class ResponseGenerator:

    def __init__(self):

        self.system_prompt = """
You are an enterprise customer support assistant.

You will receive:

1. The customer's latest message.
2. The structured state extracted from previous conversations.

Your objective is to continue the previous conversation naturally.

Rules:

• Use the previous context whenever relevant.

• Never ask the customer to repeat information
that already exists in memory.

• Continue from where the previous conversation ended.

• Use previous intent, tool execution status,
sentiment and conversation history.

• If the previous tool execution succeeded,
assume only that the lookup succeeded.
Do NOT assume the customer's issue has been resolved.

• If additional verification is needed,
ask only for the missing information.

• Be concise.

• Respond like an experienced banking
customer support agent.

Return only the assistant response.
"""

    # -------------------------------------------------------

    def _compact_transcript(
        self,
        transcript: Dict
    ) -> str:

        lines = []

        for event in transcript.get("events", []):

            role = event.get(
                "participant",
                {}
            ).get(
                "role",
                "unknown"
            )

            text = event.get(
                "text",
                ""
            )

            if text:

                lines.append(
                    f"{role.upper()}: {text}"
                )

        return "\n".join(lines)

    # -------------------------------------------------------

    def generate(

        self,

        transcript: Dict,

        previous_context: str

    ) -> str:

        conversation = self._compact_transcript(
            transcript
        )

        user_prompt = f"""
Previous Context
----------------

{previous_context}


Current Conversation
--------------------

{conversation}


Generate the next assistant response.

Do not ask the customer to repeat information
already available in the previous context.

If the previous context already contains the
customer's issue, continue the conversation
instead of restarting it.
"""

        payload = {

            "model": OLLAMA_MODEL,

            "system": self.system_prompt,

            "prompt": user_prompt,

            "stream": False,

            "options": {

                "temperature": 0.2

            }

        }

        request = urllib.request.Request(

            OLLAMA_URL,

            data=json.dumps(payload).encode("utf-8"),

            headers={

                "Content-Type": "application/json"

            },

            method="POST"

        )

        try:

            with urllib.request.urlopen(

                request,

                timeout=300

            ) as response:

                result = json.loads(

                    response.read().decode()

                )

            return result.get(

                "response",

                "No response generated."

            ).strip()

        except Exception as e:

            return (

                "Unable to generate response.\n"

                f"Reason: {e}"

            )