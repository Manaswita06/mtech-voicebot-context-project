"""
transcript_builder.py

Creates transcript JSON compatible with
state_extraction_pipeline.py
"""

from __future__ import annotations

from datetime import datetime
import json


class TranscriptBuilder:

    def __init__(self, conversation_id, customer_id):

        self.transcript = {

            "conversation_id": conversation_id,

            "customer_id": customer_id,

            "events": []

        }

    def _timestamp(self):

        return datetime.utcnow().isoformat()

    def add_customer(self, text):

        self.transcript["events"].append({

            "participant": {

                "role": "customer"

            },

            "text": text,

            "timestamp": self._timestamp()

        })

    def add_bot(self, text):

        self.transcript["events"].append({

            "participant": {

                "role": "assistant"

            },

            "text": text,

            "timestamp": self._timestamp()

        })

    def add_tool_request(self, tool):

        self.transcript["events"].append({

            "event_name": "TOOL_REQUESTED",

            "event_data": {

                "tool_name": tool

            },

            "timestamp": self._timestamp()

        })

    def add_tool_response(self, response):

        self.transcript["events"].append({

            "event_name": "TOOL_RESPONSE_RECEIVED",

            "event_data": response,

            "timestamp": self._timestamp()

        })

    def build(self):

        return self.transcript

    def save(self, path):

        with open(path, "w", encoding="utf8") as f:

            json.dump(self.transcript, f, indent=2, ensure_ascii=False)