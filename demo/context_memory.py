"""
context_memory.py

Simple persistent memory for storing and retrieving
conversation state by customer.

This module is ONLY used by the demo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


class ContextMemory:

    def __init__(self, memory_file: Path = None):

        if memory_file is None:

            memory_file = (
                Path(__file__).parent /
                "memory.json"
            )

        self.memory_file = memory_file

        self.memory = self._load()

    # -----------------------------------------------------

    def _load(self) -> Dict:

        """
        Load existing memory from disk.
        """

        if not self.memory_file.exists():

            return {}

        with open(

            self.memory_file,

            "r",

            encoding="utf-8"

        ) as f:

            return json.load(f)

    # -----------------------------------------------------

    def _save(self):

        """
        Persist memory to disk.
        """

        with open(

            self.memory_file,

            "w",

            encoding="utf-8"

        ) as f:

            json.dump(

                self.memory,

                f,

                indent=2,

                ensure_ascii=False

            )

    # -----------------------------------------------------

    def save(

        self,

        customer_id: str,

        state: Dict

    ):

        """
        Save latest state for a customer.
        """

        self.memory[customer_id] = {

            "conversation_id":

                state.get(

                    "conversation_id"

                ),

            "primary_intent":

                state.get(

                    "primary_intent"

                ),

            "secondary_intents":

                state.get(

                    "secondary_intents",

                    []

                ),

            "tool_failure":

                state.get(

                    "tool_failure"

                ),

            "failure_reason":

                state.get(

                    "failure_reasons",

                    []

                ),

            "sentiment":

                state.get(

                    "sentiment_overall"

                ),

            "ambiguity":

                state.get(

                    "ambiguity_level"

                ),

            "turn_count":

                state.get(

                    "turn_count"

                )

        }

        self._save()

    # -----------------------------------------------------

    def load(

        self,

        customer_id: str

    ) -> Optional[Dict]:

        """
        Retrieve previous conversation state.
        """

        return self.memory.get(customer_id)

    # -----------------------------------------------------

    def exists(

        self,

        customer_id: str

    ) -> bool:

        return customer_id in self.memory

    # -----------------------------------------------------

    def clear(self):

        """
        Remove all stored context.
        """

        self.memory = {}

        self._save()

    # -----------------------------------------------------

    def list_customers(self):

        return sorted(self.memory.keys())