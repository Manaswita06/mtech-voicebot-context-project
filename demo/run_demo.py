"""
run_demo.py

Main demo entry point.

This file DOES NOT implement NLP or LLM logic.

Instead, it orchestrates the existing project
components to demonstrate context-aware state
extraction and memory retrieval.
"""

from pathlib import Path

from demo.context_memory import ContextMemory
from demo.context_formatter import ContextFormatter
from demo.response_generator import ResponseGenerator
from demo.scenario_selector import ScenarioSelector
from demo.printer import Printer
from src.state_extraction_pipeline import extract_one


# --------------------------------------------------
# Update these paths according to your project
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRANSCRIPT_DIR = PROJECT_ROOT / "generated" / "transcripts"

STATE_DIR = PROJECT_ROOT / "generated" / "states"

STATE_EXTRACTION_SCRIPT = (
    PROJECT_ROOT /
    "src" /
    "state_extraction_pipeline.py"
)

# --------------------------------------------------


class DemoRunner:

    def __init__(self):

        self.memory = ContextMemory()

        self.selector = ScenarioSelector()

        self.formatter = ContextFormatter()

        self.response = ResponseGenerator()

        self.printer = Printer()

    def demo(self):
        # ----------------------------------------------
        # First conversation
        # ----------------------------------------------

        transcript = self.selector.first_conversation()

        customer_id = transcript["customer_id"]

        self.printer.section(

            "FIRST CUSTOMER CONVERSATION"

        )

        self.printer.show_transcript(

            transcript

        )

        state = extract_one(
            transcript,
            provider="ollama",
            model="qwen2.5:7b-instruct"
        )

        self.printer.show_state(

            state

        )

        self.memory.save(

            customer_id,

            state

        )

        self.printer.success(

            "Conversation context stored."

        )

        # ----------------------------------------------
        # Follow-up conversation
        # ----------------------------------------------

        print()
        print("=" * 80)
        print("ENTER A FOLLOW-UP QUESTION")
        print("=" * 80)

        user_query = input("\nCustomer: ")

        followup = {
            "conversation_id": "followup",
            "customer_id": customer_id,
            "events": [
                {
                    "participant": {
                        "role": "customer"
                    },
                    "text": user_query
                }
            ]
        }

        self.printer.section(

            "FOLLOW-UP CONVERSATION"

        )

        self.printer.show_transcript(

            followup

        )

        previous = self.memory.load(

            customer_id

        )

        formatted_context = self.formatter.format(

            previous

        )

        response_without_context = self.response.generate(
            transcript=followup,
            previous_context=''
        )

        self.printer.section(

            "ASSISTANT WITHOUT MEMORY"

        )

        print(response_without_context)

        response_with_context = self.response.generate(
            transcript=followup,
            previous_context=formatted_context
        )

        self.printer.section(

            "MEMORY RETRIEVED"

        )

        print(formatted_context)

        self.printer.section(

            "ASSISTANT WITH MEMORY"

        )

        print(response_with_context)

        self.printer.success(

            "Demo Completed Successfully."

        )


if __name__ == "__main__":

    DemoRunner().demo()