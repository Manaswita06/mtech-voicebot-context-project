"""
printer.py

Pretty console printer for the live demo.
"""

from __future__ import annotations

from typing import Dict


class Printer:

    WIDTH = 80

    # --------------------------------------------------

    def line(self):

        print("=" * self.WIDTH)

    # --------------------------------------------------

    def separator(self):

        print("-" * self.WIDTH)

    # --------------------------------------------------

    def title(self, text: str):

        print()
        self.line()
        print(text.center(self.WIDTH))
        self.line()
        print()

    # --------------------------------------------------

    def section(self, text: str):

        print()
        self.separator()
        print(text)
        self.separator()

    # --------------------------------------------------

    def success(self, text: str):

        print(f"\n[SUCCESS] {text}\n")

    # --------------------------------------------------

    def info(self, text: str):

        print(f"\n[INFO] {text}\n")

    # --------------------------------------------------

    def warning(self, text: str):

        print(f"\n[WARNING] {text}\n")

    # --------------------------------------------------

    def error(self, text: str):

        print(f"\n[ERROR] {text}\n")

    # --------------------------------------------------

    def show_transcript(self, transcript: Dict):

        print()

        print(f"Conversation ID : {transcript.get('conversation_id')}")

        print(f"Customer ID     : {transcript.get('customer_id')}")

        print()

        for event in transcript.get("events", []):

            role = (
                event.get("participant", {})
                .get("role", "unknown")
                .upper()
            )

            text = event.get("text", "")

            event_name = event.get("event_name")

            if role == "CUSTOMER":

                print(f"👤 CUSTOMER : {text}")

            elif role == "ASSISTANT":

                print(f"🤖 ASSISTANT: {text}")

            elif event_name == "TOOL_REQUEST_SENT":

                tool = event.get(
                    "event_data",
                    {}
                ).get(
                    "tool_name",
                    ""
                )

                print(f"🔧 TOOL REQUEST : {tool}")

            elif event_name == "TOOL_RESPONSE_RECEIVED":

                data = event.get("event_data", {})

                status = data.get("status")

                tool = data.get("tool_name")

                if status == "success":

                    print(
                        f"✅ TOOL RESPONSE : "
                        f"{tool} (SUCCESS)"
                    )

                else:

                    print(
                        f"❌ TOOL RESPONSE : "
                        f"{tool} ({data.get('error_code')})"
                    )

        print()

    # --------------------------------------------------

    def show_state(self, state: Dict):

        self.section("EXTRACTED STATE")

        print(
            f"Primary Intent      : "
            f"{state.get('primary_intent')}"
        )

        print(
            f"Intent Confidence   : "
            f"{state.get('primary_intent_confidence', 'N/A')}"
        )

        print(
            f"Secondary Intents   : "
            f"{state.get('secondary_intents')}"
        )

        print(
            f"Multi Intent        : "
            f"{state.get('multi_intent')}"
        )

        print(
            f"Ambiguity Level     : "
            f"{state.get('ambiguity_level')}"
        )

        print(
            f"Sentiment           : "
            f"{state.get('sentiment_overall')}"
        )

        print(
            f"Tool Failure        : "
            f"{state.get('tool_failure')}"
        )

        print(
            f"Failure Count       : "
            f"{state.get('failure_count')}"
        )

        print(
            f"Failure Reasons     : "
            f"{state.get('failure_reasons')}"
        )

        print(
            f"Turn Count          : "
            f"{state.get('turn_count')}"
        )

        print()

        evidence = state.get("extracted_evidence", {})

        if evidence:

            print("Evidence")

            print()

            for key, values in evidence.items():

                print(f"{key}")

                if not values:

                    print("   None")

                else:

                    for item in values:

                        print(f"   • {item}")

                print()

    # --------------------------------------------------

    def show_context(self, context: str):

        self.section("RETRIEVED CONTEXT")

        print(context)

        print()

    # --------------------------------------------------

    def show_response(self, response: str):

        self.section("CONTEXT-AWARE RESPONSE")

        print(response)

        print()

    # --------------------------------------------------

    def show_summary(self):

        self.line()

        print("Demo Completed Successfully".center(self.WIDTH))

        self.line()

        print()