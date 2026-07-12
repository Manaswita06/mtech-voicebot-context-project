"""
context_formatter.py

Converts stored conversational state into
a natural language context summary.

This summary is supplied to the response
generator during follow-up conversations.
"""

from __future__ import annotations

from typing import Dict, Optional


class ContextFormatter:

    def format(self, state):

        if not state:
            return (
                "No previous conversation history is available "
                "for this customer."
            )

        lines = []

        lines.append("Previous Customer Interaction Summary")
        lines.append("------------------------------------")

        if state.get("primary_intent"):
            lines.append(
                f"Primary Intent: {state['primary_intent']}"
            )

        secondary = state.get("secondary_intents", [])
        if secondary:
            lines.append(
                "Secondary Intents: " +
                ", ".join(secondary)
            )

        lines.append(
            f"Conversation Length: {state.get('turn_count',0)} turns"
        )

        if state.get("tool_failure"):

            reasons = state.get("failure_reasons", [])

            if reasons:
                lines.append(
                    "Previous Tool Status: FAILED "
                    f"({', '.join(reasons)})"
                )
            else:
                lines.append(
                    "Previous Tool Status: FAILED"
                )

        else:

            lines.append(
                "Previous Tool Status: SUCCESS"
            )

        sentiment = state.get("sentiment_overall")

        if sentiment is not None:

            if sentiment >= 0.3:
                label = "Positive"
            elif sentiment <= -0.3:
                label = "Negative"
            else:
                label = "Neutral"

            lines.append(
                f"Customer Sentiment: {label}"
            )

        ambiguity = state.get("ambiguity_level")

        if ambiguity is not None:
            lines.append(
                f"Ambiguity Score: {ambiguity:.2f}"
            )

        lines.append("")
        lines.append(
            "The customer has already provided the above information."
        )
        lines.append(
            "Do not ask for the same information again unless absolutely necessary."
        )

        return "\n".join(lines)