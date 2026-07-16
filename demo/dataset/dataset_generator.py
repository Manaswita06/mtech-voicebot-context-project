"""
dataset_generator.py

Generates a synthetic dataset of voicebot transcripts.

Usage:
    python -m demo.dataset.dataset_generator \
        --output-dir demo/data/transcripts \
        --num-transcripts 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from demo.dataset.scenario_distribution import ScenarioDistribution
from demo.generators.customers_profiles import generate_customer
from demo.generators.conversation_templates import (
    generate_opening,
    generate_clarification,
    generate_customer_answer,
)
from demo.generators.conversation_engine import ConversationEngine


class DatasetGenerator:

    def __init__(self):

        self.engine = ConversationEngine()
        self.distribution = ScenarioDistribution()

    # ---------------------------------------------------------

    def generate_dataset(
        self,
        output_dir: str,
        num_transcripts: int
    ):

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for idx in range(num_transcripts):

            customer = generate_customer()

            # Primary intent
            intent = self.distribution.sample_primary_intent()

            # Conversation text
            opening = generate_opening(intent)

            clarification = generate_clarification(intent)

            customer_answer = generate_customer_answer(intent)

            # Metadata
            secondary_intents = (
                self.distribution.sample_secondary_intent(intent)
            )

            ambiguity = (
                self.distribution.sample_ambiguity()[1]
            )

            sentiment = (
                self.distribution.sample_sentiment()[1]
            )

            tool_failure, failure_reason = (
                self.distribution.sample_tool_failure()
            )

            followup = (
                self.distribution.sample_followup_required()
            )

            transcript = self.engine.generate(

                customer=customer,

                intent=intent,

                opening=opening,

                clarification=clarification,

                customer_answer=customer_answer,

                secondary_intents=secondary_intents,

                ambiguity_level=ambiguity,

                sentiment_score=sentiment,

                tool_failure=tool_failure,

                failure_reason=failure_reason,

                followup_required=followup

            )

            filename = output_path / f"conversation_{idx+1:05d}.json"

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    transcript,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

        print(f"\nGenerated {num_transcripts} transcripts.")
        print(f"Saved to {output_path}")


# ------------------------------------------------------------------


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="demo/generated/transcripts"
    )

    parser.add_argument(
        "--num-transcripts",
        type=int,
        default=500
    )

    args = parser.parse_args()

    DatasetGenerator().generate_dataset(

        output_dir=args.output_dir,

        num_transcripts=args.num_transcripts

    )


if __name__ == "__main__":
    main()