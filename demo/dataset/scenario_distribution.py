"""
scenario_distribution.py

Controls the statistical distribution of synthetic conversations.

This module is responsible for:
1. Selecting conversation intents
2. Determining ambiguity
3. Determining tool failures
4. Determining multi-intent conversations
5. Determining customer sentiment

The transcript generator should NOT randomly decide these.
Everything should come from this distribution controller.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional


# ---------------------------------------------------------
# Distribution Configuration
# ---------------------------------------------------------

DEFAULT_INTENT_DISTRIBUTION = {
    "PAYMENT_ISSUE": 0.25,
    "CARD_REPLACEMENT": 0.20,
    "TRANSACTION_DISPUTE": 0.15,
    "ADDRESS_UPDATE": 0.15,
    "STATEMENT_REQUEST": 0.15,
    "GENERAL_QUERY": 0.10,
}


DEFAULT_SECONDARY_INTENTS = [
    "ADDRESS_UPDATE",
    "STATEMENT_REQUEST",
    "PAYMENT_ISSUE",
    "CARD_REPLACEMENT",
]


DEFAULT_FAILURE_CODES = [
    "TIMEOUT",
    "SERVICE_UNAVAILABLE",
    "AUTHORIZATION_FAILED",
    "DOWNSTREAM_ERROR",
    "INVALID_REQUEST",
]


DEFAULT_SENTIMENTS = {
    "positive": 0.20,
    "neutral": 0.45,
    "negative": 0.35,
}


# ---------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------

@dataclass
class DistributionConfig:

    intent_distribution: Dict[str, float] = None

    ambiguity_probability: float = 0.20

    tool_failure_probability: float = 0.15

    multi_intent_probability: float = 0.18

    followup_probability: float = 0.25

    sentiment_distribution: Dict[str, float] = None

    random_seed: int = 42

    def __post_init__(self):

        if self.intent_distribution is None:
            self.intent_distribution = DEFAULT_INTENT_DISTRIBUTION

        if self.sentiment_distribution is None:
            self.sentiment_distribution = DEFAULT_SENTIMENTS

        random.seed(self.random_seed)


# ---------------------------------------------------------
# Scenario Sampler
# ---------------------------------------------------------

class ScenarioDistribution:

    def __init__(self, config: Optional[DistributionConfig] = None):

        self.config = config or DistributionConfig()

    # -----------------------------------------------------

    def sample_primary_intent(self) -> str:

        intents = list(self.config.intent_distribution.keys())

        probs = list(self.config.intent_distribution.values())

        return random.choices(

            population=intents,

            weights=probs,

            k=1

        )[0]

    # -----------------------------------------------------

    def sample_secondary_intent(
        self,
        primary_intent: str
    ) -> List[str]:

        if random.random() > self.config.multi_intent_probability:

            return []

        candidates = [

            x

            for x in DEFAULT_SECONDARY_INTENTS

            if x != primary_intent

        ]

        return [random.choice(candidates)]

    # -----------------------------------------------------

    def sample_tool_failure(self):

        if random.random() <= self.config.tool_failure_probability:

            return True, random.choice(DEFAULT_FAILURE_CODES)

        return False, None

    # -----------------------------------------------------

    def sample_ambiguity(self):

        if random.random() <= self.config.ambiguity_probability:

            return True, round(random.uniform(0.6, 1.0), 2)

        return False, round(random.uniform(0.0, 0.35), 2)

    # -----------------------------------------------------

    def sample_sentiment(self):

        labels = list(

            self.config.sentiment_distribution.keys()

        )

        probs = list(

            self.config.sentiment_distribution.values()

        )

        sentiment = random.choices(

            labels,

            probs,

            k=1

        )[0]

        mapping = {

            "positive": round(random.uniform(0.25, 1.0), 2),

            "neutral": round(random.uniform(-0.2, 0.2), 2),

            "negative": round(random.uniform(-1.0, -0.25), 2),

        }

        return sentiment, mapping[sentiment]

    # -----------------------------------------------------

    def sample_followup_required(self):

        return (

            random.random()

            <= self.config.followup_probability

        )