"""
scenario_selector.py

Selects conversations from the existing generated
transcripts for the live demo.

No transcript generation happens here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List


class ScenarioSelector:

    def __init__(self, transcript_dir: Path = None):

        if transcript_dir is None:

            project_root = Path(__file__).resolve().parents[1]

            transcript_dir = (
                project_root /
                "demo" /
                "generated" /
                "transcripts"
            )

        self.transcript_dir = transcript_dir

        self.transcripts = sorted(

            self.transcript_dir.glob("*.json")

        )

        if not self.transcripts:

            raise FileNotFoundError(

                f"No transcripts found in {self.transcript_dir}"

            )

    # --------------------------------------------------------

    def _load(self, path: Path) -> Dict:

        with open(

            path,

            "r",

            encoding="utf-8"

        ) as f:

            return json.load(f)

    # --------------------------------------------------------

    def first_conversation(self) -> Dict:

        """
        Returns the first transcript.

        You can later replace this with a specific
        conversation if preferred.
        """

        return self._load(

            self.transcripts[1]

        )

    # --------------------------------------------------------

    def random_conversation(self) -> Dict:

        """
        Returns any random transcript.
        """

        return self._load(

            random.choice(

                self.transcripts

            )

        )

    # --------------------------------------------------------

    def followup_conversation(

        self,

        customer_id: str

    ) -> Dict:

        """
        Search for another conversation from
        the same customer.

        If unavailable,
        reuse the original conversation.
        """

        for file in self.transcripts:

            transcript = self._load(file)

            if (

                transcript.get("customer_id")

                == customer_id

            ):

                return transcript

        return self.first_conversation()

    # --------------------------------------------------------

    def by_intent(

        self,

        intent: str

    ) -> Dict:

        """
        Returns a transcript matching
        a requested intent.
        """

        for file in self.transcripts:

            transcript = self._load(file)

            if (

                transcript.get("gt_primary_intent")

                == intent

            ):

                return transcript

        raise ValueError(

            f"No transcript found for intent {intent}"

        )

    # --------------------------------------------------------

    def available_intents(self) -> List[str]:

        intents = set()

        for file in self.transcripts:

            transcript = self._load(file)

            intents.add(

                transcript.get(

                    "gt_primary_intent",

                    "UNKNOWN"

                )

            )

        return sorted(intents)