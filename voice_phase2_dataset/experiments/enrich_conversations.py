#!/usr/bin/env python3
"""Rewrite existing synthetic calls with scenario-grounded dialogue templates."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.lexicon import SCENARIOS


CALL_CONTEXT = {
    "Account Information": (
        "I am checking this before I make a decision based on the information in my account.",
        "I would like to understand the figures clearly before I plan my next steps.",
        "I am reviewing my account and want to make sure I have understood this correctly.",
    ),
    "Account Management": (
        "I would like to understand what this means for my account and what will happen next.",
        "I want to make sure I understand any effect this may have on my account.",
        "Could you walk me through what happens after this request?",
    ),
    "Authentication": (
        "I need to regain access without compromising the security of my account.",
        "I have not been able to get into the app, so I am checking the secure way to resolve it.",
        "I want to restore access and make sure my account stays protected.",
    ),
    "Card Delivery": (
        "I need to know when I can expect the card and whether I need to do anything in the meantime.",
        "I am checking because I need to know when I will be able to use the card.",
        "I would appreciate an update on the delivery and any next steps.",
    ),
    "Card Management": (
        "I was trying to manage the card and want to be sure the request is handled correctly.",
        "I would like to know what I can do with the card while this request is being processed.",
        "I am checking what has changed and whether I need to take another step.",
    ),
    "Card Security": (
        "I want to make sure my account is protected and understand what I should do next.",
        "I am being cautious and would like to know how to keep the card secure.",
        "Could you explain what I should do to prevent any further card activity?",
    ),
    "Cash Withdrawal": (
        "The withdrawal did not go as expected, so I would like to understand what happened and how it will be reviewed.",
        "I am checking whether the transaction can be traced and what happens next.",
        "I would like to know how this will be investigated and when I may hear back.",
    ),
    "Disputes": (
        "I want to make sure the transaction is reviewed and understand what information may be needed.",
        "I would like to know what evidence is useful and how the review will proceed.",
        "Could you explain the next steps for having this transaction looked into?",
    ),
    "Fees": (
        "I would like to understand what the charge relates to before deciding what to do next.",
        "I am checking the details so I can understand why this amount was applied.",
        "Could you help me identify the charge and explain whether it is expected?",
    ),
    "Fraud": (
        "I do not recognize the activity and want to make sure no further transactions are affected.",
        "I am concerned about the activity and want to understand how to protect the account.",
        "I would like to check whether anything else needs to be secured right away.",
    ),
    "International Services": (
        "I would like to understand how the conversion or foreign transaction was calculated.",
        "I am checking how the exchange rate and any related charges were applied.",
        "Could you explain how this transaction was converted into my account currency?",
    ),
    "Payments": (
        "I am checking the payment history so I do not take the wrong next step.",
        "I am not sure what the current payment status means and would like to clarify it.",
        "I want to understand whether any further action is needed from me.",
    ),
    "Profile Management": (
        "I want to make sure the change is recorded correctly and does not affect other account details.",
        "I would like to confirm which contact details will be used going forward.",
        "Could you check that the update will be applied to the right profile?",
    ),
    "Refunds": (
        "I have been checking my account for an update and would like to understand the expected next step.",
        "I would like to know whether the refund is still being processed or if anything is needed from me.",
        "Could you clarify when I should expect an update about the refund?",
    ),
    "Rewards": (
        "I expected this to appear in my account and want to understand how it was calculated or applied.",
        "I am checking whether the activity qualified and when the reward should show up.",
        "Could you help me understand how this reward was earned or redeemed?",
    ),
    "Subscriptions": (
        "I want to confirm when the cancellation takes effect and whether any further action is needed.",
        "I would like to know what happens to the current billing period after cancellation.",
        "Could you confirm that the subscription will not renew again?",
    ),
    "Transfers": (
        "I want to be certain this is handled correctly before I try anything else.",
        "I am checking the status so I do not accidentally repeat or duplicate the request.",
        "Could you explain what I should expect and whether I need to take another step?",
    ),
}

AGENT_ACKNOWLEDGEMENTS = (
    "I will review the details carefully and explain what I can confirm, along with any next steps.",
    "Let me check the relevant information and then I will walk you through the options.",
    "I will make sure we distinguish what is already confirmed from anything still pending.",
)
ENRICHMENT_VERSION = 1
METADATA_FIELDS = (
    "conversation_id",
    "scenario",
    "scenario_family",
    "intent",
    "customer_speaker",
    "customer_gender",
    "agent_speaker",
    "agent_gender",
    "customer_persona",
    "agent_persona",
    "num_turns",
    "resolution_status",
    "escalated",
    "split",
)


def load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"Expected a list of conversation objects in {path}")
    if len({str(row.get("conversation_id")) for row in data}) != len(data):
        raise ValueError("Conversation IDs must be unique")
    return data


def enrich(records: list[dict[str, Any]], seed: int) -> int:
    changed = 0
    for record in records:
        if record.get("dialogue_enrichment_version") == ENRICHMENT_VERSION:
            continue
        scenario = record.get("scenario")
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario in {record.get('conversation_id')}: {scenario}")
        scenario_data = SCENARIOS[scenario]
        turns = record.get("turns")
        if not isinstance(turns, list) or [
            turn.get("dialogue_act") for turn in turns
        ] != scenario_data["flow"]:
            raise ValueError(
                f"Unexpected dialogue flow in {record.get('conversation_id')}; "
                "refusing to replace its turns."
            )
        entities = record.get("entities", {})
        if not isinstance(entities, dict):
            raise ValueError(f"Invalid entities in {record.get('conversation_id')}")
        try:
            call_seed = seed + int(str(record["conversation_id"]).rsplit("_", 1)[-1])
        except (KeyError, ValueError) as error:
            raise ValueError("Conversation IDs must end in a numeric index") from error
        rng = random.Random(call_seed)
        opening_seen = False
        for turn in turns:
            role = str(turn.get("role", "")).casefold()
            act = str(turn.get("dialogue_act", ""))
            templates = scenario_data["templates"].get(act, {}).get(role, [])
            if not templates:
                raise ValueError(
                    f"No templates for {record['conversation_id']} act={act} role={role}"
                )
            turn["text"] = rng.choice(templates).format(**entities)
            if act == "problem_statement" and role == "customer":
                contexts = CALL_CONTEXT.get(scenario_data["family"])
                if not contexts:
                    raise ValueError(f"No contextual template for {scenario_data['family']}")
                context = rng.choice(contexts)
                turn["text"] = f"{turn['text'].rstrip()} {context}"
                opening_seen = True
            elif act == "acknowledge" and role == "agent":
                turn["text"] = (
                    f"{turn['text'].rstrip()} "
                    f"{rng.choice(AGENT_ACKNOWLEDGEMENTS)}"
                )
        if not opening_seen:
            raise ValueError(f"No customer problem statement in {record['conversation_id']}")
        record["dialogue_enrichment_version"] = ENRICHMENT_VERSION
        changed += 1
    return changed


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/emotional_conversations.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the source JSON, JSONL, and metadata CSV after validation.",
    )
    args = parser.parse_args()
    if not args.in_place:
        parser.error("Pass --in-place to confirm rewriting the existing dataset")
    records = load_records(args.dataset)
    changed = enrich(records, args.seed)
    if changed == 0:
        print(f"Dataset already has dialogue enrichment version {ENRICHMENT_VERSION}; no changes.")
        return

    json_path = args.dataset
    jsonl_path = json_path.with_suffix(".jsonl")
    csv_path = json_path.parent / "metadata.csv"
    json_content = json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    jsonl_content = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    csv_rows = [
        {field: record.get(field, "") for field in METADATA_FIELDS}
        for record in records
    ]
    from io import StringIO

    csv_buffer = StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=METADATA_FIELDS)
    writer.writeheader()
    writer.writerows(csv_rows)

    atomic_write(json_path, json_content)
    atomic_write(jsonl_path, jsonl_content)
    atomic_write(csv_path, csv_buffer.getvalue())
    print(
        f"Enriched {changed} calls using scenario-specific entity templates and "
        "non-sensitive incident context."
    )
    print(f"Updated: {json_path}, {jsonl_path}, {csv_path}")


if __name__ == "__main__":
    main()
