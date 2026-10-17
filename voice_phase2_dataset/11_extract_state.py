#!/usr/bin/env python3
"""
11_extract_state.py

RECAP-Voice Week-2 state-extraction baseline.

Purpose
-------
Convert the predicted speaker-attributed conversation produced by
07b_build_wordlevel_speaker_transcripts.py into a RECAP-Voice v1.0
MODEL PREDICTION packet.

This is deliberately a deterministic / retrieval-based baseline so that
Week 2 has a reproducible reference implementation before an LLM-backed
extractor is introduced. It DOES NOT read gold_state/ during inference.

Inputs
------
wordlevel_speaker_transcripts/<conversation_id>.json
    Preferred input produced by 07b_build_wordlevel_speaker_transcripts.py.

Fallback:
speaker_transcripts/<conversation_id>.json
    Older segment-level output can also be consumed when it contains
    speaker_turns or segments with text/start/end/speaker information.

data/emotional_conversations.json
    Used ONLY to build intent and entity-value lexicons from TRAIN split
    conversations. Target conversation metadata is never used to make
    predictions.

Output
------
recap_predictions/<conversation_id>.json
recap_predictions/recap_predictions.jsonl
recap_predictions/recap_prediction_summary.csv

Important experimental rule
---------------------------
Do NOT point this script at gold_state files and do NOT use the target
conversation's scenario / intent / entities as prediction hints. The gold
states are evaluation references only.

Later upgrade path
------------------
This script becomes the deterministic baseline against which a hybrid
LLM + rules extractor can be compared in the same schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


SCHEMA_NAME = "RECAP-Voice"
SCHEMA_VERSION = "1.0.0"
PACKET_TYPE = "recap_voice"
DEFAULT_INPUT_DIR = Path("wordlevel_speaker_transcripts")
FALLBACK_INPUT_DIR = Path("speaker_transcripts")
DEFAULT_METADATA = Path("data/emotional_conversations.json")
DEFAULT_OUTPUT_DIR = Path("recap_predictions")

# Conservative phrase sets. These are intentionally interpretable so that
# their impact can be ablated later.
VERIFICATION_REQUEST_PATTERNS = (
    r"\bcomplete (?:account )?verification\b",
    r"\bverify (?:your|the|my) (?:account|identity|details)\b",
    r"\bconfirm (?:your|the|my) (?:details|identity|transaction)\b",
    r"\bprovide the (?:relevant )?(?:transaction|account) information\b",
)

VERIFICATION_RESPONSE_PATTERNS = (
    r"\b(?:yes|sure),? (?:i can|please) (?:verify|provide|proceed)\b",
    r"\bi can (?:provide|verify) (?:the )?(?:details|information)\b",
    r"\bplease proceed\b",
    r"\b(?:i have|i've) verified\b",
)

COMMITMENT_PATTERNS = (
    r"\bi will\b",
    r"\bwe will\b",
    r"\bwe'll\b",
    r"\bi'll\b",
    r"\bi have (?:raised|submitted|initiated|opened|escalated)\b",
    r"\bwe have (?:raised|submitted|initiated|opened|escalated)\b",
    r"\bwill (?:check|review|notify|contact|follow up|process|raise|investigate)\b",
    r"\bnext steps? (?:will|are)\b",
)

RESOLVED_PATTERNS = (
    r"\bresolved\b",
    r"\bcompleted\b",
    r"\bsuccessfully (?:processed|completed|updated|unblocked)\b",
    r"\b(?:is|are) now available\b",
    r"\bhas been (?:processed|completed|updated|unblocked|confirmed)\b",
    r"\bhave (?:completed|processed|updated|confirmed)\b",
    r"\brequest has been completed\b",
    r"\bissue has been resolved\b",
)

UNRESOLVED_PATTERNS = (
    r"\bpending\b",
    r"\bstill pending\b",
    r"\bunder (?:review|investigation)\b",
    r"\bfurther investigation\b",
    r"\bneed(?:s)? further assistance\b",
    r"\bnot yet\b",
    r"\bunable to complete\b",
    r"\bcannot (?:confirm|complete|process)\b",
    r"\bplease contact us again if\b",
)

NEXT_ACTION_PATTERNS = (
    r"\bwill (?:check|review|notify|contact|process|raise|investigate)\b",
    r"\bi have (?:raised|submitted|initiated|opened|escalated)\b",
    r"\bplease (?:provide|complete|contact|try|wait|check)\b",
    r"\bnext step\b",
    r"\bfurther investigation\b",
)

ACTION_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("verification_request", VERIFICATION_REQUEST_PATTERNS),
    ("status_check", (r"\bcheck (?:the )?(?:status|details|information)\b", r"\blet me check\b", r"\breview the current\b")),
    ("resolve", (r"\bresolved\b", r"\bhas been (?:processed|completed|updated|unblocked)\b", r"\bi have (?:processed|completed|updated|unblocked)\b")),
    ("inform", (r"\binformation is (?:now )?available\b", r"\bthe requested .* is now available\b", r"\bapplicable .* information\b")),
    ("escalate", (r"\bescalat(?:e|ed|ion)\b", r"\braised .* for further investigation\b")),
    ("commitment", COMMITMENT_PATTERNS),
]

GENERIC_ENTITY_PATTERNS: dict[str, tuple[str, ...]] = {
    "last4": (
        r"\b(?:ending|last four|last 4)\s*(?:in|digits?)?\s*(\d{4})\b",
        r"\bcard ending (\d{4})\b",
    ),
    "amount_numeric": (
        r"(?:₹|rs\.?|inr|\$|usd|eur|€|£|gbp)\s*([0-9][0-9,]*(?:\.\d+)?)",
        r"\b([0-9][0-9,]*(?:\.\d+)?)\s*(?:rupees?|dollars?|euros?|pounds?)\b",
    ),
    "duration": (
        r"\b(?:for|since|about|around|over|nearly|almost)\s+([0-9]+(?:\.[0-9]+)?\s+(?:minutes?|hours?|days?|weeks?|months?|years?))\b",
        r"\b([0-9]+(?:\.[0-9]+)?\s+(?:minutes?|hours?|days?|weeks?|months?|years?))\s+(?:ago|back)\b",
    ),
}

STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "my", "me", "i", "we",
    "you", "your", "is", "are", "was", "were", "be", "can", "could", "would",
    "please", "for", "on", "in", "with", "this", "that", "it", "do", "does",
    "did", "how", "what", "why", "when", "where", "which", "about", "from",
    "have", "has", "had", "will", "want", "need", "like", "help", "tell", "know",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_role(role: Any) -> str | None:
    if role is None:
        return None
    value = str(role).strip().upper()
    if value in {"CUSTOMER", "CLIENT", "USER"}:
        return "CUSTOMER"
    if value in {"AGENT", "ASSOCIATE", "REPRESENTATIVE"}:
        return "AGENT"
    return value


def normalize_text(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s£€$₹]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalize_text(text)) if t not in STOP_WORDS]


def sentence_similarity(a: str, b: str) -> float:
    """Interpretable lexical similarity for a retrieval baseline."""
    a_tokens = set(tokens(a))
    b_tokens = set(tokens(b))
    if not a_tokens or not b_tokens:
        lexical = 0.0
    else:
        lexical = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    seq = SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()
    return 0.65 * lexical + 0.35 * seq


def find_patterns(text: str, patterns: Iterable[str]) -> bool:
    text = str(text or "").lower()
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def first_match(text: str, patterns: Iterable[str]) -> re.Match[str] | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match
    return None


def text_contains(text: str, value: Any) -> bool:
    if value is None:
        return False
    needle = normalize_text(value)
    haystack = normalize_text(text)
    return bool(needle) and needle in haystack


def safe_confidence(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def make_field(
    value: Any,
    evidence_ids: list[str],
    *,
    confidence: float | None,
    status: str,
    source_type: str = "derived",
) -> dict[str, Any]:
    return {
        "value": value,
        "confidence": None if confidence is None else safe_confidence(confidence),
        "evidence_ids": evidence_ids,
        "source_type": source_type,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Dataset / intent prototype index
# ---------------------------------------------------------------------------


def build_train_index(metadata: list[dict[str, Any]]) -> tuple[list[tuple[str, str]], dict[str, set[str]]]:
    """
    Build a retrieval index from TRAIN customer utterances only.

    Returns:
        examples: [(customer_text, intent), ...]
        entity_lexicon: {entity_key: {values...}}
    """
    examples: list[tuple[str, str]] = []
    entity_lexicon: dict[str, set[str]] = defaultdict(set)

    for conversation in metadata:
        if str(conversation.get("split", "")).lower() != "train":
            continue

        intent = conversation.get("intent") or conversation.get("scenario")
        if intent:
            for turn in conversation.get("turns", []):
                if normalize_role(turn.get("role")) == "CUSTOMER":
                    text = str(turn.get("text", "")).strip()
                    if text:
                        examples.append((text, str(intent)))

        for key, value in (conversation.get("entities") or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float)):
                entity_lexicon[str(key)].add(str(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, (str, int, float)):
                        entity_lexicon[str(key)].add(str(item))

    return examples, dict(entity_lexicon)


def predict_intent(customer_text: str, examples: list[tuple[str, str]]) -> tuple[str | None, float, list[str]]:
    if not customer_text.strip() or not examples:
        return None, 0.0, []

    scored: list[tuple[float, str, str]] = []
    for example_text, intent in examples:
        score = sentence_similarity(customer_text, example_text)
        scored.append((score, intent, example_text))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_intent, best_example = scored[0]

    # Aggregate the top few examples for the same intent to make the result
    # more stable than a single nearest neighbour.
    top = scored[:8]
    intent_scores: dict[str, list[float]] = defaultdict(list)
    for score, intent, _ in top:
        intent_scores[intent].append(score)

    averaged = sorted(
        ((sum(vals) / len(vals), intent) for intent, vals in intent_scores.items()),
        reverse=True,
    )

    if averaged:
        best_avg_score, best_avg_intent = averaged[0]
        if best_avg_intent == best_intent:
            confidence = 0.55 * best_score + 0.45 * best_avg_score
            return best_intent, safe_confidence(confidence), [best_example]
        return best_intent, safe_confidence(best_score), [best_example]

    return best_intent, safe_confidence(best_score), [best_example]


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def add_evidence(
    evidence: list[dict[str, Any]],
    field: str,
    turns: list[dict[str, Any]],
    *,
    source_type: str = "asr",
) -> list[str]:
    if not turns:
        return []

    ids: list[str] = []
    for turn in turns:
        ev_id = f"P-E{len(evidence) + 1:03d}"
        ids.append(ev_id)
        evidence.append(
            {
                "evidence_id": ev_id,
                "field": field,
                "source_type": source_type,
                "turn_ids": [turn.get("turn_id", turn.get("id"))],
                "text": turn.get("text", ""),
                "start": turn.get("start"),
                "end": turn.get("end"),
            }
        )
    return ids


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def normalize_turns(packet: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Prefer grouped word-level speaker_turns. If absent, fall back to
    segment-like structures.
    """
    source = packet.get("speaker_turns") or packet.get("segments") or packet.get("speaker_segments") or []
    turns: list[dict[str, Any]] = []

    for idx, item in enumerate(source, start=1):
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        speaker = item.get("speaker")
        role = normalize_role(item.get("predicted_role") or item.get("role"))
        turn_id = item.get("turn_id", idx)

        turns.append(
            {
                "turn_id": turn_id,
                "speaker": speaker,
                "role": role,
                "start": item.get("start"),
                "end": item.get("end"),
                "text": text,
            }
        )

    return turns


def summarize_speaker_resolution(packet: dict[str, Any], turns: list[dict[str, Any]]) -> tuple[str, float | None, str]:
    status = packet.get("role_mapping_status") or packet.get("mapping_status")
    detected = packet.get("num_speakers_detected")

    if status == "RESOLVED_TWO_SPEAKERS":
        margin = packet.get("mapping_margin_seconds")
        if margin is None:
            return "RESOLVED", 0.95, "Two Pyannote speakers were mapped to CUSTOMER/AGENT."
        # Mapping margin is not itself a probability; map it conservatively.
        confidence = 0.75 + 0.20 * (1.0 - math.exp(-float(margin) / 2.0))
        return "RESOLVED", safe_confidence(confidence), "Two Pyannote speakers were mapped to CUSTOMER/AGENT."

    if status in {"UNRESOLVED_ONE_SPEAKER", "ONE_SPEAKER"} or detected == 1:
        return "UNRESOLVED", 0.25, "Pyannote detected only one speaker; CUSTOMER/AGENT attribution is unresolved."

    if status in {"UNRESOLVED_AMBIGUOUS_MAPPING", "UNRESOLVED_NO_ROLE_OVERLAP", "UNRESOLVED_NO_SPEAKERS"}:
        return "UNRESOLVED", 0.25, f"Speaker-role mapping status: {status}."

    if any(turn["role"] is not None for turn in turns):
        return "PARTIAL", 0.55, "Some turns contain resolved speaker roles, but mapping is not fully resolved."

    return "UNRESOLVED", 0.20, "No reliable CUSTOMER/AGENT speaker-role mapping was available."


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def extract_entities(full_text: str, customer_turns: list[dict[str, Any]], entity_lexicon: dict[str, set[str]], evidence: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}
    searchable_turns = customer_turns or []
    searchable_text = " ".join(t["text"] for t in (searchable_turns or []))

    # Dataset-derived values from TRAIN only.
    for key, values in entity_lexicon.items():
        matches: list[str] = []
        matching_turns: list[dict[str, Any]] = []
        for value in sorted(values, key=len, reverse=True):
            if text_contains(searchable_text, value):
                matches.append(value)
                matching_turns.extend([t for t in searchable_turns if text_contains(t["text"], value)])
                break

        if matches:
            # Conservative confidence because a train lexicon match can be ambiguous.
            ev_ids = add_evidence(
                evidence,
                f"state.entities.{key}",
                matching_turns[:2],
                source_type="asr",
            )
            entities[key] = make_field(
                matches[0],
                ev_ids,
                confidence=0.88,
                status="PREDICTED",
            )

    # Generic numeric patterns not dependent on the metadata vocabulary.
    for key, patterns in GENERIC_ENTITY_PATTERNS.items():
        match = first_match(full_text, patterns)
        if match:
            raw = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if key not in entities:
                hit_turns = [t for t in customer_turns if raw.lower() in t["text"].lower()]
                ev_ids = add_evidence(
                    evidence,
                    f"state.entities.{key}",
                    hit_turns[:2] or customer_turns[:1],
                    source_type="asr",
                )
                entities[key] = make_field(
                    raw,
                    ev_ids,
                    confidence=0.82,
                    status="PREDICTED",
                )

    return entities


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------


def infer_resolution(agent_turns: list[dict[str, Any]], all_turns: list[dict[str, Any]]) -> tuple[str, float, list[dict[str, Any]]]:
    agent_text = " ".join(t["text"] for t in agent_turns)
    last_agent = agent_turns[-1] if agent_turns else None
    evidence_turns = [last_agent] if last_agent else []

    has_resolved = find_patterns(agent_text, RESOLVED_PATTERNS)
    has_unresolved = find_patterns(agent_text, UNRESOLVED_PATTERNS)

    if has_resolved and not has_unresolved:
        return "RESOLVED", 0.88, evidence_turns
    if has_unresolved and not has_resolved:
        return "UNRESOLVED", 0.84, evidence_turns
    if has_resolved and has_unresolved:
        return "PARTIALLY_RESOLVED", 0.65, evidence_turns

    # Use a conservative fallback based on explicit closing language.
    all_text = " ".join(t["text"] for t in all_turns)
    if find_patterns(all_text, (r"\bthank you\b", r"\byou are welcome\b", r"\bhappy to help\b")):
        return "RESOLVED", 0.58, evidence_turns

    return "UNRESOLVED", 0.45, evidence_turns


def infer_verification(all_turns: list[dict[str, Any]]) -> tuple[str, float, list[dict[str, Any]]]:
    request_turns = [t for t in all_turns if find_patterns(t["text"], VERIFICATION_REQUEST_PATTERNS)]
    response_turns = [t for t in all_turns if find_patterns(t["text"], VERIFICATION_RESPONSE_PATTERNS)]

    if request_turns and response_turns:
        return "COMPLETED", 0.90, (request_turns[:1] + response_turns[:1])
    if request_turns:
        return "REQUESTED", 0.85, request_turns[:2]
    return "NOT_MENTIONED", 0.90, []


def infer_agent_actions(agent_turns: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for turn in agent_turns:
        action = None
        for label, patterns in ACTION_PATTERNS:
            if find_patterns(turn["text"], patterns):
                action = label
                break
        if action:
            ev_ids = add_evidence(
                evidence,
                f"state.agent_actions.{action}",
                [turn],
                source_type="asr",
            )
            actions.append(
                {
                    "dialogue_act": action,
                    "text": turn["text"],
                    "evidence_ids": ev_ids,
                }
            )
    return actions


def infer_commitments(agent_turns: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    commitments: list[dict[str, Any]] = []
    for turn in agent_turns:
        if find_patterns(turn["text"], COMMITMENT_PATTERNS):
            ev_ids = add_evidence(
                evidence,
                "state.agent_commitments",
                [turn],
                source_type="asr",
            )
            commitments.append(
                {
                    "text": turn["text"],
                    "status": "predicted_explicit_commitment",
                    "evidence_ids": ev_ids,
                }
            )
    return commitments


def infer_next_action(agent_turns: list[dict[str, Any]], resolution: str, evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    if resolution == "RESOLVED" or not agent_turns:
        return None

    for turn in reversed(agent_turns):
        if find_patterns(turn["text"], NEXT_ACTION_PATTERNS):
            ev_ids = add_evidence(
                evidence,
                "state.next_action",
                [turn],
                source_type="asr",
            )
            return {
                "value": turn["text"],
                "evidence_ids": ev_ids,
                "status": "predicted_next_action",
            }

    return None


def build_state(
    turns: list[dict[str, Any]],
    intent: str | None,
    intent_confidence: float,
    intent_example_turn: dict[str, Any] | None,
    entity_lexicon: dict[str, set[str]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    customer_turns = [t for t in turns if t["role"] == "CUSTOMER"]
    agent_turns = [t for t in turns if t["role"] == "AGENT"]

    # When role resolution is unavailable, use the earliest utterance as a
    # weak customer-goal candidate but mark it low confidence.
    customer_goal_turn = customer_turns[0] if customer_turns else (turns[0] if turns else None)
    goal_evidence = add_evidence(
        evidence,
        "state.customer_goal",
        [customer_goal_turn] if customer_goal_turn else [],
        source_type="asr",
    )

    intent_evidence = add_evidence(
        evidence,
        "state.intent",
        [intent_example_turn] if intent_example_turn else ([customer_goal_turn] if customer_goal_turn else []),
        source_type="asr",
    )

    resolution, resolution_confidence, resolution_turns = infer_resolution(agent_turns, turns)
    resolution_evidence = add_evidence(
        evidence,
        "state.resolution_state",
        resolution_turns,
        source_type="asr",
    )

    verification, verification_confidence, verification_turns = infer_verification(turns)
    verification_evidence = add_evidence(
        evidence,
        "state.verification",
        verification_turns,
        source_type="asr",
    )

    full_text = " ".join(t["text"] for t in turns)
    entities = extract_entities(full_text, customer_turns, entity_lexicon, evidence)

    actions = infer_agent_actions(agent_turns, evidence)
    commitments = infer_commitments(agent_turns, evidence)
    next_action = infer_next_action(agent_turns, resolution, evidence)

    unresolved_items: list[dict[str, Any]] = []
    if resolution in {"UNRESOLVED", "PARTIALLY_RESOLVED"}:
        source_turn = customer_turns[0] if customer_turns else (turns[0] if turns else None)
        if source_turn:
            ev_ids = add_evidence(
                evidence,
                "state.unresolved_items",
                [source_turn],
                source_type="asr",
            )
            unresolved_items.append(
                {
                    "item": source_turn["text"],
                    "evidence_ids": ev_ids,
                    "status": "predicted_primary_issue",
                }
            )

    escalated_text = " ".join(t["text"] for t in agent_turns)
    escalated = find_patterns(
        escalated_text,
        (r"\bescalat(?:e|ed|ion)\b", r"\braised .* for further investigation\b"),
    )
    escalated_turns = [
        t for t in agent_turns
        if find_patterns(t["text"], (r"\bescalat(?:e|ed|ion)\b", r"\braised .* for further investigation\b"))
    ]
    escalated_evidence = add_evidence(
        evidence,
        "state.escalated",
        escalated_turns,
        source_type="asr",
    )

    goal_confidence = 0.76 if customer_goal_turn and customer_turns else 0.35
    if not customer_turns:
        goal_confidence *= 0.6
        intent_confidence *= 0.6

    return {
        "intent": make_field(
            intent,
            intent_evidence,
            confidence=intent_confidence if intent else 0.0,
            status="PREDICTED",
        ),
        "customer_goal": make_field(
            customer_goal_turn["text"] if customer_goal_turn else None,
            goal_evidence,
            confidence=goal_confidence if customer_goal_turn else 0.0,
            status="PREDICTED",
        ),
        "resolution_state": make_field(
            resolution,
            resolution_evidence,
            confidence=resolution_confidence,
            status="PREDICTED",
        ),
        "escalated": make_field(
            escalated,
            escalated_evidence,
            confidence=0.88,
            status="PREDICTED",
        ),
        "verification": make_field(
            verification,
            verification_evidence,
            confidence=verification_confidence,
            status="PREDICTED",
        ),
        "entities": entities,
        "agent_actions": actions,
        "agent_commitments": commitments,
        "unresolved_items": unresolved_items,
        "next_action": next_action,
    }


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------


def build_interaction(turns: list[dict[str, Any]], raw_packet: dict[str, Any]) -> dict[str, Any]:
    customer_turns = [t for t in turns if t["role"] == "CUSTOMER"]
    agent_turns = [t for t in turns if t["role"] == "AGENT"]

    timed = [t for t in turns if t.get("start") is not None and t.get("end") is not None]

    customer_duration = sum((float(t["end"]) - float(t["start"])) for t in customer_turns if t.get("start") is not None and t.get("end") is not None)
    agent_duration = sum((float(t["end"]) - float(t["start"])) for t in agent_turns if t.get("start") is not None and t.get("end") is not None)
    total_duration = customer_duration + agent_duration
    talk_ratio = customer_duration / total_duration if total_duration > 0 else None

    gaps: list[float] = []
    silence_events = 0
    overlaps: list[float] = []

    if timed:
        timed = sorted(timed, key=lambda t: float(t["start"]))
        for prev, curr in zip(timed, timed[1:]):
            gap = float(curr["start"]) - float(prev["end"])
            if gap >= 0:
                gaps.append(gap)
                if gap >= 0.5:
                    silence_events += 1
            else:
                overlaps.append(-gap)

    mean_latency_ms = (sum(gaps) / len(gaps) * 1000.0) if gaps else None
    overlap_rate = None
    if timed:
        observed = sum(float(t["end"]) - float(t["start"]) for t in timed)
        overlap_rate = min(1.0, sum(overlaps) / observed) if observed > 0 else 0.0

    speaking_rates = Counter()
    prosody = Counter()
    # voice attributes may not be present in the ASR output; if they are,
    # retain distributions without treating them as gold labels.
    for item in raw_packet.get("reference_turns", []):
        attrs = item.get("voice_attributes") or {}
        if attrs.get("speaking_rate"):
            speaking_rates[str(attrs["speaking_rate"])] += 1
        if attrs.get("prosody"):
            prosody[str(attrs["prosody"])] += 1

    return {
        "turn_count": len(turns),
        "customer_turn_count": len(customer_turns),
        "agent_turn_count": len(agent_turns),
        "customer_talk_ratio": talk_ratio,
        "mean_response_latency_ms": mean_latency_ms,
        "silence_event_count": silence_events if timed else None,
        "overlap_rate": overlap_rate,
        "speaking_rate_distribution": dict(speaking_rates) if speaking_rates else None,
        "prosody_summary": dict(prosody) if prosody else None,
    }


# ---------------------------------------------------------------------------
# Main packet construction
# ---------------------------------------------------------------------------


def build_packet(
    raw_packet: dict[str, Any],
    turns: list[dict[str, Any]],
    train_examples: list[tuple[str, str]],
    entity_lexicon: dict[str, set[str]],
) -> dict[str, Any]:
    conversation_id = str(raw_packet.get("conversation_id") or "unknown")

    customer_turns = [t for t in turns if t["role"] == "CUSTOMER"]
    intent_text = customer_turns[0]["text"] if customer_turns else " ".join(t["text"] for t in turns[:2])

    intent, intent_confidence, intent_examples = predict_intent(intent_text, train_examples)
    intent_example_turn = customer_turns[0] if customer_turns else (turns[0] if turns else None)

    evidence: list[dict[str, Any]] = []
    state = build_state(
        turns,
        intent,
        intent_confidence,
        intent_example_turn,
        entity_lexicon,
        evidence,
    )

    speaker_status, speaker_confidence, speaker_reason = summarize_speaker_resolution(raw_packet, turns)

    # Do not assign customer/agent metadata from the synthetic source file.
    # This is a predicted packet.
    conversation = {
        "conversation_id": conversation_id,
        "channel": "voice",
        "scenario": None,
        "scenario_family": None,
        "split": None,
        "num_turns": len(turns),
        "customer_speaker": None,
        "agent_speaker": None,
        "customer_gender": None,
        "agent_gender": None,
    }

    # Overall confidence is deliberately conservative: if speaker mapping is
    # unresolved, role-dependent state gets a lower packet confidence.
    overall_status = "HIGH" if speaker_status == "RESOLVED" else "MEDIUM" if speaker_status == "PARTIAL" else "UNRESOLVED"

    field_uncertainty: list[dict[str, Any]] = []
    if speaker_status != "RESOLVED":
        field_uncertainty.append(
            {
                "field": "speaker_role",
                "confidence": speaker_confidence,
                "reason": speaker_reason,
            }
        )
        field_uncertainty.append(
            {
                "field": "state.customer_goal",
                "confidence": state["customer_goal"]["confidence"],
                "reason": "Customer-role assignment is not fully resolved, so the customer-goal attribution may be uncertain.",
            }
        )

    packet = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "packet_type": PACKET_TYPE,
        "conversation": conversation,
        "state": state,
        "interaction": build_interaction(turns, raw_packet),
        "repair_events": [],
        "evidence": evidence,
        "uncertainty": {
            "overall_status": overall_status,
            "speaker_resolution": {
                "status": speaker_status,
                "confidence": speaker_confidence,
                "reason": speaker_reason,
            },
            "field_uncertainty": field_uncertainty,
        },
        "provenance": {
            "source": "wordlevel_speaker_transcripts",
            "mode": "model_prediction",
            "generated_by": "11_extract_state.py",
        },
    }

    return packet


# ---------------------------------------------------------------------------
# Structural validation (lightweight, no external dependency required)
# ---------------------------------------------------------------------------


def validate_packet(packet: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = [
        "schema_name", "schema_version", "packet_type", "conversation", "state",
        "interaction", "repair_events", "evidence", "uncertainty", "provenance",
    ]
    for key in required:
        if key not in packet:
            errors.append(f"Missing top-level field: {key}")

    if packet.get("schema_name") != SCHEMA_NAME:
        errors.append("schema_name must be RECAP-Voice")
    if packet.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version must be 1.0.0")
    if packet.get("packet_type") != PACKET_TYPE:
        errors.append("packet_type must be recap_voice")

    state = packet.get("state", {})
    for field in [
        "intent", "customer_goal", "resolution_state", "escalated", "verification",
        "entities", "agent_actions", "agent_commitments", "unresolved_items", "next_action",
    ]:
        if field not in state:
            errors.append(f"Missing state field: {field}")

    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build predicted RECAP-Voice state packets.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conversation-id", type=str, default=None)
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Build packets in memory and print summary only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    if not input_dir.exists() and FALLBACK_INPUT_DIR.exists():
        input_dir = FALLBACK_INPUT_DIR
        print(f"Preferred input dir missing; using fallback: {input_dir}")

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}. Run 07b_build_wordlevel_speaker_transcripts.py first."
        )

    metadata = load_json(args.metadata)
    if not isinstance(metadata, list):
        raise ValueError(f"Expected a list in {args.metadata}")

    train_examples, entity_lexicon = build_train_index(metadata)
    print(f"Training intent examples: {len(train_examples)}")
    print(f"Entity lexicon keys: {len(entity_lexicon)}")

    files = sorted(input_dir.glob("*.json"))
    if args.conversation_id:
        files = [p for p in files if p.stem == args.conversation_id]
    if args.max_conversations is not None:
        files = files[: args.max_conversations]

    if not files:
        raise RuntimeError(f"No JSON conversation files found in {input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "recap_predictions.jsonl"
    summary_path = args.output_dir / "recap_prediction_summary.csv"

    packets: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for index, path in enumerate(files, start=1):
        raw = load_json(path)
        turns = normalize_turns(raw)
        packet = build_packet(raw, turns, train_examples, entity_lexicon)
        errors = validate_packet(packet)

        if errors:
            raise ValueError(f"Validation errors for {path.name}: {errors}")

        packets.append(packet)

        summary_rows.append(
            {
                "conversation_id": packet["conversation"]["conversation_id"],
                "turn_count": packet["interaction"]["turn_count"],
                "intent": packet["state"]["intent"]["value"],
                "intent_confidence": packet["state"]["intent"]["confidence"],
                "resolution_state": packet["state"]["resolution_state"]["value"],
                "resolution_confidence": packet["state"]["resolution_state"]["confidence"],
                "entity_count": len(packet["state"]["entities"]),
                "agent_action_count": len(packet["state"]["agent_actions"]),
                "commitment_count": len(packet["state"]["agent_commitments"]),
                "repair_event_count": len(packet["repair_events"]),
                "speaker_resolution": packet["uncertainty"]["speaker_resolution"]["status"],
                "speaker_confidence": packet["uncertainty"]["speaker_resolution"]["confidence"],
                "overall_uncertainty": packet["uncertainty"]["overall_status"],
                "evidence_count": len(packet["evidence"]),
            }
        )

        print(
            f"[{index}/{len(files)}] {path.stem}: "
            f"intent={packet['state']['intent']['value']} "
            f"({packet['state']['intent']['confidence']:.2f}), "
            f"resolution={packet['state']['resolution_state']['value']}, "
            f"speaker={packet['uncertainty']['speaker_resolution']['status']}"
        )

        if not args.dry_run:
            output_path = args.output_dir / path.name
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(packet, f, indent=2, ensure_ascii=False)

    if not args.dry_run:
        with jsonl_path.open("w", encoding="utf-8") as f:
            for packet in packets:
                f.write(json.dumps(packet, ensure_ascii=False) + "\n")

        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\nSaved predictions to: {args.output_dir}")
        print(f"JSONL: {jsonl_path}")
        print(f"Summary: {summary_path}")
    else:
        print("\nDry run complete. No files written.")


if __name__ == "__main__":
    main()
