#!/usr/bin/env python3
"""Presentation demo for context reuse across two sessions.

This version adds:
- extract_one() wrapper
- nlp mode
- ensemble mode (NLP + LLM, with fallback logic)

It is designed to reuse your existing state_extraction_pipeline.py if available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from state_extraction_pipeline import extract_one as extract_state_one
except Exception:
    extract_state_one = None


@dataclass
class ConversationContext:
    conversation_id: str
    customer_id: str
    primary_intent: str
    secondary_intents: List[str]
    tool_failure: bool
    failure_reasons: List[str]
    ambiguity_level: float
    sentiment_overall: float
    turn_count: int
    summary: str


def simple_fallback_extract(transcript: Dict[str, Any]) -> Dict[str, Any]:
    text = " ".join((ev.get("text") or "") for ev in transcript.get("events", [])).lower()
    if "payment" in text and ("failed" in text or "not go through" in text or "missing" in text):
        primary = "PAYMENT_ISSUE"
    elif "card" in text and ("lost" in text or "replace" in text or "new card" in text):
        primary = "CARD_REPLACEMENT"
    elif "address" in text:
        primary = "ADDRESS_UPDATE"
    elif "statement" in text:
        primary = "STATEMENT_REQUEST"
    else:
        primary = "GENERAL_QUERY"
    return {
        "primary_intent": primary,
        "secondary_intents": [],
        "multi_intent": False,
        "ambiguity_level": 0.3,
        "tool_failure": False,
        "failure_reasons": [],
        "sentiment_overall": -0.1,
        "turn_count": len(transcript.get("events", [])),
        "scenario_family": "demo",
        "extracted_evidence": {"intent_support": [], "failure_support": [], "ambiguity_support": []},
    }


def normalize_prediction(pred: Dict[str, Any], transcript: Dict[str, Any]) -> Dict[str, Any]:
    """Map different extractor outputs into one common shape."""
    return {
        "primary_intent": pred.get("primary_intent", pred.get("gt_primary_intent", "UNKNOWN")),
        "secondary_intents": pred.get("secondary_intents", []),
        "multi_intent": bool(pred.get("multi_intent", len(pred.get("secondary_intents", [])) > 0)),
        "tool_failure": bool(pred.get("tool_failure", False)),
        "failure_reasons": pred.get("failure_reasons", []),
        "ambiguity_level": float(pred.get("ambiguity_level", 0.0)),
        "sentiment_overall": float(pred.get("sentiment_overall", 0.0)),
        "turn_count": int(pred.get("turn_count", len(transcript.get("events", [])))),
        "scenario_family": pred.get("scenario_family", transcript.get("gt_scenario_family", "demo")),
        "extracted_evidence": pred.get(
            "extracted_evidence",
            {"intent_support": [], "failure_support": [], "ambiguity_support": []},
        ),
    }


def ensemble_predict(nlp_pred: Dict[str, Any], llm_pred: Dict[str, Any], transcript: Dict[str, Any]) -> Dict[str, Any]:
    """Simple ensemble that combines NLP and LLM predictions."""
    nlp_pred = normalize_prediction(nlp_pred, transcript)
    llm_pred = normalize_prediction(llm_pred, transcript)

    # Intent voting: prefer agreement, else prefer LLM for harder cases.
    votes = [nlp_pred["primary_intent"], llm_pred["primary_intent"]]
    if votes[0] == votes[1]:
        final_intent = votes[0]
    else:
        final_intent = llm_pred["primary_intent"]

    # Secondary intents: union.
    secondary = list(set(nlp_pred.get("secondary_intents", []) + llm_pred.get("secondary_intents", [])))

    # Tool failure: trust LLM slightly more.
    tool_failure = bool(llm_pred.get("tool_failure", False) or nlp_pred.get("tool_failure", False))

    # Failure reasons: union.
    failure_reasons = list(set(nlp_pred.get("failure_reasons", []) + llm_pred.get("failure_reasons", [])))

    # Ambiguity and sentiment: weighted average.
    ambiguity = 0.4 * float(nlp_pred.get("ambiguity_level", 0.0)) + 0.6 * float(llm_pred.get("ambiguity_level", 0.0))
    sentiment = 0.3 * float(nlp_pred.get("sentiment_overall", 0.0)) + 0.7 * float(llm_pred.get("sentiment_overall", 0.0))

    # Turn count and evidence.
    turn_count = int(max(nlp_pred.get("turn_count", 0), llm_pred.get("turn_count", 0)))
    evidence = {
        "intent_support": list(dict.fromkeys(nlp_pred.get("extracted_evidence", {}).get("intent_support", []) +
                                             llm_pred.get("extracted_evidence", {}).get("intent_support", [])))[:10],
        "failure_support": list(dict.fromkeys(nlp_pred.get("extracted_evidence", {}).get("failure_support", []) +
                                              llm_pred.get("extracted_evidence", {}).get("failure_support", [])))[:10],
        "ambiguity_support": list(dict.fromkeys(nlp_pred.get("extracted_evidence", {}).get("ambiguity_support", []) +
                                                 llm_pred.get("extracted_evidence", {}).get("ambiguity_support", [])))[:10],
    }

    return {
        "primary_intent": final_intent,
        "secondary_intents": secondary,
        "multi_intent": len(secondary) > 0,
        "tool_failure": tool_failure,
        "failure_reasons": failure_reasons,
        "ambiguity_level": round(min(1.0, max(0.0, ambiguity)), 3),
        "sentiment_overall": round(sentiment, 3),
        "turn_count": turn_count,
        "scenario_family": transcript.get("gt_scenario_family", "demo"),
        "extracted_evidence": evidence,
        "nlp_prediction": nlp_pred,
        "llm_prediction": llm_pred,
    }


def extract_one(transcript: Dict[str, Any], provider: str = "nlp", model: str = "qwen2.5:7b-instruct") -> Dict[str, Any]:
    """Single interface for NLP, LLM, or ensemble extraction."""
    if provider == "nlp":
        if extract_state_one is not None:
            pred = extract_state_one(transcript, provider="nlp", model=None)
        else:
            pred = simple_fallback_extract(transcript)
        pred["provider"] = "nlp"
        pred["model"] = "heuristic-nlp"
        return normalize_prediction(pred, transcript) | {"provider": "nlp", "model": "heuristic-nlp"}

    if provider == "ollama":
        if extract_state_one is not None:
            pred = extract_state_one(transcript, provider="ollama", model=model)
        else:
            pred = simple_fallback_extract(transcript)
        pred["provider"] = "ollama"
        pred["model"] = model
        return normalize_prediction(pred, transcript) | {"provider": "ollama", "model": model}

    if provider == "ensemble":
        nlp_pred = extract_one(transcript, provider="nlp", model=model)
        llm_pred = extract_one(transcript, provider="ollama", model=model)
        final = ensemble_predict(nlp_pred, llm_pred, transcript)
        final["provider"] = "ensemble"
        final["model"] = model
        return final

    return simple_fallback_extract(transcript) | {"provider": "fallback", "model": "fallback"}


def make_session_1() -> Dict[str, Any]:
    return {
        "tenant_id": "demo_tenant",
        "conversation_id": "demo-session-1",
        "customer_id": "cust_1001",
        "events": [
            {"event_name": "USER_MESSAGE_RECEIVED", "participant": {"role": "user"}, "text": "My payment didn't go through."},
            {"event_name": "AGENT_MESSAGE_SENT", "participant": {"role": "agent"}, "text": "Let me check that for you."},
            {"event_name": "AGENT_TOOL_CALLED", "participant": {"role": "agent"}, "text": "Calling PAYMENT_LOOKUP", "event_data": {"tool_name": "PAYMENT_LOOKUP", "tool_call_id": "t1"}},
            {"event_name": "TOOL_RESPONSE_RECEIVED", "participant": {"role": "system"}, "text": "Tool response", "event_data": {"tool_name": "PAYMENT_LOOKUP", "tool_call_id": "t1", "status": "fail", "error_code": "TIMEOUT"}},
            {"event_name": "AGENT_MESSAGE_SENT", "participant": {"role": "agent"}, "text": "I will escalate this for manual review."},
        ],
        "gt_primary_intent": "PAYMENT_ISSUE",
        "gt_secondary_intents": [],
        "gt_multi_intent": False,
        "gt_ambiguity_level": 0.25,
        "gt_tool_failure": True,
        "gt_failure_count": 1,
        "gt_sentiment_overall": -0.5,
        "gt_turn_count": 5,
        "gt_scenario_family": "context_reuse_session_1",
    }


def make_session_2() -> Dict[str, Any]:
    return {
        "tenant_id": "demo_tenant",
        "conversation_id": "demo-session-2",
        "customer_id": "cust_1001",
        "events": [
            {"event_name": "USER_MESSAGE_RECEIVED", "participant": {"role": "user"}, "text": "Any update on that?"},
            {"event_name": "AGENT_MESSAGE_SENT", "participant": {"role": "agent"}, "text": "Could you tell me which issue you mean?"},
        ],
        "gt_primary_intent": "PAYMENT_ISSUE",
        "gt_secondary_intents": [],
        "gt_multi_intent": False,
        "gt_ambiguity_level": 0.8,
        "gt_tool_failure": False,
        "gt_failure_count": 0,
        "gt_sentiment_overall": -0.15,
        "gt_turn_count": 2,
        "gt_scenario_family": "context_reuse_session_2",
    }


def compact_text(transcript: Dict[str, Any]) -> str:
    return "\n".join(
        f"{ev.get('participant', {}).get('role', 'unknown').upper()}: {(ev.get('text') or '').strip()}"
        for ev in transcript.get("events", [])
    )


def build_context_object(transcript: Dict[str, Any], extracted: Dict[str, Any]) -> ConversationContext:
    return ConversationContext(
        conversation_id=transcript["conversation_id"],
        customer_id=transcript["customer_id"],
        primary_intent=extracted.get("primary_intent", "UNKNOWN"),
        secondary_intents=extracted.get("secondary_intents", []),
        tool_failure=bool(extracted.get("tool_failure", False)),
        failure_reasons=extracted.get("failure_reasons", []),
        ambiguity_level=float(extracted.get("ambiguity_level", 0.0)),
        sentiment_overall=float(extracted.get("sentiment_overall", 0.0)),
        turn_count=int(extracted.get("turn_count", len(transcript.get("events", [])))),
        summary=(
            f"Primary intent: {extracted.get('primary_intent', 'UNKNOWN')}; "
            f"tool_failure={extracted.get('tool_failure', False)}; "
            f"ambiguity={extracted.get('ambiguity_level', 0.0)}"
        ),
    )


def store_context_json(context: ConversationContext, store_path: Path) -> None:
    store = {}
    if store_path.exists():
        store = json.loads(store_path.read_text(encoding="utf-8"))
    store[context.customer_id] = asdict(context)
    store_path.write_text(json.dumps(store, indent=2), encoding="utf-8")


def retrieve_context_json(customer_id: str, store_path: Path) -> Optional[Dict[str, Any]]:
    if not store_path.exists():
        return None
    store = json.loads(store_path.read_text(encoding="utf-8"))
    return store.get(customer_id)


def ask_without_context(_: Dict[str, Any]) -> str:
    return "Which issue are you referring to?"


def ask_with_context(_: Dict[str, Any], prior_context: Dict[str, Any]) -> str:
    intent = prior_context.get("primary_intent", "the issue")
    if intent == "PAYMENT_ISSUE":
        return "I see this is about your payment issue. I can continue from the earlier case and check the latest status."
    if intent == "CARD_REPLACEMENT":
        return "I see this is about your card replacement request. I can continue from the earlier case."
    return f"I found your earlier context about {intent}. I can continue from there."


def run_demo(output_dir: str = "outputs/context_reuse_demo") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    store_path = out / "context_store.json"

    s1 = make_session_1()
    extracted_1 = extract_one(s1, provider="ensemble", model="qwen2.5:7b-instruct")
    context_1 = build_context_object(s1, extracted_1)
    store_context_json(context_1, store_path)

    s2 = make_session_2()
    extracted_2_without_memory = extract_one(s2, provider="nlp", model="qwen2.5:7b-instruct")
    prior = retrieve_context_json(s2["customer_id"], store_path)

    reply_without = ask_without_context(s2)
    reply_with = ask_with_context(s2, prior or {})

    report = {
        "session_1": {
            "transcript": compact_text(s1),
            "extracted_context": asdict(context_1),
        },
        "session_2": {
            "transcript": compact_text(s2),
            "without_context_reply": reply_without,
            "with_context_reply": reply_with,
            "raw_extraction_without_memory": extracted_2_without_memory,
            "retrieved_prior_context": prior,
        },
    }

    (out / "context_reuse_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== SESSION 1 ===")
    print(report["session_1"]["transcript"])
    print("\nExtracted context:")
    print(json.dumps(report["session_1"]["extracted_context"], indent=2))

    print("\n=== SESSION 2 ===")
    print(report["session_2"]["transcript"])
    print("\nWithout stored context:")
    print(reply_without)
    print("\nWith stored context:")
    print(reply_with)

    print(f"\nSaved demo report to: {out.resolve()}")
    print(f"Saved context store to: {store_path.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/context_reuse_demo")
    args = ap.parse_args()
    run_demo(args.output_dir)


if __name__ == "__main__":
    main()
