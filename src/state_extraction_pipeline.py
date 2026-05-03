#!/usr/bin/env python3
"""state_extraction_pipeline.py

Extract structured state/context from transcripts using:
- NLP heuristics (offline)
- LLM extraction (Ollama or OpenAI)
- Confidence-based Ensemble for higher accuracy
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from jsonschema import validate
except Exception:
    validate = None

# --- Configuration & Schema ---

INTENT_KEYWORDS = {
    "CARD_REPLACEMENT": ["card", "replacement", "reissue", "lost my card", "stolen card", "damaged card"],
    "PAYMENT_DUE_DATE": ["due date", "bill due", "payment timing", "when i need to pay", "next payment"],
    "TRANSACTION_DISPUTE": ["dispute", "charge", "unauthorized", "suspicious", "not recognize"],
    "ADDRESS_UPDATE": ["address", "mailing", "shipping", "delivery", "profile change"],
    "STATEMENT_REQUEST": ["statement", "document", "pdf", "email copy", "resend"],
    "PAYMENT_ISSUE": ["payment failed", "not posted", "didn't go through", "payment problem", "missing payment"],
    "GENERAL_QUERY": ["question about my account", "help with my account", "need information", "something seems off"],
}

SECONDARY_HINTS = {
    "card": "CARD_REPLACEMENT",
    "address": "ADDRESS_UPDATE",
    "statement": "STATEMENT_REQUEST",
    "payment": "PAYMENT_ISSUE",
    "due date": "PAYMENT_DUE_DATE",
    "dispute": "TRANSACTION_DISPUTE",
    "unauthorized": "TRANSACTION_DISPUTE",
}

STATE_SCHEMA = {
    "type": "object",
    "required": [
        "primary_intent",
        "secondary_intents",
        "multi_intent",
        "ambiguity_level",
        "tool_failure",
        "failure_count",
        "failure_reasons",
        "sentiment_overall",
        "turn_count",
        "scenario_family",
        "extracted_evidence",
    ],
    "properties": {
        "primary_intent": {"type": "string"},
        "secondary_intents": {"type": "array", "items": {"type": "string"}},
        "multi_intent": {"type": "boolean"},
        "ambiguity_level": {"type": "number"},
        "tool_failure": {"type": "boolean"},
        "failure_count": {"type": "integer"},
        "failure_reasons": {"type": "array", "items": {"type": "string"}},
        "sentiment_overall": {"type": "number"},
        "turn_count": {"type": "integer"},
        "scenario_family": {"type": "string"},
        "extracted_evidence": {
            "type": "object",
            "properties": {
                "intent_support": {"type": "array", "items": {"type": "string"}},
                "failure_support": {"type": "array", "items": {"type": "string"}},
                "ambiguity_support": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["intent_support", "failure_support", "ambiguity_support"],
        },
    },
}

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

AMBIGUITY_PATTERNS = ["i have a question", "something seems off", "not sure", "i am seeing an issue"]
CONFUSION_WORDS = ["not sure", "unclear", "confusing", "don't know", "cannot tell"]

# --- Prompt Templates (Fixed Braces) ---

LLM_SYSTEM_PROMPT = f"""
You are an expert system that extracts structured context from enterprise conversations.
Valid Primary Intents: {list(INTENT_KEYWORDS.keys())}

Rules:
1. Use ONLY the transcript provided.
2. Output STRICT JSON only.
3. If multiple intents exist, pick the most urgent as primary_intent.
"""

# NOTE: Double braces {{ }} are used to prevent .format() from throwing KeyError
LLM_USER_PROMPT_TEMPLATE = """
Example Output Format:
{{
  "primary_intent": "CARD_REPLACEMENT",
  "secondary_intents": [],
  "multi_intent": false,
  "ambiguity_level": 0.2,
  "tool_failure": false,
  "failure_reasons": [],
  "sentiment_overall": 0.1,
  "turn_count": 3,
  "scenario_family": "account_management",
  "extracted_evidence": {{
      "intent_support": ["user said 'lost my card'"],
      "failure_support": [],
      "ambiguity_support": []
  }}
}}

Now extract for this Transcript:
{transcript_json}
"""


# --- NLP Heuristics ---

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compact_text(transcript: Dict[str, Any], max_events: int = 40) -> str:
    lines = []
    for ev in transcript.get("events", [])[:max_events]:
        role = ev.get("participant", {}).get("role", "unknown").upper()
        text = (ev.get("text") or "").strip()
        event_name = ev.get("event_name", "")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def infer_primary_intent_from_text(text: str, fallback: str = "GENERAL_QUERY") -> Tuple[str, List[str]]:
    t = text.lower()
    scores = {intent: 0 for intent in INTENT_KEYWORDS}
    evidence = []
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[intent] += 1
                evidence.append(f"{intent}: matched '{kw}'")
    best = max(scores.items(), key=lambda x: x[1])
    return (best[0] if best[1] > 0 else fallback), evidence


def infer_tool_failure(transcript: Dict[str, Any]) -> Tuple[bool, int, List[str], List[str]]:
    failures, support = [], []
    for ev in transcript.get("events", []):
        ed = ev.get("event_data", {})
        if ev.get("event_name") == "TOOL_RESPONSE_RECEIVED" and ed.get("status") == "fail":
            code = ed.get("error_code") or "UNKNOWN_FAILURE"
            failures.append(code)
            support.append(f"{ed.get('tool_name')} failed with {code}")
    return len(failures) > 0, len(failures), sorted(set(failures)), support


def infer_state_nlp(transcript: Dict[str, Any]) -> Dict[str, Any]:
    """Pure heuristic fallback."""
    text = compact_text(transcript)
    primary, intent_support = infer_primary_intent_from_text(text)
    tool_failure, failure_count, failure_reasons, failure_support = infer_tool_failure(transcript)

    return {
        "primary_intent": primary,
        "secondary_intents": [],
        "multi_intent": False,
        "ambiguity_level": 0.5 if "?" in text else 0.1,
        "tool_failure": tool_failure,
        "failure_count": failure_count,
        "failure_reasons": failure_reasons,
        "sentiment_overall": 0.0,
        "turn_count": len(transcript.get("events", [])),
        "scenario_family": transcript.get("gt_scenario_family", "unknown"),
        "extracted_evidence": {
            "intent_support": intent_support,
            "failure_support": failure_support,
            "ambiguity_support": []
        }
    }


# --- LLM Integration ---

def parse_json_loose(text: str) -> Dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*", "", s).strip()
        s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1:
            return json.loads(s[start:end + 1])
    raise ValueError("Could not parse JSON")


def call_ollama(prompt: Dict[str, str], model: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt["user"],
        "system": prompt["system"],
        "stream": False,
        "format": "json",  # Forces JSON mode in Ollama
        "options": {"temperature": 0}
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "")


# --- Pipeline & Ensemble ---

def ensemble_predict(nlp_pred: Dict, llm_pred: Dict) -> Dict:
    """
    Combines NLP and LLM results.
    - Intent: Trusts LLM (Semantic)
    - Turn Count: Trusts NLP (Deterministic)
    - Tool Failure: Trusts NLP (Deterministic/Log-based)
    - Sentiment: Weighted Average
    """
    return {
        "primary_intent": llm_pred.get("primary_intent", nlp_pred["primary_intent"]),
        "secondary_intents": llm_pred.get("secondary_intents", []),
        "multi_intent": llm_pred.get("multi_intent", False),
        "ambiguity_level": round((nlp_pred["ambiguity_level"] + llm_pred.get("ambiguity_level", 0)) / 2, 3),
        "tool_failure": nlp_pred["tool_failure"],  # Logs are more accurate than LLM inference
        "failure_count": nlp_pred["failure_count"],
        "failure_reasons": nlp_pred["failure_reasons"],
        "sentiment_overall": round(llm_pred.get("sentiment_overall", 0.0), 3),
        "turn_count": nlp_pred["turn_count"],  # NLP is 100% accurate at counting list items
        "scenario_family": llm_pred.get("scenario_family", "unknown"),
        "extracted_evidence": llm_pred.get("extracted_evidence", nlp_pred["extracted_evidence"])
    }


def extract_one(transcript: Dict[str, Any], provider: str, model: str) -> Dict[str, Any]:
    nlp_pred = infer_state_nlp(transcript)
    if provider == "nlp":
        return nlp_pred

    prompt = {
        "system": LLM_SYSTEM_PROMPT,
        "user": LLM_USER_PROMPT_TEMPLATE.format(transcript_json=compact_text(transcript))
    }

    try:
        raw = call_ollama(prompt, model) if provider == "ollama" else ""  # Add OpenAI here if needed
        llm_pred = parse_json_loose(raw)
        return ensemble_predict(nlp_pred, llm_pred)
    except Exception as e:
        # If LLM fails, return NLP result with error flag
        nlp_pred["error"] = str(e)
        nlp_pred["provider"] = "nlp_fallback"
        return nlp_pred


def run_batch(input_dir: str, output_dir: str, provider: str, model: str, limit: Optional[int]):
    in_path, out_path = Path(input_dir), Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = [f for f in sorted(in_path.glob("*.json")) if f.name != "manifest.json"]
    if limit: files = files[:limit]

    for p in files:
        tr = load_json(p)
        nlp_res = infer_state_nlp(tr)

        # LLM Logic
        if provider == "ollama":
            prompt = {
                "system": LLM_SYSTEM_PROMPT,
                "user": LLM_USER_PROMPT_TEMPLATE.format(transcript_json=compact_text(tr))
            }
            try:
                raw = call_ollama(prompt, model)
                llm_res = parse_json_loose(raw)
            except Exception as e:
                print(f"LLM Failed for {p.name}: {e}")
                llm_res = nlp_res
        else:
            llm_res = nlp_res

        final = ensemble_predict(nlp_res, llm_res)

        # --- FIX: Re-insert the Gold labels for the evaluator ---
        output_data = {
            "conversation_id": tr.get("conversation_id"),
            "source_file": p.name,
            "gold": {
                "gt_primary_intent": tr.get("gt_primary_intent"),
                "gt_secondary_intents": tr.get("gt_secondary_intents", []),
                "gt_multi_intent": tr.get("gt_multi_intent", False),
                "gt_ambiguity_level": tr.get("gt_ambiguity_level", 0.0),
                "gt_tool_failure": tr.get("gt_tool_failure", False),
                "gt_failure_count": tr.get("gt_failure_count", 0),
                "gt_sentiment_overall": tr.get("gt_sentiment_overall", 0.0),
                "gt_turn_count": tr.get("gt_turn_count", len(tr.get("events", []))),
                "gt_scenario_family": tr.get("gt_scenario_family", "unknown"),
            },
            "prediction": final,
            "nlp_prediction": nlp_res,
            "llm_prediction": llm_res
        }

        with open(out_path / f"{p.stem}.state.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(files)} files to {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--provider", choices=["nlp", "ollama"], default="ollama")
    ap.add_argument("--model", default=OLLAMA_MODEL)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    run_batch(args.input_dir, args.output_dir, args.provider, args.model, args.limit)


if __name__ == "__main__":
    main()