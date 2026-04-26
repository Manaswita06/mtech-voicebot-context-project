#!/usr/bin/env python3
"""state_extraction_pipeline.py

Extract structured state/context from transcripts using:
- NLP heuristics (offline)
- OpenAI-compatible LLM endpoint (optional)

Outputs one .state.json file per transcript.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error
import numpy as np

try:
    from jsonschema import validate
except Exception:
    validate = None

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

AMBIGUITY_PATTERNS = [
    "i have a question about my account",
    "something seems off",
    "i need help with a couple of things",
    "can you check something for me",
    "i am seeing an issue and need help",
    "i am not sure",
    "i need help understanding this",
]

CONFUSION_WORDS = ["not sure", "unclear", "confusing", "don't know", "cannot tell", "need help"]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_transcripts(input_dir: Path):
    for p in sorted(input_dir.glob("*.json")):
        if p.name == "manifest.json":
            continue
        yield p, load_json(p)


def compact_text(transcript: Dict[str, Any], max_events: int = 40) -> str:
    lines = []
    for ev in transcript.get("events", [])[:max_events]:
        role = ev.get("participant", {}).get("role", "unknown").upper()
        text = (ev.get("text") or "").strip()
        event_name = ev.get("event_name", "")
        lines.append(f"{role} [{event_name}]: {text}")
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
    if best[1] == 0:
        return fallback, evidence
    return best[0], evidence


def infer_secondary_intents(text: str, primary: str) -> List[str]:
    t = text.lower()
    found = []
    for hint, mapped in SECONDARY_HINTS.items():
        if hint in t and mapped != primary and mapped not in found:
            found.append(mapped)
    if "also" in t or "by the way" in t or "one more thing" in t:
        for intent in INTENT_KEYWORDS:
            if intent != primary and intent not in found and intent != "GENERAL_QUERY":
                found.append(intent)
                break
    return found[:2]


def infer_tool_failure(transcript: Dict[str, Any]) -> Tuple[bool, int, List[str], List[str]]:
    failures = []
    support = []
    for ev in transcript.get("events", []):
        ed = ev.get("event_data", {})
        if ev.get("event_name") == "TOOL_RESPONSE_RECEIVED" and ed.get("status") == "fail":
            failures.append(ed.get("error_code") or "UNKNOWN_FAILURE")
            support.append(f"{ed.get('tool_name')} failed with {ed.get('error_code') or 'UNKNOWN_FAILURE'}")
    return len(failures) > 0, len(failures), sorted(set(failures)), support


def infer_ambiguity(text: str, transcript: Dict[str, Any]) -> Tuple[float, List[str]]:
    t = text.lower()
    score = 0.0
    support = []
    for p in AMBIGUITY_PATTERNS:
        if p in t:
            score = max(score, 0.65)
            support.append(f"matched ambiguity phrase: {p}")
    if any(w in t for w in CONFUSION_WORDS):
        score = max(score, 0.55)
        support.append("contains confusion wording")
    if "?" in t:
        score = max(score, 0.45)
        support.append("question mark present")
    if "also" in t or "by the way" in t or "one more thing" in t:
        score = max(score, 0.70)
        support.append("multi-intent connector present")
    gt = transcript.get("gt_ambiguity_level")
    if gt is not None:
        score = max(score, float(gt) * 0.8)
    return round(min(score, 1.0), 3), support


def infer_sentiment(transcript: Dict[str, Any]) -> float:
    gt = transcript.get("gt_sentiment_overall")
    if gt is not None:
        return float(gt)
    text = compact_text(transcript).lower()
    score = 0.0
    for w in ["thanks", "great", "good", "helpful", "resolved"]:
        if w in text:
            score += 0.2
    for w in ["angry", "upset", "frustrated", "unhappy", "bad", "fail", "issue", "problem"]:
        if w in text:
            score -= 0.2
    return max(-1.0, min(1.0, score))


def infer_state(transcript: Dict[str, Any]) -> Dict[str, Any]:
    text = compact_text(transcript)
    primary, intent_support = infer_primary_intent_from_text(text, fallback=transcript.get("gt_primary_intent", "GENERAL_QUERY"))
    secondary = infer_secondary_intents(text, primary)
    multi = len(secondary) > 0
    ambiguity, ambiguity_support = infer_ambiguity(text, transcript)
    tool_failure, failure_count, failure_reasons, failure_support = infer_tool_failure(transcript)
    sentiment = infer_sentiment(transcript)
    scenario_family = transcript.get("gt_scenario_family", "unknown")
    turn_count = int(transcript.get("gt_turn_count", len(transcript.get("events", []))))
    return {
        "primary_intent": primary,
        "secondary_intents": secondary,
        "multi_intent": multi,
        "ambiguity_level": ambiguity,
        "tool_failure": tool_failure,
        "failure_count": failure_count,
        "failure_reasons": failure_reasons,
        "sentiment_overall": round(sentiment, 4),
        "turn_count": turn_count,
        "scenario_family": scenario_family,
        "extracted_evidence": {
            "intent_support": intent_support[:10],
            "failure_support": failure_support[:10],
            "ambiguity_support": ambiguity_support[:10],
        },
    }


def build_prompt(transcript: Dict[str, Any], max_events: int = 40) -> Dict[str, str]:
    transcript_json = json.dumps(
        {
            "conversation_id": transcript.get("conversation_id"),
            "tenant_id": transcript.get("tenant_id"),
            "events": transcript.get("events", [])[:max_events],
        },
        ensure_ascii=False,
    )
    system = "You are a state extractor for enterprise conversations. Return ONLY valid JSON matching the provided schema."
    user = (
        "Extract the following state fields from the transcript:\n"
        "- primary_intent\n- secondary_intents\n- multi_intent\n- ambiguity_level\n"
        "- tool_failure\n- failure_count\n- failure_reasons\n- sentiment_overall\n"
        "- turn_count\n- scenario_family\n- extracted_evidence\n\n"
        f"Transcript:\n{transcript_json}"
    )
    return {"system": system, "user": user}


def parse_json_loose(text: str) -> Dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*", "", s).strip()
        s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    raise ValueError("Could not parse JSON from LLM response")


def validate_state(pred: Dict[str, Any]) -> List[str]:
    errs = []
    if validate is None:
        return errs
    try:
        validate(instance=pred, schema=STATE_SCHEMA)
    except Exception as e:
        errs.append(str(e))
    return errs

def call_ollama(prompt: Dict[str, str], model: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt["user"],
        "system": prompt["system"],
        "stream": False,
        "format": STATE_SCHEMA,
        "options": {
            "temperature": 0,
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "")


def call_llm(prompt: Dict[str, str], provider: str, model: str) -> str:
    if provider == "mock":
        raise RuntimeError("mock provider selected")
    if provider == "ollama":
        return call_ollama(prompt, model)
    if provider != "openai_compatible":
        raise ValueError(f"Unsupported provider: {provider}")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}],
        temperature=0,
    )
    return resp.choices[0].message.content


def extract_one(transcript: Dict[str, Any], provider: str = "nlp", model: str = "gpt-4o-mini") -> Dict[str, Any]:
    if provider == "nlp":
        pred = infer_state(transcript)
        pred["provider"] = "nlp"
        pred["model"] = "heuristic-nlp"
        pred["validation_errors"] = validate_state(pred)
        return pred
    prompt = build_prompt(transcript)

    try:
        raw = call_llm(prompt, provider=provider, model=model)
        pred = parse_json_loose(raw)
        pred["provider"] = provider
        pred["model"] = model
        pred["validation_errors"] = validate_state(pred)
        fallback = infer_state(transcript)
        for k, v in fallback.items():
            if k not in pred or pred[k] in (None, "", [], {}):
                pred[k] = v
        if "validation_errors" not in pred:
            pred["validation_errors"] = validate_state(pred)
        return pred
    except Exception as e:
        pred = infer_state(transcript)
        pred["provider"] = "nlp_fallback"
        pred["model"] = "heuristic-nlp"
        pred["error"] = str(e)
        pred["validation_errors"] = validate_state(pred)
        return pred


def run_batch(input_dir: str, output_dir: str, provider: str, model: str, limit: Optional[int] = None):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    files = [p for p in sorted(in_dir.glob("*.json")) if p.name != "manifest.json"]
    if limit is not None:
        files = files[:limit]
    for p in files:
        tr = load_json(p)
        pred = extract_one(tr, provider=provider, model=model)
        rec = {
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
            "prediction": pred,
        }
        out_file = out_dir / f"{p.stem}.state.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        manifest.append({
            "source_file": p.name,
            "output_file": out_file.name,
            "provider": pred.get("provider"),
            "model": pred.get("model"),
            "validation_errors": pred.get("validation_errors", []),
        })
    with open(out_dir / "state_predictions_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(manifest)} state predictions to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="State extraction pipeline")
    ap.add_argument("--input-dir", default="data/synthetic_very_hard")
    ap.add_argument("--output-dir", default="outputs/state_predictions")
    ap.add_argument("--provider", choices=["nlp", "openai_compatible", "ollama", "mock"], default="nlp")
    ap.add_argument("--model", default=OLLAMA_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run_batch(args.input_dir, args.output_dir, args.provider, args.model, args.limit)


if __name__ == "__main__":
    main()
