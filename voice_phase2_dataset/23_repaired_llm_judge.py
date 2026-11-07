#!/usr/bin/env python3
"""Evidence-anchored re-evaluation for RECAP-Voice acoustic-context ablation with conservative review flags.

Key repairs over the first judge:
- Scores each candidate response independently, so rationales cannot silently switch A/B.
- Requires verbatim response quotes for identified factual/re-ask/adaptation concerns.
- Requires verbatim source evidence for claims assessed as supported/contradicted.
- Verifies those quotes in Python before accepting a claim as evidence-anchored.
- Runs four pairwise comparisons with balanced response order (two each order).
- Marks non-unanimous pairwise decisions as uncertain rather than treating them as wins.

This is an automated audit helper, not human-validated ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)

ACOUSTIC_KEYS = {
    "interaction_context", "acoustic_observations", "acoustic_interaction_features", "audio_features"
}

INSTRUCTIONS = """
You are an independent evaluator of one customer-support handoff response. You do not know
which experimental condition generated it. Score only the response shown in this request.
Never refer to another response or infer its contents.

SOURCE DISCIPLINE
- Use only the supplied source transcript, explicit gold/reference state, and handover capsule.
- A general `resolved` label does not by itself prove a specific action happened; look for
  the specific action in the dialogue/reference state.
- A request being completed in a later turn means an earlier request for verification may be
  stale. Use the sequence of turns, not just one status label.
- Separate CONTRADICTED (source says the opposite) from NOT_EVIDENCED (source does not establish
  it) and AMBIGUOUS (source is incomplete/conflicting). Lack of evidence is not proof of falsity.
- For EVERY factual claim you flag or rely on, quote a short, exact, contiguous span from the
  candidate response. Do not paraphrase it. The evaluator will check that this quotation occurs
  in the candidate response. If you cannot find an exact quotation, do not include that claim.
- If assessment is SUPPORTED or CONTRADICTED, provide an exact quotation from the source transcript
  (or exact structured-state text) that supports this assessment. The evaluator will check it.
  Do not invent turn IDs or source quotations. Use empty source_quote if no appropriate quote exists.
- Do not criticize a response for asking a question or claiming an action it never mentions.
  Do not use the other response's text as evidence for this response.
- Do not infer emotion, honesty, intent, confusion, or mental state from pauses, silence, speaking
  rate, talk ratio, overlap, or acoustic cues. Interaction metadata may only support practical
  choices such as clarity, concise wording, or pacing when the relation is reasonable.
- If acoustic/interaction metadata is available, adaptation_score must be 0, 1, or 2. Use 0 when
  no helpful adaptation is demonstrated, 1 for plausible but weak adaptation, and 2 for clear,
  appropriate adaptation grounded in observable metadata. Use null ONLY when metadata is absent.
- `factual_reason` may summarize only findings represented in `factual_claims`; never introduce a
  criticism about verification, resolution, or another action unless that claim is actually present
  in candidate_response and appears in `factual_claims` with an exact response_quote.
- Be conservative. If evidence is unclear, say so in the reason and avoid a strong factual penalty.

RUBRIC
- continuity_score: 1=loses prior context; 3=adequate; 5=uses prior progress seamlessly.
- unnecessary_reask: 1 only when response explicitly requests information already present and
  still valid in the source; otherwise 0. The response quote must show the question/request.
- adaptation_score: defined above.
- unsupported_psychological_inference: 1 only if the response itself makes an unsupported claim
  about the customer's emotion, honesty, intent, confusion or mental state; quote it exactly.
- factual_consistency_score: 1=major contradicted material claims; 3=mixed/uncertain; 5=material
  claims are source-supported and appropriately calibrated.
- factual_claims: enumerate material factual/action/status/procedural claims made in this response.
  Do not create claims that are not in the response. A generic offer to help is not a factual claim.
""".strip()

CLAIM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "response_quote": {"type": "string"},
        "claim_type": {"type": "string", "enum": ["completed_action", "status", "identity_or_entity", "process_or_security", "promise_or_timeline", "other_factual"]},
        "assessment": {"type": "string", "enum": ["supported", "contradicted", "not_evidenced", "ambiguous"]},
        "severity": {"type": "string", "enum": ["low", "moderate", "high"]},
        "source_quote": {"type": "string"},
        "source_reference": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["response_quote", "claim_type", "assessment", "severity", "source_quote", "source_reference", "explanation"],
}

SCORE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "continuity_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "continuity_reason": {"type": "string"},
        "unnecessary_reask": {"type": "integer", "enum": [0, 1]},
        "reask_response_quote": {"type": "string"},
        "reask_reason": {"type": "string"},
        "adaptation_score": {"type": ["integer", "null"], "minimum": 0, "maximum": 2},
        "adaptation_response_quote": {"type": "string"},
        "adaptation_reason": {"type": "string"},
        "unsupported_psychological_inference": {"type": "integer", "enum": [0, 1]},
        "psychological_inference_response_quote": {"type": "string"},
        "psychological_inference_reason": {"type": "string"},
        "factual_consistency_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "factual_reason": {"type": "string"},
        "factual_claims": {"type": "array", "items": CLAIM_SCHEMA},
    },
    "required": [
        "continuity_score", "continuity_reason", "unnecessary_reask", "reask_response_quote",
        "reask_reason", "adaptation_score", "adaptation_response_quote", "adaptation_reason",
        "unsupported_psychological_inference", "psychological_inference_response_quote",
        "psychological_inference_reason", "factual_consistency_score", "factual_reason", "factual_claims"
    ],
}

PAIR_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "preference": {"type": "string", "enum": ["A_BETTER", "B_BETTER", "NO_MEANINGFUL_DIFFERENCE"]},
        "response_a_quote": {"type": "string"},
        "response_b_quote": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["preference", "response_a_quote", "response_b_quote", "reason"],
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_conversations(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    raw = load_json(path)
    records = raw if isinstance(raw, list) else raw.get("conversations", raw.get("data", []))
    if isinstance(raw, dict) and not records and all(isinstance(v, dict) for v in raw.values()):
        records = list(raw.values())
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        cid = str(rec.get("conversation_id") or rec.get("id") or "")
        if cid:
            out[cid] = rec
    return out


def normalize_for_match(value: str) -> str:
    """Normalize whitespace/case/punctuation for robust literal quote validation."""
    value = value.casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def quote_is_in(quote: Any, text: str) -> bool:
    if not isinstance(quote, str) or not quote.strip():
        return False
    q = normalize_for_match(quote)
    haystack = normalize_for_match(text)
    return bool(q) and q in haystack


def strip_voice_annotations(turn: dict[str, Any]) -> dict[str, Any]:
    return {k: turn.get(k) for k in ("turn_id", "role", "dialogue_act", "text") if k in turn}


def has_acoustic(obj: Any) -> bool:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).strip().lower() in ACOUSTIC_KEYS and v not in (None, {}, [], ""):
                return True
            if has_acoustic(v):
                return True
    elif isinstance(obj, list):
        return any(has_acoustic(x) for x in obj)
    return False


def build_context(cid: str, record: dict[str, Any] | None, capsule: Any, gold: Any, customer_message: str, max_chars: int) -> tuple[dict[str, Any], str, bool]:
    source: dict[str, Any] = {"conversation_id": cid, "customer_follow_up_message": customer_message}
    source_text_parts: list[str] = [f"Conversation ID: {cid}", f"Customer follow-up: {customer_message}"]
    if record:
        safe_record = {
            "scenario": record.get("scenario"),
            "scenario_family": record.get("scenario_family"),
            "intent": record.get("intent"),
            "resolution_status": record.get("resolution_status"),
            "escalated": record.get("escalated"),
            "entities": record.get("entities"),
            "turns": [strip_voice_annotations(t) for t in record.get("turns", []) if isinstance(t, dict)],
        }
        transcript_text = "\n".join(
            f"Turn {t.get('turn_id', '?')} [{t.get('role', 'unknown')}]: {t.get('text', '')}"
            for t in safe_record.get("turns", [])
        )
        source_text_parts.append(transcript_text)
        if len(transcript_text) > max_chars:
            excerpt = transcript_text[:max_chars] + "\n[TRANSCRIPT TRUNCATED]"
            source_text_parts[-1] = excerpt
            # Keep prompt transcript and local quote-validation transcript aligned.
            safe_record["turns"] = []
            safe_record["transcript_excerpt"] = excerpt
            safe_record["transcript_was_truncated"] = True
        source["conversation_reference"] = safe_record
    else:
        source["conversation_reference"] = None
    if gold is not None:
        source["gold_state_reference"] = gold
        source_text_parts.append("Gold/reference state JSON:\n" + json.dumps(gold, ensure_ascii=False))
    if capsule is not None:
        source["handover_capsule_reference"] = capsule
        source_text_parts.append("Handover capsule JSON:\n" + json.dumps(capsule, ensure_ascii=False))
    audio_available = has_acoustic(capsule) if capsule is not None else False
    audio_meta = None
    if capsule is not None:
        # Preserve the actual fields; these are evidence, not condition labels.
        audio_meta = {}
        def collect(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    p = f"{path}.{k}" if path else str(k)
                    if str(k).lower() in ACOUSTIC_KEYS and v not in (None, {}, [], ""):
                        audio_meta[p] = v
                    collect(v, p)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    collect(v, f"{path}[{i}]")
        collect(capsule)
        if not audio_meta:
            audio_meta = None
    source["observable_interaction_metadata_for_adaptation_only"] = audio_meta
    if audio_meta is not None:
        source_text_parts.append("Observable interaction metadata:\n" + json.dumps(audio_meta, ensure_ascii=False))
    source["evidence_rules"] = (
        "Transcript turn order is important. Distinguish completed actions from planned actions. "
        "`resolved` alone is not proof of every specific action. Do not infer psychology from audio."
    )
    source_text = "\n".join(p for p in source_text_parts if p)
    return source, source_text, audio_available


def parse_json(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON object in API response: {text[:300]}")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("API response was not a JSON object")
    return obj


def call_structured(client: OpenAI, model: str, name: str, schema: dict[str, Any], instructions: str, payload: dict[str, Any], max_tokens: int = 2400, retries: int = 3) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=json.dumps(payload, ensure_ascii=False),
                text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}},
                max_output_tokens=max_tokens,
            )
            out = (getattr(response, "output_text", "") or "").strip()
            if not out:
                raise ValueError("Empty judge output")
            return parse_json(out)
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                delay = min(2 ** attempt, 12) + random.random()
                print(f"  retry {attempt + 1}: {type(exc).__name__}; waiting {delay:.1f}s", file=sys.stderr)
                time.sleep(delay)
    assert last_err is not None
    raise last_err


def evaluate_one(client: OpenAI, model: str, cid: str, response: str, condition_for_internal_join: str,
                 source: dict[str, Any], source_text: str, acoustic_available: bool,
                 max_tokens: int, retries: int) -> dict[str, Any]:
    payload = {
        "task": "Score ONLY the single candidate response and audit only claims actually made in it.",
        "conversation_source": source,
        "candidate_response": response,
        "metadata_available": acoustic_available,
        "required_output_rules": [
            "Every factual_claims[].response_quote must be an exact, contiguous quote from candidate_response.",
            "Every non-empty factual_claims[].source_quote must be an exact quote from conversation_source.",
            "Do not mention or speculate about another candidate response.",
            "If metadata_available is true, adaptation_score must be numeric, including 0 when no adaptation is demonstrated.",
        ],
    }
    result = call_structured(client, model, "recap_single_response_audit", SCORE_SCHEMA, INSTRUCTIONS, payload, max_tokens, retries)
    checked_claims = []
    errors = []
    for idx, claim in enumerate(result.get("factual_claims", []), 1):
        rq = str(claim.get("response_quote", "") or "")
        sq = str(claim.get("source_quote", "") or "")
        rq_ok = quote_is_in(rq, response)
        assessment = claim.get("assessment")
        sq_required = assessment in {"supported", "contradicted"}
        sq_ok = quote_is_in(sq, source_text) if sq else False
        item = dict(claim)
        item["response_quote_valid"] = rq_ok
        item["source_quote_valid"] = (sq_ok if sq_required else (sq_ok if sq else None))
        item["claim_evidence_status"] = "validated" if (rq_ok and (not sq_required or sq_ok)) else "invalid_or_unverified"
        if not rq_ok:
            errors.append(f"factual_claim_{idx}_response_quote_not_found")
        if sq_required and not sq_ok:
            errors.append(f"factual_claim_{idx}_source_quote_not_found")
        checked_claims.append(item)
    result["factual_claims"] = checked_claims

    # Validate quotes used to justify other binary / ordinal judgements.
    reask_quote = str(result.get("reask_response_quote", "") or "")
    reask_quote_ok = quote_is_in(reask_quote, response) if reask_quote else None
    psych_quote = str(result.get("psychological_inference_response_quote", "") or "")
    psych_quote_ok = quote_is_in(psych_quote, response) if psych_quote else None
    adapt_quote = str(result.get("adaptation_response_quote", "") or "")
    adapt_quote_ok = quote_is_in(adapt_quote, response) if adapt_quote else None
    if result.get("unnecessary_reask") == 1 and not reask_quote_ok:
        errors.append("unnecessary_reask_has_no_valid_response_quote")
    if result.get("unsupported_psychological_inference") == 1 and not psych_quote_ok:
        errors.append("psychological_inference_has_no_valid_response_quote")
    if result.get("adaptation_score", None) not in (0, None) and not adapt_quote_ok:
        errors.append("adaptation_score_above_zero_has_no_valid_response_quote")
    if acoustic_available and result.get("adaptation_score") is None:
        errors.append("adaptation_score_null_despite_metadata_available")
    if (not acoustic_available) and result.get("adaptation_score") is not None:
        errors.append("adaptation_score_numeric_without_metadata")

    confirmed_contradictions = sum(
        1 for c in checked_claims
        if c.get("assessment") == "contradicted" and c.get("claim_evidence_status") == "validated"
    )
    material_unverified_claims = sum(
        1 for c in checked_claims
        if c.get("severity") == "high" and c.get("assessment") in {"not_evidenced", "ambiguous"}
        and c.get("response_quote_valid") is True
    )
    score = result.get("factual_consistency_score")
    if isinstance(score, int) and score <= 3 and confirmed_contradictions == 0:
        errors.append("low_factual_score_without_quote_validated_contradiction_check_claims")
    if isinstance(score, int) and score >= 4 and confirmed_contradictions > 0:
        errors.append("high_factual_score_despite_quote_validated_contradiction")
    if errors:
        audit_status = "needs_review"
    elif material_unverified_claims:
        audit_status = "needs_review_for_high_risk_unverified_claim"
    else:
        audit_status = "quote_checks_passed_not_semantic_proof"
    result["quote_validation"] = {
        "status": audit_status,
        "flags": errors,
        "validated_contradiction_count": confirmed_contradictions,
        "high_severity_unverified_or_ambiguous_claim_count": material_unverified_claims,
        "reask_quote_valid": reask_quote_ok,
        "psychological_inference_quote_valid": psych_quote_ok,
        "adaptation_quote_valid": adapt_quote_ok,
    }
    return result


def evaluate_pair(client: OpenAI, model: str, source: dict[str, Any], response_a: str, response_b: str, run_index: int, retries: int) -> dict[str, Any]:
    instructions = """
You are a blind pairwise evaluator of two customer-support handoff responses. Judge ONLY
substantive quality grounded in the supplied source: correct state/action claims, continuity,
asking for genuinely missing information, calibrated uncertainty, and safe handling. Do not
reward length, confidence, polish, or a claim that is not supported by the source.
If you choose A_BETTER, response_a_quote must exactly quote a decisive span from A and
response_b_quote must exactly quote a relevant span from B. Reverse the roles when B is better.
If there is no meaningful difference, return empty quotes. Do not guess which condition generated
which response. Use only the current A/B order.
""".strip()
    payload = {
        "task": f"Blinded pairwise decision, repeat {run_index}.",
        "source_context": source,
        "response_A": response_a,
        "response_B": response_b,
        "decision_criteria_priority": [
            "material factual correctness and state consistency",
            "continuity without repeating already completed actions",
            "appropriate uncertainty and safe requests for missing information",
            "interaction adaptation only when grounded in observable metadata",
        ],
    }
    out = call_structured(client, model, "recap_pairwise_quote_audit", PAIR_SCHEMA, instructions, payload, 700, retries)
    a_q = str(out.get("response_a_quote", "") or "")
    b_q = str(out.get("response_b_quote", "") or "")
    if out.get("preference") == "A_BETTER":
        valid = quote_is_in(a_q, response_a) and quote_is_in(b_q, response_b)
    elif out.get("preference") == "B_BETTER":
        valid = quote_is_in(a_q, response_a) and quote_is_in(b_q, response_b)
    else:
        valid = True
    out["quote_validation_passed"] = valid
    if not valid:
        out["validation_note"] = "Winner justification quote missing from corresponding response; preference excluded."
    return out


def to_condition(preference: str, condition_a: str) -> str:
    if preference == "NO_MEANINGFUL_DIFFERENCE":
        return "tie"
    if preference == "A_BETTER":
        return condition_a
    if preference == "B_BETTER":
        return "with_acoustic" if condition_a == "without_acoustic" else "without_acoustic"
    return "invalid"


def preference_summary(votes: list[str]) -> str:
    valid = [x for x in votes if x in {"with_acoustic", "without_acoustic", "tie"}]
    if len(valid) < 3:
        return "insufficient_valid_votes"
    counts = {k: valid.count(k) for k in ("with_acoustic", "without_acoustic", "tie")}
    if counts["with_acoustic"] == len(valid):
        return "unanimous_with_acoustic"
    if counts["without_acoustic"] == len(valid):
        return "unanimous_without_acoustic"
    if counts["tie"] == len(valid):
        return "unanimous_tie"
    max_count = max(counts.values())
    leaders = [k for k, v in counts.items() if v == max_count]
    if max_count >= 3 and len(leaders) == 1:
        return f"majority_{leaders[0]}_nonunanimous"
    return "inconsistent_or_order_sensitive"


def _as_float(value: Any) -> float | None:
    """Return a finite numeric value, or None for missing/non-numeric cells."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def build_review_decision(
    audit_by_condition: dict[str, dict[str, Any]],
    pair_runs: list[dict[str, Any]],
    summary_pref: str,
    *,
    transcript_available: bool,
    capsule_available: bool,
    major_score_diff: float = 2.0,
) -> dict[str, Any]:
    """Conservatively flag rows that need review.

    A passing literal quote check is not treated as semantic proof. Any detected
    contradiction, missing/invalid required quote, large score gap, binary
    disagreement, invalid pairwise quote, or unstable preference flags the row.
    """
    reasons: list[str] = []
    contradiction_counts: dict[str, int] = {}
    validated_contradiction_counts: dict[str, int] = {}
    missing_required_quote = False
    detected_contradiction = False
    major_differences: list[dict[str, Any]] = []
    binary_disagreements: list[dict[str, Any]] = []

    for condition in ("without_acoustic", "with_acoustic"):
        audit = audit_by_condition.get(condition, {}) or {}
        claims = audit.get("factual_claims", []) or []
        detected = [c for c in claims if c.get("assessment") == "contradicted"]
        validated = [
            c for c in detected
            if c.get("claim_evidence_status") == "validated"
            and c.get("response_quote_valid") is True
            and c.get("source_quote_valid") is True
        ]
        contradiction_counts[condition] = len(detected)
        validated_contradiction_counts[condition] = len(validated)

        # Crucial: a model-labelled contradiction triggers review even if its
        # quotes failed validation. Record separately whether it was validated.
        if detected:
            detected_contradiction = True
            reasons.append(
                f"{condition}: {len(detected)} claim(s) labelled contradicted "
                f"({len(validated)} have locally validated response and source quotes)"
            )

        quote_validation = audit.get("quote_validation", {}) or {}
        flags = quote_validation.get("flags", []) or []
        for flag in flags:
            flag_text = str(flag).lower()
            missing_required_quote = missing_required_quote or any(
                marker in flag_text
                for marker in (
                    "quote_not_found",
                    "source_quote_not_found",
                    "response_quote_not_found",
                    "no_valid_response_quote",
                    "has_no_valid_response_quote",
                )
            )
            reasons.append(f"{condition}: quote-validation flag: {flag}")

        # Re-check each claim independently instead of trusting status alone.
        for idx, claim in enumerate(claims, 1):
            response_quote_valid = claim.get("response_quote_valid") is True
            source_quote_valid = claim.get("source_quote_valid") is True
            assessment = claim.get("assessment")
            if not response_quote_valid:
                missing_required_quote = True
                reasons.append(
                    f"{condition}: factual claim {idx} has a missing/invalid candidate-response quote"
                )
            if assessment in {"supported", "contradicted"} and not source_quote_valid:
                missing_required_quote = True
                reasons.append(
                    f"{condition}: factual claim {idx} requires a source quote, but it is missing or not located"
                )

        # A high-risk claim marked not-evidenced/ambiguous warrants review even
        # if the literal quote is found: presence is not entailment.
        for idx, claim in enumerate(claims, 1):
            if (
                claim.get("severity") == "high"
                and claim.get("assessment") in {"not_evidenced", "ambiguous"}
                and claim.get("response_quote_valid") is True
            ):
                reasons.append(
                    f"{condition}: high-severity claim {idx} is not evidenced or ambiguous"
                )

        # Existing independent-audit integrity flags always carry through.
        if str(quote_validation.get("status", "")).startswith("needs_review") and not flags:
            reasons.append(f"{condition}: quote-validation status is {quote_validation.get('status')}")

    score_specs = [
        ("factual_consistency_score", "factual_consistency_score"),
        ("continuity_score", "continuity_score"),
        ("adaptation_score", "adaptation_score"),
    ]
    for output_name, audit_key in score_specs:
        left = _as_float((audit_by_condition.get("without_acoustic", {}) or {}).get(audit_key))
        right = _as_float((audit_by_condition.get("with_acoustic", {}) or {}).get(audit_key))
        if left is None or right is None:
            continue
        difference = right - left
        if abs(difference) >= major_score_diff:
            item = {
                "metric": output_name,
                "without_acoustic": left,
                "with_acoustic": right,
                "with_minus_without": difference,
                "threshold": major_score_diff,
            }
            major_differences.append(item)
            reasons.append(
                f"major score difference in {output_name}: {left:g} vs {right:g} "
                f"(absolute difference {abs(difference):g} >= {major_score_diff:g})"
            )

    binary_specs = [
        ("unnecessary_reask", "unnecessary_reask"),
        ("unsupported_psychological_inference", "unsupported_psychological_inference"),
    ]
    for output_name, audit_key in binary_specs:
        left = (audit_by_condition.get("without_acoustic", {}) or {}).get(audit_key)
        right = (audit_by_condition.get("with_acoustic", {}) or {}).get(audit_key)
        if left in (0, 1) and right in (0, 1) and left != right:
            item = {"metric": output_name, "without_acoustic": left, "with_acoustic": right}
            binary_disagreements.append(item)
            reasons.append(
                f"binary judgement differs for {output_name}: {left} vs {right}"
            )

    invalid_pair_runs = [
        i for i, run in enumerate(pair_runs, 1)
        if run.get("quote_validation_passed") is False
    ]
    if invalid_pair_runs:
        reasons.append(
            "pairwise winner quote failed local validation in run(s): "
            + ", ".join(map(str, invalid_pair_runs))
        )

    if summary_pref in {"inconsistent_or_order_sensitive", "insufficient_valid_votes"}:
        reasons.append(f"pairwise preference is {summary_pref}")
    elif summary_pref.endswith("_nonunanimous"):
        reasons.append(f"pairwise preference is non-unanimous: {summary_pref}")

    if not transcript_available:
        reasons.append("source transcript unavailable")
    if not capsule_available:
        reasons.append("handover capsule unavailable")

    # Preserve order but deduplicate repeated reasons, especially when the same
    # issue appears both in the validator flags and the explicit re-check.
    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "needs_review": bool(unique_reasons),
        "reasons": unique_reasons,
        "detected_contradiction": detected_contradiction,
        "contradiction_counts": contradiction_counts,
        "validated_contradiction_counts": validated_contradiction_counts,
        "missing_required_quote": missing_required_quote,
        "major_score_differences": major_differences,
        "binary_judgement_disagreements": binary_disagreements,
        "invalid_pairwise_quote_runs": invalid_pair_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("paired_results_for_review.csv"))
    parser.add_argument("--dataset", type=Path, default=Path("emotional_conversations.json"))
    parser.add_argument("--capsule-dir", type=Path, default=Path("handover_capsules"))
    parser.add_argument("--gold-dir", type=Path, default=Path("gold_state"))
    parser.add_argument("--previous-results", type=Path, default=None, help="Optional prior llm_judge_results.csv; used only with --audit-csv to select flagged rows")
    parser.add_argument("--audit-csv", type=Path, default=None, help="Optional targeted audit CSV; only these conversation_ids will be re-evaluated")
    parser.add_argument("--output", type=Path, default=Path("repaired_llm_judge_results"))
    parser.add_argument("--model", default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: evaluate only first N selected rows")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-transcript-chars", type=int, default=18000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--pairwise-runs", type=int, default=4, help="Use 4 for balanced two-A-first/two-B-first comparisons")
    parser.add_argument("--major-score-diff", type=float, default=2.0, help="Flag numeric score differences with absolute magnitude >= this threshold")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency. Run: pip install -r requirements_judge.txt") from exc

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY is not set. Export it before running this script.")
    if args.pairwise_runs < 2:
        raise SystemExit("Use at least two pairwise runs. Recommended: --pairwise-runs 4.")
    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")

    input_rows = load_csv(args.input)
    if args.audit_csv:
        if not args.audit_csv.exists():
            raise SystemExit(f"Audit CSV not found: {args.audit_csv}")
        audit_ids = {str(r.get("conversation_id", "")).strip() for r in load_csv(args.audit_csv)}
        audit_ids.discard("")
        input_rows = [r for r in input_rows if str(r.get("conversation_id", "")).strip() in audit_ids]
        print(f"Selected {len(input_rows)} rows from targeted audit file ({len(audit_ids)} IDs).")
    if args.limit > 0:
        input_rows = input_rows[:args.limit]
    if not input_rows:
        raise SystemExit("No input rows remain after filtering")

    conversations = load_conversations(args.dataset)
    if not conversations:
        print(f"WARNING: no transcript records found in {args.dataset}", file=sys.stderr)
    rng = random.Random(args.seed)
    client = OpenAI()
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "repaired_llm_judge_results.csv"
    summary_path = args.output / "repaired_llm_judge_summary.json"
    summary_md = args.output / "repaired_llm_judge_summary.md"
    previous = load_csv(results_path) if args.resume and results_path.exists() else []
    previous_by_id = {str(r.get("conversation_id")): r for r in previous}
    results: list[dict[str, Any]] = []

    for i, row in enumerate(input_rows, 1):
        cid = str(row.get("conversation_id", "")).strip()
        if not cid:
            continue
        if args.resume and cid in previous_by_id and previous_by_id[cid].get("status") == "complete":
            results.append(previous_by_id[cid])
            print(f"[{i}/{len(input_rows)}] {cid}: resumed")
            continue
        response_by_condition = {
            "without_acoustic": str(row.get("response_without_acoustic", "") or "").strip(),
            "with_acoustic": str(row.get("response_with_acoustic", "") or "").strip(),
        }
        if not all(response_by_condition.values()):
            results.append({"conversation_id": cid, "status": "error", "error": "Missing response text"})
            write_csv(results_path, results, ["conversation_id", "status", "error"])
            continue
        try:
            rec = conversations.get(cid)
            cap_path = args.capsule_dir / f"{cid}.json"
            gold_path = args.gold_dir / f"{cid}.json"
            capsule = load_json(cap_path) if cap_path.exists() else None
            gold = load_json(gold_path) if gold_path.exists() else None
            source, source_text, acoustic_available = build_context(
                cid, rec, capsule, gold, str(row.get("customer_message", "") or ""), args.max_transcript_chars
            )
            print(f"[{i}/{len(input_rows)}] {cid}: independent response audits")
            audit_by_condition: dict[str, dict[str, Any]] = {}
            for condition in ("without_acoustic", "with_acoustic"):
                audit_by_condition[condition] = evaluate_one(
                    client, args.model, cid, response_by_condition[condition], condition,
                    source, source_text, acoustic_available, 2400, args.retries
                )

            # Exactly half of pairwise runs place each condition in A (balanced order).
            orders = [("without_acoustic", "with_acoustic")] * (args.pairwise_runs // 2)
            orders += [("with_acoustic", "without_acoustic")] * (args.pairwise_runs // 2)
            if args.pairwise_runs % 2:
                orders.append(rng.choice([("without_acoustic", "with_acoustic"), ("with_acoustic", "without_acoustic")]))
            rng.shuffle(orders)
            pair_runs = []
            votes = []
            for run_idx, (cond_a, cond_b) in enumerate(orders, 1):
                out = evaluate_pair(
                    client, args.model, source,
                    response_by_condition[cond_a], response_by_condition[cond_b], run_idx, args.retries
                )
                condition_vote = to_condition(out["preference"], cond_a) if out.get("quote_validation_passed") else "invalid"
                out["condition_for_A"] = cond_a
                out["condition_for_B"] = cond_b
                out["condition_preference_if_valid"] = condition_vote
                pair_runs.append(out)
                if condition_vote != "invalid":
                    votes.append(condition_vote)
            summary_pref = preference_summary(votes)
            review_decision = build_review_decision(
                audit_by_condition,
                pair_runs,
                summary_pref,
                transcript_available=bool(rec and rec.get("turns")),
                capsule_available=(capsule is not None),
                major_score_diff=args.major_score_diff,
            )

            result: dict[str, Any] = {
                "conversation_id": cid,
                "split": row.get("split", ""),
                "status": "complete",
                "model": args.model,
                "source_transcript_available": str(bool(rec and rec.get("turns"))).lower(),
                "capsule_available": str(capsule is not None).lower(),
                "acoustic_metadata_available": str(acoustic_available).lower(),
                "factual_score_without_acoustic": audit_by_condition["without_acoustic"].get("factual_consistency_score"),
                "factual_score_with_acoustic": audit_by_condition["with_acoustic"].get("factual_consistency_score"),
                "continuity_without_acoustic": audit_by_condition["without_acoustic"].get("continuity_score"),
                "continuity_with_acoustic": audit_by_condition["with_acoustic"].get("continuity_score"),
                "unnecessary_reask_without_acoustic": audit_by_condition["without_acoustic"].get("unnecessary_reask"),
                "unnecessary_reask_with_acoustic": audit_by_condition["with_acoustic"].get("unnecessary_reask"),
                "adaptation_without_acoustic": audit_by_condition["without_acoustic"].get("adaptation_score"),
                "adaptation_with_acoustic": audit_by_condition["with_acoustic"].get("adaptation_score"),
                "unsupported_psych_inference_without_acoustic": audit_by_condition["without_acoustic"].get("unsupported_psychological_inference"),
                "unsupported_psych_inference_with_acoustic": audit_by_condition["with_acoustic"].get("unsupported_psychological_inference"),
                "factual_audit_without_acoustic_json": json.dumps(audit_by_condition["without_acoustic"].get("factual_claims", []), ensure_ascii=False),
                "factual_audit_with_acoustic_json": json.dumps(audit_by_condition["with_acoustic"].get("factual_claims", []), ensure_ascii=False),
                "score_reasoning_without_acoustic_json": json.dumps({k: v for k, v in audit_by_condition["without_acoustic"].items() if k != "factual_claims"}, ensure_ascii=False),
                "score_reasoning_with_acoustic_json": json.dumps({k: v for k, v in audit_by_condition["with_acoustic"].items() if k != "factual_claims"}, ensure_ascii=False),
                "quote_validation_without_acoustic_json": json.dumps(audit_by_condition["without_acoustic"].get("quote_validation", {}), ensure_ascii=False),
                "quote_validation_with_acoustic_json": json.dumps(audit_by_condition["with_acoustic"].get("quote_validation", {}), ensure_ascii=False),
                "pairwise_runs_json": json.dumps(pair_runs, ensure_ascii=False),
                "pairwise_valid_vote_count": len(votes),
                "pairwise_with_acoustic_votes": votes.count("with_acoustic"),
                "pairwise_without_acoustic_votes": votes.count("without_acoustic"),
                "pairwise_tie_votes": votes.count("tie"),
                "pairwise_preference_summary": summary_pref,
                "needs_review": str(review_decision["needs_review"]).lower(),
                "review_reasons_json": json.dumps(review_decision["reasons"], ensure_ascii=False),
                "detected_contradiction": str(review_decision["detected_contradiction"]).lower(),
                "detected_contradiction_count_without_acoustic": review_decision["contradiction_counts"]["without_acoustic"],
                "detected_contradiction_count_with_acoustic": review_decision["contradiction_counts"]["with_acoustic"],
                "validated_contradiction_count_without_acoustic": review_decision["validated_contradiction_counts"]["without_acoustic"],
                "validated_contradiction_count_with_acoustic": review_decision["validated_contradiction_counts"]["with_acoustic"],
                "missing_required_quote": str(review_decision["missing_required_quote"]).lower(),
                "major_score_differences_json": json.dumps(review_decision["major_score_differences"], ensure_ascii=False),
                "binary_judgement_disagreements_json": json.dumps(review_decision["binary_judgement_disagreements"], ensure_ascii=False),
                "invalid_pairwise_quote_runs_json": json.dumps(review_decision["invalid_pairwise_quote_runs"], ensure_ascii=False),
                "error": "",
            }
        except Exception as exc:
            print(f"  ERROR for {cid}: {type(exc).__name__}: {exc}", file=sys.stderr)
            result = {"conversation_id": cid, "split": row.get("split", ""), "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        results = [r for r in results if r.get("conversation_id") != cid]
        results.append(result)
        columns = list(dict.fromkeys(k for r in results for k in r.keys()))
        write_csv(results_path, results, columns)

    # Align output to the selected input order.
    by_id = {str(r.get("conversation_id")): r for r in results}
    ordered = [by_id[str(r.get("conversation_id"))] for r in input_rows if str(r.get("conversation_id")) in by_id]
    columns = list(dict.fromkeys(k for r in ordered for k in r.keys()))
    write_csv(results_path, ordered, columns)

    complete = [r for r in ordered if r.get("status") == "complete"]
    def avg(col: str) -> float | None:
        vals = []
        for r in complete:
            try:
                if r.get(col) not in (None, ""):
                    vals.append(float(r[col]))
            except (ValueError, TypeError):
                pass
        return round(mean(vals), 4) if vals else None
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "judge_model": args.model,
        "total_selected": len(input_rows),
        "completed": len(complete),
        "scope": "targeted audit subset" if args.audit_csv else "all selected input rows",
        "averages": {
            "factual_score_without_acoustic": avg("factual_score_without_acoustic"),
            "factual_score_with_acoustic": avg("factual_score_with_acoustic"),
            "continuity_without_acoustic": avg("continuity_without_acoustic"),
            "continuity_with_acoustic": avg("continuity_with_acoustic"),
            "adaptation_without_acoustic": avg("adaptation_without_acoustic"),
            "adaptation_with_acoustic": avg("adaptation_with_acoustic"),
        },
        "pairwise_preference_summary_counts": {},
        "needs_review_count": sum(1 for r in complete if r.get("needs_review") == "true"),
        "method_notes": [
            "Candidate responses are audited independently, not scored side-by-side.",
            "Claim quotations are verified locally against the exact response text.",
            "Supported/contradicted claims require a source quote that is locally found in the source context.",
            "Source quote occurrence is not semantic entailment; inspect flagged claims.",
            "Pairwise comparisons use balanced order and non-unanimous preferences remain uncertain.",
            "Scores remain automated estimates, not human-validated truth.",
        ],
    }
    pref_counts: dict[str, int] = {}
    for r in complete:
        k = str(r.get("pairwise_preference_summary", "unknown"))
        pref_counts[k] = pref_counts.get(k, 0) + 1
    summary["pairwise_preference_summary_counts"] = pref_counts
    write_json(summary_path, summary)
    md = [
        "# Repaired RECAP-Voice LLM judge audit", "",
        f"- Judge model: `{args.model}`",
        f"- Scope: {summary['scope']}",
        f"- Completed: {summary['completed']} / {summary['total_selected']}",
        f"- Cases flagged for review: {summary['needs_review_count']}", "",
        "## Aggregate scores (descriptive only)", "",
        "| Metric | Without acoustic | With acoustic |", "|---|---:|---:|",
    ]
    for key, label in [
        ("factual_score_without_acoustic", "Factual consistency"),
        ("continuity_without_acoustic", "Continuity"),
        ("adaptation_without_acoustic", "Adaptation"),
    ]:
        base = key
        with_key = base.replace("without_acoustic", "with_acoustic")
        md.append(f"| {label} | {summary['averages'].get(base)} | {summary['averages'].get(with_key)} |")
    md += ["", "## Pairwise stability", "", "| Outcome | Count |", "|---|---:|"]
    for k, v in pref_counts.items():
        md.append(f"| {k} | {v} |")
    md += ["", "## Guardrails", ""] + [f"- {x}" for x in summary["method_notes"]] + [""]
    summary_md.write_text("\n".join(md), encoding="utf-8")
    print("\nCompleted.")
    print(f"CSV: {results_path}")
    print(f"Summary: {summary_md}")
    print(f"Machine-readable summary: {summary_path}")
    print("Note: quote checks validate exact text presence, not whether the quote logically entails the claim.")


if __name__ == "__main__":
    main()
