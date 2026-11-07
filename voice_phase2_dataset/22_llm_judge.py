#!/usr/bin/env python3
"""Blind LLM-judge evaluation for paired RECAP-Voice responses.

Reads paired_results_for_review.csv, randomizes which condition is shown as A/B,
provides shared source evidence to the judge, and optionally repeats the pairwise
preference decision with response order reversed. Scores are mapped back to the
original conditions only after the judge returns its labels.

The judge is an automated evaluator, not a substitute for human ground truth.
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

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency. Run: pip install -r requirements.txt") from exc

ACOUSTIC_KEYS = {
    "interaction_context",
    "acoustic_observations",
    "acoustic_interaction_features",
    "audio_features",
}

SCORING_COLUMNS = [
    "continuity_score_without_acoustic_1to5",
    "continuity_score_with_acoustic_1to5",
    "unnecessary_reask_without_acoustic_0or1",
    "unnecessary_reask_with_acoustic_0or1",
    "appropriate_adaptation_without_acoustic_0to2",
    "appropriate_adaptation_with_acoustic_0to2",
    "unsupported_psychological_inference_without_acoustic_0or1",
    "unsupported_psychological_inference_with_acoustic_0or1",
    "factual_consistency_without_acoustic_1to5",
    "factual_consistency_with_acoustic_1to5",
]

RESULT_COLUMNS = [
    "conversation_id", "split", "status", "acoustic_reference_available",
    *SCORING_COLUMNS,
    "continuity_reason_without_acoustic", "continuity_reason_with_acoustic",
    "reask_reason_without_acoustic", "reask_reason_with_acoustic",
    "adaptation_reason_without_acoustic", "adaptation_reason_with_acoustic",
    "psychological_inference_reason_without_acoustic", "psychological_inference_reason_with_acoustic",
    "factual_reason_without_acoustic", "factual_reason_with_acoustic",
    "unsupported_claims_without_acoustic_json", "unsupported_claims_with_acoustic_json",
    "judge_preference_run1", "judge_preference_run2", "preference_stability",
    "pairwise_preference_summary", "preference_reason_run1", "preference_reason_run2",
    "run1_condition_for_A", "error",
]

JUDGE_INSTRUCTIONS = """
You are an independent evaluator of paired customer-support handoff responses.
Your job is to judge quality against the provided source conversation, explicit
reference state, evidence, and (when supplied) observable interaction metadata.

IMPORTANT RULES
- You do not know which response was generated with acoustic context. Do not try
  to guess the experimental condition and do not reward a response merely because
  it is longer, more polished, more confident, or more empathetic.
- Treat the supplied transcript and structured state as the available evidence.
  Do not assume an action was completed unless the transcript/state supports it.
- A handoff state saying "resolved" does not automatically justify claiming that
  a specific action (such as unlocking an account or updating a phone number) was
  completed unless that specific action is supported.
- Do not infer emotion, honesty, intent, confusion, or mental state from pauses,
  silence, speaking rate, talk ratio, overlap, or other acoustic/turn-taking cues.
  Observable interaction metadata may only support practical choices such as
  concise wording, clarity, or pacing when that connection is reasonable.
- A question is an unnecessary re-ask only if the answer is already available in
  the provided source context. Requests for missing security verification or
  genuinely missing details are not automatically unnecessary.
- Cite exact source snippets or structured facts when identifying unsupported
  claims. If evidence is insufficient, say it is insufficient rather than guessing.
- Score each response independently before making the pairwise preference choice.
""".strip()

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "response_a": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "continuity_score": {"type": "integer", "minimum": 1, "maximum": 5},
                "continuity_reason": {"type": "string"},
                "unnecessary_reask": {"type": "integer", "enum": [0, 1]},
                "reask_reason": {"type": "string"},
                "adaptation_score": {"type": ["integer", "null"], "minimum": 0, "maximum": 2},
                "adaptation_reason": {"type": "string"},
                "unsupported_psychological_inference": {"type": "integer", "enum": [0, 1]},
                "psychological_inference_reason": {"type": "string"},
                "factual_consistency_score": {"type": "integer", "minimum": 1, "maximum": 5},
                "factual_reason": {"type": "string"},
                "unsupported_claims": {
                    "type": "array",
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "claim": {"type": "string"},
                            "why_unsupported_or_contradicted": {"type": "string"},
                            "source_evidence": {"type": "string"},
                        },
                        "required": ["claim", "why_unsupported_or_contradicted", "source_evidence"],
                    },
                },
            },
            "required": [
                "continuity_score", "continuity_reason", "unnecessary_reask", "reask_reason",
                "adaptation_score", "adaptation_reason", "unsupported_psychological_inference",
                "psychological_inference_reason", "factual_consistency_score", "factual_reason",
                "unsupported_claims",
            ],
        },
        "response_b": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "continuity_score": {"type": "integer", "minimum": 1, "maximum": 5},
                "continuity_reason": {"type": "string"},
                "unnecessary_reask": {"type": "integer", "enum": [0, 1]},
                "reask_reason": {"type": "string"},
                "adaptation_score": {"type": ["integer", "null"], "minimum": 0, "maximum": 2},
                "adaptation_reason": {"type": "string"},
                "unsupported_psychological_inference": {"type": "integer", "enum": [0, 1]},
                "psychological_inference_reason": {"type": "string"},
                "factual_consistency_score": {"type": "integer", "minimum": 1, "maximum": 5},
                "factual_reason": {"type": "string"},
                "unsupported_claims": {
                    "type": "array",
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "claim": {"type": "string"},
                            "why_unsupported_or_contradicted": {"type": "string"},
                            "source_evidence": {"type": "string"},
                        },
                        "required": ["claim", "why_unsupported_or_contradicted", "source_evidence"],
                    },
                },
            },
            "required": [
                "continuity_score", "continuity_reason", "unnecessary_reask", "reask_reason",
                "adaptation_score", "adaptation_reason", "unsupported_psychological_inference",
                "psychological_inference_reason", "factual_consistency_score", "factual_reason",
                "unsupported_claims",
            ],
        },
        "pairwise_preference": {"type": "string", "enum": ["A_BETTER", "B_BETTER", "NO_MEANINGFUL_DIFFERENCE"]},
        "pairwise_reason": {"type": "string"},
    },
    "required": ["response_a", "response_b", "pairwise_preference", "pairwise_reason"],
}

PREFERENCE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "pairwise_preference": {"type": "string", "enum": ["A_BETTER", "B_BETTER", "NO_MEANINGFUL_DIFFERENCE"]},
        "pairwise_reason": {"type": "string"},
    },
    "required": ["pairwise_preference", "pairwise_reason"],
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
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_conversations(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    raw = load_json(path)
    records = raw if isinstance(raw, list) else raw.get("conversations", raw.get("data", []))
    if isinstance(raw, dict) and not records and all(isinstance(v, dict) for v in raw.values()):
        records = list(raw.values())
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if isinstance(record, dict):
            cid = str(record.get("conversation_id") or record.get("id") or "")
            if cid:
                result[cid] = record
    return result


def remove_sensitive_voice_annotations(turn: dict[str, Any]) -> dict[str, Any]:
    """Keep transcript content, but exclude synthetic emotion/speaker voice labels."""
    return {
        key: turn.get(key)
        for key in ("turn_id", "role", "dialogue_act", "text")
        if key in turn
    }


def has_nonempty_acoustic_metadata(obj: Any) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_norm = str(key).strip().lower()
            if key_norm in ACOUSTIC_KEYS and value not in (None, {}, [], ""):
                return True
            if has_nonempty_acoustic_metadata(value):
                return True
    elif isinstance(obj, list):
        return any(has_nonempty_acoustic_metadata(x) for x in obj)
    return False


def build_reference_context(
    conversation_id: str,
    dataset_record: dict[str, Any] | None,
    capsule: dict[str, Any] | None,
    gold_state: dict[str, Any] | None,
    customer_message: str,
    max_transcript_chars: int,
) -> tuple[dict[str, Any], bool]:
    conversation_ref: dict[str, Any] = {}
    if dataset_record:
        conversation_ref = {
            "conversation_id": conversation_id,
            "scenario": dataset_record.get("scenario"),
            "scenario_family": dataset_record.get("scenario_family"),
            "intent": dataset_record.get("intent"),
            "resolution_status": dataset_record.get("resolution_status"),
            "escalated": dataset_record.get("escalated"),
            "entities": dataset_record.get("entities"),
            "turns": [remove_sensitive_voice_annotations(t) for t in dataset_record.get("turns", []) if isinstance(t, dict)],
        }
        transcript_str = json.dumps(conversation_ref.get("turns", []), ensure_ascii=False)
        if len(transcript_str) > max_transcript_chars:
            conversation_ref["turns"] = []
            conversation_ref["transcript_truncated"] = True
            conversation_ref["transcript_excerpt"] = transcript_str[:max_transcript_chars]
    reference: dict[str, Any] = {
        "conversation_reference": conversation_ref or None,
        "customer_follow_up_message": customer_message,
        "instructions_for_evidence_use": (
            "Treat transcript and explicit structured reference labels as source material. "
            "A general resolved status does not prove a specific action is completed. "
            "If facts conflict or are missing, note uncertainty. Voice/emotion annotations "
            "are intentionally omitted."
        ),
    }
    if gold_state:
        reference["gold_state_reference"] = gold_state
    if capsule:
        reference["handover_capsule_reference"] = capsule
    acoustic_available = bool(capsule and has_nonempty_acoustic_metadata(capsule))
    return reference, acoustic_available


def parse_response_json(output_text: str) -> dict[str, Any]:
    try:
        value = json.loads(output_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", output_text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Judge did not return JSON: {output_text[:500]}")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Judge JSON output is not an object")
    return value


def call_judge(
    client: OpenAI,
    model: str,
    schema_name: str,
    schema: dict[str, Any],
    instructions: str,
    payload: dict[str, Any],
    max_output_tokens: int,
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=json.dumps(payload, ensure_ascii=False),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
                max_output_tokens=max_output_tokens,
            )
            text_output = (getattr(response, "output_text", "") or "").strip()
            if not text_output:
                raise ValueError("Empty output_text from judge API")
            parsed = parse_response_json(text_output)
            return parsed
        except Exception as exc:  # retry transient API / malformed-response failures
            last_error = exc
            if attempt >= retries:
                break
            delay = min(2 ** attempt, 20) + random.random()
            print(f"  API/judge attempt {attempt + 1} failed ({type(exc).__name__}); retrying in {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def preference_to_condition(preference: str, condition_for_a: str) -> str:
    if preference == "NO_MEANINGFUL_DIFFERENCE":
        return "tie"
    if preference == "A_BETTER":
        return condition_for_a
    if preference == "B_BETTER":
        return "with_acoustic" if condition_for_a == "without_acoustic" else "without_acoustic"
    return "unknown"


def as_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def average_and_delta(
    rows: list[dict[str, Any]],
    col_without: str,
    col_with: str,
    favorable_lower: bool = False,
) -> dict[str, Any]:
    pairs = []
    values_without, values_with = [], []
    for r in rows:
        a, b = as_float(r.get(col_without)), as_float(r.get(col_with))
        if a is not None:
            values_without.append(a)
        if b is not None:
            values_with.append(b)
        if a is not None and b is not None:
            pairs.append(b - a)
    out = {
        "mean_without_acoustic": round(mean(values_without), 4) if values_without else None,
        "mean_with_acoustic": round(mean(values_with), 4) if values_with else None,
        "mean_delta_with_minus_without": round(mean(pairs), 4) if pairs else None,
        "paired_n": len(pairs),
        "direction_that_favors_acoustic_condition": "negative" if favorable_lower else "positive",
    }
    if len(pairs) >= 2:
        rng = random.Random(73021)
        boot_means = []
        for _ in range(2000):
            sample = [rng.choice(pairs) for __ in range(len(pairs))]
            boot_means.append(mean(sample))
        boot_means.sort()
        out["bootstrap_95pct_ci_delta"] = [round(boot_means[int(0.025 * len(boot_means))], 4), round(boot_means[min(int(0.975 * len(boot_means)), len(boot_means)-1)], 4)]
    else:
        out["bootstrap_95pct_ci_delta"] = None
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [r for r in rows if r.get("status") == "complete"]
    score_metrics = {
        "continuity_score_1to5": ("continuity_score_without_acoustic_1to5", "continuity_score_with_acoustic_1to5", False),
        "unnecessary_reask_0or1": ("unnecessary_reask_without_acoustic_0or1", "unnecessary_reask_with_acoustic_0or1", True),
        "appropriate_adaptation_0to2": ("appropriate_adaptation_without_acoustic_0to2", "appropriate_adaptation_with_acoustic_0to2", False),
        "unsupported_psychological_inference_0or1": ("unsupported_psychological_inference_without_acoustic_0or1", "unsupported_psychological_inference_with_acoustic_0or1", True),
        "factual_consistency_score_1to5": ("factual_consistency_without_acoustic_1to5", "factual_consistency_with_acoustic_1to5", False),
    }
    metrics = {}
    for out_name, (col_without, col_with, lower) in score_metrics.items():
        metrics[out_name] = average_and_delta(complete, col_without, col_with, favorable_lower=lower)
    pref_counts = {"with_acoustic_win_stable": 0, "without_acoustic_win_stable": 0, "tie_stable": 0, "order_sensitive_or_missing": 0, "reverse_check_not_run": 0}
    for r in complete:
        p1, p2 = r.get("judge_preference_run1"), r.get("judge_preference_run2")
        if p2 == "not_run":
            pref_counts["reverse_check_not_run"] += 1
        elif p1 == p2 == "with_acoustic":
            pref_counts["with_acoustic_win_stable"] += 1
        elif p1 == p2 == "without_acoustic":
            pref_counts["without_acoustic_win_stable"] += 1
        elif p1 == p2 == "tie":
            pref_counts["tie_stable"] += 1
        else:
            pref_counts["order_sensitive_or_missing"] += 1
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "total_rows": len(rows),
        "completed_rows": len(complete),
        "automated_judge_only": True,
        "metrics": metrics,
        "pairwise_preference_after_order_reversal": pref_counts,
        "interpretation_note": (
            "LLM-judge scores are automated estimates, not human-validated labels. "
            "Positive delta favors acoustic context for continuity/adaptation/factual consistency; "
            "negative delta favors acoustic context for re-asks and unsupported psychological inference. "
            "Order-sensitive preferences should not be counted as reliable wins. Bootstrap intervals "
            "describe sampling variability, not judge validity or causal certainty."
        ),
    }


def write_summary_markdown(path: Path, summary: dict[str, Any], model: str) -> None:
    lines = [
        "# RECAP-Voice LLM judge summary", "",
        f"- Judge model: `{model}`",
        f"- Completed conversations: {summary['completed_rows']} / {summary['total_rows']}",
        "- Evaluation type: blind automated LLM judge with response-order reversal", "",
        "## Paired metric results", "",
        "| Metric | Without acoustic mean | With acoustic mean | With − without | 95% bootstrap CI for delta |",
        "|---|---:|---:|---:|---|",
    ]
    for name, item in summary["metrics"].items():
        ci = item.get("bootstrap_95pct_ci_delta")
        ci_text = f"[{ci[0]}, {ci[1]}]" if ci else "n/a"
        fmt = lambda x: "n/a" if x is None else str(x)
        lines.append(f"| {name} | {fmt(item['mean_without_acoustic'])} | {fmt(item['mean_with_acoustic'])} | {fmt(item['mean_delta_with_minus_without'])} | {ci_text} |")
    lines.extend(["", "## Pairwise preference stability", "", "| Outcome | Count |", "|---|---:|"])
    for k, v in summary["pairwise_preference_after_order_reversal"].items():
        lines.append(f"| {k} | {v} |")
    lines.extend(["", "## Interpretation guardrails", "", summary["interpretation_note"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("paired_results_for_review.csv"))
    parser.add_argument("--dataset", type=Path, default=Path("emotional_conversations.json"))
    parser.add_argument("--capsule-dir", type=Path, default=Path("handover_capsules"), help="Optional directory containing <conversation_id>.json handoff capsules")
    parser.add_argument("--gold-dir", type=Path, default=Path("gold_state"), help="Optional directory containing <conversation_id>.json reference state")
    parser.add_argument("--output", type=Path, default=Path("llm_judge_results"))
    parser.add_argument("--model", default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N rows (use 2-3 as a smoke test)")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--resume", action="store_true", help="Skip rows already marked complete in the output CSV")
    parser.add_argument("--no-reverse-check", action="store_true", help="Disable second call with A/B order reversed (cheaper, less robust)")
    parser.add_argument("--max-transcript-chars", type=int, default=18000)
    parser.add_argument("--max-output-tokens", type=int, default=2200)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY is not set. Export it before running the script.")
    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")

    input_rows = load_csv(args.input)
    if args.limit > 0:
        input_rows = input_rows[:args.limit]
    if not input_rows:
        raise SystemExit("Input CSV has no rows")

    conversations = load_conversations(args.dataset)
    if not conversations:
        print(f"WARNING: no conversation records loaded from {args.dataset}; prompts will have no original transcript unless capsule files exist.", file=sys.stderr)
    if not args.capsule_dir.exists():
        print(f"WARNING: capsule directory not found ({args.capsule_dir}); using dataset transcript and optional gold state only. Interaction-adaptation scores will be null unless acoustic metadata is present in a capsule.", file=sys.stderr)
    if not args.gold_dir.exists():
        print(f"NOTE: gold-state directory not found ({args.gold_dir}); continuing with the dataset and optional capsules.", file=sys.stderr)

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "llm_judge_results.csv"
    summary_json_path = args.output / "llm_judge_summary.json"
    summary_md_path = args.output / "llm_judge_summary.md"
    private_key_path = args.output / "private_blinding_key.csv"

    previous_rows = load_csv(results_path) if args.resume and results_path.exists() else []
    previous_by_id = {str(r.get("conversation_id")): r for r in previous_rows}
    key_rows = load_csv(private_key_path) if private_key_path.exists() else []
    key_map = {str(r["conversation_id"]): r["condition_for_A_run1"] for r in key_rows if r.get("conversation_id") and r.get("condition_for_A_run1")}
    rng = random.Random(args.seed)
    for row in input_rows:
        cid = str(row.get("conversation_id", ""))
        if cid and cid not in key_map:
            key_map[cid] = rng.choice(["without_acoustic", "with_acoustic"])
    write_csv(private_key_path, [{"conversation_id": cid, "condition_for_A_run1": cond} for cid, cond in key_map.items()], ["conversation_id", "condition_for_A_run1"])

    client = OpenAI()
    results = [previous_by_id[cid] for cid in previous_by_id if cid not in {str(r.get("conversation_id", "")) for r in input_rows}]
    # Keep input order for selected rows, reusing completed results during resume.
    for index, row in enumerate(input_rows, 1):
        cid = str(row.get("conversation_id", "")).strip()
        if not cid:
            print(f"WARNING: skipping row {index}: empty conversation_id", file=sys.stderr)
            continue
        existing = previous_by_id.get(cid)
        if args.resume and existing and existing.get("status") == "complete":
            results = [r for r in results if r.get("conversation_id") != cid]
            results.append(existing)
            print(f"[{index}/{len(input_rows)}] {cid}: already complete; skipping")
            continue

        message = str(row.get("customer_message", "") or "")
        response_without = str(row.get("response_without_acoustic", "") or "")
        response_with = str(row.get("response_with_acoustic", "") or "")
        if not response_without.strip() or not response_with.strip():
            result = {"conversation_id": cid, "split": row.get("split", ""), "status": "error", "error": "Missing one or both response texts"}
            results = [r for r in results if r.get("conversation_id") != cid] + [result]
            write_csv(results_path, results, RESULT_COLUMNS)
            continue

        dataset_record = conversations.get(cid)
        capsule_path = args.capsule_dir / f"{cid}.json"
        gold_path = args.gold_dir / f"{cid}.json"
        capsule = load_json(capsule_path) if capsule_path.exists() else None
        gold_state = load_json(gold_path) if gold_path.exists() else None
        reference, acoustic_available = build_reference_context(
            cid, dataset_record, capsule, gold_state, message, args.max_transcript_chars
        )
        condition_for_a_run1 = key_map[cid]
        condition_for_b_run1 = "with_acoustic" if condition_for_a_run1 == "without_acoustic" else "without_acoustic"
        response_by_condition = {
            "without_acoustic": response_without,
            "with_acoustic": response_with,
        }
        response_a = response_by_condition[condition_for_a_run1]
        response_b = response_by_condition[condition_for_b_run1]

        print(f"[{index}/{len(input_rows)}] {cid}: judge pass 1/" + ("2" if not args.no_reverse_check else "1"))
        try:
            first = call_judge(
                client, args.model, "recap_pairwise_judge", SCHEMA, JUDGE_INSTRUCTIONS,
                {
                    "task": "Score both responses independently and choose the better response if there is a meaningful quality difference.",
                    "shared_source_context": reference,
                    "scoring_rubric": {
                        "continuity_score": "1-5; 1 restarts/loses history, 3 adequate, 5 seamlessly preserves useful progress.",
                        "unnecessary_reask": "0=no unnecessary re-ask; 1=asks for information already present in source context.",
                        "adaptation_score": "0-2; 0 poor/inappropriate, 1 acceptable/plausible, 2 clearly helpful use of observable interaction metadata. If no acoustic observation metadata is included in source context, use null for this score.",
                        "unsupported_psychological_inference": "0=no unsupported inference; 1=unsupported inference about emotion, honesty, intent, confusion or mental state.",
                        "factual_consistency_score": "1-5; 1 major unsupported/contradictory claims, 3 mixed/uncertain, 5 claims are supported and calibrated.",
                    },
                    "response_A": response_a,
                    "response_B": response_b,
                },
                args.max_output_tokens, args.retries,
            )
            p1_condition = preference_to_condition(first["pairwise_preference"], condition_for_a_run1)
            # Translate response-by-label scores back to actual experimental condition.
            score_by_condition = {
                condition_for_a_run1: first["response_a"],
                condition_for_b_run1: first["response_b"],
            }
            second = None
            p2_condition = "not_run"
            preference_reason_2 = ""
            if not args.no_reverse_check:
                print(f"[{index}/{len(input_rows)}] {cid}: judge pass 2/2 (order reversed)")
                second = call_judge(
                    client, args.model, "recap_preference_reversed", PREFERENCE_SCHEMA, JUDGE_INSTRUCTIONS,
                    {
                        "task": "Make only a blinded pairwise quality decision. The response order is reversed from the previous evaluation. Judge the substantive customer-service quality, not position or writing polish.",
                        "shared_source_context": reference,
                        "response_A": response_b,
                        "response_B": response_a,
                        "decision_options": ["A_BETTER", "B_BETTER", "NO_MEANINGFUL_DIFFERENCE"],
                    },
                    500, args.retries,
                )
                # In reverse pass, A is original B and B is original A.
                p2_condition = preference_to_condition(second["pairwise_preference"], condition_for_b_run1)
                preference_reason_2 = second.get("pairwise_reason", "")
            if args.no_reverse_check:
                stability = "not_checked"
                preference_summary = p1_condition
            elif p1_condition == p2_condition:
                stability = "stable"
                preference_summary = p1_condition
            else:
                stability = "order_sensitive_or_inconsistent"
                preference_summary = "order_sensitive_or_inconsistent"

            without_scores = score_by_condition["without_acoustic"]
            with_scores = score_by_condition["with_acoustic"]
            result: dict[str, Any] = {
                "conversation_id": cid,
                "split": row.get("split", ""),
                "status": "complete",
                "acoustic_reference_available": str(acoustic_available).lower(),
                "continuity_score_without_acoustic_1to5": without_scores["continuity_score"],
                "continuity_score_with_acoustic_1to5": with_scores["continuity_score"],
                "unnecessary_reask_without_acoustic_0or1": without_scores["unnecessary_reask"],
                "unnecessary_reask_with_acoustic_0or1": with_scores["unnecessary_reask"],
                "appropriate_adaptation_without_acoustic_0to2": without_scores["adaptation_score"],
                "appropriate_adaptation_with_acoustic_0to2": with_scores["adaptation_score"],
                "unsupported_psychological_inference_without_acoustic_0or1": without_scores["unsupported_psychological_inference"],
                "unsupported_psychological_inference_with_acoustic_0or1": with_scores["unsupported_psychological_inference"],
                "factual_consistency_without_acoustic_1to5": without_scores["factual_consistency_score"],
                "factual_consistency_with_acoustic_1to5": with_scores["factual_consistency_score"],
                "continuity_reason_without_acoustic": without_scores["continuity_reason"],
                "continuity_reason_with_acoustic": with_scores["continuity_reason"],
                "reask_reason_without_acoustic": without_scores["reask_reason"],
                "reask_reason_with_acoustic": with_scores["reask_reason"],
                "adaptation_reason_without_acoustic": without_scores["adaptation_reason"],
                "adaptation_reason_with_acoustic": with_scores["adaptation_reason"],
                "psychological_inference_reason_without_acoustic": without_scores["psychological_inference_reason"],
                "psychological_inference_reason_with_acoustic": with_scores["psychological_inference_reason"],
                "factual_reason_without_acoustic": without_scores["factual_reason"],
                "factual_reason_with_acoustic": with_scores["factual_reason"],
                "unsupported_claims_without_acoustic_json": json.dumps(without_scores["unsupported_claims"], ensure_ascii=False),
                "unsupported_claims_with_acoustic_json": json.dumps(with_scores["unsupported_claims"], ensure_ascii=False),
                "judge_preference_run1": p1_condition,
                "judge_preference_run2": p2_condition,
                "preference_stability": stability,
                "pairwise_preference_summary": preference_summary,
                "preference_reason_run1": first.get("pairwise_reason", ""),
                "preference_reason_run2": preference_reason_2,
                "run1_condition_for_A": condition_for_a_run1,
                "error": "",
            }
        except Exception as exc:
            result = {
                "conversation_id": cid,
                "split": row.get("split", ""),
                "status": "error",
                "acoustic_reference_available": str(acoustic_available).lower(),
                "run1_condition_for_A": condition_for_a_run1,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  ERROR: {result['error']}", file=sys.stderr)

        results = [r for r in results if r.get("conversation_id") != cid]
        results.append(result)
        write_csv(results_path, results, RESULT_COLUMNS)

    # Keep the result table aligned to input order for a simpler audit trail.
    result_by_id = {str(r.get("conversation_id")): r for r in results}
    ordered_results = [result_by_id[str(r.get("conversation_id"))] for r in input_rows if str(r.get("conversation_id")) in result_by_id]
    write_csv(results_path, ordered_results, RESULT_COLUMNS)
    summary = summarize(ordered_results)
    write_json(summary_json_path, summary)
    write_summary_markdown(summary_md_path, summary, args.model)

    print("\nFinished.")
    print(f"Scored rows: {summary['completed_rows']} / {summary['total_rows']}")
    print(f"Results CSV: {results_path}")
    print(f"Summary Markdown: {summary_md_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"PRIVATE condition mapping (do not share with judge): {private_key_path}")
    if not args.no_reverse_check:
        print("Each pair used two judge calls: a scoring pass and an order-reversed preference check.")
    print("Reminder: inspect unsupported claims and do not treat automated scores as human-validated truth.")


if __name__ == "__main__":
    main()
