#!/usr/bin/env python3
"""
12_evaluate_state.py

Evaluate RECAP-Voice predicted state packets against gold reference packets.

Primary metrics:
    - Intent accuracy
    - Customer-goal token F1
    - Resolution-state accuracy
    - Verification-state accuracy
    - Escalation accuracy
    - Entity/slot micro precision, recall, F1
    - Agent-action micro precision, recall, F1
    - Commitment micro precision, recall, F1
    - Unresolved-item micro precision, recall, F1
    - Next-action accuracy
    - State-field macro accuracy
    - Speaker-resolution status
    - Prediction evidence-support rate

The evaluator intentionally does NOT score the interaction block as a gold
metric because the current synthetic gold-state builder does not contain
numeric timing/prosody ground truth. Those fields remain useful for later
end-to-end analysis.

Expected directories:
    gold_state/
        conv_0001.json
        ...

    recap_predictions/
        conv_0001.json
        ...

Outputs:
    state_evaluation/
        per_conversation.csv
        aggregate_summary.json
        aggregate_summary.csv
        errors.jsonl

Examples:
    python 12_evaluate_state.py
    python 12_evaluate_state.py --conversation-id conv_0001
    python 12_evaluate_state.py --max-conversations 20
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_GOLD_DIR = Path("gold_state")
DEFAULT_PRED_DIR = Path("recap_predictions")
DEFAULT_OUTPUT_DIR = Path("state_evaluation")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^\w\s£€$₹.-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(value: Any) -> set[str]:
    text = normalize_text(value)
    return set(re.findall(r"[a-z0-9₹£€$._-]+", text))


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    # For an empty reference set and empty prediction set, treat the
    # extraction as correct rather than penalizing a conversation simply
    # because the phenomenon is absent.
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def token_f1(reference: Any, prediction: Any) -> tuple[float, float, float]:
    ref = token_set(reference)
    pred = token_set(prediction)

    tp = len(ref & pred)
    fp = len(pred - ref)
    fn = len(ref - pred)

    return f1_from_counts(tp, fp, fn)


def exact_match(reference: Any, prediction: Any) -> bool:
    if isinstance(reference, bool) or isinstance(prediction, bool):
        return reference == prediction

    if reference is None or prediction is None:
        return reference == prediction

    return normalize_text(reference) == normalize_text(prediction)


def value_of_field(field: Any) -> Any:
    if isinstance(field, dict) and "value" in field:
        return field.get("value")
    return field


def get_state(packet: dict[str, Any]) -> dict[str, Any]:
    state = packet.get("state", {})
    return state if isinstance(state, dict) else {}


def normalize_label(value: Any) -> str:
    return normalize_text(value).replace(" ", "_")


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def list_packet_ids(directory: Path) -> set[str]:
    if not directory.exists():
        return set()

    return {
        path.stem
        for path in directory.glob("*.json")
        if path.is_file()
    }


def normalize_entities(entity_block: Any) -> dict[str, str]:
    if not isinstance(entity_block, dict):
        return {}

    result: dict[str, str] = {}

    for key, raw_value in entity_block.items():
        value = value_of_field(raw_value)

        if value is None:
            continue

        normalized_key = normalize_label(key)
        normalized_value = normalize_text(value)

        if normalized_key and normalized_value:
            result[normalized_key] = normalized_value

    return result


def action_labels(actions: Any) -> set[str]:
    if not isinstance(actions, list):
        return set()

    labels: set[str] = set()

    for item in actions:
        if not isinstance(item, dict):
            continue

        label = (
            item.get("dialogue_act")
            or item.get("action")
            or item.get("type")
        )

        if label:
            labels.add(normalize_label(label))

    return labels


def commitment_labels(commitments: Any) -> set[str]:
    if not isinstance(commitments, list):
        return set()

    values: set[str] = set()

    for item in commitments:
        if not isinstance(item, dict):
            continue

        value = item.get("text") or item.get("value")
        if value:
            values.add(normalize_text(value))

    return values


def unresolved_labels(items: Any) -> set[str]:
    if not isinstance(items, list):
        return set()

    values: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        value = item.get("item") or item.get("text") or item.get("value")
        if value:
            values.add(normalize_text(value))

    return values


def extract_evidence_ids(packet: dict[str, Any]) -> set[str]:
    evidence = packet.get("evidence", [])
    if not isinstance(evidence, list):
        return set()

    ids = set()

    for item in evidence:
        if isinstance(item, dict) and item.get("evidence_id"):
            ids.add(str(item["evidence_id"]))

    return ids


# ---------------------------------------------------------------------------
# Set metrics
# ---------------------------------------------------------------------------

def set_prf(reference: set[str], prediction: set[str]) -> tuple[float, float, float, int, int, int]:
    tp = len(reference & prediction)
    fp = len(prediction - reference)
    fn = len(reference - prediction)

    precision, recall, f1 = f1_from_counts(tp, fp, fn)

    return precision, recall, f1, tp, fp, fn


# ---------------------------------------------------------------------------
# Field-evidence support
# ---------------------------------------------------------------------------

def field_evidence_supported(
    field: Any,
    valid_evidence_ids: set[str],
) -> bool:
    if not isinstance(field, dict):
        return False

    ids = field.get("evidence_ids", [])
    if not isinstance(ids, list) or not ids:
        return False

    return all(str(item) in valid_evidence_ids for item in ids)


def count_predicted_state_items_with_evidence(packet: dict[str, Any]) -> tuple[int, int]:
    """
    Count predicted state items that have at least one evidence ID and whose
    evidence IDs point to actual evidence entries.

    This is intentionally a provenance-quality metric, not a correctness metric.
    """
    state = get_state(packet)
    evidence_ids = extract_evidence_ids(packet)

    total = 0
    supported = 0

    scalar_fields = [
        "intent",
        "customer_goal",
        "resolution_state",
        "escalated",
        "verification",
    ]

    for field_name in scalar_fields:
        field = state.get(field_name)
        if not isinstance(field, dict):
            continue

        if value_of_field(field) in (None, "", []):
            continue

        total += 1
        supported += int(
            field_evidence_supported(field, evidence_ids)
        )

    entity_block = state.get("entities", {})
    if isinstance(entity_block, dict):
        for field in entity_block.values():
            if not isinstance(field, dict):
                continue
            if value_of_field(field) in (None, "", []):
                continue
            total += 1
            supported += int(
                field_evidence_supported(field, evidence_ids)
            )

    return total, supported


# ---------------------------------------------------------------------------
# Per-conversation evaluation
# ---------------------------------------------------------------------------

def evaluate_conversation(
    gold: dict[str, Any],
    pred: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gold_state = get_state(gold)
    pred_state = get_state(pred)

    conversation_id = (
        pred.get("conversation", {}).get("conversation_id")
        or gold.get("conversation", {}).get("conversation_id")
        or "unknown"
    )

    errors: dict[str, Any] = {
        "conversation_id": conversation_id,
        "intent": None,
        "customer_goal": None,
        "resolution_state": None,
        "verification": None,
        "escalated": None,
        "entities": [],
        "agent_actions": [],
        "agent_commitments": [],
        "unresolved_items": [],
        "next_action": None,
        "speaker_resolution": None,
    }

    # ------------------------------------------------------------
    # Scalar state fields
    # ------------------------------------------------------------

    gold_intent = value_of_field(gold_state.get("intent"))
    pred_intent = value_of_field(pred_state.get("intent"))
    intent_correct = exact_match(gold_intent, pred_intent)

    goal_ref = value_of_field(gold_state.get("customer_goal"))
    goal_pred = value_of_field(pred_state.get("customer_goal"))
    goal_precision, goal_recall, goal_f1 = token_f1(goal_ref, goal_pred)

    gold_resolution = value_of_field(gold_state.get("resolution_state"))
    pred_resolution = value_of_field(pred_state.get("resolution_state"))
    resolution_correct = exact_match(gold_resolution, pred_resolution)

    gold_verification = value_of_field(gold_state.get("verification"))
    pred_verification = value_of_field(pred_state.get("verification"))
    verification_correct = exact_match(
        gold_verification,
        pred_verification,
    )

    gold_escalated = value_of_field(gold_state.get("escalated"))
    pred_escalated = value_of_field(pred_state.get("escalated"))
    escalated_correct = gold_escalated == pred_escalated

    # ------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------

    gold_entities = normalize_entities(
        gold_state.get("entities")
    )
    pred_entities = normalize_entities(
        pred_state.get("entities")
    )

    entity_precision, entity_recall, entity_f1, entity_tp, entity_fp, entity_fn = set_prf(
        set(f"{k}={v}" for k, v in gold_entities.items()),
        set(f"{k}={v}" for k, v in pred_entities.items()),
    )

    # Record entity errors by key for easier debugging.
    for key in sorted(set(gold_entities) | set(pred_entities)):
        g = gold_entities.get(key)
        p = pred_entities.get(key)

        if g != p:
            errors["entities"].append({
                "key": key,
                "gold": g,
                "predicted": p,
            })

    # ------------------------------------------------------------
    # Agent actions
    # ------------------------------------------------------------

    gold_actions = action_labels(
        gold_state.get("agent_actions")
    )
    pred_actions = action_labels(
        pred_state.get("agent_actions")
    )

    action_precision, action_recall, action_f1, action_tp, action_fp, action_fn = set_prf(
        gold_actions,
        pred_actions,
    )

    if gold_actions != pred_actions:
        errors["agent_actions"] = {
            "gold": sorted(gold_actions),
            "predicted": sorted(pred_actions),
        }

    # ------------------------------------------------------------
    # Commitments
    # ------------------------------------------------------------

    gold_commitments = commitment_labels(
        gold_state.get("agent_commitments")
    )
    pred_commitments = commitment_labels(
        pred_state.get("agent_commitments")
    )

    # Exact text matching is intentionally strict here.
    commitment_precision, commitment_recall, commitment_f1, commit_tp, commit_fp, commit_fn = set_prf(
        gold_commitments,
        pred_commitments,
    )

    if gold_commitments != pred_commitments:
        errors["agent_commitments"] = {
            "gold": sorted(gold_commitments),
            "predicted": sorted(pred_commitments),
        }

    # ------------------------------------------------------------
    # Unresolved items
    # ------------------------------------------------------------

    gold_unresolved = unresolved_labels(
        gold_state.get("unresolved_items")
    )
    pred_unresolved = unresolved_labels(
        pred_state.get("unresolved_items")
    )

    unresolved_precision, unresolved_recall, unresolved_f1, unresolved_tp, unresolved_fp, unresolved_fn = set_prf(
        gold_unresolved,
        pred_unresolved,
    )

    if gold_unresolved != pred_unresolved:
        errors["unresolved_items"] = {
            "gold": sorted(gold_unresolved),
            "predicted": sorted(pred_unresolved),
        }

    # ------------------------------------------------------------
    # Next action
    # ------------------------------------------------------------

    gold_next = gold_state.get("next_action")
    pred_next = pred_state.get("next_action")

    gold_next_value = (
        value_of_field(gold_next)
        if gold_next is not None
        else None
    )
    pred_next_value = (
        value_of_field(pred_next)
        if pred_next is not None
        else None
    )

    if gold_next_value is None and pred_next_value is None:
        next_action_correct = True
        next_action_f1 = 1.0
    elif gold_next_value is None or pred_next_value is None:
        next_action_correct = False
        next_action_f1 = 0.0
    else:
        _, _, next_action_f1 = token_f1(
            gold_next_value,
            pred_next_value,
        )
        next_action_correct = (
            next_action_f1 >= 0.95
        )

    if not next_action_correct:
        errors["next_action"] = {
            "gold": gold_next_value,
            "predicted": pred_next_value,
        }

    # ------------------------------------------------------------
    # Speaker resolution
    # ------------------------------------------------------------

    gold_speaker = (
        gold.get("uncertainty", {})
        .get("speaker_resolution", {})
        .get("status")
    )
    pred_speaker = (
        pred.get("uncertainty", {})
        .get("speaker_resolution", {})
        .get("status")
    )

    # We report both exact status and coarse resolved/unresolved agreement.
    speaker_exact = (
        normalize_label(gold_speaker)
        == normalize_label(pred_speaker)
    )

    def coarse_speaker_status(value: Any) -> str:
        label = normalize_label(value)
        if label in {
            "resolved",
            "known_from_synthetic_metadata",
        }:
            return "RESOLVED"
        if label in {
            "partial",
        }:
            return "PARTIAL"
        return "UNRESOLVED"

    speaker_coarse_match = (
        coarse_speaker_status(gold_speaker)
        == coarse_speaker_status(pred_speaker)
    )

    if not speaker_coarse_match:
        errors["speaker_resolution"] = {
            "gold_status": gold_speaker,
            "predicted_status": pred_speaker,
        }

    # ------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------

    evidence_total, evidence_supported = (
        count_predicted_state_items_with_evidence(pred)
    )
    evidence_support_rate = safe_div(
        evidence_supported,
        evidence_total,
    )

    # ------------------------------------------------------------
    # State-field macro accuracy
    # ------------------------------------------------------------

    scalar_accuracies = [
        float(intent_correct),
        float(resolution_correct),
        float(verification_correct),
        float(escalated_correct),
        goal_f1,
        entity_f1,
        action_f1,
        commitment_f1,
        unresolved_f1,
        next_action_f1,
    ]

    state_field_macro = (
        statistics.fmean(scalar_accuracies)
        if scalar_accuracies
        else 0.0
    )

    row = {
        "conversation_id": conversation_id,

        "intent_correct": int(intent_correct),

        "customer_goal_precision": round(goal_precision, 4),
        "customer_goal_recall": round(goal_recall, 4),
        "customer_goal_f1": round(goal_f1, 4),

        "resolution_correct": int(resolution_correct),
        "verification_correct": int(verification_correct),
        "escalated_correct": int(escalated_correct),

        "entity_precision": round(entity_precision, 4),
        "entity_recall": round(entity_recall, 4),
        "entity_f1": round(entity_f1, 4),
        "entity_tp": entity_tp,
        "entity_fp": entity_fp,
        "entity_fn": entity_fn,

        "action_precision": round(action_precision, 4),
        "action_recall": round(action_recall, 4),
        "action_f1": round(action_f1, 4),
        "action_tp": action_tp,
        "action_fp": action_fp,
        "action_fn": action_fn,

        "commitment_precision": round(commitment_precision, 4),
        "commitment_recall": round(commitment_recall, 4),
        "commitment_f1": round(commitment_f1, 4),
        "commitment_tp": commit_tp,
        "commitment_fp": commit_fp,
        "commitment_fn": commit_fn,

        "unresolved_precision": round(unresolved_precision, 4),
        "unresolved_recall": round(unresolved_recall, 4),
        "unresolved_f1": round(unresolved_f1, 4),
        "unresolved_tp": unresolved_tp,
        "unresolved_fp": unresolved_fp,
        "unresolved_fn": unresolved_fn,

        "next_action_f1": round(next_action_f1, 4),
        "next_action_correct": int(next_action_correct),

        "state_field_macro_score": round(state_field_macro, 4),

        "gold_speaker_status": gold_speaker,
        "predicted_speaker_status": pred_speaker,
        "speaker_resolution_exact_match": int(speaker_exact),
        "speaker_resolution_coarse_match": int(speaker_coarse_match),

        "predicted_state_items_with_evidence": evidence_total,
        "predicted_state_items_evidence_supported": evidence_supported,
        "evidence_support_rate": round(evidence_support_rate, 4),
    }

    return row, errors


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_conversations": 0,
        }

    def mean(field: str) -> float:
        values = [float(r[field]) for r in rows]
        return statistics.fmean(values) if values else 0.0

    # Aggregate entity/action/commitment/unresolved F1 from pooled counts.
    def pooled_prf(prefix: str) -> tuple[float, float, float]:
        tp = sum(int(r[f"{prefix}_tp"]) for r in rows)
        fp = sum(int(r[f"{prefix}_fp"]) for r in rows)
        fn = sum(int(r[f"{prefix}_fn"]) for r in rows)
        return f1_from_counts(tp, fp, fn)

    entity_p, entity_r, entity_f1 = pooled_prf("entity")
    action_p, action_r, action_f1 = pooled_prf("action")
    commitment_p, commitment_r, commitment_f1 = pooled_prf("commitment")
    unresolved_p, unresolved_r, unresolved_f1 = pooled_prf("unresolved")

    return {
        "n_conversations": len(rows),

        "intent_accuracy": round(mean("intent_correct"), 4),
        "customer_goal_f1_macro": round(mean("customer_goal_f1"), 4),
        "resolution_accuracy": round(mean("resolution_correct"), 4),
        "verification_accuracy": round(mean("verification_correct"), 4),
        "escalation_accuracy": round(mean("escalated_correct"), 4),

        "entity_precision_micro": round(entity_p, 4),
        "entity_recall_micro": round(entity_r, 4),
        "entity_f1_micro": round(entity_f1, 4),

        "agent_action_precision_micro": round(action_p, 4),
        "agent_action_recall_micro": round(action_r, 4),
        "agent_action_f1_micro": round(action_f1, 4),

        "commitment_precision_micro": round(commitment_p, 4),
        "commitment_recall_micro": round(commitment_r, 4),
        "commitment_f1_micro": round(commitment_f1, 4),

        "unresolved_precision_micro": round(unresolved_p, 4),
        "unresolved_recall_micro": round(unresolved_r, 4),
        "unresolved_f1_micro": round(unresolved_f1, 4),

        "next_action_f1_macro": round(mean("next_action_f1"), 4),

        "state_field_macro_score": round(
            mean("state_field_macro_score"),
            4,
        ),

        "speaker_resolution_exact_match": round(
            mean("speaker_resolution_exact_match"),
            4,
        ),

        "speaker_resolution_coarse_match": round(
            mean("speaker_resolution_coarse_match"),
            4,
        ),

        "evidence_support_rate": round(
            mean("evidence_support_rate"),
            4,
        ),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RECAP-Voice predicted state against gold state."
    )

    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=DEFAULT_GOLD_DIR,
    )

    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--conversation-id",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if not args.gold_dir.exists():
        raise FileNotFoundError(
            f"Gold directory not found: {args.gold_dir}"
        )

    if not args.pred_dir.exists():
        raise FileNotFoundError(
            f"Prediction directory not found: {args.pred_dir}"
        )

    gold_ids = list_packet_ids(args.gold_dir)
    pred_ids = list_packet_ids(args.pred_dir)

    paired_ids = sorted(gold_ids & pred_ids)

    if args.conversation_id:
        if args.conversation_id not in paired_ids:
            raise FileNotFoundError(
                f"No paired gold/prediction packet for "
                f"{args.conversation_id}"
            )
        paired_ids = [args.conversation_id]

    if args.max_conversations is not None:
        paired_ids = paired_ids[:args.max_conversations]

    if not paired_ids:
        raise RuntimeError(
            "No paired gold/prediction JSON files were found."
        )

    print("=" * 72)
    print("RECAP-VOICE STATE EVALUATION")
    print("=" * 72)
    print(f"Gold directory:       {args.gold_dir}")
    print(f"Prediction directory: {args.pred_dir}")
    print(f"Paired conversations:  {len(paired_ids)}")
    print()

    rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for index, conversation_id in enumerate(paired_ids, start=1):
        gold_path = args.gold_dir / f"{conversation_id}.json"
        pred_path = args.pred_dir / f"{conversation_id}.json"

        gold = load_json(gold_path)
        pred = load_json(pred_path)

        row, errors = evaluate_conversation(
            gold,
            pred,
        )

        rows.append(row)

        # Write only conversations that contain at least one error.
        has_error = any(
            value not in (None, [], {})
            for key, value in errors.items()
            if key != "conversation_id"
        )

        if has_error:
            error_rows.append(errors)

        print(
            f"[{index}/{len(paired_ids)}] "
            f"{conversation_id}: "
            f"intent={row['intent_correct']} | "
            f"goal_f1={row['customer_goal_f1']:.3f} | "
            f"entity_f1={row['entity_f1']:.3f} | "
            f"action_f1={row['action_f1']:.3f} | "
            f"state={row['state_field_macro_score']:.3f}"
        )

    aggregate = aggregate_rows(rows)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_conversation_path = (
        args.output_dir / "per_conversation.csv"
    )

    aggregate_json_path = (
        args.output_dir / "aggregate_summary.json"
    )

    aggregate_csv_path = (
        args.output_dir / "aggregate_summary.csv"
    )

    errors_path = (
        args.output_dir / "errors.jsonl"
    )

    write_csv(
        per_conversation_path,
        rows,
    )

    with aggregate_json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            aggregate,
            f,
            indent=2,
            ensure_ascii=False,
        )

    write_csv(
        aggregate_csv_path,
        [aggregate],
    )

    write_jsonl(
        errors_path,
        error_rows,
    )

    print()
    print("=" * 72)
    print("AGGREGATE RESULTS")
    print("=" * 72)

    for key, value in aggregate.items():
        print(f"{key}: {value}")

    print()
    print("Saved:")
    print(f"  {per_conversation_path}")
    print(f"  {aggregate_json_path}")
    print(f"  {aggregate_csv_path}")
    print(f"  {errors_path}")


if __name__ == "__main__":
    main()
