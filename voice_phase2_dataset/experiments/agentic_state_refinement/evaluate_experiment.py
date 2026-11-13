#!/usr/bin/env python3
"""Paired evaluation of deterministic baseline and refined RECAP-Voice states."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


FIELD_METRICS = {
    "state.intent": ("intent_correct", "intent"),
    "state.customer_goal": ("customer_goal_f1", "customer_goal"),
    "state.resolution_state": ("resolution_correct", "resolution_state"),
    "state.verification": ("verification_correct", "verification"),
    "state.escalated": ("escalated_correct", "escalated"),
    "state.entities": ("entity_f1", "entities"),
    "state.agent_actions": ("action_f1", "agent_actions"),
    "state.agent_commitments": ("commitment_f1", "agent_commitments"),
    "state.unresolved_items": ("unresolved_f1", "unresolved_items"),
    "state.next_action": ("next_action_f1", "next_action"),
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_stage(dataset_root: Path, module_name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        module_name, dataset_root / filename
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        empty_headers = {
            "evidence_audit.csv": (
                "conversation_id,field,evidence_id,id_exists,field_match,"
                "source_turn_valid,structurally_valid,validation_outcome\n"
            ),
        }
        path.write_text(
            empty_headers.get(path.name, ""),
            encoding="utf-8",
        )
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def value_of(field: Any) -> Any:
    return field.get("value") if isinstance(field, dict) and "value" in field else field


def bootstrap_mean_ci(values: list[float], seed: int, n_bootstrap: int = 2000):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(n_bootstrap)
    )
    return samples[int(0.025 * n_bootstrap)], samples[
        min(n_bootstrap - 1, int(0.975 * n_bootstrap))
    ]


def paired_test(deltas: list[float]) -> dict[str, Any]:
    nonzero = [value for value in deltas if value != 0]
    if len(deltas) < 5 or len(nonzero) < 5:
        return {
            "test": "NOT AVAILABLE",
            "statistic": None,
            "p_value": None,
            "reason": (
                "Fewer than five non-zero paired differences; "
                "a paired test is not informative."
            ),
        }
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {
            "test": "NOT AVAILABLE",
            "statistic": None,
            "p_value": None,
            "reason": "scipy is not installed.",
        }
    statistic, p_value = wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
    return {
        "test": "Wilcoxon signed-rank (two-sided)",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "reason": None,
    }


def validate_evidence_ref(
    packet: dict[str, Any], field: str, evidence_id: str, turn_count: int
) -> dict[str, bool]:
    path = field
    if not path.startswith("state."):
        path = f"state.{path}"
    evidence = next(
        (
            item for item in packet.get("evidence", [])
            if str(item.get("evidence_id")) == evidence_id
        ),
        None,
    )
    if path == "state.agent_actions":
        field_match = bool(
            evidence
            and (
                evidence.get("field") == path
                or str(evidence.get("field", "")).startswith(path + ".")
            )
        )
    else:
        field_match = bool(evidence and evidence.get("field") == path)
    turn_ids = evidence.get("turn_ids") if evidence else None
    turns_valid = bool(
        isinstance(turn_ids, list)
        and turn_ids
        and all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 1 <= value <= turn_count
            for value in turn_ids
        )
    )
    return {
        "id_exists": evidence is not None,
        "field_match": field_match,
        "source_turn_valid": turns_valid,
        "structurally_valid": bool(evidence and field_match and turns_valid),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "gold_state_voice_heldout",
    )
    parser.add_argument("--role-summary", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    gold_dir = args.gold_dir.resolve()
    dataset_root = Path(__file__).resolve().parents[2]
    stage12 = load_stage(dataset_root, "state_refine_eval_stage12", "12_evaluate_state.py")
    stage13 = load_stage(dataset_root, "state_refine_eval_stage13", "13_verify_handover.py")
    stage16 = load_stage(dataset_root, "state_refine_eval_stage16", "16_build_handoff_benchmark.py")
    role_accuracy: dict[str, float | None] = {}
    if args.role_summary and args.role_summary.is_file():
        with args.role_summary.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    role_accuracy[row["conversation_id"]] = float(row["role_accuracy"])
                except (KeyError, TypeError, ValueError):
                    role_accuracy[row.get("conversation_id", "")] = None

    records = {
        row["conversation_id"]: row
        for row in read_jsonl(run_dir / "agent_runs.jsonl")
    }
    ids = sorted(records)
    if not ids:
        raise RuntimeError(f"No agent traces found under {run_dir}")
    per_conversation = []
    field_rows = []
    evidence_rows = []
    tool_rows = []
    condition_metrics: dict[str, list[dict[str, Any]]] = {
        "baseline": [],
        "agentic": [],
    }
    paired_fields: dict[str, list[tuple[float, float]]] = {
        field: [] for field in FIELD_METRICS
    }
    capsule_pairs: list[tuple[float, float]] = []
    aggregate_tool_counts: dict[str, int] = {}

    for cid in ids:
        trace = records[cid]
        baseline = load_json(run_dir / "baseline_predictions" / f"{cid}.json")
        refined = load_json(
            run_dir / "final_refined_predictions" / f"{cid}.json"
        )
        baseline_capsule = load_json(
            run_dir / "baseline_capsules" / f"{cid}.json"
        )
        refined_capsule = load_json(
            run_dir / "final_refined_capsules" / f"{cid}.json"
        )
        gold_path = gold_dir / f"{cid}.json"
        if not gold_path.is_file():
            raise FileNotFoundError(f"Missing gold state: {gold_path}")
        gold = load_json(gold_path)
        gold_slots = stage16.extract_state_slots(gold)
        outcome_metrics = {}
        for condition, packet, capsule in (
            ("baseline", baseline, baseline_capsule),
            ("agentic", refined, refined_capsule),
        ):
            metric, _ = stage12.evaluate_conversation(gold, packet)
            slot_score = stage16.score_condition(
                gold_slots, stage16.extract_capsule_slots(capsule)
            )["slot_carryover_f1"]
            verifier = stage13.verify_packet(packet)
            evidence_ids = {
                str(item.get("evidence_id"))
                for item in packet.get("evidence", [])
                if item.get("evidence_id")
            }
            claims = verifier.get("claims", [])
            invalid_refs = sum(
                1 for claim in claims
                for ref in claim.get("evidence_ids", [])
                if str(ref) not in evidence_ids
            )
            metrics = {
                **metric,
                "capsule_slot_carryover_f1": slot_score,
                "capsule_bytes": len(
                    json.dumps(capsule, ensure_ascii=False).encode("utf-8")
                ),
                "claim_count": len(claims),
                "claims_without_evidence": sum(
                    not claim.get("evidence_ids") for claim in claims
                ),
                "invalid_evidence_reference_count": invalid_refs,
                "structural_evidence_support_rate": (
                    sum(claim.get("verification") == "VERIFIED" for claim in claims)
                    / len(claims)
                    if claims else 1.0
                ),
                "role_accuracy": role_accuracy.get(cid),
            }
            outcome_metrics[condition] = metrics
            condition_metrics[condition].append(metrics)
        base_metric = outcome_metrics["baseline"]
        agent_metric = outcome_metrics["agentic"]
        orch = trace.get("orchestrator", {})
        tools = orch.get("tool_call_counts", {})
        decision_fields = {
            item.get("field")
            for item in orch.get("validated_changes", [])
            if item.get("applied")
        }
        for field_path, (metric_name, state_key) in FIELD_METRICS.items():
            before = float(base_metric[metric_name])
            after = float(agent_metric[metric_name])
            changed = field_path in decision_fields or (
                field_path == "state.entities"
                and any(
                    str(path).startswith("state.entities.")
                    for path in decision_fields
                )
            )
            outcome = {
                "conversation_id": cid,
                "field": field_path,
                "baseline_score": before,
                "agentic_score": after,
                "delta": after - before,
                "field_changed": changed,
                "baseline_correct": before >= 1.0,
                "agentic_correct": after >= 1.0,
                "improved": after > before,
                "unchanged_correct": before >= 1.0 and after >= 1.0 and not changed,
                "unchanged_incorrect": before < 1.0 and after == before and not changed,
                "incorrectly_changed": before >= 1.0 and changed and after < 1.0,
                "corrected": before < 1.0 and after >= 1.0,
                "regressed": after < before,
            }
            field_rows.append(outcome)
            paired_fields[field_path].append((before, after))

        base_capsule_score = float(base_metric["capsule_slot_carryover_f1"])
        agent_capsule_score = float(agent_metric["capsule_slot_carryover_f1"])
        capsule_pairs.append((base_capsule_score, agent_capsule_score))
        decisions = orch.get("decisions", [])
        validated = [
            item for item in orch.get("validated_changes", [])
            if item.get("applied")
        ]
        rejected = [
            item for item in orch.get("rejected_changes", [])
            if item.get("action") == "PROPOSE_CORRECTION"
        ]
        proposed = [
            item for item in decisions
            if item.get("action") == "PROPOSE_CORRECTION"
        ]
        for proposal in proposed:
            refs = proposal.get("evidence_refs", [])
            if not refs:
                evidence_rows.append({
                    "conversation_id": cid,
                    "field": proposal.get("field"),
                    "evidence_id": "",
                    "id_exists": False,
                    "field_match": False,
                    "source_turn_valid": False,
                    "structurally_valid": False,
                    "validation_outcome": "REJECTED_NO_EVIDENCE",
                })
            for ref in refs:
                validity = validate_evidence_ref(
                    baseline,
                    str(proposal.get("field", "")),
                    str(ref),
                    int(trace.get("source_turn_count", 0)),
                )
                evidence_rows.append({
                    "conversation_id": cid,
                    "field": proposal.get("field"),
                    "evidence_id": ref,
                    **validity,
                    "validation_outcome": (
                        "VALIDATED" if validity["structurally_valid"] else "INVALID"
                    ),
                })

        all_tool_events = orch.get("tool_events", [])
        event_counts: dict[str, int] = {}
        for event in all_tool_events:
            event_counts[event.get("tool", "unknown")] = (
                event_counts.get(event.get("tool", "unknown"), 0) + 1
            )
        for name, count in event_counts.items():
            aggregate_tool_counts[name] = aggregate_tool_counts.get(name, 0) + count
        tool_rows.append({
            "conversation_id": cid,
            "status": trace.get("agent_status"),
            **{f"calls_{name}": count for name, count in event_counts.items()},
            "tool_calls_total": sum(event_counts.values()),
            "tool_rounds": orch.get("tool_rounds"),
            "acoustic_context_retrieved": trace.get("acoustic_context_retrieved", False),
            "input_tokens": orch.get("usage", {}).get("input_tokens"),
            "output_tokens": orch.get("usage", {}).get("output_tokens"),
            "api_latency_seconds": orch.get("api_latency_seconds"),
            "agent_wall_latency_seconds": trace.get("agent_total_latency_seconds"),
        })

        base_score = float(base_metric["state_field_macro_score"])
        agent_score = float(agent_metric["state_field_macro_score"])
        per_conversation.append({
            "conversation_id": cid,
            "agent_status": trace.get("agent_status"),
            "baseline_score": base_score,
            "agentic_score": agent_score,
            "delta": agent_score - base_score,
            "baseline_context_bytes": trace.get("baseline_context_bytes"),
            "agent_retrieved_context_bytes": trace.get(
                "agent_retrieved_context_bytes"
            ),
            "available_context_bytes": trace.get("available_context_bytes"),
            "agent_context_retrieval_rate": trace.get(
                "agent_context_retrieval_percent"
            ),
            "tool_calls": sum(event_counts.values()),
            "latency_seconds": trace.get("agent_total_latency_seconds"),
            "input_tokens": orch.get("usage", {}).get("input_tokens"),
            "output_tokens": orch.get("usage", {}).get("output_tokens"),
            "num_proposed_changes": len(proposed),
            "num_applied_changes": len(validated),
            "num_rejected_changes": len(rejected),
            "num_rejected_decisions": len(
                orch.get("rejected_changes", [])
            ),
            "num_corrected_fields": sum(
                1 for row in field_rows[-len(FIELD_METRICS):]
                if row["conversation_id"] == cid and row["corrected"]
            ),
            "num_regressions": sum(
                1 for row in field_rows[-len(FIELD_METRICS):]
                if row["conversation_id"] == cid and row["regressed"]
            ),
            "evidence_support_rate": agent_metric["evidence_support_rate"],
            "capsule_slot_carryover_f1": agent_metric[
                "capsule_slot_carryover_f1"
            ],
            "role_accuracy": agent_metric["role_accuracy"],
            "baseline_state_macro": base_score,
            "agentic_state_macro": agent_score,
            "acoustic_context_calls": event_counts.get("get_acoustic_context", 0),
            "acoustic_context_retrieved": trace.get(
                "acoustic_context_retrieved", False
            ),
            "baseline_capsule_latency_seconds": trace.get(
                "baseline_capsule_latency_seconds"
            ),
        })

    write_csv(run_dir / "per_conversation.csv", per_conversation)
    write_csv(run_dir / "field_level_audit.csv", field_rows)
    write_csv(run_dir / "evidence_audit.csv", evidence_rows)
    write_csv(run_dir / "tool_usage.csv", tool_rows)

    summary_rows = []
    scalar_metrics = [
        "intent_correct", "customer_goal_f1", "resolution_correct",
        "verification_correct", "escalated_correct", "entity_precision",
        "entity_recall", "entity_f1", "action_precision", "action_recall",
        "action_f1", "commitment_precision", "commitment_recall",
        "commitment_f1", "unresolved_precision", "unresolved_recall",
        "unresolved_f1", "next_action_f1", "state_field_macro_score",
        "speaker_resolution_exact_match", "speaker_resolution_coarse_match",
        "evidence_support_rate", "capsule_slot_carryover_f1", "capsule_bytes",
        "claim_count", "claims_without_evidence",
        "invalid_evidence_reference_count", "structural_evidence_support_rate",
        "role_accuracy",
    ]
    for condition in ("baseline", "agentic"):
        rows = condition_metrics[condition]
        aggregate = {
            "condition": condition,
            "n_conversations": len(rows),
            "successful_agentic_predictions": (
                sum(r["agent_status"] == "COMPLETED" for r in per_conversation)
                if condition == "agentic" else len(rows)
            ),
            "failed_agentic_predictions": (
                sum(r["agent_status"] != "COMPLETED" for r in per_conversation)
                if condition == "agentic" else 0
            ),
        }
        for metric in scalar_metrics:
            values = [
                float(row[metric]) for row in rows
                if isinstance(row.get(metric), (int, float))
                and not (isinstance(row[metric], float) and math.isnan(row[metric]))
            ]
            aggregate[f"mean_{metric}"] = (
                statistics.fmean(values) if values else None
            )
        summary_rows.append(aggregate)
    write_csv(run_dir / "summary.csv", summary_rows)

    pair_rows = []
    for metric in [
        "intent_correct", "customer_goal_f1", "resolution_correct",
        "verification_correct", "escalated_correct", "entity_f1",
        "action_f1", "commitment_f1", "unresolved_f1", "next_action_f1",
        "state_field_macro_score", "evidence_support_rate",
        "capsule_slot_carryover_f1",
    ]:
        baseline_values = [float(row[metric]) for row in condition_metrics["baseline"]]
        agent_values = [float(row[metric]) for row in condition_metrics["agentic"]]
        deltas = [agent - baseline for baseline, agent in zip(baseline_values, agent_values)]
        low, high = bootstrap_mean_ci(deltas, args.seed)
        pair_rows.append({
            "metric": metric,
            "n_pairs": len(deltas),
            "baseline_mean": statistics.fmean(baseline_values),
            "agentic_mean": statistics.fmean(agent_values),
            "mean_delta": statistics.fmean(deltas),
            "median_delta": statistics.median(deltas),
            "delta_standard_deviation": (
                statistics.stdev(deltas) if len(deltas) > 1 else 0.0
            ),
            "bootstrap_95ci_low": low,
            "bootstrap_95ci_high": high,
            "bootstrap_seed": args.seed,
            **paired_test(deltas),
        })
    for field_path, (metric_name, _state_key) in FIELD_METRICS.items():
        pairs = paired_fields[field_path]
        deltas = [agent - baseline for baseline, agent in pairs]
        low, high = bootstrap_mean_ci(deltas, args.seed)
        pair_rows.append({
            "metric": f"field:{field_path}",
            "n_pairs": len(deltas),
            "baseline_mean": (
                statistics.fmean(before for before, _after in pairs)
                if pairs else None
            ),
            "agentic_mean": (
                statistics.fmean(after for _before, after in pairs)
                if pairs else None
            ),
            "mean_delta": statistics.fmean(deltas) if deltas else None,
            "median_delta": statistics.median(deltas) if deltas else None,
            "delta_standard_deviation": (
                statistics.stdev(deltas) if len(deltas) > 1 else 0.0
            ),
            "bootstrap_95ci_low": low,
            "bootstrap_95ci_high": high,
            "bootstrap_seed": args.seed,
            **paired_test(deltas),
        })
    write_csv(run_dir / "paired_comparison.csv", pair_rows)

    total_fields = len(field_rows)
    wrong_baseline = [row for row in field_rows if not row["baseline_correct"]]
    correct_baseline = [row for row in field_rows if row["baseline_correct"]]
    changed = [
        row for row in field_rows if row["field_changed"]
    ]
    applied_count = sum(len([
        change for change in record.get("orchestrator", {}).get(
            "validated_changes", []
        ) if change.get("applied")
    ]) for record in records.values())
    proposed_count = sum(
        sum(
            decision.get("action") == "PROPOSE_CORRECTION"
            for decision in record.get("orchestrator", {}).get("decisions", [])
        )
        for record in records.values()
    )
    invalid_refs = sum(
        not row.get("structurally_valid", False)
        for row in evidence_rows if row.get("evidence_id")
    )
    total_refs = sum(bool(row.get("evidence_id")) for row in evidence_rows)
    applied_deltas = [
        (cid, change)
        for cid, record in records.items()
        for change in record.get("orchestrator", {}).get(
            "validated_changes", []
        )
        if change.get("applied")
    ]
    improved_applied = 0
    fully_correct_applied = 0
    structurally_unsupported_applied = 0
    for cid, change in applied_deltas:
        normalized_field = str(change.get("field", ""))
        metric_key = (
            "state.entities"
            if normalized_field.startswith("state.entities.")
            else normalized_field
        )
        mapping = FIELD_METRICS.get(metric_key)
        if mapping is not None:
            metric_name = mapping[0]
            row_index = ids.index(cid)
            before_score = float(condition_metrics["baseline"][row_index][metric_name])
            after_score = float(condition_metrics["agentic"][row_index][metric_name])
            if after_score > before_score:
                improved_applied += 1
            if after_score >= 1.0 and after_score >= before_score:
                fully_correct_applied += 1
        validation = change.get("validation", {})
        if not all(
            validation.get(key) is True
            for key in (
                "evidence_reference_valid",
                "source_turns_valid",
                "field_scope_valid",
                "schema_valid",
            )
        ):
            structurally_unsupported_applied += 1
    macro_deltas = [row["delta"] for row in per_conversation]
    macro_ci = bootstrap_mean_ci(macro_deltas, args.seed)
    change_metrics = {
        "field_conversation_pairs": total_fields,
        "improved_field_pairs": sum(row["improved"] for row in field_rows),
        "unchanged_correct_field_pairs": sum(
            row["unchanged_correct"] for row in field_rows
        ),
        "unchanged_incorrect_field_pairs": sum(
            row["unchanged_incorrect"] for row in field_rows
        ),
        "incorrectly_changed_field_pairs": sum(
            row["incorrectly_changed"] for row in field_rows
        ),
        "corrected_to_fully_correct_field_pairs": sum(
            row["corrected"] for row in field_rows
        ),
        "regressed_field_pairs": sum(row["regressed"] for row in field_rows),
        "baseline_wrong_field_pairs": len(wrong_baseline),
        "baseline_correct_field_pairs": len(correct_baseline),
        "baseline_error_improvement_rate": (
            sum(row["improved"] for row in wrong_baseline) / len(wrong_baseline)
            if wrong_baseline else None
        ),
        "baseline_error_full_correction_rate": (
            sum(row["corrected"] for row in wrong_baseline) / len(wrong_baseline)
            if wrong_baseline else None
        ),
        "preservation_rate_among_baseline_correct_fields": (
            sum(row["unchanged_correct"] for row in correct_baseline)
            / len(correct_baseline)
            if correct_baseline else None
        ),
        "regression_rate_among_baseline_correct_fields": (
            sum(row["incorrectly_changed"] for row in correct_baseline)
            / len(correct_baseline)
            if correct_baseline else None
        ),
        "net_mean_field_score_delta": (
            statistics.fmean(row["delta"] for row in field_rows)
            if field_rows else None
        ),
        "correction_precision_by_gold_score_improvement": (
            improved_applied / applied_count
            if applied_count else "NOT AVAILABLE"
        ),
        "correction_precision_by_fully_correct_final_field": (
            fully_correct_applied / applied_count
            if applied_count else "NOT AVAILABLE"
        ),
        "correction_recall_among_evidence_supportable_errors": "NOT AVAILABLE",
        "reason_correction_recall_unavailable": (
            "No independent annotation identifies which baseline errors were "
            "semantically correctable from their evidence; reference validity "
            "alone cannot establish that."
        ),
        "validated_change_rate": (
            applied_count / proposed_count
            if proposed_count else "NOT AVAILABLE"
        ),
        "unsupported_applied_change_rate_structural": (
            structurally_unsupported_applied / applied_count
            if applied_count else "NOT AVAILABLE"
        ),
        "unsupported_applied_changes_structural": structurally_unsupported_applied,
        "unsupported_applied_change_rate_semantic": "NOT AVAILABLE",
        "invalid_evidence_reference_rate": (
            invalid_refs / total_refs
            if total_refs else "NOT AVAILABLE"
        ),
        "invalid_proposed_evidence_refs": invalid_refs,
        "proposed_evidence_refs": total_refs,
        "applied_changes": applied_count,
        "proposed_corrections": proposed_count,
        "conversations_with_no_applied_changes": sum(
            row["num_applied_changes"] == 0 for row in per_conversation
        ),
        "correctly_unchanged_when_no_correction_is_justified": "NOT AVAILABLE",
        "reason_unchanged_metric_unavailable": (
            "The gold state does not annotate whether each candidate correction "
            "was semantically justified; no-change alone cannot establish that."
        ),
        "field_score_delta_mean": (
            statistics.fmean(macro_deltas) if macro_deltas else None
        ),
        "field_score_delta_median": (
            statistics.median(macro_deltas) if macro_deltas else None
        ),
        "field_score_delta_standard_deviation": (
            statistics.stdev(macro_deltas) if len(macro_deltas) > 1 else 0.0
        ),
        "field_score_delta_bootstrap_95ci": macro_ci,
        "field_score_paired_test": paired_test(macro_deltas),
        "acoustic_context_calls": aggregate_tool_counts.get(
            "get_acoustic_context", 0
        ),
        "conversations_with_acoustic_context": sum(
            row["acoustic_context_retrieved"] for row in per_conversation
        ),
        "acoustic_associated_state_changes": sum(
            bool(
                record.get("acoustic_context_retrieved")
                and any(
                    change.get("applied")
                    for change in record.get("orchestrator", {}).get(
                        "validated_changes", []
                    )
                )
            )
            for record in records.values()
        ),
        "conversations_acoustic_retrieved_without_state_change": sum(
            bool(
                record.get("acoustic_context_retrieved")
                and not any(
                    change.get("applied")
                    for change in record.get("orchestrator", {}).get(
                        "validated_changes", []
                    )
                )
            )
            for record in records.values()
        ),
        "acoustic_retrieved_without_state_change": sum(
            bool(
                record.get("acoustic_context_retrieved")
                and not any(
                    change.get("applied")
                    for change in record.get("orchestrator", {}).get(
                        "validated_changes", []
                    )
                )
            )
            for record in records.values()
        ),
        "acoustic_attributable_state_changes": 0,
        "irrelevant_acoustic_retrievals": "NOT AVAILABLE",
        "acoustic_state_change_note": (
            "The validator forbids acoustic features as supporting evidence for "
            "state changes; contextual association is not causal attribution."
        ),
        "context_retrieval_rate_mean": (
            statistics.fmean(
                float(row["agent_context_retrieval_rate"] or 0)
                for row in per_conversation
            )
        ),
        "baseline_full_context_reference_percent": 100.0,
        "mean_baseline_context_bytes": statistics.fmean(
            float(row["baseline_context_bytes"] or 0) for row in per_conversation
        ),
        "mean_agent_retrieved_context_bytes": statistics.fmean(
            float(row["agent_retrieved_context_bytes"] or 0)
            for row in per_conversation
        ),
        "mean_tool_calls": statistics.fmean(
            float(row["tool_calls"] or 0) for row in per_conversation
        ),
        "mean_input_tokens": statistics.fmean(
            float(row["input_tokens"] or 0) for row in per_conversation
        ),
        "mean_output_tokens": statistics.fmean(
            float(row["output_tokens"] or 0) for row in per_conversation
        ),
        "mean_agent_wall_latency_seconds": statistics.fmean(
            float(row["latency_seconds"] or 0) for row in per_conversation
        ),
        "tool_call_counts": aggregate_tool_counts,
        "executed_tool_calls": sum(
            event.get("status") in {"success", "error"}
            for record in records.values()
            for event in record.get("orchestrator", {}).get("tool_events", [])
        ),
        "budget_exhausted_tool_requests": sum(
            event.get("status") == "budget_exhausted"
            for record in records.values()
            for event in record.get("orchestrator", {}).get("tool_events", [])
        ),
        "available_existing_state_evaluation_metrics": scalar_metrics,
        "semantic_evidence_support": "NOT AVAILABLE",
        "speaker_role_metric_note": (
            "Role accuracy is the existing stage-07b diagnostic; it is not "
            "recomputed for a state-only refinement."
        ),
    }
    summary = {
        "experiment": "agentic_state_refinement",
        "n_conversations": len(ids),
        "conversation_ids": ids,
        "conditions": summary_rows,
        "paired_comparisons": pair_rows,
        "refinement_effectiveness": change_metrics,
        "interpretation_limits": [
            "The existing evaluator is reused without changes.",
            "Evidence validation establishes existing ID, field association, source-turn index, and packet-schema validity, not semantic entailment.",
            "Correction recall among semantically correctable errors and correctness of a no-change decision require annotations not present in the dataset.",
            "Acoustic context is not accepted as evidence for transcript-derived state changes.",
            "The hard-case test sample contains 10 conversations and is a controlled pilot.",
            "Baseline full-context bytes are a reference inventory of all available state, transcript, evidence, and acoustic units; they are not an LLM run with that prompt.",
            "Token usage and latency include agent/API work; upstream ASR, diarization, and state extraction are precomputed and excluded.",
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Evaluated {len(ids)} paired conversations: {run_dir}")
    for row in summary_rows:
        print(
            f"{row['condition']}: n={row['n_conversations']} "
            f"state_macro={row.get('mean_state_field_macro_score')} "
            f"capsule_slot_f1={row.get('mean_capsule_slot_carryover_f1')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
