#!/usr/bin/env python3
"""Run evidence-grounded field refinement on existing hard-case test artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

EXPERIMENT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = EXPERIMENT_DIR.parents[1]
PREVIOUS_RUN = (
    DATASET_ROOT / "robustness" / "agentic_context_orchestration"
    / "seed_42_test10_run4"
)
DEFAULT_DATASET = DATASET_ROOT / "data/experiments/voice_heldout_dataset.json"
DEFAULT_BASELINE = DATASET_ROOT / "robustness/hard_cases_pipeline/seed_42"
DEFAULT_GOLD = DATASET_ROOT / "gold_state_voice_heldout"
DEFAULT_ACOUSTIC = PREVIOUS_RUN / "baseline/acoustic_features"
DEFAULT_OUTPUT_ROOT = DATASET_ROOT / "robustness/agentic_state_refinement"
AGENTIC_DIR = DATASET_ROOT / "experiments/agentic_context_orchestration"
sys.path.insert(0, str(AGENTIC_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

from orchestrator import (  # noqa: E402
    ContextTools,
    LLMConfig,
    _load_openai_client,
    _redact,
    load_stage_module,
)
from state_refinement import RefinementOrchestrator  # noqa: E402


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else DATASET_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(DATASET_ROOT.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def select_ids(
    dataset: list[dict[str, Any]],
    baseline_dir: Path,
    gold_dir: Path,
    acoustic_dir: Path,
    limit: int,
    seed: int,
    selected: list[str] | None,
) -> list[str]:
    eligible = []
    for row in dataset:
        cid = str(row.get("conversation_id", ""))
        if (
            row.get("split") == "test"
            and cid.startswith("hard_")
            and (baseline_dir / "transcripts_whisperx" / f"{cid}.json").is_file()
            and (baseline_dir / "wordlevel_speaker_transcripts" / f"{cid}.json").is_file()
            and (baseline_dir / "recap_predictions" / f"{cid}.json").is_file()
            and (gold_dir / f"{cid}.json").is_file()
            and (acoustic_dir / f"{cid}.json").is_file()
        ):
            eligible.append(cid)
    eligible.sort()
    if selected:
        missing = sorted(set(selected) - set(eligible))
        if missing:
            raise FileNotFoundError(
                "Requested test conversations lack required artifacts: "
                + ", ".join(missing)
            )
        return list(dict.fromkeys(selected))
    if limit < 0:
        raise ValueError("--limit must be zero or a positive integer")
    if limit and limit < len(eligible):
        return sorted(random.Random(seed).sample(eligible, limit))
    return eligible


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--acoustic-dir", type=Path, default=DEFAULT_ACOUSTIC)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conversation-id", action="append", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument("--max-tool-calls", type=int, default=10)
    args = parser.parse_args()
    if args.max_tool_rounds < 1 or args.max_tool_calls < 1:
        parser.error("tool budgets must be positive")

    load_dotenv(DATASET_ROOT / ".env", override=False)
    dataset_path = resolve_path(args.dataset)
    baseline_dir = resolve_path(args.baseline_dir)
    gold_dir = resolve_path(args.gold_dir)
    acoustic_dir = resolve_path(args.acoustic_dir)
    dataset = load_json(dataset_path)
    if not isinstance(dataset, list):
        raise ValueError("Held-out dataset must contain a JSON list")
    conversation_ids = select_ids(
        dataset,
        baseline_dir,
        gold_dir,
        acoustic_dir,
        args.limit,
        args.seed,
        args.conversation_id,
    )
    if not conversation_ids:
        raise RuntimeError("No matching held-out test artifacts were found")

    if args.output:
        output_dir = resolve_path(args.output)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
        output_dir = DEFAULT_OUTPUT_ROOT / f"seed_{args.seed}_run_{timestamp}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "baseline_predictions", "agentic_predictions",
        "final_refined_predictions", "baseline_capsules",
        "final_refined_capsules",
    ):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    stage11 = load_stage_module("refine_stage11", "11_extract_state.py")
    stage13 = load_stage_module("refine_stage13", "13_verify_handover.py")
    stage15 = load_stage_module("refine_stage15", "15_build_handover_capsules.py")
    config = LLMConfig.from_env()
    client = None
    client_error: dict[str, Any] | None = None
    try:
        client = _load_openai_client(config)
    except Exception as exc:
        client_error = {
            "error_type": type(exc).__name__,
            "error": _redact(str(exc))[:1000],
        }

    trace_path = output_dir / "agent_runs.jsonl"
    errors_path = output_dir / "errors.jsonl"
    changes_path = output_dir / "field_level_changes.jsonl"
    for path in (trace_path, errors_path, changes_path):
        path.touch()
    records: list[dict[str, Any]] = []
    all_tool_counts: dict[str, int] = {}
    started_all = time.perf_counter()

    for index, cid in enumerate(conversation_ids, start=1):
        print(f"[{index}/{len(conversation_ids)}] {cid}", flush=True)
        raw_transcript = load_json(
            baseline_dir / "transcripts_whisperx" / f"{cid}.json"
        )
        speaker_transcript = load_json(
            baseline_dir / "wordlevel_speaker_transcripts" / f"{cid}.json"
        )
        baseline_packet = load_json(
            baseline_dir / "recap_predictions" / f"{cid}.json"
        )
        if (
            baseline_packet.get("packet_type") != "recap_voice"
            or baseline_packet.get("conversation", {}).get("conversation_id") != cid
        ):
            raise ValueError(f"Invalid baseline packet for {cid}")
        acoustic = load_json(acoustic_dir / f"{cid}.json")
        tools = ContextTools(
            cid, raw_transcript, speaker_transcript, baseline_packet, acoustic
        )
        baseline_context_bytes = tools.available_context_bytes
        for key, value in tools.available_units.items():
            tools.retrieved_units[key] = value
        baseline_context_bytes = tools.retrieved_context_bytes
        tools.retrieved_units.clear()
        tools.retrieved_units["state:initial"] = tools.available_units["state:initial"]

        baseline_errors = stage11.validate_packet(baseline_packet)
        if baseline_errors:
            raise ValueError(
                f"Baseline packet for {cid} fails existing schema validation: "
                + "; ".join(baseline_errors)
            )
        baseline_capsule_started = time.perf_counter()
        baseline_verification = stage13.verify_packet(baseline_packet)
        baseline_capsule = stage15.build_capsule(
            baseline_packet, baseline_verification, acoustic
        )
        baseline_capsule_seconds = time.perf_counter() - baseline_capsule_started
        write_json(output_dir / "baseline_predictions" / f"{cid}.json", baseline_packet)
        write_json(output_dir / "baseline_capsules" / f"{cid}.json", baseline_capsule)

        agent_started = time.perf_counter()
        if client_error:
            result = {
                "status": "FAILED",
                "error": client_error,
                "errors": [client_error],
                "decisions": [],
                "validated_changes": [],
                "rejected_changes": [],
                "final_state": copy.deepcopy(baseline_packet["state"]),
                "fallback_used": True,
                "tool_events": [],
                "tool_call_counts": {},
                "usage": {},
                "api_latency_seconds": 0.0,
                "retrieved_context_bytes": tools.retrieved_context_bytes,
                "available_context_bytes": tools.available_context_bytes,
                "context_retrieval_percent": (
                    100 * tools.retrieved_context_bytes / tools.available_context_bytes
                    if tools.available_context_bytes else 0
                ),
            }
        else:
            agent = RefinementOrchestrator(
                config,
                tools,
                stage11,
                client,
                max_tool_rounds=args.max_tool_rounds,
                max_tool_calls=args.max_tool_calls,
            )
            result = agent.run()
        final_packet = copy.deepcopy(baseline_packet)
        final_packet["state"] = copy.deepcopy(result["final_state"])
        schema_errors = stage11.validate_packet(final_packet)
        if schema_errors:
            result["status"] = "FAILED"
            result["fallback_used"] = True
            result["error"] = {
                "error_type": "FinalSchemaValidationError",
                "error": "; ".join(schema_errors),
            }
            final_packet = copy.deepcopy(baseline_packet)
            result["final_state"] = copy.deepcopy(baseline_packet["state"])
        if result["status"] != "COMPLETED":
            final_packet = copy.deepcopy(baseline_packet)
            result["final_state"] = copy.deepcopy(baseline_packet["state"])

        final_verification = stage13.verify_packet(final_packet)
        acoustic_called = result.get("tool_call_counts", {}).get(
            "get_acoustic_context", 0
        ) > 0
        final_capsule = stage15.build_capsule(
            final_packet,
            final_verification,
            acoustic if acoustic_called else None,
        )
        final_capsule["state_refinement"] = {
            "status": result["status"],
            "fallback_used": result.get("fallback_used", False),
            "applied_change_count": sum(
                1 for change in result.get("validated_changes", [])
                if change.get("applied")
            ),
        }
        write_json(output_dir / "agentic_predictions" / f"{cid}.json", {
            "conversation_id": cid,
            "agent_status": result["status"],
            "summary": result.get("summary", ""),
            "decisions": result.get("decisions", []),
            "validated_changes": result.get("validated_changes", []),
            "rejected_changes": result.get("rejected_changes", []),
        })
        write_json(
            output_dir / "final_refined_predictions" / f"{cid}.json",
            final_packet,
        )
        write_json(
            output_dir / "final_refined_capsules" / f"{cid}.json",
            final_capsule,
        )
        elapsed = time.perf_counter() - agent_started

        record = {
            "conversation_id": cid,
            "agent_status": result["status"],
            "fallback_used": result.get("fallback_used", False),
            "baseline_state": baseline_packet["state"],
            "initial_uncertainty": baseline_packet.get("uncertainty", {}),
            "decisions": result.get("decisions", []),
            "validated_changes": result.get("validated_changes", []),
            "rejected_changes": result.get("rejected_changes", []),
            "final_state": final_packet["state"],
            "baseline_context_bytes": baseline_context_bytes,
            "available_context_bytes": tools.available_context_bytes,
            "source_turn_count": len(tools.speaker_turns),
            "agent_retrieved_context_bytes": result.get(
                "retrieved_context_bytes", tools.retrieved_context_bytes
            ),
            "agent_context_retrieval_percent": result.get(
                "context_retrieval_percent"
            ),
            "agent_total_latency_seconds": round(elapsed, 6),
            "baseline_capsule_latency_seconds": round(baseline_capsule_seconds, 6),
            "baseline_capsule": baseline_capsule,
            "final_capsule": final_capsule,
            "acoustic_context_retrieved": acoustic_called,
            "orchestrator": result,
        }
        records.append(record)
        append_jsonl(trace_path, record)
        append_jsonl(changes_path, {
            "conversation_id": cid,
            "decisions": record["decisions"],
            "validated_changes": record["validated_changes"],
            "rejected_changes": record["rejected_changes"],
        })
        for name, count in result.get("tool_call_counts", {}).items():
            all_tool_counts[name] = all_tool_counts.get(name, 0) + int(count)
        for error in result.get("errors", []):
            append_jsonl(errors_path, {"conversation_id": cid, **error})
        if result.get("error"):
            append_jsonl(errors_path, {
                "conversation_id": cid,
                **result["error"],
            })

    llm_metadata = config.public_metadata()
    llm_metadata["max_tool_rounds"] = args.max_tool_rounds
    llm_metadata["max_tool_calls"] = args.max_tool_calls
    source_files = (
        "run_experiment.py",
        "evaluate_experiment.py",
        "state_refinement.py",
        "system_prompt.md",
        "test_state_refinement.py",
    )
    metadata = {
        "experiment": "agentic_state_refinement",
        "experiment_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dataset_split": "hard-case held-out test",
        "dataset_path": str(resolve_path(args.dataset)),
        "dataset_sha256": sha256_file(resolve_path(args.dataset)),
        "baseline_artifact_root": str(baseline_dir),
        "gold_reference_root": str(gold_dir),
        "acoustic_context_root": str(acoustic_dir),
        "conversation_ids": conversation_ids,
        "n_conversations": len(conversation_ids),
        "baseline_predictions": len(records),
        "successful_agentic_predictions": sum(
            row["agent_status"] == "COMPLETED" for row in records
        ),
        "failed_agentic_predictions": sum(
            row["agent_status"] != "COMPLETED" for row in records
        ),
        "tool_configuration": {
            "max_tool_rounds": args.max_tool_rounds,
            "max_tool_calls": args.max_tool_calls,
            "parallel_tool_calls": False,
            "tools": [
                "get_baseline_state", "get_evidence", "search_transcript",
                "get_speaker_aware_transcript", "get_acoustic_context",
                "get_unresolved_items", "get_commitments",
                "get_next_action_context", "validate_evidence",
                "validate_state_change",
            ],
        },
        "llm": llm_metadata,
        "api_key_configured": bool(config.api_key),
        "code_revision": git_revision(),
        "experiment_source_sha256": {
            name: sha256_file(EXPERIMENT_DIR / name)
            for name in source_files
        },
        "state_policy": "Baseline packet is immutable; only deterministic validated field deltas are applied; any agent failure falls back to baseline.",
        "evidence_validation_scope": (
            "Evidence ID existence, packet/conversation scope, state-field "
            "association, source-turn index validity, and packet structure. "
            "Semantic entailment is not automatically established."
        ),
        "elapsed_seconds": round(time.perf_counter() - started_all, 3),
        "tool_call_counts": all_tool_counts,
        "total_tool_calls": sum(all_tool_counts.values()),
        "total_input_tokens": sum(
            int(row["orchestrator"].get("usage", {}).get("input_tokens", 0))
            for row in records
        ),
        "total_output_tokens": sum(
            int(row["orchestrator"].get("usage", {}).get("output_tokens", 0))
            for row in records
        ),
        "total_api_latency_seconds": round(sum(
            float(row["orchestrator"].get("api_latency_seconds", 0))
            for row in records
        ), 3),
        "total_agent_wall_latency_seconds": round(sum(
            float(row["agent_total_latency_seconds"]) for row in records
        ), 3),
    }
    write_json(output_dir / "metadata.json", metadata)
    evaluator = EXPERIMENT_DIR / "evaluate_experiment.py"
    subprocess.run(
        [
            sys.executable, str(evaluator),
            "--run-dir", str(output_dir),
            "--gold-dir", str(gold_dir),
            "--role-summary",
            str(baseline_dir / "wordlevel_speaker_summary.csv"),
            "--seed", str(args.seed),
        ],
        check=True,
        cwd=DATASET_ROOT,
    )
    print(f"Experiment outputs: {output_dir}")
    print(
        "Agentic predictions: "
        f"{metadata['successful_agentic_predictions']} successful, "
        f"{metadata['failed_agentic_predictions']} failed."
    )
    return 0 if metadata["failed_agentic_predictions"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
