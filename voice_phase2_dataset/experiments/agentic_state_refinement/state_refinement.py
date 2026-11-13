from __future__ import annotations

import copy
import json
import math
import re
import time
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = PACKAGE_DIR / "system_prompt.md"

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "field": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["KEEP_BASELINE", "PROPOSE_CORRECTION"],
                    },
                    "old_value_json": {"type": "string"},
                    "new_value_json": {"type": "string"},
                    "reason": {"type": "string"},
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number"},
                },
                "required": [
                    "field", "action", "old_value_json", "new_value_json",
                    "reason", "evidence_refs", "confidence",
                ],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["decisions", "summary"],
}

BASE_FIELDS = {
    "intent", "customer_goal", "resolution_state", "verification",
    "escalated", "entities", "agent_actions", "agent_commitments",
    "unresolved_items", "next_action",
}
SCALAR_TYPES: dict[str, tuple[type, ...]] = {
    "intent": (str,),
    "customer_goal": (str,),
    "resolution_state": (str,),
    "verification": (str,),
    "escalated": (bool,),
    "next_action": (str,),
}
SCALAR_VALUES: dict[str, set[str]] = {
    "resolution_state": {"RESOLVED", "UNRESOLVED", "PARTIALLY_RESOLVED"},
    "verification": {"COMPLETED", "REQUESTED", "NOT_MENTIONED"},
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_field(field: str) -> str:
    value = field.strip()
    if value.startswith("state."):
        value = value[6:]
    if value in BASE_FIELDS - {"entities"}:
        return f"state.{value}"
    if value.startswith("entities."):
        value = value[len("entities."):]
    if value.startswith("state.entities."):
        value = value[len("state.entities."):]
    if value and "." not in value:
        return f"state.entities.{value}"
    return f"state.{value}"


def _field_parent(packet: dict[str, Any], field: str) -> tuple[Any, str]:
    path = normalize_field(field)
    if path.startswith("state.entities."):
        key = path[len("state.entities."):]
        if not key or "." in key:
            raise ValueError("Only one entity-key path is allowed")
        return packet["state"].setdefault("entities", {}), key
    name = path.removeprefix("state.")
    if name not in BASE_FIELDS - {"entities"}:
        raise ValueError(f"State field is not refinable: {field}")
    return packet["state"], name


def _current_value(packet: dict[str, Any], field: str) -> Any:
    parent, key = _field_parent(packet, field)
    wrapped = parent.get(key)
    return wrapped.get("value") if isinstance(wrapped, dict) and "value" in wrapped else wrapped


def _evidence_field_matches(evidence_field: str, field: str) -> bool:
    path = normalize_field(field)
    if path == "state.agent_actions":
        return evidence_field == path or evidence_field.startswith(path + ".")
    return evidence_field == path


def validate_decision(
    baseline_packet: dict[str, Any],
    decision: dict[str, Any],
    speaker_turn_count: int,
    stage11: Any,
) -> dict[str, Any]:
    """Validate a proposed delta and return it without mutating baseline_packet."""
    field = normalize_field(str(decision.get("field", "")))
    old_value_json = decision.get("old_value_json")
    new_value_json = decision.get("new_value_json")
    evidence_refs = decision.get("evidence_refs")
    confidence = decision.get("confidence")
    action = decision.get("action")
    try:
        parent, key = _field_parent(baseline_packet, field)
        current = _current_value(baseline_packet, field)
        if not isinstance(old_value_json, str) or json.loads(old_value_json) != current:
            raise ValueError("old_value_json does not match the baseline value")
        if action != "PROPOSE_CORRECTION":
            raise ValueError("Only correction proposals can be validated here")
        if not isinstance(new_value_json, str):
            raise ValueError("new_value_json must be a JSON string")
        proposed_value = json.loads(new_value_json)
        if proposed_value == current:
            raise ValueError("Proposed value is identical to the baseline")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not math.isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            raise ValueError("confidence must be a finite number between 0 and 1")
        if (
            not isinstance(evidence_refs, list)
            or not evidence_refs
            or not all(isinstance(item, str) for item in evidence_refs)
        ):
            raise ValueError("A correction requires existing evidence_refs")
        if len(set(evidence_refs)) != len(evidence_refs):
            raise ValueError("Duplicate evidence references are not allowed")

        evidence_by_id = {
            str(item.get("evidence_id")): item
            for item in baseline_packet.get("evidence", [])
            if isinstance(item, dict) and item.get("evidence_id")
        }
        source_turns = [
            item for item in (
                baseline_packet.get("interaction", {}).get("speaker_turns", [])
            ) if isinstance(item, dict)
        ]
        if not source_turns:
            source_turns = [None] * speaker_turn_count
        validated_evidence = []
        for evidence_id in evidence_refs:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                raise ValueError(f"Evidence ID does not exist: {evidence_id}")
            if not _evidence_field_matches(str(evidence.get("field", "")), field):
                raise ValueError(
                    f"Evidence {evidence_id} is not associated with {field}"
                )
            turn_ids = evidence.get("turn_ids")
            if (
                not isinstance(turn_ids, list)
                or not turn_ids
                or any(
                    isinstance(turn_id, bool)
                    or not isinstance(turn_id, int)
                    or turn_id < 1
                    or turn_id > speaker_turn_count
                    for turn_id in turn_ids
                )
            ):
                raise ValueError(
                    f"Evidence {evidence_id} does not point to a valid source turn"
                )
            validated_evidence.append(evidence)

        state_field = field.removeprefix("state.")
        if state_field in SCALAR_TYPES:
            if not isinstance(proposed_value, SCALAR_TYPES[state_field]):
                raise ValueError(f"Invalid value type for {field}")
            allowed_values = SCALAR_VALUES.get(state_field)
            if (
                allowed_values is not None
                and proposed_value not in allowed_values
            ):
                raise ValueError(
                    f"Value is outside the existing {state_field} label set"
                )
        elif state_field.startswith("entities."):
            if not isinstance(proposed_value, (str, int, float, bool)):
                raise ValueError("Entity value must be a scalar")
            if isinstance(proposed_value, str) and not proposed_value.strip():
                raise ValueError("Entity value cannot be empty")
        elif state_field in {"agent_actions", "agent_commitments", "unresolved_items"}:
            if not isinstance(proposed_value, list):
                raise ValueError(f"{field} must be a JSON array")
            for item in proposed_value:
                if not isinstance(item, dict):
                    raise ValueError(f"Every {field} item must be an object")
                item_refs = item.get("evidence_ids")
                if (
                    not isinstance(item_refs, list)
                    or not item_refs
                    or not set(item_refs).issubset(set(evidence_refs))
                ):
                    raise ValueError(
                        f"Every proposed {field} item must cite validated evidence_refs"
                    )
                if state_field == "agent_actions" and not all(
                    isinstance(item.get(name), str)
                    for name in ("dialogue_act", "text")
                ):
                    raise ValueError("Agent actions require dialogue_act and text")
                if state_field == "agent_commitments" and not all(
                    isinstance(item.get(name), str)
                    for name in ("text", "status")
                ):
                    raise ValueError("Commitments require text and status")
                if state_field == "unresolved_items" and not isinstance(
                    item.get("item"), str
                ):
                    raise ValueError("Unresolved items require an item string")
        elif state_field == "entities":
            raise ValueError("Refine entities one key at a time")
        else:
            raise ValueError(f"Unsupported field: {field}")

        candidate = copy.deepcopy(baseline_packet)
        candidate_parent, candidate_key = _field_parent(candidate, field)
        previous = candidate_parent.get(candidate_key)
        if isinstance(previous, dict) and "value" in previous:
            previous["value"] = proposed_value
            previous["confidence"] = float(confidence)
            previous["evidence_ids"] = list(evidence_refs)
            previous["source_type"] = "agentic_refinement"
            previous["status"] = "PREDICTED"
        elif state_field.startswith("entities."):
            candidate_parent[candidate_key] = {
                "value": proposed_value,
                "confidence": float(confidence),
                "evidence_ids": list(evidence_refs),
                "source_type": "agentic_refinement",
                "status": "PREDICTED",
            }
        else:
            candidate_parent[candidate_key] = proposed_value
        structural_errors = stage11.validate_packet(candidate)
        if structural_errors:
            raise ValueError(
                "Refined packet fails RECAP-Voice schema validation: "
                + "; ".join(structural_errors)
            )
        return {
            "status": "VALIDATED",
            "field": field,
            "old_value": current,
            "new_value": proposed_value,
            "evidence_refs": list(evidence_refs),
            "evidence_reference_valid": True,
            "source_turns_valid": True,
            "field_scope_valid": True,
            "schema_valid": True,
            "semantic_entailment": "NOT_VALIDATED",
            "candidate_packet": candidate,
            "rejection_reason": None,
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "REJECTED",
            "field": field,
            "old_value": None,
            "new_value": None,
            "evidence_refs": (
                evidence_refs if isinstance(evidence_refs, list) else []
            ),
            "evidence_reference_valid": False,
            "source_turns_valid": False,
            "field_scope_valid": False,
            "schema_valid": False,
            "semantic_entailment": "NOT_VALIDATED",
            "candidate_packet": None,
            "rejection_reason": str(exc),
        }


def apply_decisions(
    baseline_packet: dict[str, Any],
    decisions: list[dict[str, Any]],
    speaker_turn_count: int,
    stage11: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply only individually validated field deltas to a deep copy."""
    final_packet = copy.deepcopy(baseline_packet)
    validated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_fields: set[str] = set()
    for raw in decisions:
        if not isinstance(raw, dict):
            rejected.append({"decision": raw, "reason": "Decision is not an object"})
            continue
        field = normalize_field(str(raw.get("field", "")))
        if field in seen_fields:
            rejected.append({
                "decision": raw,
                "field": field,
                "reason": "Duplicate decision for state field",
            })
            continue
        seen_fields.add(field)
        if raw.get("action") == "KEEP_BASELINE":
            try:
                _field_parent(baseline_packet, field)
                if json.loads(raw.get("old_value_json", "")) != _current_value(
                    baseline_packet, field
                ):
                    raise ValueError("old_value_json does not match the baseline value")
                if json.loads(raw.get("new_value_json", "")) is not None:
                    raise ValueError("KEEP_BASELINE requires new_value_json=null")
                confidence = raw.get("confidence")
                if (
                    not isinstance(confidence, (int, float))
                    or isinstance(confidence, bool)
                    or not math.isfinite(float(confidence))
                    or not 0 <= float(confidence) <= 1
                ):
                    raise ValueError(
                        "confidence must be a finite number between 0 and 1"
                    )
                if not isinstance(raw.get("reason"), str):
                    raise ValueError("Decision reason must be a string")
                if not isinstance(raw.get("evidence_refs"), list) or not all(
                    isinstance(ref, str) for ref in raw["evidence_refs"]
                ):
                    raise ValueError("evidence_refs must be an array of strings")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                rejected.append({
                    **raw,
                    "field": field,
                    "status": "REJECTED",
                    "applied": False,
                    "rejection_reason": str(exc),
                })
            else:
                validated.append({
                    **raw,
                    "field": field,
                    "status": "KEEP_BASELINE",
                    "applied": False,
                })
            continue
        result = validate_decision(
            baseline_packet, raw, speaker_turn_count, stage11
        )
        if result["status"] != "VALIDATED":
            rejected.append({
                **raw,
                "field": field,
                "status": "REJECTED",
                "applied": False,
                "rejection_reason": result["rejection_reason"],
            })
            continue
        candidate_packet = result.pop("candidate_packet")
        candidate_parent, candidate_key = _field_parent(candidate_packet, field)
        final_parent, final_key = _field_parent(final_packet, field)
        final_parent[final_key] = copy.deepcopy(candidate_parent[candidate_key])
        combined_errors = stage11.validate_packet(final_packet)
        if combined_errors:
            final_parent[final_key] = copy.deepcopy(
                _field_parent(baseline_packet, field)[0].get(
                    _field_parent(baseline_packet, field)[1]
                )
            )
            rejected.append({
                **raw,
                "field": field,
                "status": "REJECTED",
                "applied": False,
                "rejection_reason": (
                    "Combined refined packet fails RECAP-Voice schema "
                    "validation: " + "; ".join(combined_errors)
                ),
            })
            continue
        validated.append({
            **raw,
            "field": field,
            "status": "VALIDATED",
            "applied": True,
            "validation": result,
        })
    return final_packet, validated, rejected


def tool_schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        },
        "strict": True,
    }


COMMON = {
    "conversation_id": {"type": "string"},
    "reason": {"type": "string"},
}
COMMON_REQUIRED = ["conversation_id", "reason"]
REFINEMENT_TOOLS = [
    tool_schema(
        "get_baseline_state",
        "Read the immutable baseline state and its uncertainty metadata.",
        COMMON,
        COMMON_REQUIRED,
    ),
    tool_schema(
        "get_evidence",
        "Retrieve existing evidence by ID or state field; cannot create evidence.",
        {
            **COMMON,
            "evidence_id": {"type": ["string", "null"]},
            "state_field": {"type": ["string", "null"]},
        },
        ["conversation_id", "reason", "evidence_id", "state_field"],
    ),
    tool_schema(
        "search_transcript",
        "Return only matching speaker-aware transcript turns.",
        {**COMMON, "query": {"type": "string"}},
        ["conversation_id", "reason", "query"],
    ),
    tool_schema(
        "get_speaker_aware_transcript",
        "Read the existing transcript with speaker roles and timestamps.",
        COMMON,
        COMMON_REQUIRED,
    ),
    tool_schema(
        "get_acoustic_context",
        "Read existing operational timing, pause, or overlap measurements only.",
        COMMON,
        COMMON_REQUIRED,
    ),
    tool_schema(
        "get_unresolved_items",
        "Read baseline unresolved items and commitments.",
        COMMON,
        COMMON_REQUIRED,
    ),
    tool_schema(
        "get_commitments",
        "Read existing commitments without changing them.",
        COMMON,
        COMMON_REQUIRED,
    ),
    tool_schema(
        "get_next_action_context",
        "Read the baseline next action and related handoff context.",
        COMMON,
        COMMON_REQUIRED,
    ),
    tool_schema(
        "validate_evidence",
        "Check structural validity of existing evidence references; not semantic entailment.",
        {
            **COMMON,
            "state_field": {"type": "string"},
            "claim_value_json": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
        },
        ["conversation_id", "reason", "state_field", "claim_value_json", "evidence_refs"],
    ),
    tool_schema(
        "validate_state_change",
        "Non-mutating preview of field scope, evidence references, source turns, and packet schema.",
        {
            **COMMON,
            "field": {"type": "string"},
            "old_value_json": {"type": "string"},
            "new_value_json": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
        [
            "conversation_id", "reason", "field", "old_value_json",
            "new_value_json", "evidence_refs", "confidence",
        ],
    ),
]


class RefinementOrchestrator:
    def __init__(
        self,
        config: Any,
        tools: Any,
        stage11: Any,
        client: Any,
        max_tool_rounds: int = 4,
        max_tool_calls: int = 10,
    ) -> None:
        self.config = config
        self.tools = tools
        self.stage11 = stage11
        self.client = client
        self.max_tool_rounds = max_tool_rounds
        self.max_tool_calls = max_tool_calls
        self.instructions = PROMPT_PATH.read_text(encoding="utf-8")
        self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.api_calls = 0
        self.api_attempts = 0
        self.api_latency_seconds = 0.0
        self.retry_count = 0
        self.temperature_supported: bool | None = None
        self.errors: list[dict[str, Any]] = []
        self.tool_events: list[dict[str, Any]] = []
        self.tool_counts: dict[str, int] = {}

    def _request(self, **kwargs: Any) -> Any:
        request = dict(kwargs)
        tool_choice = request.pop("tool_choice", "auto")
        if self.temperature_supported is not False:
            request["temperature"] = self.config.temperature
        for attempt in range(self.config.max_retries + 1):
            started = time.perf_counter()
            self.api_attempts += 1
            try:
                response = self.client.responses.create(
                    model=self.config.model,
                    instructions=self.instructions,
                    tools=REFINEMENT_TOOLS,
                    tool_choice=tool_choice,
                    parallel_tool_calls=False,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "state_refinement_decisions",
                            "strict": True,
                            "schema": DECISION_SCHEMA,
                        }
                    },
                    **request,
                )
                self.api_calls += 1
                if "temperature" in request:
                    self.temperature_supported = True
                self.api_latency_seconds += time.perf_counter() - started
                usage = getattr(response, "usage", None)
                for key in self.usage:
                    value = getattr(usage, key, None) if usage else None
                    if value is not None:
                        self.usage[key] += int(value)
                return response
            except Exception as exc:
                self.api_latency_seconds += time.perf_counter() - started
                self.errors.append({
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "attempt": attempt + 1,
                })
                message = str(exc).casefold()
                if (
                    "temperature" in request
                    and "temperature" in message
                    and "not supported" in message
                    and attempt < self.config.max_retries
                ):
                    request.pop("temperature")
                    self.temperature_supported = False
                    self.retry_count += 1
                    continue
                if attempt >= self.config.max_retries:
                    raise
                status = getattr(exc, "status_code", None)
                retryable = (
                    status == 429
                    or isinstance(status, int) and status >= 500
                    or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
                )
                if not retryable:
                    raise
                self.retry_count += 1
                time.sleep(min(2 ** attempt, 4))
        raise RuntimeError("LLM request retries exhausted")

    @staticmethod
    def _calls(response: Any) -> list[Any]:
        return [
            item for item in getattr(response, "output", []) or []
            if getattr(item, "type", None) == "function_call"
        ]

    def _initial_payload(self) -> dict[str, Any]:
        context = self.tools.initial_context()
        return {
            "conversation_id": self.tools.conversation_id,
            "conversation_metadata": self.tools.packet.get("conversation", {}),
            "baseline_state": context.get("initial_state", {}),
            "speaker_resolution": context.get("speaker_resolution", {}),
            "potentially_uncertain_fields": context.get(
                "potentially_uncertain_fields", []
            ),
            "available_resources": context.get("available_resources", {}),
            "baseline_uncertainty": self.tools.packet.get("uncertainty", {}),
            "policy": (
                "Propose only evidence-grounded field deltas. State/evidence "
                "are unchanged unless deterministic validation accepts a delta."
            ),
        }

    def _dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if args.get("conversation_id") != self.tools.conversation_id:
            raise ValueError("Tool conversation_id must match current conversation")
        if name == "get_baseline_state":
            return {
                "conversation_id": self.tools.conversation_id,
                "state": self.tools.packet.get("state", {}),
                "uncertainty": self.tools.packet.get("uncertainty", {}),
            }
        if name == "get_commitments":
            return {
                "conversation_id": self.tools.conversation_id,
                "agent_commitments": self.tools.packet.get("state", {}).get(
                    "agent_commitments", []
                ),
            }
        if name == "get_next_action_context":
            return self.tools.invoke(
                "get_next_best_action_context",
                {
                    "conversation_id": self.tools.conversation_id,
                    "reason": args.get("reason"),
                },
            )
        if name == "validate_evidence":
            value = json.loads(args["claim_value_json"])
            claim = next(
                (
                    item for item in self.tools._field_claims().values()
                    if item.get("field") == normalize_field(args["state_field"])
                ),
                None,
            )
            if claim is None:
                return {"valid": False, "error": "Unknown state claim"}
            current_field = normalize_field(args["state_field"])
            current_value = _current_value(self.tools.packet, current_field)
            if value != current_value:
                return {"valid": False, "error": "Value differs from baseline"}
            records = {
                str(item.get("evidence_id")): item
                for item in self.tools.evidence
            }
            refs = args.get("evidence_refs", [])
            matches = []
            for ref in refs:
                record = records.get(ref)
                matches.append(bool(
                    record
                    and _evidence_field_matches(
                        str(record.get("field", "")), current_field
                    )
                ))
            return {
                "valid": bool(refs) and all(matches),
                "evidence_reference_validity": matches,
                "scope": "structural field/reference check only",
            }
        if name == "validate_state_change":
            candidate = {
                "field": args["field"],
                "action": "PROPOSE_CORRECTION",
                "old_value_json": args["old_value_json"],
                "new_value_json": args["new_value_json"],
                "evidence_refs": args["evidence_refs"],
                "confidence": args["confidence"],
            }
            result = validate_decision(
                self.tools.packet,
                candidate,
                len(self.tools.speaker_turns),
                self.stage11,
            )
            result.pop("candidate_packet", None)
            return result
        allowed = {
            "get_evidence",
            "search_transcript",
            "get_speaker_aware_transcript",
            "get_acoustic_context",
            "get_unresolved_items",
        }
        if name not in allowed:
            raise ValueError(f"Unknown tool: {name}")
        return self.tools.invoke(name, args)

    @staticmethod
    def _result_summary(output: Any) -> str:
        if not isinstance(output, dict):
            return "Returned a tool result"
        if output.get("error"):
            return str(output["error"])[:300]
        if "evidence" in output:
            return f"Returned {len(output['evidence'])} evidence records"
        if "matches" in output:
            return f"Returned {len(output['matches'])} transcript matches"
        if "speaker_turns" in output:
            return f"Returned {len(output['speaker_turns'])} speaker-aware turns"
        if "features" in output:
            return "Returned operational acoustic features"
        if "state" in output:
            return "Returned baseline state"
        if "not_found" in output and output["not_found"]:
            return "No matching evidence found"
        if "status" in output:
            return f"Validation status: {output['status']}"
        return "Returned a tool result"

    def run(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "FAILED",
            "summary": "",
            "decisions": [],
            "validated_changes": [],
            "rejected_changes": [],
            "final_state": copy.deepcopy(self.tools.packet.get("state", {})),
        }
        try:
            response = self._request(input=json.dumps(
                self._initial_payload(), ensure_ascii=False
            ))
            rounds = 0
            call_count = 0
            budget_final_requested = False
            while True:
                calls = self._calls(response)
                if not calls:
                    text = (getattr(response, "output_text", "") or "").strip()
                    parsed = json.loads(text)
                    if (
                        not isinstance(parsed, dict)
                        or set(parsed) != {"decisions", "summary"}
                        or not isinstance(parsed["decisions"], list)
                    ):
                        raise ValueError("Malformed refinement decision output")
                    if len(parsed["decisions"]) > len(BASE_FIELDS) + 32:
                        raise ValueError("Too many field decisions")
                    final_packet, accepted, rejected = apply_decisions(
                        self.tools.packet,
                        parsed["decisions"],
                        len(self.tools.speaker_turns),
                        self.stage11,
                    )
                    self.tools.stage11.validate_packet(final_packet)
                    result.update({
                        "status": "COMPLETED",
                        "summary": str(parsed["summary"]),
                        "decisions": parsed["decisions"],
                        "validated_changes": accepted,
                        "rejected_changes": rejected,
                        "final_state": final_packet["state"],
                    })
                    break
                if (
                    rounds >= self.max_tool_rounds
                    or call_count + len(calls) > self.max_tool_calls
                ):
                    if budget_final_requested:
                        raise RuntimeError(
                            "Model continued requesting tools after budget exhaustion"
                        )
                    budget_final_requested = True
                    outputs = []
                    for call in calls:
                        name = str(getattr(call, "name", ""))
                        call_id = str(getattr(call, "call_id", ""))
                        raw_args = getattr(call, "arguments", "{}")
                        try:
                            args = json.loads(raw_args)
                        except (TypeError, ValueError):
                            args = {}
                        if not isinstance(args, dict):
                            args = {}
                        output = {
                            "error": (
                                "Tool budget exhausted. Use only retrieved "
                                "context and return KEEP_BASELINE where a "
                                "correction cannot be established."
                            ),
                            "status": "BUDGET_EXHAUSTED",
                        }
                        self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
                        call_count += 1
                        self.tool_events.append({
                            "round": rounds + 1,
                            "tool": name,
                            "tool_input": args if args else raw_args,
                            "reason": args.get("reason"),
                            "result_summary": output["error"],
                            "tool_output": output,
                            "status": "budget_exhausted",
                            "latency_seconds": 0.0,
                            "decision_changed": None,
                            "decision_change_assessment": (
                                "Not causally attributable; requested context "
                                "was not retrieved."
                            ),
                        })
                        outputs.append({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(output),
                        })
                    response = self._request(
                        tool_choice="none",
                        previous_response_id=response.id,
                        input=outputs,
                    )
                    continue
                rounds += 1
                outputs = []
                for call in calls:
                    name = str(getattr(call, "name", ""))
                    call_id = str(getattr(call, "call_id", ""))
                    raw_args = getattr(call, "arguments", "{}")
                    started = time.perf_counter()
                    try:
                        args = json.loads(raw_args)
                        if not isinstance(args, dict):
                            raise ValueError("Tool arguments must be a JSON object")
                        output = self._dispatch(name, args)
                        status = "success" if "error" not in output else "error"
                    except Exception as exc:
                        args = {}
                        output = {
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:1000],
                        }
                        status = "error"
                    elapsed = round(time.perf_counter() - started, 6)
                    self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
                    call_count += 1
                    self.tool_events.append({
                        "round": rounds,
                        "tool": name,
                        "tool_input": args if args else raw_args,
                        "reason": args.get("reason") if isinstance(args, dict) else None,
                        "result_summary": self._result_summary(output),
                        "tool_output": output,
                        "status": status,
                        "latency_seconds": elapsed,
                        "decision_changed": None,
                    })
                    outputs.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(output, ensure_ascii=False),
                    })
                response = self._request(
                    previous_response_id=response.id,
                    input=outputs,
                )
            final_decisions = {
                normalize_field(str(item.get("field", ""))): item
                for item in result.get("decisions", [])
                if isinstance(item, dict)
            }
            for event in self.tool_events:
                if event["tool"] != "validate_state_change":
                    event["decision_change_assessment"] = (
                        "Not causally attributable from a single observed trace"
                    )
                    continue
                candidate_field = normalize_field(
                    str(event.get("tool_input", {}).get("field", ""))
                )
                final_decision = final_decisions.get(candidate_field)
                rejected_by_validator = (
                    event.get("tool_output", {}).get("status") == "REJECTED"
                )
                kept_after_rejection = (
                    rejected_by_validator
                    and (
                        final_decision is None
                        or final_decision.get("action") == "KEEP_BASELINE"
                    )
                )
                event["decision_changed"] = bool(kept_after_rejection)
                event["decision_change_assessment"] = (
                    "Observed proxy: validator rejected the candidate and the "
                    "final decision kept the baseline; causal attribution is not established."
                    if kept_after_rejection
                    else "No validator-caused final-decision change observed"
                )
            result["tool_rounds"] = rounds
            result["tool_call_count"] = call_count
        except Exception as exc:
            result["error"] = {
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
            self.errors.append(result["error"])
        result.update({
            "api_calls": self.api_calls,
            "api_attempts": self.api_attempts,
            "retry_count": self.retry_count,
            "temperature_parameter_supported": self.temperature_supported,
            "usage": self.usage,
            "api_latency_seconds": round(self.api_latency_seconds, 6),
            "tool_call_count": self.tool_counts and sum(self.tool_counts.values()) or 0,
            "tool_call_counts": self.tool_counts,
            "tool_events": self.tool_events,
            "errors": self.errors,
            "retrieved_context_bytes": self.tools.retrieved_context_bytes,
            "available_context_bytes": self.tools.available_context_bytes,
            "context_retrieval_percent": (
                round(
                    100 * self.tools.retrieved_context_bytes
                    / self.tools.available_context_bytes,
                    4,
                )
                if self.tools.available_context_bytes else 0.0
            ),
        })
        if result["status"] != "COMPLETED":
            result["final_state"] = copy.deepcopy(
                self.tools.packet.get("state", {})
            )
            result["fallback_used"] = True
        else:
            result["fallback_used"] = False
        return result
