from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from state_refinement import (
    RefinementOrchestrator,
    apply_decisions,
    validate_decision,
)


def packet() -> dict:
    return {
        "schema_name": "RECAP-Voice",
        "schema_version": "1.0.0",
        "packet_type": "recap_voice",
        "conversation": {"conversation_id": "c1"},
        "state": {
            "intent": {
                "value": "PAYMENT_FAILED",
                "confidence": 0.8,
                "evidence_ids": ["E1"],
                "source_type": "rule",
                "status": "PREDICTED",
            },
            "customer_goal": {
                "value": "Resolve the payment issue",
                "confidence": 0.7,
                "evidence_ids": ["E1"],
                "source_type": "asr",
                "status": "PREDICTED",
            },
            "resolution_state": {
                "value": "UNRESOLVED",
                "confidence": 0.6,
                "evidence_ids": ["E2"],
                "source_type": "asr",
                "status": "PREDICTED",
            },
            "escalated": {
                "value": False,
                "confidence": 0.9,
                "evidence_ids": [],
                "source_type": "rule",
                "status": "PREDICTED",
            },
            "verification": {
                "value": "NOT_MENTIONED",
                "confidence": 0.8,
                "evidence_ids": [],
                "source_type": "rule",
                "status": "PREDICTED",
            },
            "entities": {},
            "agent_actions": [],
            "agent_commitments": [],
            "unresolved_items": [],
            "next_action": {
                "value": "Wait for review",
                "evidence_ids": ["E3"],
                "status": "predicted_next_action",
            },
        },
        "evidence": [
            {
                "evidence_id": "E1",
                "field": "state.intent",
                "turn_ids": [1],
                "text": "My card payment was declined.",
            },
            {
                "evidence_id": "E2",
                "field": "state.resolution_state",
                "turn_ids": [2],
                "text": "The fee is resolved, but the payment is still being reviewed.",
            },
            {
                "evidence_id": "E3",
                "field": "state.next_action",
                "turn_ids": [2],
                "text": "Please wait for the review.",
            },
        ],
        "interaction": {},
        "repair_events": [],
        "uncertainty": {},
        "provenance": {},
    }


class Stage11:
    def __init__(self, errors=None):
        self.errors = errors or []

    def validate_packet(self, _packet):
        return list(self.errors)


class BudgetResponses:
    def __init__(self):
        self.calls = 0
        self.tool_choices = []

    def create(self, **kwargs):
        self.calls += 1
        self.tool_choices.append(kwargs.get("tool_choice"))
        if self.calls <= 2:
            return SimpleNamespace(
                id=f"response_{self.calls}",
                output=[SimpleNamespace(
                    type="function_call",
                    name="get_unresolved_items",
                    call_id=f"call_{self.calls}",
                    arguments=json.dumps({
                        "conversation_id": "c1",
                        "reason": "Check unresolved work",
                    }),
                )],
                output_text="",
                usage=None,
            )
        return SimpleNamespace(
            id="final",
            output=[],
            output_text=json.dumps({"decisions": [], "summary": "Keep baseline."}),
            usage=None,
        )


def correction(**updates):
    value = {
        "field": "state.resolution_state",
        "action": "PROPOSE_CORRECTION",
        "old_value_json": json.dumps("UNRESOLVED"),
        "new_value_json": json.dumps("PARTIALLY_RESOLVED"),
        "reason": "The cited response states that one issue is resolved while payment review remains.",
        "evidence_refs": ["E2"],
        "confidence": 0.82,
    }
    value.update(updates)
    return value


class StateDeltaTests(unittest.TestCase):
    def test_valid_evidence_grounded_correction_is_applied(self):
        original = packet()
        final, accepted, rejected = apply_decisions(
            original, [correction()], 2, Stage11()
        )
        self.assertEqual(
            final["state"]["resolution_state"]["value"], "PARTIALLY_RESOLVED"
        )
        self.assertEqual(final["state"]["resolution_state"]["evidence_ids"], ["E2"])
        self.assertEqual(original["state"]["resolution_state"]["value"], "UNRESOLVED")
        self.assertTrue(accepted[0]["applied"])
        self.assertEqual(rejected, [])

    def test_invalid_evidence_reference_is_rejected(self):
        _final, _accepted, rejected = apply_decisions(
            packet(), [correction(evidence_refs=["FAKE"])], 2, Stage11()
        )
        self.assertEqual(rejected[0]["status"], "REJECTED")
        self.assertIn("does not exist", rejected[0]["rejection_reason"])

    def test_wrong_field_evidence_is_rejected_as_unsupported(self):
        _final, _accepted, rejected = apply_decisions(
            packet(), [correction(evidence_refs=["E1"])], 2, Stage11()
        )
        self.assertIn("not associated", rejected[0]["rejection_reason"])

    def test_keep_baseline_does_not_change_any_field(self):
        original = packet()
        final, accepted, rejected = apply_decisions(
            original,
            [{
                "field": "state.resolution_state",
                "action": "KEEP_BASELINE",
                "old_value_json": json.dumps("UNRESOLVED"),
                "new_value_json": "null",
                "reason": "Available evidence does not establish a correction.",
                "evidence_refs": [],
                "confidence": 0.5,
            }],
            2,
            Stage11(),
        )
        self.assertEqual(final, original)
        self.assertEqual(accepted[0]["status"], "KEEP_BASELINE")
        self.assertEqual(rejected, [])

    def test_schema_validation_rejects_delta(self):
        _final, _accepted, rejected = apply_decisions(
            packet(), [correction()], 2, Stage11(["invalid state schema"])
        )
        self.assertIn("schema validation", rejected[0]["rejection_reason"])

    def test_invalid_value_type_is_rejected(self):
        _final, _accepted, rejected = apply_decisions(
            packet(),
            [correction(new_value_json=json.dumps(True))],
            2,
            Stage11(),
        )
        self.assertIn("Invalid value type", rejected[0]["rejection_reason"])

    def test_out_of_schema_verification_label_is_rejected(self):
        baseline = packet()
        baseline["evidence"].append({
            "evidence_id": "E4",
            "field": "state.verification",
            "turn_ids": [1],
            "text": "Please verify.",
        })
        _final, _accepted, rejected = apply_decisions(
            baseline,
            [correction(
                field="state.verification",
                old_value_json=json.dumps("NOT_MENTIONED"),
                new_value_json=json.dumps("PENDING"),
                evidence_refs=["E4"],
            )],
            2,
            Stage11(),
        )
        self.assertIn("existing verification label set", rejected[0]["rejection_reason"])

    def test_unchanged_fields_are_preserved_exactly(self):
        original = packet()
        final, _accepted, _rejected = apply_decisions(
            original, [correction()], 2, Stage11()
        )
        for field in (
            "intent", "customer_goal", "escalated", "verification",
            "entities", "agent_actions", "agent_commitments",
            "unresolved_items", "next_action",
        ):
            self.assertEqual(final["state"][field], original["state"][field])

    def test_duplicate_decision_is_rejected(self):
        _final, _accepted, rejected = apply_decisions(
            packet(), [correction(), correction()], 2, Stage11()
        )
        self.assertEqual(len(rejected), 1)
        self.assertIn("Duplicate", rejected[0]["reason"])

    def test_fallback_after_agent_api_failure_preserves_baseline(self):
        class BadResponses:
            def create(self, **_kwargs):
                raise RuntimeError("API unavailable")

        baseline = packet()
        fake_tools = SimpleNamespace(
            conversation_id="c1",
            packet=baseline,
            initial_context=lambda: {
                "conversation_id": "c1",
                "initial_state": baseline["state"],
            },
            retrieved_context_bytes=1,
            available_context_bytes=10,
            speaker_turns=[{}, {}],
            evidence=baseline["evidence"],
        )
        config = SimpleNamespace(
            model="test",
            temperature=0,
            max_retries=0,
        )
        result = RefinementOrchestrator(
            config,
            fake_tools,
            Stage11(),
            SimpleNamespace(responses=BadResponses()),
        ).run()
        self.assertEqual(result["status"], "FAILED")
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["final_state"], baseline["state"])

    def test_tool_budget_exhaustion_requests_a_final_conservative_decision(self):
        baseline = packet()
        fake_tools = SimpleNamespace(
            conversation_id="c1",
            packet=baseline,
            initial_context=lambda: {
                "conversation_id": "c1",
                "initial_state": baseline["state"],
            },
            retrieved_context_bytes=1,
            available_context_bytes=10,
            speaker_turns=[{}, {}],
            evidence=baseline["evidence"],
            invoke=lambda *_args: {"unresolved_items": []},
            stage11=Stage11(),
        )
        responses = BudgetResponses()
        agent = RefinementOrchestrator(
            SimpleNamespace(model="test", temperature=0, max_retries=0),
            fake_tools,
            Stage11(),
            SimpleNamespace(responses=responses),
            max_tool_rounds=1,
            max_tool_calls=3,
        )
        result = agent.run()
        self.assertEqual(result["status"], "COMPLETED")
        self.assertFalse(result["fallback_used"])
        self.assertIn("none", responses.tool_choices)
        self.assertEqual(
            [event["status"] for event in result["tool_events"]],
            ["success", "budget_exhausted"],
        )

    def test_delta_application_is_deterministic(self):
        decisions = [correction()]
        first = apply_decisions(packet(), decisions, 2, Stage11())
        second = apply_decisions(packet(), decisions, 2, Stage11())
        self.assertEqual(first, second)

    def test_multiple_valid_deltas_are_applied_cumulatively(self):
        second = correction(
            field="state.intent",
            old_value_json=json.dumps("PAYMENT_FAILED"),
            new_value_json=json.dumps("PAYMENT_DECLINED"),
            evidence_refs=["E1"],
        )
        final, accepted, rejected = apply_decisions(
            packet(), [correction(), second], 2, Stage11()
        )
        self.assertEqual(
            final["state"]["resolution_state"]["value"], "PARTIALLY_RESOLVED"
        )
        self.assertEqual(final["state"]["intent"]["value"], "PAYMENT_DECLINED")
        self.assertEqual(sum(bool(item.get("applied")) for item in accepted), 2)
        self.assertEqual(rejected, [])

    def test_malformed_keep_decision_is_rejected(self):
        _final, _accepted, rejected = apply_decisions(
            packet(),
            [{
                "field": "state.intent",
                "action": "KEEP_BASELINE",
                "old_value_json": "not-json",
                "new_value_json": "null",
                "reason": "keep",
                "evidence_refs": [],
                "confidence": 0,
            }],
            2,
            Stage11(),
        )
        self.assertEqual(rejected[0]["status"], "REJECTED")


if __name__ == "__main__":
    unittest.main()
