# Agentic evidence-grounded state refinement

This is the project's final agentic experiment. It extends the prior
`agentic_context_orchestration` pilot by allowing constrained, field-level
state corrections after selective read-only retrieval. It does not create a
generic multi-agent system and does not alter baseline pipeline stages or
previous experiment outputs.

## Research design

The paired conditions use the same hard-case held-out test calls:

- **Baseline:** existing deterministic RECAP-Voice prediction packet.
- **Agentic refinement:** the same packet, reviewed with selective context
  tools, followed by deterministic validation and application of individual
  field deltas. A failed agent call preserves the complete baseline packet.

The agent never writes a packet. It returns `KEEP_BASELINE` or
`PROPOSE_CORRECTION` decisions. The validator checks the original value,
permitted state-field path, JSON value type, existing evidence ID, field
association, source-turn index, confidence range, and RECAP-Voice packet
structure. Only accepted deltas are applied to a deep copy of the baseline;
all other fields remain unchanged.

Resolution and verification labels are restricted to values emitted by the
existing state extractor. An out-of-schema label such as `PENDING` is rejected
even if a model proposes it.

Evidence checks establish reference existence, same-packet scope,
field-association, source-turn-index validity, and packet structure. They do
**not** establish semantic entailment. The evaluator therefore reports
semantic support and recall among semantically correctable errors as
`NOT AVAILABLE`. Gold labels are used only during evaluation and never appear
in the agent prompt or tools.

## Existing inputs and reused modules

The default run selects the same fixed-seed 10 conversations as the previous
agentic pilot:

`hard_0004, hard_0014, hard_0015, hard_0018, hard_0029, hard_0032, hard_0036, hard_0082, hard_0087, hard_0095`

Inputs are read-only:

- `data/experiments/voice_heldout_dataset.json` for test split membership.
- `robustness/hard_cases_pipeline/seed_42/recap_predictions/` for baseline
  state/evidence packets.
- `transcripts_whisperx/` and `wordlevel_speaker_transcripts/` for selective
  transcript retrieval.
- `gold_state_voice_heldout/` for evaluation only.
- `robustness/agentic_context_orchestration/seed_42_test10_run4/baseline/acoustic_features/`
  for existing acoustic-feature outputs.
- `wordlevel_speaker_summary.csv` for the existing role-accuracy diagnostic.

The implementation reuses the prior experiment's `ContextTools`,
`LLMConfig`, OpenAI-compatible client setup, and tool-safe context accounting;
the existing stages 11, 12, 13, 15, and 16 provide packet validation, state
metrics, evidence-reference verification, capsule generation, and capsule
metrics. No dependency or baseline/evaluation script is changed.

## Agent tools

All tools are conversation-scoped. `get_evidence`, transcript search,
speaker-aware transcript retrieval, and acoustic retrieval reuse existing
artifacts. `get_baseline_state`, `get_commitments`, unresolved-item and
next-action accessors are read-only. `validate_evidence` performs structural
reference checks; `validate_state_change` is a non-mutating delta-validation
preview. The same deterministic validator is always run on final proposed
changes, whether or not the model calls the preview tool.

The agent is limited to 4 tool rounds and 10 tool calls by default. Tool events
record the round, input, reason, output/result summary, latency, and whether a
validator rejection was followed by a keep decision. For retrieval calls,
whether the result caused a decision cannot be attributed causally from a
single run; those entries explicitly say so. If the budget is exhausted, the
runner blocks further retrieval and requests a conservative final decision; a
failed final response falls back to baseline.

Acoustic features are available only as operational context (timing, pauses,
overlap, and related measures). Acoustic information is not accepted as
evidence for transcript-derived state changes and cannot support emotion,
intent, honesty, deception, personality, or mental-state claims.

## Run

From the repository root:

```bash
python voice_phase2_dataset/experiments/agentic_state_refinement/run_experiment.py \
  --limit 10 --seed 42
```

The runner loads `LLM_API_KEY`/`OPENAI_API_KEY` and model settings from the
environment or `voice_phase2_dataset/.env`. `LLM_PROVIDER`, `LLM_MODEL`,
`LLM_TEMPERATURE`, and `LLM_BASE_URL` follow the prior experiment's
configuration. The default acoustic path points to the existing prior-run
features; `--acoustic-dir` can select another complete feature directory.
Use `--conversation-id hard_0004` to run a single eligible test case or
`--limit 0` to select all eligible hard-case test calls. Fresh, non-empty
output directories are never overwritten.

Tool limits can be changed explicitly:

```bash
python voice_phase2_dataset/experiments/agentic_state_refinement/run_experiment.py \
  --limit 10 --seed 42 --max-tool-rounds 4 --max-tool-calls 10 \
  --output robustness/agentic_state_refinement/seed_42_final
```

To evaluate a completed run without API calls:

```bash
python voice_phase2_dataset/experiments/agentic_state_refinement/evaluate_experiment.py \
  --run-dir voice_phase2_dataset/robustness/agentic_state_refinement/<run> \
  --gold-dir voice_phase2_dataset/gold_state_voice_heldout \
  --role-summary voice_phase2_dataset/robustness/hard_cases_pipeline/seed_42/wordlevel_speaker_summary.csv \
  --seed 42
```

## Outputs

```text
robustness/agentic_state_refinement/<run>/
├── metadata.json
├── agent_runs.jsonl
├── errors.jsonl
├── field_level_changes.jsonl
├── per_conversation.csv
├── summary.csv
├── summary.json
├── paired_comparison.csv
├── field_level_audit.csv
├── evidence_audit.csv
├── tool_usage.csv
├── baseline_predictions/
├── agentic_predictions/       # decisions and validation outcomes
├── final_refined_predictions/
├── baseline_capsules/
└── final_refined_capsules/
```

`summary.json` includes existing stage-12 metrics, refinement outcomes,
paired bootstrap intervals and Wilcoxon tests when informative, evidence
reference audits, context retrieval, tool/token/latency costs, and limitations.
The baseline context percentage is a full-available-context byte inventory
(100%), not a separate baseline LLM run. Agent retrieval percentage counts
initial state plus retrieved existing context; model prompt/reasoning tokens
and API latency are reported separately.

## Tests

The isolated tests require no network/API key:

```bash
python -m unittest discover \
  -s voice_phase2_dataset/experiments/agentic_state_refinement \
  -p 'test_*.py' -v
```

## Interpretation limits

This is a fixed-seed, 10-conversation hard-case pilot unless run with
`--limit 0`; it is not representative evidence for all 500 source calls.
State quality is scored with existing evaluator definitions. Exact
correctness/preservation and F1 changes are computed from the gold references.
The dataset has no annotation of which incorrect baseline fields were
semantically correctable from their evidence, nor of whether a KEEP decision
was semantically mandatory. These safety/recall claims are reported as
unavailable rather than inferred from structural reference checks.

The agent's field-level edits are evaluated against gold, but a citation that
is valid and field-associated can still fail to entail the proposed value.
Human or independently validated semantic judgments would be needed to claim
semantic grounding. No conclusion that agentic refinement is superior should
be drawn unless paired quality metrics improve without unacceptable regression
or overhead.
