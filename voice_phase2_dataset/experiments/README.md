# Voice Phase 2 experiments

The numbered scripts at the dataset root remain the canonical pipeline. This
directory contains experiment runners and analysis utilities that use those
stages without copying the audio corpus or changing the source labels.

## Speech pipeline environment

Use Python 3.12 for ASR and diarization. These pinned speech dependencies are
separate from the main environment used for TTS and other experiments. On
macOS:

```bash
brew install python@3.12 ffmpeg@7
python3.12 -m venv ~/.venvs/voice-phase2-speech
~/.venvs/voice-phase2-speech/bin/python -m pip install \
  -r requirements_speech.txt
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib"
```

The diarization and robustness scripts load `HF_TOKEN` from
`voice_phase2_dataset/.env` automatically. An `HF_TOKEN` already exported in
the shell takes precedence. The token must have access to the required
pyannote models. Run the robustness pipeline with this environment's Python,
not the project's general `.env` Python; the runner checks Python and key
speech-package versions before starting. Diarization reads WAVs with
`soundfile` and passes in-memory waveforms to pyannote, so TorchCodec's
FFmpeg-backed file decoder is not required for this stage. Pyannote may still
print a TorchCodec warning at startup; it does not prevent this in-memory path
from running.

## Enrich and re-render the existing calls

The current audio is synthetic and scripted. To add scenario-specific incident
context while preserving conversation IDs, scenario, entities, split, outcome,
speaker assignments, and dialogue-act flow, rewrite the existing text metadata:

```bash
python experiments/enrich_conversations.py --in-place --seed 42
```

This regenerates each utterance from its scenario/entity templates, adds a
non-sensitive context sentence to the customer opening and a varied review
statement to the agent acknowledgement, and updates the JSON, JSONL, and
metadata CSV. Re-render all turns and rebuild the merged calls:

```bash
VOICE_MAX_CONVERSATIONS=0 VOICE_OVERWRITE_TURNS=1 \
  python 02_generate_emotional_audio.py
python 03_merge_conversations.py
python 10_build_gold_state.py
```

Then regenerate the single telephone condition and run the clean-vs-telephone
pipeline below. The richer calls are still TTS-generated synthetic speech, not
recordings of real incidents.

## 8 kHz telephone experiment

Generate only the 8 kHz telephone-bandwidth variants. The original WAVs remain
the clean baseline; they are not copied into the experiment directory.

```bash
cd voice_phase2_dataset
python 24_generate_robustness_audio.py --condition telephone_8khz --seed 42
```

Run the clean baseline and 8 kHz audio through the same ASR → two-speaker
diarization → role mapping → DER → state evaluation pipeline. Start with a
small smoke test; omit `--limit` to run the full 500-call corpus.

```bash
~/.venvs/voice-phase2-speech/bin/python experiments/run_robustness_pipeline.py \
  --limit 3 \
  --output robustness/pipeline/seed_42_smoke
```

`HF_TOKEN` from `voice_phase2_dataset/.env` is needed for pyannote. WhisperX
and pyannote outputs are stored under each run/condition; existing run outputs
are never overwritten. Without `--limit`, it processes every call in the
selected split (75 test calls by default); use `--split all` to run all 500
calls.

```bash
python experiments/analyze_robustness.py \
  --run robustness/pipeline/seed_42 \
  --run robustness/pipeline/seed_43 \
  --output robustness/analysis
```

The analysis writes per-call WER, DER, and the evaluator's state-field macro
score (a mean of accuracy and F1 components), paired clean-versus-telephone
bootstrap intervals and Wilcoxon tests (Holm-adjusted), an error-propagation
plot, and an ECE report for predicted state confidence. Telephone conversion
is duration-preserving and stores only the 8 kHz WAVs in
`robustness/audio/telephone_8khz/`.

## Error propagation

After generating clean `wordlevel_speaker_transcripts/`, compare state
extraction with oracle text, ASR text plus oracle speaker roles, and ASR text
plus the predicted language-cue roles:

```bash
python experiments/error_propagation.py --split test --limit 10
```

The script uses only held-out records by default. The gold-role condition is
an oracle diagnostic, never a deployable result. It writes separate state
predictions/evaluations and `error_propagation_metrics.csv`.

## Hard outcomes and voice-held-out evaluation

The source dataset currently has only resolved, non-escalated outcomes. Create
a separate evaluation dataset with unresolved/escalated calls, deterministic
customer paraphrases, and speaker-disjoint train/test partitions:

```bash
python experiments/create_hard_cases.py
```

This writes a held-out dataset plus a separate audio-only subset; it does not
alter the existing 500 records. The default held-out voices are `aiden` and
`ono_anna`; all original records containing either voice are moved out of the
train partition in the new metadata. Render and merge the new calls into
separate directories:

```bash
VOICE_DATASET_FILE=data/experiments/hard_cases_audio.json \
VOICE_TURN_AUDIO_DIR=new_generated_turns/hard_cases \
VOICE_MAX_CONVERSATIONS=100 \
python 02_generate_emotional_audio.py

VOICE_DATASET_FILE=data/experiments/hard_cases_audio.json \
VOICE_TURN_AUDIO_DIR=new_generated_turns/hard_cases \
VOICE_AUDIO_DIR=conversations_audio/hard_cases \
python 03_merge_conversations.py

python 10_build_gold_state.py \
  --metadata data/experiments/voice_heldout_dataset.json \
  --output-dir gold_state_voice_heldout
```

Point ASR/diarization inputs at `conversations_audio/hard_cases`, preserve the
separate `gold_state_voice_heldout/` labels, and pass
`data/experiments/voice_heldout_dataset.json` as the metadata argument to state
extraction. These synthetic examples test controlled outcome/voice shift;
they are not a substitute for a small consented real-speech evaluation set.

## Customer behavior analysis

Compare pre-outcome customer interaction patterns between resolved and
unresolved/escalated conversations, and estimate outcome prediction using
scenario-grouped cross-validation. Generate the hard-case dataset above, then
run:

```bash
python experiments/customer_behavior_analysis.py
```

By default, the script reads `data/experiments/voice_heldout_dataset.json` and
writes `customer_behavior_analysis/per_conversation.csv`,
`group_comparisons.csv`, and `summary.json`. The group comparisons use
Mann–Whitney tests with Holm-adjusted p-values. The logistic-regression
baseline uses only pre-outcome text/turn features and holds out entire
scenarios in `GroupKFold`; it reports standard classification metrics. A
lexical-overlap heuristic estimates issue restatements and is not a semantic
re-ask annotation. Because these data are synthetic and scripted, results are
exploratory and do not establish causal or real-customer behavior.

## LLM and fine-tuned state extractors

Compare zero-shot and train-only few-shot OpenAI structured extraction on the
same held-out transcripts. The script loads `OPENAI_API_KEY` from
`voice_phase2_dataset/.env`; `--limit` defaults to 10 calls to control cost.
Only use approved, appropriately de-identified data with external APIs:

```bash
python experiments/compare_llm_state_extractor.py \
  --dataset data/experiments/voice_heldout_dataset.json \
  --wordlevel-dir wordlevel_speaker_transcripts \
  --gold-dir gold_state_voice_heldout \
  --output llm_state_comparison/seed_42 --limit 10 --seed 42

python experiments/calibration_report.py \
  --gold-dir gold_state_voice_heldout \
  --pred-dir llm_state_comparison/seed_42/zero_shot/predictions \
  --output llm_state_comparison/seed_42/zero_shot/ece.csv

python experiments/calibration_report.py \
  --gold-dir gold_state_voice_heldout \
  --pred-dir llm_state_comparison/seed_42/few_shot/predictions \
  --output llm_state_comparison/seed_42/few_shot/ece.csv
```

For a local fine-tuned `google/flan-t5-small` comparison, train only on
speaker-disjoint training records from the generated held-out metadata, then
evaluate the model on the held-out test records:

```bash
python experiments/train_small_state_model.py \
  --dataset data/experiments/voice_heldout_dataset.json \
  --transcript-dir wordlevel_speaker_transcripts \
  --gold-dir gold_state_voice_heldout \
  --output models/state_extractor_flan_t5_small

python experiments/evaluate_small_state_model.py \
  --dataset data/experiments/voice_heldout_dataset.json \
  --transcript-dir wordlevel_speaker_transcripts \
  --gold-dir gold_state_voice_heldout \
  --model models/state_extractor_flan_t5_small \
  --output llm_state_comparison/seed_42/fine_tuned --limit 10

python experiments/calibration_report.py \
  --gold-dir gold_state_voice_heldout \
  --pred-dir llm_state_comparison/seed_42/fine_tuned/predictions \
  --output llm_state_comparison/seed_42/fine_tuned/ece.csv
```

The small-model confidence is mean generated-token probability and is not
calibrated; ECE is a diagnostic, not calibration itself. Use a separate
validation split to fit a calibration mapping before reporting calibrated
confidence. Repeat each comparison with several seeds and report paired
uncertainty intervals.

## Role mapping and exactly-two-speaker evaluation

`06_run_diarization.py` already constrains calls to exactly two speakers.
`07b_build_wordlevel_speaker_transcripts.py` now infers CUSTOMER/AGENT roles
from agent-specific transcript cues and abstains when the evidence is weak.
Ground-truth overlap mapping is retained only as a separately named evaluation
diagnostic; it no longer supplies `predicted_role` to state extraction. This
role heuristic is a baseline. Compare it with ECAPA speaker embeddings on
pyannote's speech segments:

```bash
python experiments/compare_ecapa_speaker_embeddings.py \
  --audio-dir conversations_audio \
  --diarization-dir diarization \
  --wordlevel-dir wordlevel_speaker_transcripts \
  --output experiments/ecapa_comparison.csv
```

The ECAPA experiment replaces speaker labels by clustering embeddings but
retains pyannote's speech boundaries. Its ARI score uses gold word-role labels
only for evaluation; it is not an end-to-end deployable diarizer or a role
prediction input. The pretrained checkpoint may download from Hugging Face.

## Emotion and human-rating checks

The pretrained speech-emotion check is optional and may download a model:

```bash
python experiments/check_tts_emotion.py --split test --limit 10
```

It compares only coarse classes supported by the checkpoint. Fine-grained
requested styles and intensity require human listening/annotation; classifier
agreement is not proof that the requested emotion was rendered.

For the repaired blind judge in `23_repaired_llm_judge.py`, have at least two
raters independently complete a CSV with columns
`conversation_id,rater_id,metric,rating`. Then calculate pairwise Cohen's kappa:

```bash
python experiments/human_agreement.py \
  --annotations human_study/ratings.csv \
  --output human_study/agreement
```

The code reports observed agreement and a conversation-bootstrap 95% interval
alongside kappa. Preserve the raw, de-identified ratings and report the number
of raters, cases, rubric, and missing ratings.

## Existing stages to reuse

- Rule baseline and state metrics: `11_extract_state.py`, `12_evaluate_state.py`.
- Two-speaker diarization metrics and baseline/E1 comparison: `08_evaluate_diarization.py`, `09_compare_baseline_vs_e1.py` (the latter already reports bootstrap CIs and Wilcoxon tests).
- Evidence verification and capsule construction: `13_verify_handover.py`, `14_extract_acoustic_context.py`, `15_build_handover_capsules.py`.
- Context-size conditions: `17_ablation_compression.py`. `21_acoustic_context_ablation.py` records API input/output tokens and latency and creates human-review sheets. Run it once per compression directory using distinct output directories; fill ratings before claiming a quality effect.
- Blind LLM judging: `22_llm_judge.py` is the initial judge; `23_repaired_llm_judge.py` is the stricter quote-audited implementation and should be the default report.

Keep `gold_state/` out of inference inputs except for explicit oracle conditions
and evaluation. Do not send company/customer recordings or transcripts to an
external model unless the company has approved that processing.

## Agentic context orchestration

The isolated [`agentic_context_orchestration`](agentic_context_orchestration/README.md)
experiment tests selective tool-based review on existing hard-case test
artifacts. It reuses stages 11–16, preserves the baseline state and evidence
unchanged, and writes new outputs under
`robustness/agentic_context_orchestration/`. It currently uses a configurable
OpenAI Responses API-compatible provider and only the project's synthetic
held-out data; see that experiment README for the tool list, run commands,
guardrails, and interpretation limits.
