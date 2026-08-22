# Voice Phase 2 Dataset (Qwen3-TTS)

Synthetic spoken enterprise customer-support conversations for Phase 2 experiments involving:
- Automatic Speech Recognition (ASR)
- Speaker diarization
- Intent detection
- Emotion recognition
- Context and state tracking
- Multi-turn conversational analysis

The dataset is generated through a three-stage pipeline:
1. Generate coherent scenario-based conversations with speaker, gender, and emotional metadata.
2. Render each conversation turn using Qwen3-TTS CustomVoice.
3. Merge individual turn-level WAV files into complete two-speaker conversations.

---

## Pipeline overview

```
data/lexicon.py
        │  01_generate_dataset.py
        ▼
data/emotional_conversations.json
data/emotional_conversations.jsonl
data/metadata.csv
        │  02_generate_emotional_audio.py   (Qwen3-TTS)
        ▼
new_generated_turns/<conversation_id>/turn_<n>_<role>_<speaker>.wav
        │  03_merge_conversations.py        (pydub)
        ▼
conversations_audio/<conversation_id>.wav
```

---

## Folder structure

```
voice_phase2_dataset/
├── 01_generate_dataset.py         # text conversations from different templates
├── 02_generate_emotional_audio.py # per-turn TTS with Qwen3-TTS
├── 03_merge_conversations.py      # merge turns into one WAV per conversation
├── data/
│   ├── emotional_conversations.json              # generated text conversations + voice_attributes (500)
│   ├── metadata.csv                    # flat per-conversation metadata
│   ├── emotional_conversations.jsonl    # conversations + voice_attributes
│   └── lexicon.py       # speaker chosen per conversation
├── new_generated_turns/               # per-turn WAV files
└── conversations_audio/           # final merged conversation WAVs
```

---

## Requirements

* Python 3.10+
* `torch`, `soundfile`, `qwen_tts` (Qwen3-TTS inference package)
* `pydub` for merging, plus `ffmpeg` on the system

```bash
pip install torch soundfile pydub
brew install ffmpeg          # macOS
```

The model `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` is downloaded from Hugging Face
on the first run of stage 03 and loaded in `float32` (CPU-friendly).

---

## Usage

Run the stages in order **from inside this folder** (all paths are relative):

```bash
cd voice_phase2_dataset

python 01_generate_dataset.py
python 02_generate_emotional_audio.py
python 03_merge_conversations.py
```

### 01 — `01_generate_dataset.py`
Generates 10 conversations for each scenario in `data/scenarios.py`
(50 scenarios → **500 conversations**, ids `conv_0001` … `conv_0500`).
Each conversation has multiple alternating turns built from scenario-specific templates, agent openings and closings, and a
`customer_name` drawn from `["aiden", "serena", "eric", "sohee"]`. Adds a `voice_attributes` block to every turn.
Customer emotions are restricted to those plausible for the scenario
(`SCENARIO_EMOTIONS`, e.g. `FRAUD_DISPUTE → anxious / angry / worried`), agents
get a professional style (`calm`, `empathetic`, `reassuring`, `apologetic`,
`professional`). Writes `data/emotional_conversations.json`, `data/emotional_conversations.jsonl` and
`data/metadata.csv` (`conversation_id, scenario, scenario_family, num_turns`).


### 02 — `02_generate_emotional_audio.py`
Renders each turn with Qwen3-TTS.
* **Speakers.** The customer voice is the conversation's `customer_name`; the
  agent voice is picked at random from
  `["dylan", "ono_anna", "ryan", "uncle_fu", "vivian"]`.
  The choice is stored in `data/speaker_assignments.json` (written immediately
  after each new assignment) so the same voices are reused on later runs.
  `random.seed(42)` keeps assignments reproducible.
* **Style control.** `build_voice_instruction()` converts the voice attributes
  into a natural-language instruction ("Speak as a customer … the emotional
  style is frustrated … speak noticeably faster than normal …") passed to the
  model via `instruct`, together with `temperature=0.7`, `top_p=0.9`,
  `repetition_penalty=1.05`.
* **Output.** `generated_turns/<conversation_id>/turn_<turn_id>_<role>_<speaker>.wav`.
  Existing files are skipped, so generation is resumable; errors on a single turn
  are reported and do not stop the run.

> The loop is currently limited to the first conversations
> (`for ... in enumerate(conversations[:2], start=1)`).
> Remove the `[:2]` slice to render the whole dataset.

### 04 — `04_merge_conversations.py`
Rebuilds the speaker-suffixed turn file names from
`emotional_conversations.json` + `speaker_assignments.json`, inserts
`pause_before` / `pause_after` silence around each turn and exports
`conversations_audio/<conversation_id>.wav`. Missing turn files are printed as
`Missing: ...` and skipped instead of aborting the run.

---

## Data schema

`data/conversations.json` — list of conversation objects:

| Field | Description |
|---|---|
| `conversation_id` | unique id (`conv_0001`), also the output WAV name |
| `scenario` | scenario key, e.g. `PAYMENT_FAILED` |
| `scenario_family` | higher-level grouping, e.g. `Payments` |
| `customer_name` | speaker id used for the customer voice |
| `turns` | list of turns |

Each turn:

| Field | Description |
|---|---|
| `turn_id` | 1-based turn index |
| `role` | `customer` or `agent` |
| `text` | utterance to synthesize |
| `voice_attributes` | added by stage 02 (see below) |

`voice_attributes`:

| Field | Description |
|---|---|
| `emotion` | customer emotion or agent style |
| `emotion_intensity` | 0.0–1.0, sampled from the emotion profile |
| `speaking_rate` | `slow` / `medium` / `normal` / `fast` |
| `prosody` | e.g. `stable`, `high_variation`, `hesitant`, `controlled` |
| `pause_before` | seconds of silence before the turn |
| `pause_after` | seconds of silence after the turn |

`data/speaker_assignments.json` — `conv_XXXX → {scenario, customer_speaker, agent_speaker}`.

---

## Notes and troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: data/...` | Run the scripts from inside `voice_phase2_dataset` |
| Only two conversations rendered | Remove the `[:2]` slice in stage 03 |
| `Missing: generated_turns/...` in stage 04 | That turn was not generated yet — rerun stage 03 |
| `Couldn't find ffmpeg or avconv` | Install `ffmpeg`, required by `pydub` |
| Want to regenerate a turn | Delete the WAV — existing files are skipped |
| Want different voices | Edit `CUSTOMER_SPEAKERS` / `AGENT_SPEAKERS` in stage 03 and delete `data/speaker_assignments.json` |
