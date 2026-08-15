# Voice Phase 2 Dataset (Qwen3-TTS)

Synthetic **spoken customer-support conversations** for Phase 2 experiments
(ASR, diarization, intent detection, emotion and context tracking).

The dataset is built by a four-stage pipeline: text conversations are generated
from a scenario catalogue, annotated with emotion/prosody attributes, rendered
turn by turn with the **Qwen3-TTS** custom-voice model, and finally merged into a
single two-speaker WAV per conversation.

---

## Pipeline overview

```
data/scenarios.py
        │  01_generate_dataset.py
        ▼
data/conversations.json (+ .jsonl, metadata.csv)
        │  02_add_voice_attributes.py
        ▼
data/emotional_conversations.json
        │  03_generate_emotional_audio.py   (Qwen3-TTS)
        ▼
generated_turns/<conversation_id>/turn_<n>_<role>_<speaker>.wav
        │  04_merge_conversations.py        (pydub)
        ▼
conversations_audio/<conversation_id>.wav
```

---

## Folder structure

```
voice_phase2_dataset/
├── 01_generate_dataset.py         # text conversations from scenarios
├── 02_add_voice_attributes.py     # emotion / prosody / pause annotation
├── 03_generate_emotional_audio.py # per-turn TTS with Qwen3-TTS
├── 04_merge_conversations.py      # merge turns into one WAV per conversation
├── data/
│   ├── scenarios.py                    # scenario catalogue (scenario -> family)
│   ├── conversations.json              # generated text conversations (480)
│   ├── conversations.jsonl             # same data, one conversation per line
│   ├── metadata.csv                    # flat per-conversation metadata
│   ├── emotional_conversations.json    # conversations + voice_attributes
│   └── speaker_assignments.json        # speaker chosen per conversation
├── generated_turns/               # per-turn WAV files
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
python 02_add_voice_attributes.py
python 03_generate_emotional_audio.py
python 04_merge_conversations.py
```

### 01 — `01_generate_dataset.py`
Generates 10 conversations for each scenario in `data/scenarios.py`
(48 scenarios → **480 conversations**, ids `conv_0001` … `conv_0480`).
Each conversation has 4 alternating turns (customer → agent → customer → agent)
built from scenario-specific templates, agent openings and closings, and a
`customer_name` drawn from `["aiden", "serena", "eric", "sohee"]`.
Writes `data/conversations.json`, `data/conversations.jsonl` and
`data/metadata.csv` (`conversation_id, scenario, scenario_family, num_turns`).

### 02 — `02_add_voice_attributes.py`
Adds a `voice_attributes` block to every turn.
Customer emotions are restricted to those plausible for the scenario
(`SCENARIO_EMOTIONS`, e.g. `FRAUD_DISPUTE → anxious / angry / worried`), agents
get a professional style (`calm`, `empathetic`, `reassuring`, `apologetic`,
`professional`). Writes `data/emotional_conversations.json`.

### 03 — `03_generate_emotional_audio.py`
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
