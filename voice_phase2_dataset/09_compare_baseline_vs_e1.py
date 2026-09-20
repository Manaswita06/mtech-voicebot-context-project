#!/usr/bin/env python3

"""
09_compare_baseline_vs_e1.py

Direct paired comparison:

BASELINE
Whisper base.en + Pyannote + segment-level attribution

vs

E1
Whisper large-v3-turbo + WhisperX word alignment
+ same Pyannote + word-level attribution

Main common metric:
    Reference-turn speaker-role accuracy

Additional metrics:
    unresolved turn rate
    turn coverage
    speaker detection status
    paired delta
    bootstrap CI
    Wilcoxon signed-rank test

IMPORTANT:
- Same 500 conversations are evaluated in both pipelines.
- Same reference turns are used.
- Pyannote is NOT rerun.
"""

import csv
import glob
import json
import os
import wave

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


# ============================================================
# CONFIG
# ============================================================

BASELINE_DIR = "speaker_transcripts"
E1_DIR = "wordlevel_speaker_transcripts"

METADATA_FILE = "data/emotional_conversations.json"
TURN_AUDIO_DIR = "new_generated_turns"

OUTPUT_FILE = "baseline_vs_e1_per_conversation.csv"
SUMMARY_FILE = "baseline_vs_e1_summary.csv"


# None = all conversations
MAX_CONVERSATIONS = None


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def wav_duration_seconds(path):

    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()

    return frames / float(rate)


def normalize_role(role):

    if role is None:
        return None

    role = str(role).strip().upper()

    if role in {"CUSTOMER", "CLIENT", "USER"}:
        return "CUSTOMER"

    if role in {
        "AGENT",
        "ASSOCIATE",
        "REPRESENTATIVE"
    }:
        return "AGENT"

    return role


def find_turn_wav(conversation_id, turn_id):

    conv_dir = os.path.join(
        TURN_AUDIO_DIR,
        conversation_id
    )

    patterns = [
        os.path.join(
            conv_dir,
            f"turn_{turn_id}_*.wav"
        )
    ]

    try:
        tid = int(turn_id)

        patterns.extend([
            os.path.join(
                conv_dir,
                f"turn_{tid:03d}_*.wav"
            ),
            os.path.join(
                conv_dir,
                f"turn_{tid:04d}_*.wav"
            ),
        ])

    except (ValueError, TypeError):
        pass

    for pattern in patterns:

        matches = sorted(glob.glob(pattern))

        if matches:
            return matches[0]

    return None


def overlap(a_start, a_end, b_start, b_end):

    return max(
        0.0,
        min(a_end, b_end)
        - max(a_start, b_start)
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_conversations():

    data = load_json(METADATA_FILE)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        if "conversations" in data:
            return data["conversations"]

        return list(data.values())

    raise ValueError(
        "Unsupported metadata format"
    )


def build_lookup(conversations):

    return {
        c["conversation_id"]: c
        for c in conversations
        if "conversation_id" in c
    }


# ============================================================
# GROUND-TRUTH TIMELINE
# ============================================================

def build_reference_turns(conversation):

    turns = conversation["turns"]

    reference = []

    current_time = 0.0

    for turn in turns:

        turn_id = turn["turn_id"]

        role = normalize_role(
            turn.get("role")
        )

        wav_path = find_turn_wav(
            conversation["conversation_id"],
            turn_id
        )

        if wav_path is None:

            raise FileNotFoundError(
                f"Missing WAV for "
                f"{conversation['conversation_id']} "
                f"turn {turn_id}"
            )

        duration = wav_duration_seconds(
            wav_path
        )

        start = current_time
        end = current_time + duration

        reference.append({
            "turn_id": turn_id,
            "role": role,
            "start": start,
            "end": end,
        })

        current_time = end

    return reference


# ============================================================
# EXTRACT BASELINE PREDICTION INTERVALS
# ============================================================

def extract_baseline_intervals(data):

    candidates = []

    for key in [
        "segments",
        "speaker_segments",
        "speaker_turns"
    ]:

        value = data.get(key)

        if isinstance(value, list):

            candidates = value

            if candidates:
                break

    intervals = []

    for item in candidates:

        if (
            "start" not in item
            or "end" not in item
        ):
            continue

        predicted_role = (
            item.get("predicted_role")
            or item.get("speaker_role")
            or item.get("role")
        )

        predicted_role = normalize_role(
            predicted_role
        )

        if predicted_role is None:
            continue

        intervals.append({
            "start": float(item["start"]),
            "end": float(item["end"]),
            "role": predicted_role,
        })

    return intervals


# ============================================================
# EXTRACT E1 WORD INTERVALS
# ============================================================

def extract_e1_intervals(data):

    intervals = []

    for word in data.get("words", []):

        if (
            "start" not in word
            or "end" not in word
        ):
            continue

        role = normalize_role(
            word.get("predicted_role")
        )

        if role is None:
            continue

        intervals.append({
            "start": float(word["start"]),
            "end": float(word["end"]),
            "role": role,
        })

    return intervals


# ============================================================
# EVALUATE ONE PIPELINE
# ============================================================

def evaluate_turn_accuracy(
    reference_turns,
    predicted_intervals
):

    total_turns = len(reference_turns)

    correct_turns = 0
    unresolved_turns = 0

    turn_results = []

    for ref in reference_turns:

        role_overlap = {
            "CUSTOMER": 0.0,
            "AGENT": 0.0,
        }

        for pred in predicted_intervals:

            ov = overlap(
                ref["start"],
                ref["end"],
                pred["start"],
                pred["end"],
            )

            if ov > 0:

                if pred["role"] in role_overlap:

                    role_overlap[
                        pred["role"]
                    ] += ov

        total_predicted_overlap = sum(
            role_overlap.values()
        )

        if total_predicted_overlap <= 0:

            predicted_role = None

            unresolved_turns += 1

            correct = False

        else:

            sorted_roles = sorted(
                role_overlap.items(),
                key=lambda x: x[1],
                reverse=True
            )

            predicted_role = sorted_roles[0][0]

            # Tie handling
            if (
                len(sorted_roles) > 1
                and sorted_roles[0][1]
                == sorted_roles[1][1]
                and sorted_roles[0][1] > 0
            ):

                predicted_role = None
                unresolved_turns += 1
                correct = False

            else:

                correct = (
                    predicted_role
                    == ref["role"]
                )

                if correct:
                    correct_turns += 1

        turn_results.append({
            "turn_id": ref["turn_id"],
            "ground_truth_role": ref["role"],
            "predicted_role": predicted_role,
            "correct": correct,
            "customer_overlap":
                role_overlap["CUSTOMER"],
            "agent_overlap":
                role_overlap["AGENT"],
        })

    accuracy = (
        correct_turns / total_turns
        if total_turns > 0
        else 0.0
    )

    unresolved_rate = (
        unresolved_turns / total_turns
        if total_turns > 0
        else 0.0
    )

    return {
        "total_turns": total_turns,
        "correct_turns": correct_turns,
        "turn_accuracy": accuracy,
        "unresolved_turns": unresolved_turns,
        "unresolved_turn_rate": unresolved_rate,
        "turn_results": turn_results,
    }


# ============================================================
# SPEAKER DETECTION
# ============================================================

def get_num_speakers(data):

    if "num_speakers_detected" in data:

        return int(
            data["num_speakers_detected"]
        )

    speakers = data.get(
        "speakers_detected",
        []
    )

    return len(speakers)


# ============================================================
# VOICE-PAIR INFORMATION
# ============================================================

def get_voice_pair_type(conversation):

    customer_gender = str(
        conversation.get(
            "customer_gender",
            ""
        )
    ).strip().lower()

    agent_gender = str(
        conversation.get(
            "agent_gender",
            ""
        )
    ).strip().lower()

    if customer_gender and agent_gender:

        if customer_gender == agent_gender:

            if customer_gender == "female":
                return "Female+Female"

            if customer_gender == "male":
                return "Male+Male"

            return "Same-Gender"

        return "Female+Male"

    return "Unknown"


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_mean_ci(
    values,
    n_boot=10000,
    seed=42
):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(
        n_boot
    )

    n = len(values)

    for i in range(n_boot):

        sample = rng.choice(
            values,
            size=n,
            replace=True
        )

        bootstrap_means[i] = np.mean(
            sample
        )

    lower = np.percentile(
        bootstrap_means,
        2.5
    )

    upper = np.percentile(
        bootstrap_means,
        97.5
    )

    return (
        float(np.mean(values)),
        float(lower),
        float(upper),
    )


# ============================================================
# MAIN COMPARISON
# ============================================================

def main():

    conversations = load_conversations()

    lookup = build_lookup(
        conversations
    )

    baseline_files = sorted(
        glob.glob(
            os.path.join(
                BASELINE_DIR,
                "*.json"
            )
        )
    )

    rows = []

    for baseline_path in baseline_files:

        conversation_id = os.path.splitext(
            os.path.basename(
                baseline_path
            )
        )[0]

        if (
            MAX_CONVERSATIONS is not None
            and len(rows) >= MAX_CONVERSATIONS
        ):
            break

        e1_path = os.path.join(
            E1_DIR,
            f"{conversation_id}.json"
        )

        if not os.path.exists(e1_path):
            print(
                f"Skipping {conversation_id}: "
                f"E1 output missing"
            )
            continue

        conversation = lookup.get(
            conversation_id
        )

        if conversation is None:
            continue

        baseline = load_json(
            baseline_path
        )

        e1 = load_json(
            e1_path
        )

        reference_turns = (
            build_reference_turns(
                conversation
            )
        )

        baseline_intervals = (
            extract_baseline_intervals(
                baseline
            )
        )

        e1_intervals = (
            extract_e1_intervals(
                e1
            )
        )

        baseline_eval = (
            evaluate_turn_accuracy(
                reference_turns,
                baseline_intervals
            )
        )

        e1_eval = (
            evaluate_turn_accuracy(
                reference_turns,
                e1_intervals
            )
        )

        baseline_speakers = (
            get_num_speakers(baseline)
        )

        e1_speakers = (
            get_num_speakers(e1)
        )

        row = {
            "conversation_id":
                conversation_id,

            "scenario":
                conversation.get(
                    "scenario",
                    ""
                ),

            "voice_pair":
                get_voice_pair_type(
                    conversation
                ),

            "baseline_num_speakers":
                baseline_speakers,

            "e1_num_speakers":
                e1_speakers,

            "baseline_total_turns":
                baseline_eval[
                    "total_turns"
                ],

            "e1_total_turns":
                e1_eval[
                    "total_turns"
                ],

            "baseline_correct_turns":
                baseline_eval[
                    "correct_turns"
                ],

            "e1_correct_turns":
                e1_eval[
                    "correct_turns"
                ],

            "baseline_turn_accuracy":
                baseline_eval[
                    "turn_accuracy"
                ],

            "e1_turn_accuracy":
                e1_eval[
                    "turn_accuracy"
                ],

            "delta_turn_accuracy":
                e1_eval[
                    "turn_accuracy"
                ]
                -
                baseline_eval[
                    "turn_accuracy"
                ],

            "baseline_unresolved_turns":
                baseline_eval[
                    "unresolved_turns"
                ],

            "e1_unresolved_turns":
                e1_eval[
                    "unresolved_turns"
                ],

            "baseline_unresolved_rate":
                baseline_eval[
                    "unresolved_turn_rate"
                ],

            "e1_unresolved_rate":
                e1_eval[
                    "unresolved_turn_rate"
                ],

            "delta_unresolved_rate":
                e1_eval[
                    "unresolved_turn_rate"
                ]
                -
                baseline_eval[
                    "unresolved_turn_rate"
                ],
        }

        rows.append(row)

    if not rows:

        raise RuntimeError(
            "No paired Baseline/E1 conversations found."
        )

    df = pd.DataFrame(rows)

    # ========================================================
    # SAVE PER-CONVERSATION RESULTS
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("PAIRED BASELINE VS E1 COMPARISON")
    print("=" * 70)

    print(
        f"Paired conversations: {len(df)}"
    )

    # ========================================================
    # SANITY CHECK: SAME DIARIZATION
    # ========================================================

    same_speaker_count = (
        df["baseline_num_speakers"]
        ==
        df["e1_num_speakers"]
    ).mean()

    print(
        f"Same Pyannote speaker count: "
        f"{same_speaker_count * 100:.2f}%"
    )

    # ========================================================
    # BASIC RESULTS
    # ========================================================

    baseline_mean = (
        df["baseline_turn_accuracy"]
        .mean()
    )

    e1_mean = (
        df["e1_turn_accuracy"]
        .mean()
    )

    delta = (
        df["delta_turn_accuracy"]
    )

    print()
    print(
        f"Baseline mean turn accuracy: "
        f"{baseline_mean * 100:.2f}%"
    )

    print(
        f"E1 mean turn accuracy: "
        f"{e1_mean * 100:.2f}%"
    )

    print(
        f"Mean paired difference: "
        f"{delta.mean() * 100:.2f} percentage points"
    )

    print(
        f"Median paired difference: "
        f"{delta.median() * 100:.2f} percentage points"
    )

    # ========================================================
    # CONVERSATION-LEVEL CHANGES
    # ========================================================

    improved = (delta > 0).sum()
    unchanged = (delta == 0).sum()
    decreased = (delta < 0).sum()

    print()
    print(
        f"E1 higher than Baseline: "
        f"{improved}"
    )

    print(
        f"No change: "
        f"{unchanged}"
    )

    print(
        f"E1 lower than Baseline: "
        f"{decreased}"
    )

    # ========================================================
    # BOOTSTRAP CI
    # ========================================================

    mean_delta, ci_low, ci_high = (
        bootstrap_mean_ci(
            delta.values
        )
    )

    print()
    print(
        f"Mean delta: "
        f"{mean_delta * 100:.2f} pp"
    )

    print(
        f"95% bootstrap CI: "
        f"[{ci_low * 100:.2f}, "
        f"{ci_high * 100:.2f}] pp"
    )

    # ========================================================
    # WILCOXON
    # ========================================================

    try:

        statistic, pvalue = wilcoxon(
            delta,
            zero_method="wilcox",
            alternative="two-sided"
        )

        print()
        print(
            f"Wilcoxon statistic: "
            f"{statistic:.4f}"
        )

        print(
            f"Wilcoxon p-value: "
            f"{pvalue:.6g}"
        )

    except ValueError:

        print(
            "\nWilcoxon test could not be calculated "
            "because there were no non-zero paired differences."
        )

    # ========================================================
    # UNRESOLVED RATE
    # ========================================================

    print()
    print(
        "Mean baseline unresolved-turn rate: "
        f"{df['baseline_unresolved_rate'].mean() * 100:.2f}%"
    )

    print(
        "Mean E1 unresolved-turn rate: "
        f"{df['e1_unresolved_rate'].mean() * 100:.2f}%"
    )

    # ========================================================
    # VOICE-PAIR BREAKDOWN
    # ========================================================

    print()
    print("=" * 70)
    print("VOICE-PAIR BREAKDOWN")
    print("=" * 70)

    group_rows = []

    for voice_pair, group in (
        df.groupby("voice_pair")
    ):

        group_delta = group[
            "delta_turn_accuracy"
        ]

        group_rows.append({
            "voice_pair":
                voice_pair,

            "n":
                len(group),

            "baseline_mean_accuracy":
                group[
                    "baseline_turn_accuracy"
                ].mean(),

            "e1_mean_accuracy":
                group[
                    "e1_turn_accuracy"
                ].mean(),

            "mean_delta":
                group_delta.mean(),

            "improved":
                (group_delta > 0).sum(),

            "unchanged":
                (group_delta == 0).sum(),

            "decreased":
                (group_delta < 0).sum(),
        })

        print(
            f"{voice_pair}: "
            f"n={len(group)}, "
            f"Baseline="
            f"{group['baseline_turn_accuracy'].mean()*100:.2f}%, "
            f"E1="
            f"{group['e1_turn_accuracy'].mean()*100:.2f}%, "
            f"Delta="
            f"{group_delta.mean()*100:.2f} pp"
        )

    pd.DataFrame(
        group_rows
    ).to_csv(
        SUMMARY_FILE,
        index=False
    )

    print()
    print(
        f"Per-conversation results: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Group summary: "
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()