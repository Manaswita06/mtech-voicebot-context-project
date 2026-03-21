#!/usr/bin/env python3
"""
dataset_builder.py

Scan a directory of synthetic transcript JSON files and build a CSV dataset for training/evaluation.

Output CSV columns:
- conversation_id
- conversation_text (all turns concatenated, speaker-prefixed)
- user_text (only user utterances concatenated)
- gt_primary_intent
- gt_authenticated
- gt_tool_failure
- gt_sentiment_overall
- gt_turn_count
- events_json (raw events as JSON string)
"""

import argparse
import json
import csv
from pathlib import Path
from tqdm import tqdm


def load_transcripts(input_dir: str):
    p = Path(input_dir)
    files = sorted(p.glob("*.json"))
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                yield json.load(fh)
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}")


def build_record(transcript: dict):
    conv_id = transcript.get("conversation_id") or transcript.get("conversation_id")
    events = transcript.get("events", [])
    # build conversation text with speaker prefixes
    conversation_text = []
    user_texts = []
    for ev in events:
        role = ev.get("participant", {}).get("role", "user")
        text = ev.get("text") or ev.get("event_data", {}).get("message") or ""
        if text is None:
            text = ""
        # prefix speaker and join
        conversation_text.append(f"{role.upper()}: {text}")
        if role == "user":
            user_texts.append(text)
    conv_text = " \n ".join(conversation_text)
    user_text = " \n ".join(user_texts)
    record = {
        "conversation_id": conv_id,
        "conversation_text": conv_text,
        "user_text": user_text,
        "gt_primary_intent": transcript.get("gt_primary_intent"),
        "gt_authenticated": transcript.get("gt_authenticated"),
        "gt_tool_failure": transcript.get("gt_tool_failure"),
        "gt_sentiment_overall": transcript.get("gt_sentiment_overall"),
        "gt_turn_count": transcript.get("gt_turn_count"),
        "events_json": json.dumps(events, ensure_ascii=False)
    }
    return record


def main(input_dir: str, output_csv: str):
    transcripts = list(load_transcripts(input_dir))
    print(f"Found {len(transcripts)} transcript files in {input_dir}")
    fieldnames = [
        "conversation_id",
        "conversation_text",
        "user_text",
        "gt_primary_intent",
        "gt_authenticated",
        "gt_tool_failure",
        "gt_sentiment_overall",
        "gt_turn_count",
        "events_json",
    ]
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for t in tqdm(transcripts, desc="Processing transcripts"):
            try:
                rec = build_record(t)
                writer.writerow(rec)
            except Exception as e:
                print(f"Error processing transcript {t.get('conversation_id','<unknown>')}: {e}")
    print(f"Wrote dataset to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset CSV from synthetic transcripts")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="../data/synthetic_advanced",
        help="Directory with transcript JSON files",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="../data/processed/conversation_dataset.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_csv)