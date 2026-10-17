#!/usr/bin/env python3
"""Paired real-LLM ablation: RECAP semantic context with vs. without acoustics.

Default is held-out test split. Creates:
  responses.csv                 one real LLM response per condition/conversation
  paired_results_for_review.csv paired replies + blank scoring columns
  run_metadata.json             experiment settings

Run a small smoke test before the full test set.
"""
from __future__ import annotations
import argparse, csv, json, os, re, time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit("Install requirements first: pip install -r requirements.txt") from exc

DEFAULT_MESSAGE = "I'd like to continue from where we left off. What should happen next?"
INSTRUCTIONS = """
You are Agent B taking over a customer-support conversation. You will receive a
structured handover context and the customer's new message.

Use semantic state to preserve continuity and avoid making the customer repeat
available information. Do not invent facts, actions, commitments, or outcomes.
Respect uncertainty and review flags; ask a concise clarification if necessary.
If acoustic/turn-taking observations are present, treat them only as observable
interaction metadata. They may guide pacing or clarity when relevant. Never infer
or state emotion, honesty, intent, or mental state from silence, rate, pauses,
talk ratio, or overlap. Do not mention internal processing or experiment conditions.
Respond naturally and concisely.
""".strip()
ACOUSTIC_KEYS = {"interaction_context", "acoustic_observations",
                 "acoustic_interaction_features"}

def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def dump_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def strip_acoustics(capsule: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(capsule))
    for key in ACOUSTIC_KEYS:
        out.pop(key, None)
    # Support older layouts as well.
    for parent_key in ("handover", "interaction"):
        parent = out.get(parent_key)
        if isinstance(parent, dict):
            for key in ACOUSTIC_KEYS:
                parent.pop(key, None)
    return out

def split_from_json(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        x = read_json(path)
    except Exception:
        return None
    for val in [
        x.get("conversation", {}).get("split") if isinstance(x.get("conversation"), dict) else None,
        x.get("split"),
        x.get("metadata", {}).get("split") if isinstance(x.get("metadata"), dict) else None,
    ]:
        if val:
            return str(val).lower().strip()
    return None

def load_dataset_splits(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    try:
        x = read_json(path)
    except Exception:
        return {}
    records = x if isinstance(x, list) else (x.get("conversations") or x.get("data") or [])
    if isinstance(x, dict) and not records and all(isinstance(v, dict) for v in x.values()):
        records = list(x.values())
    return {
        str(r.get("conversation_id") or r.get("id")): str(r["split"]).lower().strip()
        for r in records if isinstance(r, dict) and (r.get("conversation_id") or r.get("id")) and r.get("split")
    }

def get_candidates(capsule_dir: Path, gold_dir: Path, dataset: Path | None, split: str):
    split_map = load_dataset_splits(dataset)
    candidates = []
    for p in sorted(capsule_dir.glob("*.json")):
        cid = p.stem
        sp = split_from_json(gold_dir / f"{cid}.json") or split_map.get(cid)
        if split == "all" or sp == split:
            candidates.append((cid, sp or "unknown"))
    if not candidates:
        raise SystemExit(f"No capsules found for split={split!r}. Check split metadata in gold_state/ or dataset. Use --split all only for a smoke test.")
    return candidates

def run_response(client: OpenAI, model: str, capsule: dict[str, Any], message: str) -> dict[str, Any]:
    ctx = json.dumps(capsule, ensure_ascii=False, separators=(",", ":"))
    inp = f"PREVIOUS-CALL HANDOVER CONTEXT (JSON):\n{ctx}\n\nCUSTOMER'S NEW MESSAGE:\n{message}"
    start = time.perf_counter()
    r = client.responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=[{"role": "user", "content": inp}],
    )
    elapsed = time.perf_counter() - start
    usage = getattr(r, "usage", None)
    return {
        "response_text": str(getattr(r, "output_text", "") or "").strip(),
        "response_id": getattr(r, "id", ""),
        "context_bytes": len(ctx.encode("utf-8")),
        "approx_context_tokens_whitespace": len(re.findall(r"\S+", ctx)),
        "input_tokens": getattr(usage, "input_tokens", "") if usage else "",
        "output_tokens": getattr(usage, "output_tokens", "") if usage else "",
        "elapsed_seconds": round(elapsed, 3),
        "error": "",
    }

RESPONSE_COLUMNS = [
    "conversation_id","split","condition","model","customer_message",
    "context_bytes","approx_context_tokens_whitespace","input_tokens",
    "output_tokens","elapsed_seconds","response_id","response_text","error"
]
PAIRED_COLUMNS = [
    "conversation_id","split","customer_message","response_without_acoustic","response_with_acoustic",
    "bytes_without_acoustic","bytes_with_acoustic","input_tokens_without_acoustic","input_tokens_with_acoustic",
    "latency_without_acoustic_s","latency_with_acoustic_s",
    "continuity_score_without_acoustic_1to5","continuity_score_with_acoustic_1to5",
    "unnecessary_reask_without_acoustic_0or1","unnecessary_reask_with_acoustic_0or1",
    "appropriate_adaptation_without_acoustic_0to2","appropriate_adaptation_with_acoustic_0to2",
    "unsupported_psychological_inference_without_acoustic_0or1","unsupported_psychological_inference_with_acoustic_0or1",
    "factual_consistency_without_acoustic_1to5","factual_consistency_with_acoustic_1to5","reviewer_notes"
]

def prior_rows(path: Path) -> dict[tuple[str,str], dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return {(str(r["conversation_id"]), str(r["condition"])): r for r in rows}
    except Exception:
        return {}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capsule-dir", type=Path, default=Path("handover_capsules"))
    ap.add_argument("--gold-dir", type=Path, default=Path("gold_state"))
    ap.add_argument("--dataset", type=Path, default=Path("data/emotional_conversations.json"))
    ap.add_argument("--output", type=Path, default=Path("acoustic_ablation_results"))
    ap.add_argument("--split", choices=["test","validation","train","all"], default="test")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    ap.add_argument("--message", default=DEFAULT_MESSAGE)
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test limit; 0 runs all selected conversations.")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set OPENAI_API_KEY before running.")
    if not args.capsule_dir.exists():
        raise SystemExit(f"Capsule directory not found: {args.capsule_dir}")

    candidates = get_candidates(args.capsule_dir, args.gold_dir, args.dataset, args.split)
    if args.limit > 0:
        candidates = candidates[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    response_path = args.output / "responses.csv"
    paired_path = args.output / "paired_results_for_review.csv"
    metadata_path = args.output / "run_metadata.json"

    old = prior_rows(response_path) if args.resume else {}
    client = OpenAI(api_key=key)
    total = len(candidates) * 2
    done = 0
    print(f"Model={args.model}; split={args.split}; conversations={len(candidates)}; planned calls={total}")

    for i, (cid, split) in enumerate(candidates, 1):
        capsule = read_json(args.capsule_dir / f"{cid}.json")
        conditions = [
            ("without_acoustic", strip_acoustics(capsule)),
            ("with_acoustic", json.loads(json.dumps(capsule))),
        ]
        for cond, context in conditions:
            k = (cid, cond)
            if args.resume and old.get(k, {}).get("response_text") and not old[k].get("error"):
                done += 1
                continue
            print(f"[{i}/{len(candidates)}] {cid}: {cond}")
            try:
                output = run_response(client, args.model, context, args.message)
            except Exception as exc:
                ctx = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
                output = {
                    "response_text": "", "response_id": "",
                    "context_bytes": len(ctx.encode("utf-8")),
                    "approx_context_tokens_whitespace": len(re.findall(r"\S+", ctx)),
                    "input_tokens": "", "output_tokens": "", "elapsed_seconds": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            old[k] = {
                "conversation_id": cid, "split": split, "condition": cond,
                "model": args.model, "customer_message": args.message, **output
            }
            done += 1
            dump_csv(response_path, list(old.values()), RESPONSE_COLUMNS)
            print(f"  saved {done}/{total}")

    # Build one paired row per selected conversation; quality scores are intentionally blank.
    paired = []
    for cid, split in candidates:
        a, b = old.get((cid, "without_acoustic"), {}), old.get((cid, "with_acoustic"), {})
        paired.append({
            "conversation_id": cid, "split": split, "customer_message": args.message,
            "response_without_acoustic": a.get("response_text",""),
            "response_with_acoustic": b.get("response_text",""),
            "bytes_without_acoustic": a.get("context_bytes",""),
            "bytes_with_acoustic": b.get("context_bytes",""),
            "input_tokens_without_acoustic": a.get("input_tokens",""),
            "input_tokens_with_acoustic": b.get("input_tokens",""),
            "latency_without_acoustic_s": a.get("elapsed_seconds",""),
            "latency_with_acoustic_s": b.get("elapsed_seconds",""),
            "continuity_score_without_acoustic_1to5": "",
            "continuity_score_with_acoustic_1to5": "",
            "unnecessary_reask_without_acoustic_0or1": "",
            "unnecessary_reask_with_acoustic_0or1": "",
            "appropriate_adaptation_without_acoustic_0to2": "",
            "appropriate_adaptation_with_acoustic_0to2": "",
            "unsupported_psychological_inference_without_acoustic_0or1": "",
            "unsupported_psychological_inference_with_acoustic_0or1": "",
            "factual_consistency_without_acoustic_1to5": "",
            "factual_consistency_with_acoustic_1to5": "",
            "reviewer_notes": "",
        })
    dump_csv(paired_path, paired, PAIRED_COLUMNS)
    dump_json(metadata_path, {
        "experiment": "RECAP-Voice acoustic-context ablation",
        "model": args.model, "requested_split": args.split,
        "conversation_count": len(candidates), "planned_llm_calls": total,
        "customer_message": args.message,
        "condition_A": "Same capsule with interaction_context/acoustic_observations removed",
        "condition_B": "Unmodified capsule with acoustic/turn-taking context",
        "controlled_variables": ["same model","same instructions","same customer message","same semantic/evidence/uncertainty fields"],
        "quality_metrics": "Human annotation columns remain blank. Fill them before making a response-quality claim.",
        "caution": "Acoustic signals are not interpreted as emotion, honesty, intent, or mental state."
    })
    print(f"\nDone.\nResponses: {response_path}\nPaired review sheet: {paired_path}\nMetadata: {metadata_path}")

if __name__ == "__main__":
    main()
