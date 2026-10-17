#!/usr/bin/env python3
"""
17_ablation_compression.py

RECAP-Voice leave-one-block-out and compression study.

Ablations:
- full
- no_repair
- no_interaction
- no_evidence
- no_uncertainty

Compression:
- 100%
- 75%
- 50%
- 25%
- 10%

This script focuses on packet-size accounting and structural integrity.
Use 16_build_handoff_benchmark.py after each ablation/compression condition
to evaluate handoff quality with the same metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_ablation(packet: dict[str, Any], condition: str) -> dict[str, Any]:
    out = json.loads(json.dumps(packet))

    if condition == "no_repair":
        out["repair_events"] = []

    elif condition == "no_interaction":
        out.pop("interaction", None)
        if isinstance(out.get("handover"), dict):
            out["handover"].pop("interaction_context", None)
            out["handover"].pop("acoustic_observations", None)

    elif condition == "no_evidence":
        out["evidence"] = []
        for field in [
            "intent",
            "customer_goal",
            "resolution_state",
            "escalated",
            "verification",
        ]:
            item = out.get("state", {}).get(field)
            if isinstance(item, dict):
                item["evidence_ids"] = []

        if isinstance(out.get("handover"), dict):
            for key in (
                "customer_goal",
                "detected_intent",
                "resolution_state",
            ):
                item = out["handover"].get(key)
                if isinstance(item, dict):
                    item["evidence"] = []
                    item["evidence_ids"] = []

    elif condition == "no_uncertainty":
        out.pop("uncertainty", None)
        if isinstance(out.get("handover"), dict):
            out["handover"].pop("uncertainty", None)

    elif condition != "full":
        raise ValueError(f"Unknown ablation condition: {condition}")

    return out


def truncate_json_by_budget(data: dict[str, Any], keep_fraction: float) -> dict[str, Any]:
    """
    Simple, deterministic packet compression for the first experiment.

    It keeps top-level semantic content but removes lower-priority blocks
    once the serialized packet exceeds the requested budget.

    This is intentionally not a semantic compression algorithm. It is a
    controlled budget study. A later thesis experiment can replace this
    with a learned/LLM compressor.
    """
    priority_order = [
        "state",
        "evidence",
        "uncertainty",
        "repair_events",
        "interaction",
        "provenance",
        "conversation",
    ]

    base = json.loads(json.dumps(data))
    full = len(json.dumps(base, ensure_ascii=False).encode("utf-8"))
    budget = max(1, int(full * keep_fraction))

    while len(json.dumps(base, ensure_ascii=False).encode("utf-8")) > budget:
        removed = False
        for key in priority_order[::-1]:
            if key in base and key not in {"state", "conversation"}:
                del base[key]
                removed = True
                break
        if not removed:
            break

    base["_compression"] = {
        "target_fraction": keep_fraction,
        "original_bytes": full,
        "compressed_bytes": len(
            json.dumps(base, ensure_ascii=False).encode("utf-8")
        ),
    }
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("recap_predictions"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ablation_packets"),
    )
    args = parser.parse_args()

    ablations = [
        "full",
        "no_repair",
        "no_interaction",
        "no_evidence",
        "no_uncertainty",
    ]
    compression = [1.0, 0.75, 0.50, 0.25, 0.10]

    files = sorted(args.input.glob("*.json"))

    for path in files:
        cid = path.stem
        packet = load_json(path)

        for condition in ablations:
            packet_variant = build_ablation(packet, condition)
            dump_json(
                packet_variant,
                args.output / "ablations" / condition / f"{cid}.json",
            )

        for fraction in compression:
            packet_variant = truncate_json_by_budget(
                packet, fraction
            )
            label = f"{int(fraction * 100)}pct"
            dump_json(
                packet_variant,
                args.output / "compression" / label / f"{cid}.json",
            )

    print(
        f"Generated {len(files)} packets across "
        f"{len(ablations)} ablations and {len(compression)} compression levels."
    )


if __name__ == "__main__":
    main()
