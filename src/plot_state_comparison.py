#!/usr/bin/env python3
"""plot_state_comparison.py

Generate comparison plots for state extraction results.
Inputs: one or more state_eval_summary.json files.
Outputs: bar charts and optional gold-vs-pred scatter plots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def load_summary(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_metric_table(method_summaries: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for method, summ in method_summaries.items():
        rows.append({
            "method": method,
            "primary_intent_accuracy": summ["primary_intent"]["accuracy"],
            "secondary_intents_f1": summ["secondary_intents"]["f1"],
            "secondary_intents_exact_match": summ["secondary_intents"]["exact_match"],
            "multi_intent_accuracy": summ["multi_intent_accuracy"],
            "tool_failure_accuracy": summ["tool_failure_accuracy"],
            "ambiguity_mae": summ["ambiguity"]["mae"],
            "sentiment_mae": summ["sentiment_overall"]["mae"],
            "schema_valid_rate": summ["schema_valid_rate"],
        })
    return pd.DataFrame(rows)


def plot_metric_bars(df: pd.DataFrame, out_dir: Path):
    metrics = [
        "primary_intent_accuracy",
        "secondary_intents_f1",
        "secondary_intents_exact_match",
        "multi_intent_accuracy",
        "tool_failure_accuracy",
        "schema_valid_rate",
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        ax.bar(df["method"], df[metric])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(df[metric]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(out_dir / "state_metrics_comparison.png", dpi=200)
    plt.close(fig)


def plot_error_metrics(df: pd.DataFrame, out_dir: Path):
    metrics = ["ambiguity_mae", "sentiment_mae"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        ax.bar(df["method"], df[metric])
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(df[metric]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(out_dir / "state_error_comparison.png", dpi=200)
    plt.close(fig)


def plot_gold_vs_pred(rows_df: pd.DataFrame, out_dir: Path, method_name: str):
    pairs = [
        ("gold_ambiguity_level", "pred_ambiguity_level", "Ambiguity"),
        ("gold_sentiment_overall", "pred_sentiment_overall", "Sentiment"),
        ("gold_turn_count", "pred_turn_count", "Turn Count"),
    ]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), constrained_layout=True)
    if len(pairs) == 1:
        axes = [axes]
    for ax, (g, p, title) in zip(axes, pairs):
        ax.scatter(rows_df[g], rows_df[p], alpha=0.6)
        mn = min(rows_df[g].min(), rows_df[p].min())
        mx = max(rows_df[g].max(), rows_df[p].max())
        ax.plot([mn, mx], [mn, mx], linestyle="--")
        ax.set_xlabel(f"Gold {title}")
        ax.set_ylabel(f"Pred {title}")
        ax.set_title(title)
    fig.savefig(out_dir / f"{method_name}_gold_vs_pred.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot comparison charts for state extraction")
    ap.add_argument("--summaries", nargs="+", required=True, help="Paths to state_eval_summary.json files")
    ap.add_argument("--rows", nargs="*", default=None, help="Optional paths to state_eval_rows.csv files (same order)")
    ap.add_argument("--labels", nargs="*", default=None, help="Method labels in same order as summaries")
    ap.add_argument("--output-dir", default="results/state_plots")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels or [Path(p).parent.name for p in args.summaries]
    method_summaries = {lab: load_summary(Path(p)) for lab, p in zip(labels, args.summaries)}
    table = build_metric_table(method_summaries)
    table.to_csv(out_dir / "state_metrics_table.csv", index=False)

    plot_metric_bars(table, out_dir)
    plot_error_metrics(table, out_dir)

    if args.rows:
        for label, row_path in zip(labels, args.rows):
            plot_gold_vs_pred(load_rows(Path(row_path)), out_dir, label)

    print(f"Saved plots and tables to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
