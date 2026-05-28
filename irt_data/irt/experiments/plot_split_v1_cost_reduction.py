#!/usr/bin/env python3
"""Plot benchmark-size reduction from split-v1 feature discrimination."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = ROOT / "irt_data" / "irt" / "experiments"
PAPER_FIGURE_DIR = ROOT / "irt_data" / "paper" / "final_report" / "figures"
MODEL_TAG = "k1_features"
MODEL_LABEL = "MIRT K=1 feature-informed"
DISCRIMINATION_CSV = EXPERIMENT_DIR / f"split_v1_{MODEL_TAG}_task_discrimination.csv"
THRESHOLDS = [0.5, 0.7, 0.8, 0.9]


def build_cost_table(discrimination: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for benchmark, group in discrimination.groupby("benchmark", sort=True):
        ranked = group.sort_values("discrimination", ascending=False).reset_index(drop=True)
        total_discrimination = ranked["discrimination"].sum()
        cumulative = ranked["discrimination"].cumsum() / total_discrimination
        n_total = len(ranked)
        for threshold in THRESHOLDS:
            n_needed = int((cumulative.values <= threshold).sum()) + 1
            n_needed = min(n_needed, n_total)
            rows.append(
                {
                    "benchmark": benchmark,
                    "threshold": f"{int(threshold * 100)}pct",
                    "discrimination_coverage": threshold,
                    "n_total": n_total,
                    "n_needed": n_needed,
                    "n_removed": n_total - n_needed,
                    "fraction_needed": n_needed / n_total,
                    "cost_reduction": 1 - (n_needed / n_total),
                }
            )
    return pd.DataFrame(rows)


def write_subsets(discrimination: pd.DataFrame, cost_table: pd.DataFrame) -> None:
    subsets = {}
    for benchmark, group in discrimination.groupby("benchmark", sort=True):
        ranked = group.sort_values("discrimination", ascending=False).reset_index(drop=True)
        thresholds = {}
        for row in cost_table[cost_table["benchmark"] == benchmark].itertuples():
            task_ids = ranked["task_id"].iloc[: row.n_needed].astype(str).tolist()
            thresholds[row.threshold] = {
                "n_tasks": int(row.n_needed),
                "n_total": int(row.n_total),
                "fraction": round(float(row.fraction_needed), 4),
                "cost_reduction": round(float(row.cost_reduction), 4),
                "task_ids": task_ids,
            }
        subsets[benchmark] = {
            "source": str(DISCRIMINATION_CSV.relative_to(ROOT)),
            "model": MODEL_LABEL,
            "thresholds": thresholds,
        }
    with (EXPERIMENT_DIR / f"split_v1_{MODEL_TAG}_adaptive_task_subsets.json").open("w") as handle:
        json.dump(subsets, handle, indent=2)
        handle.write("\n")


def plot_cost_reduction(cost_table: pd.DataFrame) -> None:
    benchmarks = sorted(cost_table["benchmark"].unique())
    pretty = [b.replace("_", "\n") for b in benchmarks]
    x = np.arange(len(benchmarks))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = plt.cm.viridis(np.linspace(0.18, 0.82, len(THRESHOLDS)))
    for i, threshold in enumerate(THRESHOLDS):
        label = f"{int(threshold * 100)}%"
        subset = cost_table[cost_table["threshold"] == f"{int(threshold * 100)}pct"]
        subset = subset.set_index("benchmark").loc[benchmarks].reset_index()
        offset = (i - (len(THRESHOLDS) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            subset["fraction_needed"].values,
            width,
            label=label,
            color=colors[i],
            alpha=0.9,
        )
        for bar, row in zip(bars, subset.itertuples()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(bar.get_height() + 0.02, 1.03),
                str(row.n_needed),
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(pretty, fontsize=8)
    ax.set_ylabel("Fraction of benchmark tasks retained")
    ax.set_title("Tasks Needed to Retain Fixed Fractions of K=1 Discrimination Mass")
    ax.set_ylim(0, 1.12)
    ax.legend(title="Discrimination retained", ncols=len(THRESHOLDS), fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    experiment_png = EXPERIMENT_DIR / f"split_v1_{MODEL_TAG}_cost_reduction.png"
    paper_png = PAPER_FIGURE_DIR / f"split_v1_{MODEL_TAG}_cost_reduction.png"
    fig.savefig(experiment_png, dpi=180, bbox_inches="tight")
    fig.savefig(paper_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    discrimination = pd.read_csv(DISCRIMINATION_CSV)
    cost_table = build_cost_table(discrimination)
    cost_table.to_csv(EXPERIMENT_DIR / f"split_v1_{MODEL_TAG}_cost_reduction.csv", index=False)
    write_subsets(discrimination, cost_table)
    plot_cost_reduction(cost_table)

    summary = (
        cost_table.groupby("threshold")
        .agg(
            mean_fraction_needed=("fraction_needed", "mean"),
            median_fraction_needed=("fraction_needed", "median"),
            mean_cost_reduction=("cost_reduction", "mean"),
            median_cost_reduction=("cost_reduction", "median"),
        )
        .reset_index()
    )
    summary.to_csv(EXPERIMENT_DIR / f"split_v1_{MODEL_TAG}_cost_reduction_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
