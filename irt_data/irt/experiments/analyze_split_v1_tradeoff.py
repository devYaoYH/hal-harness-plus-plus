#!/usr/bin/env python3
"""Join split-v1 holdout metrics with task-reduction summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "irt_data" / "irt" / "experiments"
FIG_DIR = ROOT / "irt_data" / "paper" / "final_report" / "figures"
THRESHOLDS = [0.7, 0.8, 0.9]


def cost_summary(discrimination_csv: Path, threshold: float) -> dict[str, float]:
    df = pd.read_csv(discrimination_csv)
    rows = []
    for _, group in df.groupby("benchmark", sort=True):
        ranked = group.sort_values("discrimination", ascending=False)
        cumsum = ranked["discrimination"].cumsum() / ranked["discrimination"].sum()
        n_total = len(ranked)
        n_needed = min(int((cumsum.values <= threshold).sum()) + 1, n_total)
        rows.append(
            {
                "fraction_needed": n_needed / n_total,
                "cost_reduction": 1 - n_needed / n_total,
            }
        )
    out = pd.DataFrame(rows)
    return {
        f"mean_fraction_needed_{int(threshold * 100)}pct": out["fraction_needed"].mean(),
        f"median_fraction_needed_{int(threshold * 100)}pct": out["fraction_needed"].median(),
        f"mean_cost_reduction_{int(threshold * 100)}pct": out["cost_reduction"].mean(),
        f"median_cost_reduction_{int(threshold * 100)}pct": out["cost_reduction"].median(),
    }


def build_tradeoff_table() -> pd.DataFrame:
    metrics = pd.read_csv(EXP_DIR / "split_v1_sweep_results.csv")
    rows = []
    for row in metrics.itertuples(index=False):
        variant = "features" if row.use_features else "latent_only"
        disc_path = EXP_DIR / f"split_v1_k{row.k}_{variant}_task_discrimination.csv"
        entry = row._asdict()
        for threshold in THRESHOLDS:
            entry.update(cost_summary(disc_path, threshold))
        entry["variant"] = f"K={row.k} {'features' if row.use_features else 'latent-only'}"
        rows.append(entry)
    return pd.DataFrame(rows)


def plot_tradeoff(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for use_features, group in table.groupby("use_features"):
        marker = "o" if use_features else "s"
        label = "feature-informed" if use_features else "latent-only"
        ax.scatter(
            group["mean_cost_reduction_80pct"],
            group["holdout_auc"],
            s=90,
            marker=marker,
            label=label,
        )
        for r in group.itertuples(index=False):
            ax.annotate(
                f"K={r.k}",
                (r.mean_cost_reduction_80pct, r.holdout_auc),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
    ax.set_xlabel("Mean task-count reduction at 80% discrimination coverage")
    ax.set_ylabel("Holdout AUC")
    ax.set_title("Prediction vs. Benchmark-Reduction Tradeoff")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for path in [
        EXP_DIR / "split_v1_auc_reduction_tradeoff.png",
        FIG_DIR / "split_v1_auc_reduction_tradeoff.png",
    ]:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    table = build_tradeoff_table()
    table = table.sort_values(["holdout_auc", "mean_cost_reduction_80pct"], ascending=False)
    table.to_csv(EXP_DIR / "split_v1_auc_reduction_tradeoff.csv", index=False)
    plot_tradeoff(table)
    cols = [
        "k",
        "use_features",
        "holdout_bce",
        "holdout_auc",
        "holdout_accuracy",
        "mean_cost_reduction_70pct",
        "mean_cost_reduction_80pct",
        "mean_cost_reduction_90pct",
    ]
    print(table[cols].to_string(index=False))


if __name__ == "__main__":
    main()
