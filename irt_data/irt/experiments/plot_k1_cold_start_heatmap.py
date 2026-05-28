#!/usr/bin/env python3
"""Plot K=1 feature-informed cold-start predictions for one benchmark."""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from irt_data.irt.experiments.run_split_v1_sweep import (
    HOLDOUT_CSV,
    OUT_DIR,
    build_explicit_split_dataset,
    logits_for_triplets,
    train_fixed,
)
from irt_data.irt.features import BENCHMARK_REGISTRY


ROOT = Path(__file__).resolve().parents[3]
PAPER_FIGURE_DIR = ROOT / "irt_data" / "paper" / "final_report" / "figures"


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace("-", "_")


def _label(row: pd.Series) -> str:
    model = row.get("model_name") or row.get("model") or "unknown model"
    harness = row.get("scaffold_name") or row.get("scaffold") or "unknown harness"
    return f"{model}\n{harness}"


def cold_start_predictions(benchmark: str, epochs: int) -> pd.DataFrame:
    dataset, holdout_triplets, train_agent_indices, _ = build_explicit_split_dataset()
    model, _ = train_fixed(dataset, k=1, use_features=True, device="cpu", epochs=epochs)
    logits, labels = logits_for_triplets(
        model,
        holdout_triplets,
        dataset,
        train_agent_indices,
        "cpu",
        zero_unseen_agent_theta=True,
    )

    holdout = pd.read_csv(HOLDOUT_CSV)
    holdout = holdout[holdout["benchmark"].isin(BENCHMARK_REGISTRY)].copy()
    if len(holdout) != len(logits):
        raise RuntimeError(
            f"Holdout row mismatch: {len(holdout)} rows, {len(logits)} predictions"
        )

    holdout["predicted_prob"] = 1 / (1 + np.exp(-logits))
    holdout["label"] = labels

    agent_meta = dataset.agent_features.reset_index().rename(columns={"index": "agent_id"})
    keep_agent_cols = [
        "agent_id",
        "model_name",
        "model_family",
        "scaffold_name",
        "scaffold_tool_exposure_mechanism",
    ]
    holdout = holdout.merge(
        agent_meta[[c for c in keep_agent_cols if c in agent_meta.columns]],
        on="agent_id",
        how="left",
    )

    task_disc_path = OUT_DIR / "split_v1_k1_features_task_discrimination.csv"
    if task_disc_path.exists():
        task_disc = pd.read_csv(task_disc_path)
        holdout = holdout.merge(
            task_disc[["benchmark", "task_id", "discrimination", "easiness"]],
            on=["benchmark", "task_id"],
            how="left",
        )
    else:
        holdout["discrimination"] = np.nan
        holdout["easiness"] = np.nan

    preds = holdout[holdout["benchmark"] == benchmark].copy()
    if preds.empty:
        available = ", ".join(sorted(holdout["benchmark"].unique()))
        raise ValueError(f"No holdout rows for {benchmark}. Available: {available}")
    preds["semantic_unit"] = preds.apply(_label, axis=1)
    return preds


def plot_heatmap(preds: pd.DataFrame, benchmark: str, top_tasks: int | None) -> dict:
    summary = (
        preds.groupby(["semantic_unit", "model_name", "scaffold_name"], dropna=False)
        .agg(
            predicted_accuracy=("predicted_prob", "mean"),
            observed_accuracy=("correct", "mean"),
            rows=("correct", "size"),
        )
        .reset_index()
        .sort_values(["scaffold_name", "model_name", "semantic_unit"])
    )

    task_order = (
        preds.groupby("task_id", dropna=False)
        .agg(
            mean_predicted=("predicted_prob", "mean"),
            discrimination=("discrimination", "mean"),
        )
        .reset_index()
        .sort_values(["discrimination", "mean_predicted", "task_id"], ascending=[False, False, True])
    )
    if top_tasks is not None:
        task_order = task_order.head(top_tasks)

    plot_df = preds[preds["task_id"].isin(task_order["task_id"])].copy()
    heat = (
        plot_df.groupby(["task_id", "semantic_unit"], dropna=False)["predicted_prob"]
        .mean()
        .unstack("semantic_unit")
    )
    heat = heat.reindex(index=task_order["task_id"], columns=summary["semantic_unit"])

    fig_width = max(6.5, 1.35 * len(heat.columns))
    fig_height = min(10.0, max(5.5, 0.13 * len(heat.index) + 2.4))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heat.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")

    ax.set_xlabel("Held-out model-harness unit", fontsize=10)
    ax.set_ylabel("Task rank by estimated discrimination", fontsize=10)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(
        ["\n".join(textwrap.wrap(label.replace("\n", " / "), width=22)) for label in heat.columns],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    tick_count = min(10, len(heat.index))
    tick_positions = np.linspace(0, len(heat.index) - 1, tick_count, dtype=int)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(pos + 1) for pos in tick_positions], fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Predicted probability of pass", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()

    tag = f"split_v1_k1_features_cold_start_heatmap_{_safe_name(benchmark)}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    experiment_png = OUT_DIR / f"{tag}.png"
    paper_png = PAPER_FIGURE_DIR / f"{tag}.png"
    fig.savefig(experiment_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    shutil.copyfile(experiment_png, paper_png)

    pred_csv = OUT_DIR / f"split_v1_k1_features_cold_start_predictions_{_safe_name(benchmark)}.csv"
    summary_csv = OUT_DIR / f"split_v1_k1_features_cold_start_summary_{_safe_name(benchmark)}.csv"
    preds.sort_values(["semantic_unit", "discrimination", "task_id"], ascending=[True, False, True]).to_csv(
        pred_csv, index=False
    )
    summary.to_csv(summary_csv, index=False)

    report = {
        "benchmark": benchmark,
        "task_count": int(preds["task_id"].nunique()),
        "semantic_units": int(summary["semantic_unit"].nunique()),
        "rows": int(len(preds)),
        "mean_predicted_accuracy": float(preds["predicted_prob"].mean()),
        "observed_accuracy": float(preds["correct"].mean()),
        "figure": str(paper_png.relative_to(ROOT)),
        "prediction_csv": str(pred_csv.relative_to(ROOT)),
        "summary_csv": str(summary_csv.relative_to(ROOT)),
        "unit_summary": summary.to_dict(orient="records"),
    }
    json_path = OUT_DIR / f"split_v1_k1_features_cold_start_summary_{_safe_name(benchmark)}.json"
    with json_path.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="taubench_airline")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--top-tasks", type=int, default=None)
    args = parser.parse_args()

    preds = cold_start_predictions(args.benchmark, args.epochs)
    report = plot_heatmap(preds, args.benchmark, args.top_tasks)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
