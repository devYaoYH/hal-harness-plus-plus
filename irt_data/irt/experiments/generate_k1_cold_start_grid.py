#!/usr/bin/env python3
"""Generate K=1 feature-informed cold-start normalized model x harness predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from irt_data.irt.data import IRTDataset, _encode_agent_features, _encode_task_features
from irt_data.irt.experiments.run_split_v1_sweep import (
    HOLDOUT_CSV,
    OUT_DIR,
    TRAIN_CSV,
    build_explicit_split_dataset,
    logits_for_triplets,
    train_fixed,
)
from irt_data.irt.features import BENCHMARK_REGISTRY, build_feature_tables


ROOT = Path(__file__).resolve().parents[3]

DISPLAY_MODELS = [
    ("DeepSeek-R1", "together_ai/deepseek-ai/DeepSeek-R1"),
    ("DeepSeek-V3", "together_ai/deepseek-ai/DeepSeek-V3"),
    ("claude-3-7-sonnet", "claude-3-7-sonnet-20250219"),
    ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
    ("claude-opus-4-1", "claude-opus-4-1-20250805"),
    ("claude-opus-4", "claude-opus-4-20250514"),
    ("claude-sonnet-4-5", "claude-sonnet-4-5-20250929"),
    ("gemini-2.0-flash", "gemini/gemini-2.0-flash"),
    ("gpt-4.1", "gpt-4.1"),
    ("gpt-4o", "gpt-4o"),
    ("gpt-4o-2024-11-20", "gpt-4o-2024-11-20"),
    ("gpt-5", "gpt-5-2025-08-07"),
    ("o1", "o1"),
    ("o3", "o3-2025-04-16"),
    ("o3-mini", "o3-mini"),
    ("o3-mini-2025-01-31", "o3-mini-2025-01-31"),
    ("o4-mini", "o4-mini-2025-04-16"),
]

DISPLAY_HARNESSES = [
    ("HAL Generalist Agent", "HAL Generalist Agent"),
    ("SWE-Agent", "SWE-Agent"),
    ("SAB Self-Debug", "SAB Self-Debug Claude-3-7"),
]


def build_dataset_with_synthetic_agents(
    benchmark: str,
) -> tuple[IRTDataset, pd.DataFrame, set[int]]:
    train_df = pd.read_csv(TRAIN_CSV)
    holdout_df = pd.read_csv(HOLDOUT_CSV)
    train_df = train_df[train_df["benchmark"].isin(BENCHMARK_REGISTRY)].copy()
    holdout_df = holdout_df[holdout_df["benchmark"].isin(BENCHMARK_REGISTRY)].copy()

    task_ids = sorted(train_df.loc[train_df["benchmark"] == benchmark, "task_id"].unique())
    synthetic_rows = []
    for model_name, raw_model in DISPLAY_MODELS:
        for harness_name, raw_harness in DISPLAY_HARNESSES:
            agent_id = f"synthetic::{benchmark}::{model_name}::{harness_name}"
            for task_id in task_ids:
                synthetic_rows.append(
                    {
                        "agent_id": agent_id,
                        "scaffold": raw_harness,
                        "model": raw_model,
                        "benchmark": benchmark,
                        "task_id": task_id,
                        "correct": 0,
                        "run_id": "synthetic_cold_start",
                        "display_model": model_name,
                        "display_harness": harness_name,
                    }
                )
    synthetic_df = pd.DataFrame(synthetic_rows)

    all_df = pd.concat(
        [
            train_df,
            holdout_df,
            synthetic_df.drop(columns=["display_model", "display_harness"]),
        ],
        ignore_index=True,
    )
    agent_features, task_features, schema = build_feature_tables(all_df)

    agent_ids = sorted(agent_features.index.tolist())
    task_keys = sorted(task_features.index.tolist())
    agent_id_to_idx = {agent_id: i for i, agent_id in enumerate(agent_ids)}
    task_key_to_idx = {task_key: i for i, task_key in enumerate(task_keys)}

    def encode_response(df: pd.DataFrame) -> np.ndarray:
        encoded = df.copy()
        encoded["agent_idx"] = encoded["agent_id"].map(agent_id_to_idx)
        encoded["task_idx"] = encoded.apply(
            lambda row: task_key_to_idx[(row["benchmark"], row["task_id"])],
            axis=1,
        )
        return encoded[["agent_idx", "task_idx", "correct"]].to_numpy()

    train_triplets = encode_response(train_df)
    holdout_triplets = encode_response(holdout_df)

    agent_cat, agent_cont = _encode_agent_features(agent_features, schema, agent_ids)
    task_cat, task_cont = _encode_task_features(task_features, schema, task_keys)

    dataset = IRTDataset(
        train_triplets=train_triplets,
        val_triplets=holdout_triplets,
        agent_features=agent_features,
        task_features=task_features,
        schema=schema,
        agent_id_to_idx=agent_id_to_idx,
        task_key_to_idx=task_key_to_idx,
        n_agents=len(agent_ids),
        n_tasks=len(task_keys),
        agent_cat_tensor=agent_cat,
        agent_cont_tensor=agent_cont,
        task_cat_tensor=task_cat,
        task_cont_tensor=task_cont,
    )
    train_agent_indices = set(train_triplets[:, 0].astype(int).tolist())

    synthetic_df["agent_idx"] = synthetic_df["agent_id"].map(agent_id_to_idx)
    synthetic_df["task_idx"] = synthetic_df.apply(
        lambda row: task_key_to_idx[(row["benchmark"], row["task_id"])],
        axis=1,
    )
    return dataset, synthetic_df, train_agent_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="swebench_verified_mini")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--min-observed-tasks", type=int, default=25)
    args = parser.parse_args()

    baseline_dataset, _, _, _ = build_explicit_split_dataset()
    model, _ = train_fixed(baseline_dataset, k=1, use_features=True, device="cpu", epochs=args.epochs)

    dataset, synthetic_df, train_agent_indices = build_dataset_with_synthetic_agents(args.benchmark)
    if dataset.n_agents != baseline_dataset.n_agents or dataset.n_tasks != baseline_dataset.n_tasks:
        # Refit only when synthetic feature tables changed the categorical vocabularies.
        torch.manual_seed(20260601)
        model, _ = train_fixed(dataset, k=1, use_features=True, device="cpu", epochs=args.epochs)

    triplets = synthetic_df[["agent_idx", "task_idx", "correct"]].to_numpy()
    logits, _ = logits_for_triplets(
        model,
        triplets,
        dataset,
        train_agent_indices,
        "cpu",
        zero_unseen_agent_theta=True,
    )
    synthetic_df["predicted_prob"] = 1 / (1 + np.exp(-logits))

    grid = (
        synthetic_df.groupby(["display_harness", "display_model"], as_index=False)
        .agg(predicted_accuracy=("predicted_prob", "mean"), tasks=("task_id", "nunique"))
        .sort_values(["display_harness", "display_model"])
    )

    actual = pd.read_csv(ROOT / "irt_data" / "response_matrix.csv")
    actual = actual[actual["benchmark"] == args.benchmark].copy()
    actual["display_harness"] = actual["scaffold"].map(
        lambda s: "SWE-Agent"
        if s in {"My Agent", "SWE-Agent"}
        else (
            "HAL Generalist Agent"
            if str(s).startswith(("HAL Generalist", "hal_generalist_agent"))
            else ("SAB Self-Debug" if str(s).startswith(("SAB Self-Debug", "sab_selfdebug")) else s)
        )
    )
    model_lookup = {raw: name for name, raw in DISPLAY_MODELS}
    actual["display_model"] = actual["model"].map(model_lookup)
    actual.loc[actual["model"].str.contains("DeepSeek-R1", na=False), "display_model"] = "DeepSeek-R1"
    actual.loc[actual["model"].str.contains("DeepSeek-V3", na=False), "display_model"] = "DeepSeek-V3"
    actual.loc[actual["model"].str.contains("claude-3-7", na=False), "display_model"] = "claude-3-7-sonnet"
    actual.loc[actual["model"].str.contains("claude-haiku-4-5", na=False), "display_model"] = "claude-haiku-4-5"
    actual.loc[actual["model"].str.contains("claude-opus-4-1", na=False), "display_model"] = "claude-opus-4-1"
    actual.loc[actual["model"].str.contains("claude-opus-4-20250514", na=False), "display_model"] = "claude-opus-4"
    actual.loc[actual["model"].str.contains("claude-sonnet-4-5", na=False), "display_model"] = "claude-sonnet-4-5"
    actual.loc[actual["model"].str.contains("gemini-2.0-flash", na=False), "display_model"] = "gemini-2.0-flash"
    actual.loc[actual["model"].str.contains("gpt-4.1", na=False), "display_model"] = "gpt-4.1"
    actual.loc[actual["model"].eq("gpt-4o"), "display_model"] = "gpt-4o"
    actual.loc[actual["model"].str.contains("gpt-4o-2024", na=False), "display_model"] = "gpt-4o-2024-11-20"
    actual.loc[actual["model"].str.contains("gpt-5", na=False), "display_model"] = "gpt-5"
    actual.loc[actual["model"].eq("o1"), "display_model"] = "o1"
    actual.loc[actual["model"].str.contains("o3-mini-2025", na=False), "display_model"] = "o3-mini-2025-01-31"
    actual.loc[actual["model"].eq("o3-mini"), "display_model"] = "o3-mini"
    actual.loc[actual["model"].str.contains("o3-2025", na=False), "display_model"] = "o3"
    actual.loc[actual["model"].str.contains("o4-mini", na=False), "display_model"] = "o4-mini"
    actual_grid = (
        actual.dropna(subset=["display_model"])
        .groupby(["display_harness", "display_model"], as_index=False)
        .agg(actual_accuracy=("correct", "mean"), observed_tasks=("task_id", "nunique"))
    )
    grid = grid.merge(actual_grid, on=["display_harness", "display_model"], how="left")
    grid["is_observed"] = grid["observed_tasks"].fillna(0) >= args.min_observed_tasks

    out_csv = OUT_DIR / f"k1_cold_start_model_harness_grid_{args.benchmark}.csv"
    out_json = OUT_DIR / f"k1_cold_start_model_harness_grid_{args.benchmark}.json"
    grid.to_csv(out_csv, index=False)
    with out_json.open("w") as handle:
        json.dump(grid.to_dict(orient="records"), handle, indent=2)
        handle.write("\n")
    print(grid.to_string(index=False))
    print(f"\nWrote {out_csv.relative_to(ROOT)}")
    print(f"Wrote {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
