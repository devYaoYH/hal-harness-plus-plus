#!/usr/bin/env python3
"""Ablation attribution for the split-v1 K=1 feature-informed MIRT model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from irt_data.irt.experiments.run_split_v1_sweep import (
    OUT_DIR,
    build_explicit_split_dataset,
    evaluate_triplets,
    feature_kwargs,
    train_fixed,
)


ROOT = Path(__file__).resolve().parents[3]
ATTRIBUTION_DIR = ROOT / "irt_data" / "irt" / "experiments"


def evaluate_with_feature_ablation(
    model,
    dataset,
    holdout_triplets,
    train_agent_indices,
    feature_group: str,
    feature_name: str,
):
    saved_agent_cat = dataset.agent_cat_tensor.clone()
    saved_agent_cont = dataset.agent_cont_tensor.clone()
    saved_task_cat = dataset.task_cat_tensor.clone()
    saved_task_cont = dataset.task_cont_tensor.clone()

    try:
        if feature_group == "agent_categorical":
            idx = dataset.schema.categorical_agent.index(feature_name)
            dataset.agent_cat_tensor[:, idx] = 0
        elif feature_group == "agent_continuous":
            idx = dataset.schema.continuous_agent.index(feature_name)
            dataset.agent_cont_tensor[:, idx] = 0.0
        elif feature_group == "agent_boolean":
            offset = len(dataset.schema.continuous_agent)
            idx = offset + dataset.schema.boolean_agent.index(feature_name)
            dataset.agent_cont_tensor[:, idx] = 0.0
        elif feature_group == "task_categorical":
            idx = dataset.schema.categorical_task.index(feature_name)
            dataset.task_cat_tensor[:, idx] = 0
        elif feature_group == "task_continuous":
            idx = dataset.schema.continuous_task.index(feature_name)
            dataset.task_cont_tensor[:, idx] = 0.0
        elif feature_group == "task_boolean":
            offset = len(dataset.schema.continuous_task)
            idx = offset + dataset.schema.boolean_task.index(feature_name)
            dataset.task_cont_tensor[:, idx] = 0.0
        else:
            raise ValueError(feature_group)

        return evaluate_triplets(
            model,
            holdout_triplets,
            dataset,
            train_agent_indices,
            "cpu",
            zero_unseen_agent_theta=True,
        )
    finally:
        dataset.agent_cat_tensor = saved_agent_cat
        dataset.agent_cont_tensor = saved_agent_cont
        dataset.task_cat_tensor = saved_task_cat
        dataset.task_cont_tensor = saved_task_cont


def projection_values(model, dataset, train_agent_indices):
    feats = feature_kwargs(dataset, "cpu")
    with torch.no_grad():
        all_agents = torch.arange(dataset.n_agents)
        all_tasks = torch.arange(dataset.n_tasks)
        f_agent = model._encode_agent_features(
            feats["agent_cat"][all_agents], feats["agent_cont"][all_agents]
        ).squeeze(-1).numpy()
        f_task = model._encode_task_features(
            feats["task_cat"][all_tasks], feats["task_cont"][all_tasks]
        ).squeeze(-1).numpy()

    idx_to_agent = {idx: agent_id for agent_id, idx in dataset.agent_id_to_idx.items()}
    agent_rows = []
    for idx, value in enumerate(f_agent):
        row = dataset.agent_features.loc[idx_to_agent[idx]].to_dict()
        agent_rows.append(
            {
                "agent_id": idx_to_agent[idx],
                "is_train_agent": idx in train_agent_indices,
                "f_agent": float(value),
                "model_name": row.get("model_name"),
                "model_provider": row.get("model_provider"),
                "model_family": row.get("model_family"),
                "scaffold_name": row.get("scaffold_name"),
                "scaffold_tool_exposure_mechanism": row.get(
                    "scaffold_tool_exposure_mechanism"
                ),
            }
        )

    idx_to_task = {idx: key for key, idx in dataset.task_key_to_idx.items()}
    task_rows = []
    for idx, value in enumerate(f_task):
        benchmark, task_id = idx_to_task[idx]
        row = dataset.task_features.loc[(benchmark, task_id)].to_dict()
        task_rows.append(
            {
                "benchmark": benchmark,
                "task_id": task_id,
                "f_task": float(value),
                "bench_domain": row.get("bench_domain"),
                "bench_task_type": row.get("bench_task_type"),
                "bench_requires_code": row.get("bench_requires_code"),
                "bench_requires_web": row.get("bench_requires_web"),
                "bench_requires_reasoning": row.get("bench_requires_reasoning"),
            }
        )

    return pd.DataFrame(agent_rows), pd.DataFrame(task_rows)


def main() -> None:
    dataset, holdout_triplets, train_agent_indices, _ = build_explicit_split_dataset()
    model, _ = train_fixed(dataset, k=1, use_features=True, device="cpu", epochs=120)

    baseline = evaluate_triplets(
        model,
        holdout_triplets,
        dataset,
        train_agent_indices,
        "cpu",
        zero_unseen_agent_theta=True,
    )

    specs = []
    specs.extend(("agent_categorical", f) for f in dataset.schema.categorical_agent)
    specs.extend(("agent_continuous", f) for f in dataset.schema.continuous_agent)
    specs.extend(("agent_boolean", f) for f in dataset.schema.boolean_agent)
    specs.extend(("task_categorical", f) for f in dataset.schema.categorical_task)
    specs.extend(("task_continuous", f) for f in dataset.schema.continuous_task)
    specs.extend(("task_boolean", f) for f in dataset.schema.boolean_task)

    rows = []
    for feature_group, feature_name in specs:
        metrics = evaluate_with_feature_ablation(
            model,
            dataset,
            holdout_triplets,
            train_agent_indices,
            feature_group,
            feature_name,
        )
        rows.append(
            {
                "feature_group": feature_group,
                "feature_name": feature_name,
                "baseline_bce": baseline["bce"],
                "ablated_bce": metrics["bce"],
                "delta_bce": metrics["bce"] - baseline["bce"],
                "baseline_auc": baseline["auc"],
                "ablated_auc": metrics["auc"],
                "delta_auc": metrics["auc"] - baseline["auc"],
                "baseline_accuracy": baseline["accuracy"],
                "ablated_accuracy": metrics["accuracy"],
                "delta_accuracy": metrics["accuracy"] - baseline["accuracy"],
                "baseline_brier": baseline["brier"],
                "ablated_brier": metrics["brier"],
                "delta_brier": metrics["brier"] - baseline["brier"],
            }
        )

    attribution = pd.DataFrame(rows).sort_values(
        ["delta_bce", "delta_auc"], ascending=[False, True]
    )
    attribution.to_csv(ATTRIBUTION_DIR / "split_v1_k1_feature_ablation_attribution.csv", index=False)

    agent_values, task_values = projection_values(model, dataset, train_agent_indices)
    agent_values.to_csv(ATTRIBUTION_DIR / "split_v1_k1_f_agent_values.csv", index=False)
    task_values.to_csv(ATTRIBUTION_DIR / "split_v1_k1_f_task_values.csv", index=False)

    summaries = {
        "baseline": baseline,
        "top_positive_delta_bce": attribution.head(15).to_dict(orient="records"),
        "top_negative_delta_bce": attribution.tail(15).to_dict(orient="records"),
        "f_agent_by_harness": agent_values.groupby("scaffold_name")["f_agent"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .to_dict(orient="records"),
        "f_agent_by_model_family": agent_values.groupby("model_family")["f_agent"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .to_dict(orient="records"),
        "f_task_by_task_type": task_values.groupby("bench_task_type")["f_task"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .to_dict(orient="records"),
        "f_task_by_domain": task_values.groupby("bench_domain")["f_task"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .to_dict(orient="records"),
    }
    with (ATTRIBUTION_DIR / "split_v1_k1_feature_attribution_summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)
        handle.write("\n")

    print("Baseline:", baseline)
    print("\nTop positive delta BCE:")
    print(attribution.head(15).to_string(index=False))
    print("\nf_agent by harness:")
    print(
        agent_values.groupby("scaffold_name")["f_agent"]
        .agg(["count", "mean", "std", "min", "max"])
        .sort_values("mean", ascending=False)
        .head(12)
        .to_string()
    )
    print("\nf_task by task type:")
    print(
        task_values.groupby("bench_task_type")["f_task"]
        .agg(["count", "mean", "std", "min", "max"])
        .sort_values("mean", ascending=False)
        .to_string()
    )


if __name__ == "__main__":
    main()
