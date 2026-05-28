#!/usr/bin/env python3
"""Mechanism-level logit contributions for split-v1 K=1 feature MIRT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from irt_data.irt.experiments.run_split_v1_sweep import (
    HOLDOUT_CSV,
    build_explicit_split_dataset,
    feature_kwargs,
    train_fixed,
)


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "irt_data" / "irt" / "experiments"


def main() -> None:
    dataset, holdout_triplets, train_agent_indices, _ = build_explicit_split_dataset()
    model, _ = train_fixed(dataset, k=1, use_features=True, device="cpu", epochs=120)
    feats = feature_kwargs(dataset, "cpu")

    with torch.no_grad():
        agent_idx = torch.arange(dataset.n_agents)
        task_idx = torch.arange(dataset.n_tasks)
        f_agent = model._encode_agent_features(
            feats["agent_cat"][agent_idx], feats["agent_cont"][agent_idx]
        ).squeeze(-1).numpy()
        a_effective = (
            model.a(task_idx)
            + model._encode_task_features(
                feats["task_cat"][task_idx], feats["task_cont"][task_idx]
            )
        ).squeeze(-1).numpy()
        easiness = model.d(task_idx).squeeze(-1).numpy()

    holdout = pd.read_csv(HOLDOUT_CSV)
    model_meta = pd.read_csv(ROOT / "irt_data" / "model_metadata.csv")
    harness_meta = pd.read_csv(ROOT / "irt_data" / "harness_metadata.csv")
    holdout = holdout.merge(
        model_meta[["raw_model", "normalized_model", "model_family"]],
        left_on="model",
        right_on="raw_model",
        how="left",
    ).merge(
        harness_meta[
            [
                "raw_scaffold",
                "normalized_harness",
                "tool_exposure_mechanism",
            ]
        ],
        left_on="scaffold",
        right_on="raw_scaffold",
        how="left",
    )

    agent_lookup = {idx: agent_id for agent_id, idx in dataset.agent_id_to_idx.items()}
    task_lookup = {idx: key for key, idx in dataset.task_key_to_idx.items()}
    agent_index_by_id = dataset.agent_id_to_idx
    task_index_by_key = dataset.task_key_to_idx

    rows = []
    for row in holdout.itertuples(index=False):
        ai = agent_index_by_id[row.agent_id]
        ti = task_index_by_key[(row.benchmark, row.task_id)]
        contribution = float(f_agent[ai] * a_effective[ti])
        logit = contribution + float(easiness[ti])
        prob = 1 / (1 + np.exp(-logit))
        rows.append(
            {
                "agent_id": row.agent_id,
                "benchmark": row.benchmark,
                "task_id": row.task_id,
                "correct": row.correct,
                "normalized_model": row.normalized_model,
                "model_family": row.model_family,
                "normalized_harness": row.normalized_harness,
                "tool_exposure_mechanism": row.tool_exposure_mechanism,
                "f_agent": float(f_agent[ai]),
                "effective_task_discrimination": float(a_effective[ti]),
                "task_easiness": float(easiness[ti]),
                "agent_task_logit_contribution": contribution,
                "logit": logit,
                "probability": prob,
            }
        )

    detailed = pd.DataFrame(rows)
    detailed.to_csv(OUT_DIR / "split_v1_k1_mechanism_logit_contributions_by_row.csv", index=False)

    summary = (
        detailed.groupby("tool_exposure_mechanism")
        .agg(
            rows=("correct", "size"),
            agent_configs=("agent_id", "nunique"),
            mean_correct=("correct", "mean"),
            mean_probability=("probability", "mean"),
            mean_f_agent=("f_agent", "mean"),
            mean_effective_task_discrimination=("effective_task_discrimination", "mean"),
            mean_logit_contribution=("agent_task_logit_contribution", "mean"),
            median_logit_contribution=("agent_task_logit_contribution", "median"),
            min_logit_contribution=("agent_task_logit_contribution", "min"),
            max_logit_contribution=("agent_task_logit_contribution", "max"),
            mean_task_easiness=("task_easiness", "mean"),
        )
        .reset_index()
        .sort_values("mean_logit_contribution", ascending=False)
    )
    summary.to_csv(OUT_DIR / "split_v1_k1_mechanism_logit_contributions.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
