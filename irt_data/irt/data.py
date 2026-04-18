"""
Data loading, filtering, and train/validation splitting for MIRT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import FeatureSchema, build_feature_tables


@dataclass
class IRTDataset:
    """Container for train/val split with feature tables and index mappings."""

    train_triplets: np.ndarray  # (N, 3): [agent_idx, task_idx, correct]
    val_triplets: np.ndarray

    agent_features: pd.DataFrame
    task_features: pd.DataFrame
    schema: FeatureSchema

    agent_id_to_idx: dict[str, int]
    task_key_to_idx: dict[tuple[str, str], int]

    n_agents: int
    n_tasks: int

    # For encoding categorical/continuous features into tensors
    agent_cat_tensor: torch.Tensor | None = None
    agent_cont_tensor: torch.Tensor | None = None
    task_cat_tensor: torch.Tensor | None = None
    task_cont_tensor: torch.Tensor | None = None

    def summary(self) -> str:
        lines = [
            f"Agents: {self.n_agents}, Tasks: {self.n_tasks}",
            f"Train: {len(self.train_triplets)} observations",
            f"Val:   {len(self.val_triplets)} observations",
        ]
        return "\n".join(lines)


class TripletDataset(Dataset):
    """PyTorch dataset wrapping (agent_idx, task_idx, correct) triplets."""

    def __init__(self, triplets: np.ndarray):
        self.agent_idx = torch.from_numpy(triplets[:, 0].astype(np.int64))
        self.task_idx = torch.from_numpy(triplets[:, 1].astype(np.int64))
        self.correct = torch.from_numpy(triplets[:, 2].astype(np.float32))

    def __len__(self):
        return len(self.correct)

    def __getitem__(self, idx):
        return self.agent_idx[idx], self.task_idx[idx], self.correct[idx]


def load_and_split(
    response_csv: str | Path = "irt_data/response_matrix.csv",
    min_tasks_per_agent: int = 10,
    min_agents_per_task: int = 3,
    val_agent_fraction: float = 0.2,
    val_item_fraction: float = 0.3,
    seed: int = 42,
) -> IRTDataset:
    """
    Load response matrix, filter, extract features, and split into train/val.

    Holdout strategy:
      1. Select val agents (those with enough data across benchmarks).
      2. For each val agent, hold out val_item_fraction of their responses.
      3. The held-in portion is added to training (so we can fit θ for val agents).
      4. Held-out portion is used for evaluation.
    """
    rng = np.random.RandomState(seed)

    df = pd.read_csv(response_csv)

    # Filter to known benchmarks
    from .features import BENCHMARK_REGISTRY
    df = df[df["benchmark"].isin(BENCHMARK_REGISTRY)]

    # Iterative filtering: agents with enough tasks, tasks with enough agents
    for _ in range(3):
        agent_counts = df.groupby("agent_id")["task_id"].nunique()
        keep_agents = agent_counts[agent_counts >= min_tasks_per_agent].index
        df = df[df["agent_id"].isin(keep_agents)]

        task_counts = df.groupby(["benchmark", "task_id"])["agent_id"].nunique()
        keep_tasks = task_counts[task_counts >= min_agents_per_task].index
        df = df[df.set_index(["benchmark", "task_id"]).index.isin(keep_tasks)].reset_index(drop=True)

    # Build feature tables
    agent_features, task_features, schema = build_feature_tables(df)

    # Build index mappings
    agent_ids = sorted(agent_features.index.tolist())
    agent_id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}

    task_keys = sorted(task_features.index.tolist())
    task_key_to_idx = {tk: i for i, tk in enumerate(task_keys)}

    # Map responses to indices
    df["agent_idx"] = df["agent_id"].map(agent_id_to_idx)
    df["task_idx"] = df.apply(
        lambda r: task_key_to_idx.get((r["benchmark"], r["task_id"])), axis=1
    )
    df = df.dropna(subset=["agent_idx", "task_idx"])
    df["agent_idx"] = df["agent_idx"].astype(int)
    df["task_idx"] = df["task_idx"].astype(int)

    # Select validation agents: prefer those appearing across multiple benchmarks
    agent_bench_counts = df.groupby("agent_id")["benchmark"].nunique()
    eligible = agent_bench_counts[agent_bench_counts >= 1].index.tolist()
    # Sort by number of benchmarks (descending) so multi-benchmark agents are preferred
    eligible.sort(key=lambda a: -agent_bench_counts[a])
    n_val = max(1, int(len(eligible) * val_agent_fraction))
    val_agents = set(eligible[:n_val])

    # Split: for val agents, hold out a fraction of their responses
    val_mask = df["agent_id"].isin(val_agents)
    val_df = df[val_mask]
    train_df = df[~val_mask]

    # For each val agent, split their responses
    val_held_out_rows = []
    val_held_in_rows = []
    for agent_id, grp in val_df.groupby("agent_id"):
        indices = grp.index.tolist()
        rng.shuffle(indices)
        split_point = max(1, int(len(indices) * (1 - val_item_fraction)))
        val_held_in_rows.extend(indices[:split_point])
        val_held_out_rows.extend(indices[split_point:])

    # Training = non-val agents + held-in portion of val agents
    train_indices = train_df.index.tolist() + val_held_in_rows
    val_indices = val_held_out_rows

    train_triplets = df.loc[train_indices, ["agent_idx", "task_idx", "correct"]].values
    val_triplets = df.loc[val_indices, ["agent_idx", "task_idx", "correct"]].values

    # Encode features into tensors
    agent_cat, agent_cont = _encode_agent_features(agent_features, schema, agent_ids)
    task_cat, task_cont = _encode_task_features(task_features, schema, task_keys)

    return IRTDataset(
        train_triplets=train_triplets,
        val_triplets=val_triplets,
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


def _encode_agent_features(
    features: pd.DataFrame, schema: FeatureSchema, ordered_ids: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode agent features into categorical (int) and continuous (float) tensors."""
    features = features.loc[ordered_ids]

    cat_cols = [c for c in schema.categorical_agent if c in features.columns]
    cont_cols = [c for c in schema.continuous_agent + schema.boolean_agent if c in features.columns]

    # Categorical: label encode, NaN → 0 (reserved for unknown)
    cat_data = []
    for col in cat_cols:
        codes, _ = pd.factorize(features[col], sort=True)
        cat_data.append(codes + 1)  # shift so 0 = unknown/NaN
    cat_tensor = torch.from_numpy(np.column_stack(cat_data).astype(np.int64)) if cat_data else torch.zeros(len(features), 0, dtype=torch.long)

    # Continuous + boolean: fill NaN with 0, normalize
    cont_data = []
    for col in cont_cols:
        vals = features[col].copy()
        if vals.dtype == object or vals.dtype == bool:
            vals = vals.map({True: 1.0, False: 0.0, np.nan: 0.0}).astype(float)
        else:
            vals = vals.fillna(0.0).astype(float)
        cont_data.append(vals.values)
    cont_tensor = torch.from_numpy(np.column_stack(cont_data).astype(np.float32)) if cont_data else torch.zeros(len(features), 0)

    return cat_tensor, cont_tensor


def _encode_task_features(
    features: pd.DataFrame, schema: FeatureSchema, ordered_keys: list[tuple],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode task features into categorical and continuous tensors."""
    features = features.loc[ordered_keys]

    cat_cols = [c for c in schema.categorical_task if c in features.columns]
    cont_cols = [c for c in schema.continuous_task + schema.boolean_task if c in features.columns]

    cat_data = []
    for col in cat_cols:
        codes, _ = pd.factorize(features[col], sort=True)
        cat_data.append(codes + 1)
    cat_tensor = torch.from_numpy(np.column_stack(cat_data).astype(np.int64)) if cat_data else torch.zeros(len(features), 0, dtype=torch.long)

    cont_data = []
    for col in cont_cols:
        vals = features[col].copy()
        if vals.dtype == object or vals.dtype == bool:
            vals = vals.map({True: 1.0, False: 0.0, np.nan: 0.0}).astype(float)
        else:
            vals = vals.fillna(0.0).astype(float)
        cont_data.append(vals.values)
    cont_tensor = torch.from_numpy(np.column_stack(cont_data).astype(np.float32)) if cont_data else torch.zeros(len(features), 0)

    return cat_tensor, cont_tensor
