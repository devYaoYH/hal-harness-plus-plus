#!/usr/bin/env python3
"""Run MIRT K sweep on the canonical split-v1 train/holdout files."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader

from irt_data.irt.data import (
    IRTDataset,
    TripletDataset,
    _encode_agent_features,
    _encode_task_features,
)
from irt_data.irt.features import BENCHMARK_REGISTRY, build_feature_tables
from irt_data.irt.model import MIRT
from irt_data.irt.train import TrainConfig


ROOT = Path(__file__).resolve().parents[3]
SPLIT_DIR = ROOT / "irt_data" / "eval_splits"
OUT_DIR = ROOT / "irt_data" / "irt" / "experiments"
TRAIN_CSV = SPLIT_DIR / "response_matrix_train.csv"
HOLDOUT_CSV = SPLIT_DIR / "response_matrix_holdout.csv"
SPLIT_SUMMARY_JSON = SPLIT_DIR / "split_summary.json"
K_GRID = [1, 2, 4, 8]

logger = logging.getLogger(__name__)


@dataclass
class HoldoutMetrics:
    split_name: str
    k: int
    use_features: bool
    cold_start_note: str
    train_rows: int
    holdout_rows: int
    train_agent_configs: int
    holdout_agent_configs: int
    holdout_unseen_agent_configs: int
    train_tasks: int
    holdout_tasks: int
    holdout_unseen_tasks: int
    epochs: int
    train_bce: float
    holdout_bce: float
    holdout_auc: float
    holdout_accuracy: float
    holdout_brier: float
    holdout_calibration_error: float


def calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    errors = []
    for lo, hi in zip(np.linspace(0, 1, n_bins, endpoint=False), np.linspace(0.1, 1, n_bins)):
        if hi == 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if mask.any():
            errors.append(abs(float(probs[mask].mean()) - float(labels[mask].mean())))
    return float(np.mean(errors)) if errors else 0.0


def build_explicit_split_dataset() -> tuple[IRTDataset, np.ndarray, set[int], set[int]]:
    train_df = pd.read_csv(TRAIN_CSV)
    holdout_df = pd.read_csv(HOLDOUT_CSV)

    train_df = train_df[train_df["benchmark"].isin(BENCHMARK_REGISTRY)].copy()
    holdout_df = holdout_df[holdout_df["benchmark"].isin(BENCHMARK_REGISTRY)].copy()
    all_df = pd.concat([train_df, holdout_df], ignore_index=True)

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
    train_task_indices = set(train_triplets[:, 1].astype(int).tolist())
    return dataset, holdout_triplets, train_agent_indices, train_task_indices


def feature_kwargs(dataset: IRTDataset, device: str) -> dict[str, torch.Tensor]:
    return {
        "agent_cat": dataset.agent_cat_tensor.to(device),
        "agent_cont": dataset.agent_cont_tensor.to(device),
        "task_cat": dataset.task_cat_tensor.to(device),
        "task_cont": dataset.task_cont_tensor.to(device),
    }


def logits_for_triplets(
    model: MIRT,
    triplets: np.ndarray,
    dataset: IRTDataset,
    train_agent_indices: set[int],
    device: str,
    zero_unseen_agent_theta: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(TripletDataset(triplets), batch_size=4096, shuffle=False)
    feats = feature_kwargs(dataset, device) if model.use_features else {}
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for agent_idx, task_idx, correct in loader:
            agent_idx = agent_idx.to(device)
            task_idx = task_idx.to(device)

            theta = model.theta(agent_idx)
            if zero_unseen_agent_theta:
                seen = torch.tensor(
                    [int(idx) in train_agent_indices for idx in agent_idx.cpu().numpy()],
                    device=device,
                    dtype=torch.bool,
                )
                theta = theta.clone()
                theta[~seen] = 0.0

            a = model.a(task_idx)
            d = model.d(task_idx).squeeze(-1)

            if model.use_features and model.agent_proj is not None:
                theta = theta + model._encode_agent_features(
                    feats["agent_cat"][agent_idx], feats["agent_cont"][agent_idx]
                )
            if model.use_features and model.task_proj is not None:
                a = a + model._encode_task_features(
                    feats["task_cat"][task_idx], feats["task_cont"][task_idx]
                )

            all_logits.append(((theta * a).sum(dim=-1) + d).cpu())
            all_labels.append(correct)

    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def evaluate_triplets(
    model: MIRT,
    triplets: np.ndarray,
    dataset: IRTDataset,
    train_agent_indices: set[int],
    device: str,
    zero_unseen_agent_theta: bool,
) -> dict[str, float]:
    logits, labels = logits_for_triplets(
        model, triplets, dataset, train_agent_indices, device, zero_unseen_agent_theta
    )
    probs = 1 / (1 + np.exp(-logits))
    preds = probs >= 0.5
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = 0.5
    return {
        "bce": float(log_loss(labels, probs, labels=[0, 1])),
        "auc": auc,
        "accuracy": float((preds == labels).mean()),
        "brier": float(np.mean((probs - labels) ** 2)),
        "calibration_error": calibration_error(labels, probs),
    }


def train_fixed(
    dataset: IRTDataset,
    k: int,
    use_features: bool,
    device: str,
    epochs: int = 120,
) -> tuple[MIRT, list[float]]:
    torch.manual_seed(20260527 + k + (100 if use_features else 0))
    np.random.seed(20260527 + k + (100 if use_features else 0))

    model = MIRT(
        n_agents=dataset.n_agents,
        n_tasks=dataset.n_tasks,
        k=k,
        schema=dataset.schema if use_features else None,
        use_features=use_features,
    ).to(device)
    config = TrainConfig(k=k, use_features=use_features, device=device, epochs=epochs)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loader = DataLoader(
        TripletDataset(dataset.train_triplets),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    feats = feature_kwargs(dataset, device) if model.use_features else {}
    losses = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for agent_idx, task_idx, correct in loader:
            agent_idx = agent_idx.to(device)
            task_idx = task_idx.to(device)
            correct = correct.to(device)
            logits = model(agent_idx, task_idx, **feats)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, correct)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        losses.append(float(np.mean(batch_losses)))
        if epoch % 20 == 0 or epoch == epochs - 1:
            logger.info(
                "K=%s features=%s epoch=%03d train_bce=%.4f",
                k,
                use_features,
                epoch,
                losses[-1],
            )
    return model, losses


def effective_task_discrimination(
    model: MIRT,
    dataset: IRTDataset,
    k: int,
    device: str,
) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        task_indices = torch.arange(dataset.n_tasks, device=device)
        a = model.a(task_indices)
        if model.use_features and model.task_proj is not None:
            feats = feature_kwargs(dataset, device)
            a = a + model._encode_task_features(
                feats["task_cat"][task_indices], feats["task_cont"][task_indices]
            )
        a_np = a.cpu().numpy()
        d_np = model.d(task_indices).squeeze(-1).cpu().numpy()

    idx_to_task = {idx: key for key, idx in dataset.task_key_to_idx.items()}
    rows = []
    for idx in range(dataset.n_tasks):
        benchmark, task_id = idx_to_task[idx]
        row = {
            "benchmark": benchmark,
            "task_id": task_id,
            "discrimination": float(np.linalg.norm(a_np[idx])),
            "easiness": float(d_np[idx]),
        }
        for dim in range(k):
            row[f"a{dim}"] = float(a_np[idx, dim])
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["benchmark", "discrimination"], ascending=[True, False])
        .reset_index(drop=True)
    )


def plot_curve(losses: list[float], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(losses)), losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training BCE")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset, holdout_triplets, train_agent_indices, train_task_indices = (
        build_explicit_split_dataset()
    )

    holdout_agent_indices = set(holdout_triplets[:, 0].astype(int).tolist())
    holdout_task_indices = set(holdout_triplets[:, 1].astype(int).tolist())
    unseen_holdout_agents = holdout_agent_indices - train_agent_indices
    unseen_holdout_tasks = holdout_task_indices - train_task_indices

    logger.info("Train rows: %d", len(dataset.train_triplets))
    logger.info("Holdout rows: %d", len(holdout_triplets))
    logger.info("Unseen holdout agents: %d", len(unseen_holdout_agents))
    logger.info("Unseen holdout tasks: %d", len(unseen_holdout_tasks))

    results = []
    for k in K_GRID:
        for use_features in [False, True]:
            model, losses = train_fixed(dataset, k, use_features, device="cpu")
            tag = f"split_v1_k{k}_{'features' if use_features else 'latent_only'}"
            plot_curve(
                losses,
                OUT_DIR / f"{tag}_training_curve.png",
                f"Split v1 MIRT K={k} {'features' if use_features else 'latent only'}",
            )

            zero_unseen = True
            train_metrics = evaluate_triplets(
                model,
                dataset.train_triplets,
                dataset,
                train_agent_indices,
                "cpu",
                zero_unseen_agent_theta=False,
            )
            holdout_metrics = evaluate_triplets(
                model,
                holdout_triplets,
                dataset,
                train_agent_indices,
                "cpu",
                zero_unseen_agent_theta=zero_unseen,
            )
            cold_start_note = (
                "metadata-only theta for unseen holdout agents"
                if use_features
                else "unseen holdout agents scored with zero theta; logits reduce to task easiness"
            )

            metrics = HoldoutMetrics(
                split_name="benchmark_normalized_model_harness_holdout_v1",
                k=k,
                use_features=use_features,
                cold_start_note=cold_start_note,
                train_rows=int(len(dataset.train_triplets)),
                holdout_rows=int(len(holdout_triplets)),
                train_agent_configs=len(train_agent_indices),
                holdout_agent_configs=len(holdout_agent_indices),
                holdout_unseen_agent_configs=len(unseen_holdout_agents),
                train_tasks=len(train_task_indices),
                holdout_tasks=len(holdout_task_indices),
                holdout_unseen_tasks=len(unseen_holdout_tasks),
                epochs=len(losses),
                train_bce=train_metrics["bce"],
                holdout_bce=holdout_metrics["bce"],
                holdout_auc=holdout_metrics["auc"],
                holdout_accuracy=holdout_metrics["accuracy"],
                holdout_brier=holdout_metrics["brier"],
                holdout_calibration_error=holdout_metrics["calibration_error"],
            )
            results.append(asdict(metrics))
            logger.info(
                "RESULT K=%d features=%s holdout_bce=%.4f auc=%.3f acc=%.3f brier=%.4f cal=%.4f",
                k,
                use_features,
                metrics.holdout_bce,
                metrics.holdout_auc,
                metrics.holdout_accuracy,
                metrics.holdout_brier,
                metrics.holdout_calibration_error,
            )

            disc = effective_task_discrimination(model, dataset, k, "cpu")
            disc.to_csv(OUT_DIR / f"{tag}_task_discrimination.csv", index=False)

    result_df = pd.DataFrame(results).sort_values(["holdout_bce", "k", "use_features"])
    result_df.to_csv(OUT_DIR / "split_v1_sweep_results.csv", index=False)
    with (OUT_DIR / "split_v1_sweep_results.json").open("w") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    best = result_df.iloc[0].to_dict()
    feature_best = result_df[result_df["use_features"]].iloc[0].to_dict()
    notes = {
        "split_summary": json.loads(SPLIT_SUMMARY_JSON.read_text()),
        "selection_metric": "lowest holdout_bce",
        "best_overall": best,
        "recommended_for_cold_start": feature_best,
        "latent_only_note": (
            "All holdout agent configurations are unseen under the split-v1 "
            "leakage unit. Latent-only models have no learned theta for those "
            "agents, so holdout scoring zeros unseen theta and effectively uses "
            "task easiness only."
        ),
        "feature_informed_note": (
            "Feature-informed models also zero unseen theta, but retain the "
            "metadata projection f_agent(features), giving a cold-start theta "
            "without fitting on held-out response rows."
        ),
    }
    with (OUT_DIR / "split_v1_recommendation.json").open("w") as handle:
        json.dump(notes, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
