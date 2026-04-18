"""Training loop and evaluation for MIRT."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import IRTDataset, TripletDataset
from .model import MIRT

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    k: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 2048
    use_features: bool = True
    patience: int = 20
    device: str = "cpu"


@dataclass
class EvalMetrics:
    loss: float
    accuracy: float
    auc: float
    calibration_error: float

    def __str__(self):
        return (
            f"loss={self.loss:.4f} acc={self.accuracy:.3f} "
            f"auc={self.auc:.3f} cal_err={self.calibration_error:.3f}"
        )


def evaluate(
    model: MIRT,
    triplets: np.ndarray,
    dataset: IRTDataset,
    device: str = "cpu",
) -> EvalMetrics:
    """Evaluate model on a set of triplets."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    ds = TripletDataset(triplets)
    loader = DataLoader(ds, batch_size=4096, shuffle=False)

    all_logits = []
    all_labels = []

    feat_kwargs = {}
    if model.use_features:
        feat_kwargs = {
            "agent_cat": dataset.agent_cat_tensor.to(device),
            "agent_cont": dataset.agent_cont_tensor.to(device),
            "task_cat": dataset.task_cat_tensor.to(device),
            "task_cont": dataset.task_cont_tensor.to(device),
        }

    with torch.no_grad():
        for agent_idx, task_idx, correct in loader:
            agent_idx = agent_idx.to(device)
            task_idx = task_idx.to(device)
            logits = model(agent_idx, task_idx, **feat_kwargs)
            all_logits.append(logits.cpu())
            all_labels.append(correct)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits)

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    preds = (probs > 0.5).float()
    accuracy = (preds == labels).float().mean().item()

    labels_np = labels.numpy()
    probs_np = probs.numpy()

    try:
        auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        auc = 0.5

    # Calibration: bin predictions, compare mean predicted vs actual
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    cal_errors = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs_np >= lo) & (probs_np < hi)
        if mask.sum() > 0:
            cal_errors.append(abs(probs_np[mask].mean() - labels_np[mask].mean()))
    calibration_error = np.mean(cal_errors) if cal_errors else 0.0

    return EvalMetrics(loss=loss, accuracy=accuracy, auc=auc, calibration_error=calibration_error)


def train(
    dataset: IRTDataset,
    config: TrainConfig | None = None,
) -> tuple[MIRT, list[EvalMetrics], list[EvalMetrics]]:
    """
    Train MIRT model on the dataset.

    Returns (model, train_metrics_per_epoch, val_metrics_per_epoch).
    """
    if config is None:
        config = TrainConfig()

    device = config.device
    model = MIRT(
        n_agents=dataset.n_agents,
        n_tasks=dataset.n_tasks,
        k=config.k,
        schema=dataset.schema if config.use_features else None,
        use_features=config.use_features,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    train_ds = TripletDataset(dataset.train_triplets)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False
    )

    feat_kwargs = {}
    if model.use_features:
        feat_kwargs = {
            "agent_cat": dataset.agent_cat_tensor.to(device),
            "agent_cont": dataset.agent_cont_tensor.to(device),
            "task_cat": dataset.task_cat_tensor.to(device),
            "task_cont": dataset.task_cont_tensor.to(device),
        }

    train_history = []
    val_history = []
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for agent_idx, task_idx, correct in train_loader:
            agent_idx = agent_idx.to(device)
            task_idx = task_idx.to(device)
            correct = correct.to(device)

            logits = model(agent_idx, task_idx, **feat_kwargs)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, correct)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Evaluate
        train_metrics = evaluate(model, dataset.train_triplets, dataset, device)
        val_metrics = evaluate(model, dataset.val_triplets, dataset, device)
        train_history.append(train_metrics)
        val_history.append(val_metrics)

        if epoch % 20 == 0 or epoch == config.epochs - 1:
            logger.info(
                f"Epoch {epoch:3d} | train: {train_metrics} | val: {val_metrics}"
            )

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, train_history, val_history
