"""
Main entry point for fitting and evaluating MIRT models.

Usage:
  python -m irt_data.irt.run                          # default K=4 with features
  python -m irt_data.irt.run --k 1 --no-features      # Rasch-like baseline
  python -m irt_data.irt.run --sweep                   # sweep K ∈ {1,2,4,8}
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load_and_split
from .train import TrainConfig, evaluate, train

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent


def plot_training_curves(train_hist, val_hist, title: str, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    epochs = range(len(train_hist))

    axes[0].plot(epochs, [m.loss for m in train_hist], label="train")
    axes[0].plot(epochs, [m.loss for m in val_hist], label="val")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [m.auc for m in train_hist], label="train")
    axes[1].plot(epochs, [m.auc for m in val_hist], label="val")
    axes[1].set_ylabel("AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, [m.accuracy for m in train_hist], label="train")
    axes[2].plot(epochs, [m.accuracy for m in val_hist], label="val")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def run_single(k: int, use_features: bool, dataset, device: str = "cpu") -> dict:
    label = f"K={k} {'+ features' if use_features else 'latent only'}"
    logger.info(f"\n{'='*60}")
    logger.info(f"Training MIRT: {label}")
    logger.info(f"{'='*60}")

    config = TrainConfig(k=k, use_features=use_features, device=device)
    model, train_hist, val_hist = train(dataset, config)

    final_train = train_hist[-1]
    final_val = evaluate(model, dataset.val_triplets, dataset, device)

    logger.info(f"\nFinal results ({label}):")
    logger.info(f"  Train: {final_train}")
    logger.info(f"  Val:   {final_val}")

    tag = f"k{k}_{'feat' if use_features else 'nofeat'}"
    plot_training_curves(
        train_hist, val_hist,
        f"MIRT {label}", OUTPUT_DIR / f"training_curves_{tag}.png",
    )

    # Compute marginal baseline for comparison
    train_mean = dataset.train_triplets[:, 2].mean()
    from sklearn.metrics import roc_auc_score
    val_labels = dataset.val_triplets[:, 2]
    baseline_preds = np.full_like(val_labels, train_mean, dtype=float)
    try:
        baseline_auc = roc_auc_score(val_labels, baseline_preds)
    except ValueError:
        baseline_auc = 0.5
    baseline_acc = max(val_labels.mean(), 1 - val_labels.mean())
    logger.info(f"  Baseline (marginal): acc={baseline_acc:.3f} auc={baseline_auc:.3f}")

    return {
        "k": k,
        "use_features": use_features,
        "val_loss": final_val.loss,
        "val_acc": final_val.accuracy,
        "val_auc": final_val.auc,
        "val_cal_err": final_val.calibration_error,
        "baseline_acc": float(baseline_acc),
        "baseline_auc": float(baseline_auc),
        "n_epochs": len(train_hist),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--no-features", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="Sweep K ∈ {1,2,4,8}")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--response-csv", default="irt_data/response_matrix.csv")
    parser.add_argument("--min-tasks", type=int, default=10)
    parser.add_argument("--min-agents", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    logger.info("Loading data...")
    dataset = load_and_split(
        args.response_csv,
        min_tasks_per_agent=args.min_tasks,
        min_agents_per_task=args.min_agents,
        val_agent_fraction=args.val_fraction,
    )
    logger.info(dataset.summary())

    if args.sweep:
        results = []
        for k in [1, 2, 4, 8]:
            for use_feat in [False, True]:
                r = run_single(k, use_feat, dataset, args.device)
                results.append(r)

        logger.info(f"\n{'='*60}")
        logger.info("Sweep results:")
        logger.info(f"{'K':>3} {'Features':>10} {'Val Loss':>10} {'Val AUC':>10} {'Val Acc':>10} {'Cal Err':>10}")
        for r in results:
            logger.info(
                f"{r['k']:3d} {'yes' if r['use_features'] else 'no':>10} "
                f"{r['val_loss']:10.4f} {r['val_auc']:10.3f} "
                f"{r['val_acc']:10.3f} {r['val_cal_err']:10.3f}"
            )

        serializable = [
            {k: float(v) if isinstance(v, (np.floating,)) else v for k, v in r.items()}
            for r in results
        ]
        with open(OUTPUT_DIR / "sweep_results.json", "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Sweep results saved to {OUTPUT_DIR / 'sweep_results.json'}")
    else:
        run_single(args.k, not args.no_features, dataset, args.device)


if __name__ == "__main__":
    main()
