"""
Cold-start ability estimation for a novel scaffold × model combination.

For an agent never seen during training, we have no latent θ embedding.
Instead we use the feature projection alone (the f_agent(features) term
from the MIRT model) as the θ estimate, then compute P(correct) for every
task on the relevant benchmarks.

Usage:
  python -m irt_data.irt.predict_new_agent \
      --scaffold "SWE-Agent" \
      --model "deepseek-r1" \
      --benchmarks swebench_verified_mini \
      --threshold 0.8
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import _encode_agent_features, load_and_split
from .features import (
    FeatureSchema,
    extract_model_features,
    extract_scaffold_features,
)
from .train import TrainConfig, train

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).resolve().parent.parent


def predict_new_agent(
    scaffold: str,
    model: str,
    benchmarks: list[str],
    threshold: str = "80pct",
    k: int = 2,
    response_csv: str = "irt_data/response_matrix.csv",
    subsets_json: str = "irt_data/adaptive_task_subsets.json",
) -> pd.DataFrame:
    """
    Fit MIRT on existing data, then cold-start predict for an unseen agent.

    Returns a DataFrame with columns:
        benchmark, task_id, predicted_prob, discrimination, easiness
    sorted by benchmark + task_id.
    """
    logger.info("Loading data and fitting model...")
    dataset = load_and_split(response_csv)
    config = TrainConfig(k=k, use_features=True)
    model_obj, _, _ = train(dataset, config)
    model_obj.eval()

    # Build feature vector for the new agent
    agent_row = {**extract_model_features(model), **extract_scaffold_features(scaffold), "agent_id": "__new__"}
    agent_df = pd.DataFrame([agent_row]).set_index("agent_id")

    # Remove metadata columns not in the schema
    schema = dataset.schema
    agent_cat_t, agent_cont_t = _encode_agent_features(agent_df, schema, ["__new__"])

    # Feature-only θ estimate (no latent embedding — this is the cold-start prior).
    # Rescale to the norm distribution of f_agent outputs seen during training so
    # that extrapolated feature combinations don't saturate predictions.
    with torch.no_grad():
        theta_feat = model_obj._encode_agent_features(agent_cat_t, agent_cont_t)  # (1, k)
        train_feat = model_obj._encode_agent_features(
            dataset.agent_cat_tensor, dataset.agent_cont_tensor
        )  # (n_agents, k)
        train_mean_norm = train_feat.norm(dim=1).mean().item()
        new_norm = theta_feat.norm().item()
        if new_norm > 0:
            theta_feat = theta_feat * (train_mean_norm / new_norm)

    logger.info(
        f"Cold-start θ for {scaffold} + {model}: {theta_feat.numpy().round(3)}"
        f"  (rescaled from norm {new_norm:.2f} → {train_mean_norm:.2f})"
    )

    # Load task parameters and subset list
    with open(subsets_json) as f:
        subsets = json.load(f)

    idx_to_key = {v: k for k, v in dataset.task_key_to_idx.items()}
    a_mat = model_obj.a.weight.detach()   # (n_tasks, k)
    d_vec = model_obj.d.weight.detach().squeeze()  # (n_tasks,)

    rows = []
    for bench in benchmarks:
        if bench not in subsets:
            logger.warning(f"Benchmark {bench!r} not in subsets JSON — skipping")
            continue
        task_ids = set(subsets[bench]["thresholds"][threshold]["task_ids"])
        n_total = subsets[bench]["n_total"]
        n_subset = subsets[bench]["thresholds"][threshold]["n_tasks"]
        logger.info(f"{bench}: predicting on {n_subset}/{n_total} tasks ({threshold} coverage)")

        for task_idx, (b, tid) in idx_to_key.items():
            if b != bench or tid not in task_ids:
                continue
            a_j = a_mat[task_idx]        # (k,)
            d_j = d_vec[task_idx]        # scalar
            logit = (theta_feat[0] * a_j).sum() + d_j
            prob = torch.sigmoid(logit).item()
            rows.append({"benchmark": bench, "task_id": tid, "predicted_prob": prob})

    pred_df = pd.DataFrame(rows).sort_values(["benchmark", "task_id"]).reset_index(drop=True)
    return pred_df, theta_feat.numpy()


def compare_with_actuals(pred_df: pd.DataFrame, actuals_csv: str) -> pd.DataFrame:
    """
    Merge cold-start predictions with actual run results.

    actuals_csv must have columns: benchmark, task_id, correct (0/1).
    Returns merged DataFrame with brier_score column added.
    """
    actuals = pd.read_csv(actuals_csv)
    merged = pred_df.merge(actuals[["benchmark", "task_id", "correct"]], on=["benchmark", "task_id"], how="inner")
    merged["brier_score"] = (merged["predicted_prob"] - merged["correct"]) ** 2
    return merged


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--threshold", default="80pct")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--response-csv", default="irt_data/response_matrix.csv")
    parser.add_argument("--subsets-json", default="irt_data/adaptive_task_subsets.json")
    parser.add_argument("--actuals-csv", default=None,
                        help="If provided, compare predictions against actual results")
    parser.add_argument("--out", default=None, help="Save predictions to this CSV path")
    args = parser.parse_args()

    pred_df, theta = predict_new_agent(
        scaffold=args.scaffold,
        model=args.model,
        benchmarks=args.benchmarks,
        threshold=args.threshold,
        k=args.k,
        response_csv=args.response_csv,
        subsets_json=args.subsets_json,
    )

    logger.info(f"\nPredicted accuracy: {pred_df['predicted_prob'].mean():.3f}")
    logger.info(f"θ vector: {theta.round(3)}")

    if args.actuals_csv:
        merged = compare_with_actuals(pred_df, args.actuals_csv)
        brier = merged["brier_score"].mean()
        corr = merged[["predicted_prob", "correct"]].corr().iloc[0, 1]
        actual_acc = merged["correct"].mean()
        pred_acc = merged["predicted_prob"].mean()
        logger.info(f"\n--- Generalization evaluation ---")
        logger.info(f"Tasks matched:      {len(merged)}")
        logger.info(f"Actual accuracy:    {actual_acc:.3f}")
        logger.info(f"Predicted accuracy: {pred_acc:.3f}")
        logger.info(f"Brier score:        {brier:.4f}  (lower = better; 0.25 = chance)")
        logger.info(f"Point-biserial r:   {corr:.3f}   (predicted_prob vs correct)")
        pred_df = merged

    if args.out:
        pred_df.to_csv(args.out, index=False)
        logger.info(f"Saved: {args.out}")
    else:
        print(pred_df.to_string(index=False))


if __name__ == "__main__":
    main()
