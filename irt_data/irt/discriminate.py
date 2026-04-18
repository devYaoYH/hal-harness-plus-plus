"""
Fit a MIRT model and output ranked task discrimination lists per benchmark.

Produces:
  irt_data/task_discrimination.csv       — all tasks ranked by ‖a‖
  irt_data/adaptive_task_subsets.json    — task_id shortlists per benchmark × threshold

Usage:
  python -m irt_data.irt.discriminate
  python -m irt_data.irt.discriminate --k 4 --thresholds 0.7 0.8 0.9
  python -m irt_data.irt.discriminate --benchmarks swebench_verified_mini gaia
  python -m irt_data.irt.discriminate --top-n 20   # fixed top-N per benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_and_split
from .train import TrainConfig, train

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).resolve().parent.parent


def fit_and_rank(
    k: int = 2,
    use_features: bool = True,
    response_csv: str = "irt_data/response_matrix.csv",
    benchmarks: list[str] | None = None,
    thresholds: list[float] | None = None,
    top_n: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Fit MIRT and return (task_discrimination_df, adaptive_subsets_dict).

    task_discrimination_df columns:
        benchmark, task_id, discrimination, a0, ..., aK-1, easiness

    adaptive_subsets_dict structure:
        {benchmark: {n_total: int, thresholds: {
            "80pct": {n_tasks, fraction, task_ids},
            ...
        }}}
    """
    if thresholds is None:
        thresholds = [0.5, 0.7, 0.8, 0.9]

    logger.info("Loading data...")
    dataset = load_and_split(response_csv)
    logger.info(dataset.summary())

    logger.info(f"Fitting MIRT K={k}, features={'yes' if use_features else 'no'}...")
    config = TrainConfig(k=k, use_features=use_features)
    model, _, _ = train(dataset, config)

    # Extract parameters
    a = model.a.weight.detach().numpy()       # (n_tasks, k)
    d = model.d.weight.detach().numpy().squeeze()  # (n_tasks,)
    disc = np.linalg.norm(a, axis=1)

    idx_to_key = {v: k for k, v in dataset.task_key_to_idx.items()}
    rows = []
    for i in range(dataset.n_tasks):
        bench, tid = idx_to_key[i]
        row = {"benchmark": bench, "task_id": tid, "discrimination": float(disc[i])}
        for dim in range(k):
            row[f"a{dim}"] = float(a[i, dim])
        row["easiness"] = float(d[i])
        rows.append(row)

    task_df = (
        pd.DataFrame(rows)
        .sort_values(["benchmark", "discrimination"], ascending=[True, False])
        .reset_index(drop=True)
    )

    if benchmarks:
        task_df = task_df[task_df["benchmark"].isin(benchmarks)]

    # Build adaptive subsets
    subsets = {}
    for bench, bg in task_df.groupby("benchmark"):
        bg = bg.sort_values("discrimination", ascending=False)
        n_total = len(bg)
        cumsum = bg["discrimination"].cumsum() / bg["discrimination"].sum()
        thresh_entries = {}

        for thresh in thresholds:
            n = int((cumsum.values <= thresh).sum()) + 1
            thresh_entries[f"{int(thresh * 100)}pct"] = {
                "n_tasks": n,
                "fraction": round(n / n_total, 3),
                "task_ids": bg["task_id"].iloc[:n].tolist(),
            }

        if top_n is not None:
            thresh_entries[f"top{top_n}"] = {
                "n_tasks": min(top_n, n_total),
                "fraction": round(min(top_n, n_total) / n_total, 3),
                "task_ids": bg["task_id"].iloc[:top_n].tolist(),
            }

        subsets[bench] = {"n_total": n_total, "thresholds": thresh_entries}

    return task_df, subsets


def print_summary(task_df: pd.DataFrame, subsets: dict, thresholds: list[float]):
    logger.info("\n" + "=" * 70)
    logger.info("Task discrimination summary")
    logger.info("=" * 70)
    header = f"{'Benchmark':<35} {'Total':>6}" + "".join(
        f"  {int(t*100)}%: N (reduc.)" for t in thresholds
    )
    logger.info(header)
    for bench, meta in subsets.items():
        n_total = meta["n_total"]
        row = f"{bench:<35} {n_total:>6}"
        for t in thresholds:
            key = f"{int(t * 100)}pct"
            n = meta["thresholds"][key]["n_tasks"]
            reduc = 1 - n / n_total
            row += f"  {n:>3} ({reduc:.0%})"
        logger.info(row)


def main():
    parser = argparse.ArgumentParser(description="MIRT discrimination analysis")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--no-features", action="store_true")
    parser.add_argument("--response-csv", default="irt_data/response_matrix.csv")
    parser.add_argument("--benchmarks", nargs="*")
    parser.add_argument("--thresholds", nargs="*", type=float,
                        default=[0.5, 0.7, 0.8, 0.9])
    parser.add_argument("--top-n", type=int, default=None,
                        help="Also output a fixed top-N shortlist per benchmark")
    parser.add_argument("--out-dir", default="irt_data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    task_df, subsets = fit_and_rank(
        k=args.k,
        use_features=not args.no_features,
        response_csv=args.response_csv,
        benchmarks=args.benchmarks,
        thresholds=args.thresholds,
        top_n=args.top_n,
    )

    print_summary(task_df, subsets, args.thresholds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    disc_path = out_dir / "task_discrimination.csv"
    task_df.to_csv(disc_path, index=False)
    logger.info(f"\nSaved: {disc_path} ({len(task_df)} tasks)")

    subsets_path = out_dir / "adaptive_task_subsets.json"
    with open(subsets_path, "w") as f:
        json.dump(subsets, f, indent=2)
    logger.info(f"Saved: {subsets_path}")


if __name__ == "__main__":
    main()
