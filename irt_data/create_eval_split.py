#!/usr/bin/env python3
"""Create the canonical train/holdout split for IRT evaluation.

The leakage unit is a semantic run-configuration cell:

    benchmark x normalized_model x normalized_harness

All raw runs and task outcomes belonging to a selected cell are held out
together. The holdout selector takes one representative normalized model per
benchmark x normalized_harness cell, choosing the model whose pass rate is
closest to that cell's median pass rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IRT_DIR = ROOT / "irt_data"
OUTPUT_DIR = IRT_DIR / "eval_splits"

RESPONSE_COLUMNS = [
    "agent_id",
    "scaffold",
    "model",
    "benchmark",
    "task_id",
    "correct",
    "run_id",
]

UNIT_COLUMNS = ["benchmark", "normalized_model", "normalized_harness"]


def selection_unit_id(row: pd.Series) -> str:
    return " :: ".join(str(row[col]) for col in UNIT_COLUMNS)


def load_joined_response_matrix() -> pd.DataFrame:
    response = pd.read_csv(IRT_DIR / "response_matrix.csv")
    models = pd.read_csv(IRT_DIR / "model_metadata.csv")
    harnesses = pd.read_csv(IRT_DIR / "harness_metadata.csv")

    joined = response.merge(
        models[["raw_model", "normalized_model", "provider", "model_family"]],
        left_on="model",
        right_on="raw_model",
        how="left",
        validate="many_to_one",
    )
    joined = joined.merge(
        harnesses[
            [
                "raw_scaffold",
                "normalized_harness",
                "tool_exposure_mechanism",
                "tool_exposure_strength",
            ]
        ],
        left_on="scaffold",
        right_on="raw_scaffold",
        how="left",
        validate="many_to_one",
    )

    missing_models = joined["normalized_model"].isna().sum()
    missing_harnesses = joined["normalized_harness"].isna().sum()
    if missing_models or missing_harnesses:
        raise ValueError(
            "Metadata join failed: "
            f"{missing_models} rows missing normalized_model, "
            f"{missing_harnesses} rows missing normalized_harness"
        )

    joined["selection_unit_id"] = joined.apply(selection_unit_id, axis=1)
    return joined


def summarize_units(joined: pd.DataFrame) -> pd.DataFrame:
    return (
        joined.groupby(
            [
                "selection_unit_id",
                "benchmark",
                "normalized_model",
                "normalized_harness",
                "tool_exposure_mechanism",
            ],
            dropna=False,
        )
        .agg(
            outcomes=("correct", "size"),
            configs=("agent_id", "nunique"),
            pass_rate=("correct", "mean"),
            raw_models=("model", lambda values: "; ".join(sorted(set(values)))),
            raw_scaffolds=("scaffold", lambda values: "; ".join(sorted(set(values)))),
            run_ids=("run_id", lambda values: "; ".join(sorted(set(values)))),
            agent_ids=("agent_id", lambda values: "; ".join(sorted(set(values)))),
        )
        .reset_index()
    )


def choose_holdout_units(units: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    for _, cell in units.groupby(["benchmark", "normalized_harness"], sort=True):
        median_pass_rate = cell["pass_rate"].median()
        ranked = cell.assign(
            distance_from_cell_median=(cell["pass_rate"] - median_pass_rate).abs()
        ).sort_values(
            [
                "distance_from_cell_median",
                "outcomes",
                "normalized_model",
                "selection_unit_id",
            ],
            ascending=[True, False, True, True],
        )
        chosen.append(ranked.head(1))

    holdout = pd.concat(chosen, ignore_index=True)
    holdout["split"] = "holdout"
    holdout["selection_reason"] = (
        "one representative normalized model per benchmark x normalized_harness "
        "cell; chosen by closest pass rate to that cell median"
    )
    return holdout


def write_outputs(joined: pd.DataFrame, holdout_units: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    holdout_ids = set(holdout_units["selection_unit_id"])
    with_split = joined.copy()
    with_split["split"] = with_split["selection_unit_id"].map(
        lambda unit_id: "holdout" if unit_id in holdout_ids else "train"
    )

    train = with_split[with_split["split"] == "train"]
    holdout = with_split[with_split["split"] == "holdout"]

    train[RESPONSE_COLUMNS].to_csv(OUTPUT_DIR / "response_matrix_train.csv", index=False)
    holdout[RESPONSE_COLUMNS].to_csv(
        OUTPUT_DIR / "response_matrix_holdout.csv", index=False
    )
    with_split[
        RESPONSE_COLUMNS
        + [
            "split",
            "selection_unit_id",
            "normalized_model",
            "normalized_harness",
            "tool_exposure_mechanism",
        ]
    ].to_csv(OUTPUT_DIR / "response_matrix_with_split.csv", index=False)

    holdout_units.to_csv(OUTPUT_DIR / "holdout_configurations.csv", index=False)

    train_units = set(train["selection_unit_id"])
    leaked_units = sorted(train_units & holdout_ids)
    if leaked_units:
        raise ValueError(f"Holdout units leaked into train: {leaked_units[:5]}")
    if len(train) + len(holdout) != len(joined):
        raise ValueError("Train and holdout row counts do not sum to source rows")

    summary = {
        "split_name": "benchmark_normalized_model_harness_holdout_v1",
        "source_response_matrix": "irt_data/response_matrix.csv",
        "source_model_metadata": "irt_data/model_metadata.csv",
        "source_harness_metadata": "irt_data/harness_metadata.csv",
        "leakage_unit": "benchmark x normalized_model x normalized_harness",
        "selection_rule": (
            "Within each benchmark x normalized_harness cell, hold out the "
            "normalized model whose pass rate is closest to that cell's median "
            "pass rate. Ties prefer larger outcome coverage, then stable labels."
        ),
        "source_rows": int(len(joined)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "holdout_row_fraction": round(len(holdout) / len(joined), 6),
        "source_selection_units": int(joined["selection_unit_id"].nunique()),
        "train_selection_units": int(train["selection_unit_id"].nunique()),
        "holdout_selection_units": int(holdout["selection_unit_id"].nunique()),
        "source_agent_configs": int(joined["agent_id"].nunique()),
        "train_agent_configs": int(train["agent_id"].nunique()),
        "holdout_agent_configs": int(holdout["agent_id"].nunique()),
        "no_selection_unit_overlap_between_train_and_holdout": not leaked_units,
    }
    with (OUTPUT_DIR / "split_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    by_benchmark = (
        with_split.groupby(["split", "benchmark"])
        .agg(
            outcomes=("correct", "size"),
            configs=("agent_id", "nunique"),
            selection_units=("selection_unit_id", "nunique"),
            pass_rate=("correct", "mean"),
        )
        .reset_index()
    )
    by_benchmark.to_csv(OUTPUT_DIR / "split_summary_by_benchmark.csv", index=False)


def main() -> None:
    joined = load_joined_response_matrix()
    units = summarize_units(joined)
    holdout_units = choose_holdout_units(units)
    write_outputs(joined, holdout_units)


if __name__ == "__main__":
    main()
