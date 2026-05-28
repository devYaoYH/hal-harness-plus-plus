"""G-study-inspired variance audit for HAL binary response rows.

This is intentionally lightweight: it materializes the row-level binary dataset
and fits regularized logistic fixed-effect models to estimate how much outcome
deviance is explained by benchmark/task, model, harness, and coarse
interactions. The resulting table is a practical variance audit for the sparse
HAL matrix rather than a full balanced random-effects G-theory estimator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "irt_data" / "gstudy"
RESPONSE_CSV = REPO_ROOT / "irt_data" / "response_matrix.csv"


MODEL_SPECS: list[tuple[str, list[str]]] = [
    ("null", []),
    ("benchmark", ["benchmark"]),
    ("task_nested", ["task_nested"]),
    ("task_nested + model", ["task_nested", "model_name"]),
    ("task_nested + scaffold", ["task_nested", "scaffold_name"]),
    ("task_nested + model + scaffold", ["task_nested", "model_name", "scaffold_name"]),
    (
        "task_nested + model + scaffold + model:harness",
        ["task_nested", "model_name", "scaffold_name", "model_harness"],
    ),
    (
        "task_nested + model + scaffold + model:harness + model:benchmark + harness:benchmark",
        [
            "task_nested",
            "model_name",
            "scaffold_name",
            "model_harness",
            "model_benchmark",
            "harness_benchmark",
        ],
    ),
]

STAGED_SPECS: list[tuple[str, list[str]]] = [
    ("null", []),
    ("benchmark", ["benchmark"]),
    ("benchmark + model", ["benchmark", "model_name"]),
    ("benchmark + harness", ["benchmark", "scaffold_name"]),
    ("benchmark + model + harness", ["benchmark", "model_name", "scaffold_name"]),
    (
        "benchmark + model + harness + model:harness",
        ["benchmark", "model_name", "scaffold_name", "model_harness"],
    ),
    (
        "benchmark + model + harness + benchmark:harness",
        ["benchmark", "model_name", "scaffold_name", "harness_benchmark"],
    ),
    (
        "benchmark + model + harness + model:harness + benchmark:harness",
        ["benchmark", "model_name", "scaffold_name", "model_harness", "harness_benchmark"],
    ),
    (
        "task_nested + model + harness",
        ["task_nested", "model_name", "scaffold_name"],
    ),
    (
        "task_nested + model + harness + model:harness",
        ["task_nested", "model_name", "scaffold_name", "model_harness"],
    ),
    (
        "task_nested + model + harness + benchmark:harness",
        ["task_nested", "model_name", "scaffold_name", "harness_benchmark"],
    ),
    (
        "task_nested + model + harness + model:harness + benchmark:harness",
        ["task_nested", "model_name", "scaffold_name", "model_harness", "harness_benchmark"],
    ),
    (
        "task_nested + model + harness + task:harness",
        ["task_nested", "model_name", "scaffold_name", "task_harness"],
    ),
]


def _clean_part(value: object) -> str:
    return str(value).replace("|", "/").strip()


def build_rows() -> pd.DataFrame:
    df = pd.read_csv(RESPONSE_CSV)

    model_meta = pd.read_csv(REPO_ROOT / "irt_data" / "model_metadata.csv")
    harness_meta = pd.read_csv(REPO_ROOT / "irt_data" / "harness_metadata.csv")

    df = df.copy()
    df = df.merge(model_meta, left_on="model", right_on="raw_model", how="left")
    df = df.merge(harness_meta, left_on="scaffold", right_on="raw_scaffold", how="left")
    df["model_name"] = df["normalized_model"].fillna(df["model"])
    df["model_family"] = df["model_family"].fillna("unknown")
    df["model_provider"] = df["provider"].fillna("unknown")
    df["scaffold_name"] = df["normalized_harness"].fillna(df["scaffold"])
    df["scaffold_type"] = df["tool_exposure_mechanism"].fillna("unknown")

    df["task_nested"] = df["benchmark"].map(_clean_part) + "|" + df["task_id"].map(_clean_part)
    df["model_harness"] = df["model_name"].map(_clean_part) + "|" + df["scaffold_name"].map(_clean_part)
    df["model_benchmark"] = df["model_name"].map(_clean_part) + "|" + df["benchmark"].map(_clean_part)
    df["harness_benchmark"] = df["scaffold_name"].map(_clean_part) + "|" + df["benchmark"].map(_clean_part)
    df["task_harness"] = df["task_nested"].map(_clean_part) + "|" + df["scaffold_name"].map(_clean_part)

    cols = [
        "correct",
        "agent_id",
        "run_id",
        "benchmark",
        "task_id",
        "task_nested",
        "model",
        "model_name",
        "model_family",
        "model_provider",
        "scaffold",
        "scaffold_name",
        "scaffold_type",
        "model_harness",
        "model_benchmark",
        "harness_benchmark",
        "task_harness",
    ]
    return df[cols]


def _null_metrics(y: np.ndarray) -> dict[str, float]:
    p = float(np.mean(y))
    prob = np.repeat(p, len(y))
    return {
        "log_loss": log_loss(y, prob, labels=[0, 1]),
        "auc": 0.5,
        "accuracy": float(np.mean((prob >= 0.5) == y)),
        "n_features": 1,
    }


def fit_model(rows: pd.DataFrame, columns: list[str]) -> tuple[dict[str, float], np.ndarray]:
    y = rows["correct"].astype(int).to_numpy()
    if not columns:
        p = np.repeat(float(np.mean(y)), len(y))
        return _null_metrics(y), p

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    x = encoder.fit_transform(rows[columns].astype(str))
    x = sparse.csr_matrix(x)

    # L2 regularization prevents complete separation on tiny task cells while
    # still letting high-cardinality task effects absorb obvious difficulty.
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="saga",
        max_iter=2000,
        n_jobs=-1,
        random_state=0,
    )
    clf.fit(x, y)
    prob = clf.predict_proba(x)[:, 1]
    metrics = {
        "log_loss": log_loss(y, prob, labels=[0, 1]),
        "auc": roc_auc_score(y, prob),
        "accuracy": float(np.mean((prob >= 0.5) == y)),
        "n_features": int(x.shape[1]),
    }
    return metrics, prob


def summarize_groups(rows: pd.DataFrame, prob: np.ndarray) -> dict[str, pd.DataFrame]:
    residuals = rows.copy()
    residuals["predicted"] = prob
    residuals["residual"] = residuals["correct"] - residuals["predicted"]

    def agg(group_cols: list[str]) -> pd.DataFrame:
        grouped = (
            residuals.groupby(group_cols, dropna=False)
            .agg(
                n=("correct", "size"),
                accuracy=("correct", "mean"),
                mean_predicted=("predicted", "mean"),
                mean_residual=("residual", "mean"),
                abs_mean_residual=("residual", lambda x: float(np.abs(np.mean(x)))),
            )
            .reset_index()
            .sort_values(["abs_mean_residual", "n"], ascending=[False, False])
        )
        return grouped

    return {
        "residuals_by_harness_benchmark": agg(["scaffold_name", "benchmark"]),
        "residuals_by_model_harness": agg(["model_name", "scaffold_name"]),
        "residuals_by_model_benchmark": agg(["model_name", "benchmark"]),
    }


def run_specs(rows: pd.DataFrame, specs: list[tuple[str, list[str]]]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    y = rows["correct"].astype(int).to_numpy()
    null = _null_metrics(y)
    null_deviance = 2 * len(y) * null["log_loss"]

    records = []
    fitted_probs: dict[str, np.ndarray] = {}
    previous_deviance = None
    benchmark_main_deviance = None
    task_nested_deviance = None
    for name, columns in specs:
        metrics, prob = fit_model(rows, columns)
        fitted_probs[name] = prob
        deviance = 2 * len(y) * metrics["log_loss"]
        if name == "benchmark + model + harness":
            benchmark_main_deviance = deviance
        if name == "task_nested":
            task_nested_deviance = deviance
        records.append(
            {
                "model": name,
                "terms": " + ".join(columns) if columns else "(intercept)",
                "n_features": metrics["n_features"],
                "log_loss": metrics["log_loss"],
                "deviance": deviance,
                "pseudo_r2_vs_null": 1 - deviance / null_deviance,
                "deviance_drop_vs_benchmark_main_effects": (
                    np.nan if benchmark_main_deviance is None else benchmark_main_deviance - deviance
                ),
                "deviance_drop_vs_task_nested_main_effects": (
                    np.nan if task_nested_deviance is None else task_nested_deviance - deviance
                ),
                "incremental_deviance_drop": (
                    np.nan if previous_deviance is None else previous_deviance - deviance
                ),
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
            }
        )
        previous_deviance = deviance

    return pd.DataFrame(records), fitted_probs


def _comparison_to_markdown(comparison: pd.DataFrame) -> str:
    table_cols = [
        "model",
        "n_features",
        "log_loss",
        "pseudo_r2_vs_null",
        "deviance_drop_vs_benchmark_main_effects",
        "deviance_drop_vs_task_nested_main_effects",
        "incremental_deviance_drop",
        "auc",
        "accuracy",
    ]
    cols = [col for col in table_cols if col in comparison.columns]
    table = comparison[cols].copy()
    for col in [
        "log_loss",
        "pseudo_r2_vs_null",
        "deviance_drop_vs_benchmark_main_effects",
        "deviance_drop_vs_task_nested_main_effects",
        "incremental_deviance_drop",
        "auc",
        "accuracy",
    ]:
        if col in table:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    markdown_table = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in table.iterrows():
        markdown_table.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(markdown_table)


def write_report(rows: pd.DataFrame, comparison: pd.DataFrame, staged_comparison: pd.DataFrame) -> None:
    n_models = rows["model_name"].nunique()
    n_scaffolds = rows["scaffold_name"].nunique()
    n_model_harness = rows["model_harness"].nunique()
    n_benchmarks = rows["benchmark"].nunique()
    n_tasks = rows["task_nested"].nunique()

    lines = [
        "# HAL G-Study Variance Audit",
        "",
        "This is a fast G-study-inspired audit over the fixed HAL response matrix. "
        "Rows are binary pass/fail observations; factors are estimated with "
        "regularized logistic fixed effects because the design is sparse and unbalanced.",
        "",
        "## Binary row dataset",
        "",
        f"- Rows: {len(rows):,}",
        f"- Canonical models: {n_models:,}",
        f"- Canonical scaffolds: {n_scaffolds:,}",
        f"- Model × harness combinations: {n_model_harness:,}",
        f"- Benchmarks: {n_benchmarks:,}",
        f"- Tasks nested in benchmark: {n_tasks:,}",
        f"- Mean accuracy: {rows['correct'].mean():.3f}",
        "",
        "## Incremental logistic deviance audit",
        "",
        _comparison_to_markdown(comparison),
        "",
        "## Staged G-study audit",
        "",
        _comparison_to_markdown(staged_comparison),
        "",
        "Interpretation: larger drops in log loss / deviance indicate factors that "
        "explain more structure in the binary outcomes. This is not a balanced "
        "random-effects G-study; it is the practical first pass we can run on the "
        "observed HAL matrix before deciding the IRT and cold-start model structure.",
        "",
        "Residual summary CSVs are computed from the staged "
        "`benchmark + model + harness + model:harness + benchmark:harness` "
        "model. These residuals answer the requested question: what remains "
        "after accounting for benchmark, model, harness, model-harness "
        "compatibility, and benchmark-specific harness effects?",
        "",
        "## Immediate read",
        "",
        "- Task identity explains the largest single chunk, which supports item-level modeling rather than aggregate benchmark scores.",
        "- Model and scaffold both add signal after task difficulty, so the object of measurement should remain model × harness.",
        "- Model × harness and benchmark-specific interactions add further signal, which motivates cold-start features and targeted ablations.",
    ]
    (OUT_DIR / "gstudy_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    rows.to_csv(OUT_DIR / "hal_binary_rows.csv", index=False)

    comparison, fitted_probs = run_specs(rows, MODEL_SPECS)
    staged_comparison, staged_probs = run_specs(rows, STAGED_SPECS)
    comparison.to_csv(OUT_DIR / "gstudy_model_comparison.csv", index=False)
    staged_comparison.to_csv(OUT_DIR / "gstudy_staged_model_comparison.csv", index=False)

    diagnostic_name = "benchmark + model + harness + model:harness + benchmark:harness"
    summaries = summarize_groups(rows, staged_probs[diagnostic_name])
    for name, df in summaries.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)

    summary = {
        "n_rows": int(len(rows)),
        "n_models": int(rows["model_name"].nunique()),
        "n_scaffolds": int(rows["scaffold_name"].nunique()),
        "n_model_harness": int(rows["model_harness"].nunique()),
        "n_benchmarks": int(rows["benchmark"].nunique()),
        "n_tasks": int(rows["task_nested"].nunique()),
        "mean_accuracy": float(rows["correct"].mean()),
    }
    (OUT_DIR / "gstudy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(rows, comparison, staged_comparison)

    print(staged_comparison.to_string(index=False))
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
