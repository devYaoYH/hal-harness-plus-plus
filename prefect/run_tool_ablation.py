"""
Tool ablation experiment — parallel Daytona runner.

Runs HAL Generalist Agent with a specific tool disabled against the IRT
adaptive task subset for a target benchmark, then compares against the
full-tool baseline from the response matrix.

Default experiment: remove vision_query on scicode.
Predicted effect: +1.2% accuracy (model-matched observational estimate).

Usage:
    cd /path/to/hal-harness
    python3 -c "
    import os, sys
    sys.path.insert(0, 'prefect')
    os.environ.setdefault('SSL_CERT_FILE', __import__('certifi').where())
    from dotenv import load_dotenv; load_dotenv()
    import run_tool_ablation
    run_tool_ablation.main()
    "

Override via env vars:
    DISABLE_TOOLS   comma-separated tool names to remove (default: vision_query)
    MODEL           model string (default: anthropic/claude-3-7-sonnet-20250219)
    BENCHMARK       target benchmark (default: scicode)
    THRESHOLD       IRT coverage threshold (default: 80pct)
    JOB_ID          unique run identifier
    MAX_WORKERS     parallel sandboxes (default: 10)
"""

from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────

DISABLE_TOOLS = os.getenv("DISABLE_TOOLS", "web_search,page_browse,text_inspect,vision_query")
MODEL         = os.getenv("MODEL",         "anthropic/claude-3-7-sonnet-20250219")
BENCHMARK     = os.getenv("BENCHMARK",     "swebench_verified_mini")
THRESHOLD     = os.getenv("THRESHOLD",     "80pct")
JOB_ID        = os.getenv("JOB_ID",        f"ablation-{DISABLE_TOOLS.replace(',','-')}-{BENCHMARK}-001")
MAX_WORKERS   = int(os.getenv("MAX_WORKERS", "10"))

REPO_ROOT    = Path(__file__).resolve().parent.parent
SUBSETS      = REPO_ROOT / "irt_data" / "adaptive_task_subsets.json"
RESPONSE_CSV = REPO_ROOT / "irt_data" / "response_matrix.csv"
RESULTS_OUT  = REPO_ROOT / "irt_data" / f"actuals_{JOB_ID}.csv"

_AGENT = {
    "name":            "hal_generalist_agent",
    "function":        "main.run",
    "dir":             "agents/hal_generalist_agent",
    "benchmark_extra": BENCHMARK,
}

# ── Baseline from response matrix ───────────────────────────────────────────

def load_observational_baseline(model: str, benchmark: str) -> dict:
    """
    Return model-matched baseline stats from the response matrix:
      - HAL Generalist Agent with the tool ON (full config)
      - Subset accuracy on the IRT task subset
    """
    import pandas as pd
    from irt_data.irt.features import extract_scaffold_features, _normalize_model_name

    df = pd.read_csv(RESPONSE_CSV)
    df = df[df['benchmark'] == benchmark]

    with open(SUBSETS) as f:
        subset_tasks = set(json.load(f)[benchmark]["thresholds"][THRESHOLD]["task_ids"])

    norm_model = _normalize_model_name(model)
    hal = df[
        df['scaffold'].map(lambda s: extract_scaffold_features(s).get('scaffold_name'))
              .isin(['HAL Generalist Agent']) &
        df['model'].map(_normalize_model_name).eq(norm_model) &
        df['task_id'].isin(subset_tasks)
    ]
    if hal.empty:
        return {"baseline_acc": None, "baseline_n": 0}
    return {"baseline_acc": hal['correct'].mean(), "baseline_n": len(hal)}

# ── Task runner ──────────────────────────────────────────────────────────────

def run_one(task_id: str) -> dict:
    from config_daytona import DaytonaEvalSpec
    from daytona_runner import run_eval_on_daytona

    spec = DaytonaEvalSpec(
        agent=_AGENT["name"],
        agent_function=_AGENT["function"],
        agent_dir=_AGENT["dir"],
        benchmark=BENCHMARK,
        task_id=task_id,
        model=MODEL,
        job_id=JOB_ID,
        benchmark_extra=_AGENT["benchmark_extra"],
    )
    # disable_tools is forwarded as an -A kwarg via the hal.cli call inside
    # run_eval_on_daytona → _build_sandbox_command. We patch spec dynamically
    # by monkey-patching the command builder for this run.
    _orig_build = None
    try:
        import daytona_runner as dr
        _orig_build = dr._build_sandbox_command

        def _patched_build(s, run_id):
            cmd = _orig_build(s, run_id)
            # Insert disable_tools arg before the trailing semicolon
            return cmd.replace(
                f"  -A 'model_name={s.model}' ;",
                f"  -A 'model_name={s.model}'"
                f"  -A 'disable_tools={DISABLE_TOOLS}' ;",
            )
        dr._build_sandbox_command = _patched_build

        result = run_eval_on_daytona(spec)
        return {"task_id": task_id, "correct": int(bool(result.get("score", 0))), "error": None}
    except Exception as exc:
        print(f"[error] {task_id}: {exc}")
        return {"task_id": task_id, "correct": 0, "error": str(exc)}
    finally:
        if _orig_build is not None:
            import daytona_runner as dr
            dr._build_sandbox_command = _orig_build


def compare(actuals: list[dict], baseline: dict, predicted_delta: float) -> None:
    n = len(actuals)
    actual_acc = sum(r["correct"] for r in actuals) / n if n else float("nan")
    errors = sum(1 for r in actuals if r["error"])

    print()
    print("=" * 60)
    print("Tool Ablation Experiment Results")
    print("=" * 60)
    print(f"Agent     : HAL Generalist Agent (disable_tools={DISABLE_TOOLS})")
    print(f"Model     : {MODEL}")
    print(f"Benchmark : {BENCHMARK} ({THRESHOLD} subset, {n} tasks)")
    print()
    print(f"Predicted Δ (observational, model-matched) : {predicted_delta:+.3f}")
    if baseline["baseline_acc"] is not None:
        delta = actual_acc - baseline["baseline_acc"]
        print(f"Baseline accuracy (tool ON, same model)    : {baseline['baseline_acc']:.3f}  (n={baseline['baseline_n']})")
        print(f"Ablated accuracy  (tool OFF)               : {actual_acc:.3f}  (n={n})")
        print(f"Observed Δ                                 : {delta:+.3f}")
        direction = "✓ matches prediction" if (delta > 0) == (predicted_delta > 0) else "✗ opposite direction"
        print(f"Direction                                  : {direction}")
    else:
        print(f"Ablated accuracy (tool OFF) : {actual_acc:.3f}  (n={n})")
        print("No baseline found in response matrix for this model.")
    if errors:
        print(f"[warn] {errors}/{n} tasks errored")
    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    with open(SUBSETS) as f:
        task_ids = json.load(f)[BENCHMARK]["thresholds"][THRESHOLD]["task_ids"]

    baseline = load_observational_baseline(MODEL, BENCHMARK)

    # Observational predicted delta for the default experiment
    PREDICTED_DELTAS = {
        ("web_search,page_browse,text_inspect,vision_query", "swebench_verified_mini"): +0.18,
        ("vision_query", "scicode"):  +0.012,
        ("web_search",   "scicode"):  +0.011,
    }
    predicted_delta = PREDICTED_DELTAS.get((DISABLE_TOOLS, BENCHMARK), float("nan"))

    print(f"Tool ablation: disable_tools={DISABLE_TOOLS}")
    print(f"Benchmark: {BENCHMARK} ({THRESHOLD} subset, {len(task_ids)} tasks)")
    print(f"Model: {MODEL}")
    print(f"Observational predicted Δ: {predicted_delta:+.3f}")
    if baseline["baseline_acc"] is not None:
        print(f"Baseline (tool ON, same model): {baseline['baseline_acc']:.3f}  n={baseline['baseline_n']}")
    print(f"Launching {len(task_ids)} tasks | parallelism={MAX_WORKERS}")

    actuals: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, tid): tid for tid in task_ids}
        for fut in as_completed(futures):
            result = fut.result()
            actuals.append(result)
            status = "✓" if result["correct"] else "✗"
            err = f"  [{result['error'][:60]}]" if result["error"] else ""
            print(f"  {status} {result['task_id']}{err}")

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark", "task_id", "correct", "error"])
        writer.writeheader()
        for r in actuals:
            writer.writerow({"benchmark": BENCHMARK, **r})
    print(f"\nActuals saved: {RESULTS_OUT}")

    compare(actuals, baseline, predicted_delta)


if __name__ == "__main__":
    main()
