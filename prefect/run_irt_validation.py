"""
IRT generalization validation — parallel Daytona runner.

Runs a novel scaffold × model combination (never seen during IRT training)
on the IRT-selected adaptive task subset, then compares actual results
against cold-start predictions from the MIRT model.

Usage:
    cd /path/to/hal-harness
    python3 -c "
    import os, sys
    sys.path.insert(0, 'prefect')
    os.environ.setdefault('SSL_CERT_FILE', __import__('certifi').where())
    from dotenv import load_dotenv; load_dotenv()
    import run_irt_validation
    run_irt_validation.main()
    "

Config (edit below or override via env vars):
    AGENT            — scaffold key (must exist in agents/)
    MODEL            — provider-prefixed model string
    BENCHMARK        — HAL benchmark name
    THRESHOLD        — IRT discrimination coverage threshold (default: 80pct)
    JOB_ID           — unique run identifier
    MAX_WORKERS      — parallel sandboxes (default: 10)
"""

from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────

AGENT       = os.getenv("AGENT",     "SWE-agent")
MODEL       = os.getenv("MODEL",     "together_ai/deepseek-ai/DeepSeek-R1")
BENCHMARK   = os.getenv("BENCHMARK", "swebench_verified_mini")
THRESHOLD   = os.getenv("THRESHOLD", "80pct")
JOB_ID      = os.getenv("JOB_ID",   "irt-val-sweagent-deepseekr1-001")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))

# Agent registry — add entries as new scaffolds are validated
_AGENT_REGISTRY: dict[str, dict] = {
    "SWE-agent": {
        "function": "main.run",
        "dir":      "agents/SWE-agent",
        "benchmark_extra": "swebench",
    },
    "hal_generalist_agent": {
        "function": "main.run",
        "dir":      "agents/hal_generalist_agent",
        "benchmark_extra": "swebench",
    },
}

REPO_ROOT  = Path(__file__).resolve().parent.parent
SUBSETS    = REPO_ROOT / "irt_data" / "adaptive_task_subsets.json"
PREDS_CSV  = REPO_ROOT / "irt_data" / f"pred_{AGENT.replace('-','').lower()}_{MODEL.split('/')[-1].replace('-','').lower()}.csv"
RESULTS_OUT = REPO_ROOT / "irt_data" / f"actuals_{JOB_ID}.csv"

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_task_ids() -> list[str]:
    with open(SUBSETS) as f:
        subsets = json.load(f)
    return subsets[BENCHMARK]["thresholds"][THRESHOLD]["task_ids"]


def load_predictions() -> dict[str, float]:
    """Returns {task_id: predicted_prob}. Empty dict if predictions CSV missing."""
    if not PREDS_CSV.exists():
        print(f"[warn] predictions CSV not found at {PREDS_CSV} — skipping comparison")
        return {}
    preds = {}
    with open(PREDS_CSV) as f:
        for row in csv.DictReader(f):
            if row["benchmark"] == BENCHMARK:
                preds[row["task_id"]] = float(row["predicted_prob"])
    return preds


def run_one(task_id: str, agent_cfg: dict) -> dict:
    from config_daytona import DaytonaEvalSpec
    from daytona_runner import run_eval_on_daytona

    spec = DaytonaEvalSpec(
        agent=AGENT,
        agent_function=agent_cfg["function"],
        agent_dir=agent_cfg["dir"],
        benchmark=BENCHMARK,
        task_id=task_id,
        model=MODEL,
        job_id=JOB_ID,
        benchmark_extra=agent_cfg["benchmark_extra"],
    )
    try:
        result = run_eval_on_daytona(spec)
        return {"task_id": task_id, "correct": int(bool(result.get("score", 0))), "error": None}
    except Exception as exc:
        print(f"[error] {task_id}: {exc}")
        return {"task_id": task_id, "correct": 0, "error": str(exc)}


def compare(actuals: list[dict], predictions: dict[str, float]) -> None:
    if not predictions:
        return
    matched = [(r["task_id"], predictions[r["task_id"]], r["correct"])
               for r in actuals if r["task_id"] in predictions]
    if not matched:
        print("[warn] no tasks matched between actuals and predictions")
        return

    n = len(matched)
    brier = sum((p - c) ** 2 for _, p, c in matched) / n
    actual_acc = sum(c for _, _, c in matched) / n
    pred_acc   = sum(p for _, p, _ in matched) / n

    # Point-biserial correlation
    import math
    mean_p = pred_acc
    var_p = sum((p - mean_p) ** 2 for _, p, _ in matched) / n
    std_p = math.sqrt(var_p) if var_p > 0 else 1.0
    cov = sum((p - mean_p) * (c - actual_acc) for _, p, c in matched) / n
    std_c = math.sqrt(actual_acc * (1 - actual_acc)) if 0 < actual_acc < 1 else 1.0
    r = cov / (std_p * std_c) if std_p * std_c > 0 else float("nan")

    print()
    print("=" * 50)
    print("IRT Generalization Evaluation")
    print("=" * 50)
    print(f"Agent / Model  : {AGENT} × {MODEL}")
    print(f"Benchmark      : {BENCHMARK} ({THRESHOLD} subset, {n} tasks)")
    print(f"Actual accuracy : {actual_acc:.3f}")
    print(f"Predicted acc   : {pred_acc:.3f}  (IRT cold-start)")
    print(f"Brier score     : {brier:.4f}  (chance = 0.25)")
    print(f"Point-biserial r: {r:.3f}    (predicted_prob vs correct)")
    print("=" * 50)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    agent_cfg = _AGENT_REGISTRY.get(AGENT)
    if agent_cfg is None:
        sys.exit(f"Unknown agent {AGENT!r}. Add it to _AGENT_REGISTRY in run_irt_validation.py")

    task_ids   = load_task_ids()
    predictions = load_predictions()

    print(f"Launching {len(task_ids)} tasks | agent={AGENT} model={MODEL} job={JOB_ID}")
    print(f"Parallelism: {MAX_WORKERS} sandboxes")

    actuals: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, tid, agent_cfg): tid for tid in task_ids}
        for fut in as_completed(futures):
            result = fut.result()
            actuals.append(result)
            status = "✓" if result["correct"] else "✗"
            err = f"  [err: {result['error'][:60]}]" if result["error"] else ""
            print(f"  {status} {result['task_id']}{err}")

    # Save actuals
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark", "task_id", "correct", "error"])
        writer.writeheader()
        for r in actuals:
            writer.writerow({"benchmark": BENCHMARK, **r})
    print(f"\nActuals saved: {RESULTS_OUT}")

    compare(actuals, predictions)


if __name__ == "__main__":
    main()
