"""
Runs INSIDE a Daytona sandbox after hal.cli completes.
Collects results and agent traces, writes them to /home/daytona/reporter_output.json.

The runner (daytona_runner.py) reads this file via sandbox.fs and writes to TiDB
from the local machine, avoiding the need for the sandbox to reach port 4000.

Invoked by the sandbox entrypoint — requires HAL_* env vars.

Usage (inside sandbox):
    python sandbox_reporter.py
"""

import glob
import json
import os
from datetime import datetime, timezone

_OUTPUT_PATH = "/home/daytona/reporter_output.json"


def _find_file(pattern):
    matches = glob.glob(f"results/**/{pattern}", recursive=True)
    return matches[0] if matches else None


def _read_file(path):
    if path and os.path.exists(path):
        with open(path, "r", errors="replace") as f:
            return f.read()
    return ""


def main():
    job_id = os.environ["HAL_JOB_ID"]
    task_key = os.environ["HAL_TASK_KEY"]
    scaffold = os.environ["HAL_SCAFFOLD"]
    model = os.environ["HAL_MODEL"]
    benchmark = os.environ["HAL_BENCHMARK"]
    task_id = os.environ["HAL_TASK_ID"]
    run_id = os.environ["HAL_RUN_ID"]
    exit_code = int(os.environ.get("HAL_EXIT_CODE", "1"))

    agent_id = f"{scaffold}__{model}__{run_id}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    status = "completed" if exit_code == 0 else "failed"

    # Collect _UPLOAD.json
    upload_path = _find_file("*_UPLOAD.json")
    result = {}
    if upload_path:
        with open(upload_path, "r") as f:
            result = json.load(f)

    # Collect agent trace log
    trace_path = _find_file("*_log.log")
    trace_log = _read_file(trace_path)

    # Collect raw submission (first line of JSONL)
    sub_path = _find_file("*_RAW_SUBMISSIONS.jsonl")
    raw_submission = None
    if sub_path:
        first_line = _read_file(sub_path).strip().split("\n")[0]
        if first_line:
            raw_submission = json.loads(first_line)

    # Collect wall clock
    wc_path = _find_file("*_WALL_CLOCK_TIMES.jsonl")
    wall_clock = None
    if wc_path:
        first_line = _read_file(wc_path).strip().split("\n")[0]
        if first_line:
            wall_clock = json.loads(first_line).get("total_time")

    # Determine correct from results
    correct = None
    results_data = result.get("results", {})
    successful = [str(t) for t in results_data.get("successful_tasks", [])]
    failed = [str(t) for t in results_data.get("failed_tasks", [])]
    if task_id in successful:
        correct = 1
    elif task_id in failed:
        correct = 0

    total_cost = result.get("total_cost", results_data.get("total_cost"))

    output = {
        "job_id": job_id,
        "task_key": task_key,
        "agent_id": agent_id,
        "scaffold": scaffold,
        "model": model,
        "benchmark": benchmark,
        "task_id": task_id,
        "run_id": run_id,
        "status": status,
        "completed_at": now,
        "correct": correct,
        "total_cost": total_cost,
        "wall_clock_seconds": wall_clock,
        "trace_log": trace_log,
        "raw_submission": raw_submission,
        "result_json": result,
    }

    with open(_OUTPUT_PATH, "w") as f:
        json.dump(output, f)

    print(f"[reporter] Written to {_OUTPUT_PATH} | status={status} correct={correct}")


if __name__ == "__main__":
    main()
