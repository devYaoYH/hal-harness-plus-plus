"""
Runs INSIDE a Daytona sandbox after hal.cli completes.
Collects results and agent traces, writes them directly to TiDB.

Invoked by the sandbox entrypoint — requires TIDB_* and HAL_* env vars.

Usage (inside sandbox):
    python sandbox_reporter.py
"""

import glob
import json
import os
import sys
from datetime import datetime, timezone

import pymysql


def _connect():
    ssl_config = {"ssl_verify_cert": True, "ssl_verify_identity": True}
    ca = os.environ.get("TIDB_SSL_CA", "")
    if ca:
        ssl_config["ca"] = ca
    return pymysql.connect(
        host=os.environ["TIDB_HOST"],
        port=int(os.environ.get("TIDB_PORT", "4000")),
        user=os.environ["TIDB_USER"],
        password=os.environ["TIDB_PASSWORD"],
        database=os.environ["TIDB_DATABASE"],
        ssl=ssl_config,
        autocommit=True,
    )


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

    # Write to TiDB
    conn = _connect()
    try:
        with conn.cursor() as cur:
            # Update eval_runs
            cur.execute(
                """UPDATE eval_runs
                   SET status=%s, completed_at=%s, result_json=%s,
                       total_cost=%s, stdout=%s, stderr=%s
                   WHERE job_id=%s AND task_key=%s""",
                (status, now, json.dumps(result) if result else None,
                 total_cost, "", "", job_id, task_key),
            )
            print(f"[reporter] eval_runs updated: status={status}")

            # Insert agent trace
            cur.execute(
                """INSERT INTO agent_traces
                   (job_id, task_key, agent_id, scaffold, model, benchmark,
                    task_id, run_id, correct, trace_log, raw_submission,
                    wall_clock_seconds, total_cost)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (job_id, task_key, agent_id, scaffold, model, benchmark,
                 task_id, run_id, correct, trace_log,
                 json.dumps(raw_submission) if raw_submission else None,
                 wall_clock, total_cost),
            )
            print(f"[reporter] agent_trace inserted: correct={correct} wall_clock={wall_clock}")
    finally:
        conn.close()

    print(f"[reporter] Done. agent_id={agent_id}")


if __name__ == "__main__":
    main()
