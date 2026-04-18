"""
TiDB storage backend — stores eval metadata, results, and logs.

Replaces Azure Blob Storage for the Daytona flow.
"""

import json
import ssl
from datetime import datetime, timezone
from pathlib import Path

import pymysql

from config_daytona import (
    TIDB_DATABASE,
    TIDB_HOST,
    TIDB_PASSWORD,
    TIDB_PORT,
    TIDB_SSL_CA,
    TIDB_USER,
)

_SCHEMA_EVAL_RUNS = """
CREATE TABLE IF NOT EXISTS eval_runs (
    job_id       VARCHAR(255) NOT NULL,
    task_key     VARCHAR(255) NOT NULL,
    scaffold     VARCHAR(255),
    model        VARCHAR(255),
    benchmark    VARCHAR(255),
    task_id      VARCHAR(255),
    run_id       VARCHAR(255),
    submitted_at DATETIME,
    completed_at DATETIME,
    status       VARCHAR(32) DEFAULT 'pending',
    result_json  JSON,
    total_cost   FLOAT,
    stdout       LONGTEXT,
    stderr       LONGTEXT,
    PRIMARY KEY  (job_id, task_key),
    INDEX idx_job (job_id),
    INDEX idx_scaffold_model (scaffold, model),
    INDEX idx_benchmark_task (benchmark, task_id)
);
"""

_SCHEMA_AGENT_TRACES = """
CREATE TABLE IF NOT EXISTS agent_traces (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    job_id              VARCHAR(255) NOT NULL,
    task_key            VARCHAR(255) NOT NULL,
    agent_id            VARCHAR(512) NOT NULL,
    scaffold            VARCHAR(255) NOT NULL,
    model               VARCHAR(255) NOT NULL,
    benchmark           VARCHAR(255) NOT NULL,
    task_id             VARCHAR(255) NOT NULL,
    run_id              VARCHAR(255),
    correct             TINYINT,
    trace_log           LONGTEXT,
    raw_submission      JSON,
    wall_clock_seconds  FLOAT,
    total_cost          FLOAT,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_agent_id (agent_id),
    INDEX idx_scaffold_model (scaffold, model),
    INDEX idx_benchmark_task (benchmark, task_id),
    INDEX idx_job_task (job_id, task_key)
);
"""


def _connect() -> pymysql.Connection:
    ssl_config = {"ssl_verify_cert": True, "ssl_verify_identity": True}
    if TIDB_SSL_CA:
        ssl_config["ca"] = TIDB_SSL_CA
    return pymysql.connect(
        host=TIDB_HOST,
        port=TIDB_PORT,
        user=TIDB_USER,
        password=TIDB_PASSWORD,
        database=TIDB_DATABASE,
        ssl=ssl_config,
        autocommit=True,
    )


def ensure_schema() -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_EVAL_RUNS)
            cur.execute(_SCHEMA_AGENT_TRACES)
    finally:
        conn.close()
    print(f"TiDB schema ensured | host={TIDB_HOST} db={TIDB_DATABASE}")


def insert_task_metadata(spec, task_key: str, run_id: str, submitted_at: str) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO eval_runs
                   (job_id, task_key, scaffold, model, benchmark, task_id, run_id,
                    submitted_at, status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'running')
                   ON DUPLICATE KEY UPDATE submitted_at=VALUES(submitted_at), status='running'""",
                (
                    spec.job_id,
                    task_key,
                    spec.agent,
                    spec.model,
                    spec.benchmark,
                    spec.task_id,
                    run_id,
                    submitted_at,
                ),
            )
    finally:
        conn.close()


def update_task_result(
    job_id: str,
    task_key: str,
    *,
    status: str,
    result: dict | None = None,
    total_cost: float | None = None,
    stdout: str = "",
    stderr: str = "",
) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE eval_runs
                   SET status=%s,
                       completed_at=%s,
                       result_json=%s,
                       total_cost=%s,
                       stdout=%s,
                       stderr=%s
                   WHERE job_id=%s AND task_key=%s""",
                (
                    status,
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    json.dumps(result) if result else None,
                    total_cost,
                    stdout,
                    stderr,
                    job_id,
                    task_key,
                ),
            )
    finally:
        conn.close()


def insert_agent_trace(
    *,
    job_id: str,
    task_key: str,
    scaffold: str,
    model: str,
    benchmark: str,
    task_id: str,
    run_id: str,
    correct: int | None = None,
    trace_log: str = "",
    raw_submission: dict | None = None,
    wall_clock_seconds: float | None = None,
    total_cost: float | None = None,
) -> None:
    agent_id = f"{scaffold}__{model}__{run_id}"
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO agent_traces
                   (job_id, task_key, agent_id, scaffold, model, benchmark, task_id,
                    run_id, correct, trace_log, raw_submission, wall_clock_seconds,
                    total_cost)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    job_id,
                    task_key,
                    agent_id,
                    scaffold,
                    model,
                    benchmark,
                    task_id,
                    run_id,
                    correct,
                    trace_log,
                    json.dumps(raw_submission) if raw_submission else None,
                    wall_clock_seconds,
                    total_cost,
                ),
            )
    finally:
        conn.close()


def fetch_task_result(job_id: str, task_key: str) -> dict | None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT result_json FROM eval_runs WHERE job_id=%s AND task_key=%s",
                (job_id, task_key),
            )
            row = cur.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return None
    finally:
        conn.close()


# Local cache (same as Azure flow)
_RESULTS_DIR = Path(__file__).resolve().parent.parent / ".prefect_results"


def save_task_results_local(job_id: str, task_key: str, result: dict) -> Path:
    out = _RESULTS_DIR / job_id / task_key / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return out
