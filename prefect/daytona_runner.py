"""
Daytona sandbox runner — submit, execute, and collect results for eval tasks.

Each sandbox writes its own results directly to TiDB via sandbox_reporter.py,
so the pipeline is resilient to local machine crashes.
"""

import datetime
import io
import json
import re
import uuid
import zipfile
from pathlib import Path

from daytona_sdk import Daytona, DaytonaConfig, CreateSandboxFromImageParams

from config_daytona import (
    DAYTONA_API_KEY,
    DAYTONA_SERVER_URL,
    DAYTONA_TARGET,
    TASK_ENV_VARS,
    TIDB_DATABASE,
    TIDB_HOST,
    TIDB_PASSWORD,
    TIDB_PORT,
    TIDB_SSL_CA,
    TIDB_USER,
    DaytonaEvalSpec,
)
from tidb_storage import (
    fetch_task_result,
    insert_task_metadata,
    save_task_results_local,
)

_ZIP_EXCLUDES = {
    ".git",
    ".venv",
    "results",
    "__pycache__",
    ".mypy_cache",
    "hal/benchmarks/corebench",
    "hal/benchmarks/USACO",
    "hal/benchmarks/appworld",
    "hal/benchmarks/scienceagentbench",
    "hal/benchmarks/taubench/taubench_setup.sh",
}


def _daytona_client() -> Daytona:
    return Daytona(
        DaytonaConfig(
            api_key=DAYTONA_API_KEY,
            api_url=DAYTONA_SERVER_URL,
            target=DAYTONA_TARGET,
        )
    )


def _zip_repo() -> bytes:
    repo_root = Path(__file__).resolve().parent.parent
    buf = io.BytesIO()
    with zipfile.ZipFile(
        buf, mode="w", compression=zipfile.ZIP_DEFLATED, strict_timestamps=False
    ) as zf:
        for path in sorted(repo_root.rglob("*")):
            rel = path.relative_to(repo_root)
            if any(str(rel).startswith(ex) for ex in _ZIP_EXCLUDES):
                continue
            if path.is_file():
                zf.write(path, rel)
    return buf.getvalue()


def _task_key(spec: DaytonaEvalSpec) -> str:
    raw = f"{spec.model}-{spec.agent}-{spec.benchmark}-{spec.task_id}"
    sanitized = re.sub(r"[^a-zA-Z0-9\-_]", "-", raw)
    suffix = uuid.uuid4().hex[:8]
    return f"{sanitized[:55]}-{suffix}"


def _make_run_id(spec: DaytonaEvalSpec) -> str:
    sanitized_model = re.sub(r"[^a-z0-9]", "_", spec.model.lower())
    return (
        f"{spec.benchmark}_{spec.agent}_{sanitized_model}"
        f"_task{spec.task_id}_{int(datetime.datetime.now().timestamp())}"
    )


def _sandbox_env_vars(spec: DaytonaEvalSpec, task_key: str, run_id: str) -> dict[str, str]:
    """Env vars forwarded into the sandbox: LLM keys + TiDB creds + task metadata."""
    env = {**TASK_ENV_VARS}
    env.update({
        "TIDB_HOST": TIDB_HOST,
        "TIDB_PORT": str(TIDB_PORT),
        "TIDB_USER": TIDB_USER,
        "TIDB_PASSWORD": TIDB_PASSWORD,
        "TIDB_DATABASE": TIDB_DATABASE,
    })
    if TIDB_SSL_CA:
        env["TIDB_SSL_CA"] = TIDB_SSL_CA
    env.update({
        "HAL_JOB_ID": spec.job_id,
        "HAL_TASK_KEY": task_key,
        "HAL_SCAFFOLD": spec.agent,
        "HAL_MODEL": spec.model,
        "HAL_BENCHMARK": spec.benchmark,
        "HAL_TASK_ID": spec.task_id,
        "HAL_RUN_ID": run_id,
    })
    return env


def _build_sandbox_command(spec: DaytonaEvalSpec, run_id: str) -> str:
    """Command that runs inside the sandbox: install, eval, then report to TiDB."""
    return (
        "cd /home/daytona/hal-harness && "
        "pip install --quiet -e '.[dev]' && "
        "pip install --quiet pymysql && "
        f"if [ -f '{spec.agent_dir}/requirements.txt' ]; then "
        f"  pip install --quiet -r '{spec.agent_dir}/requirements.txt'; "
        "fi && "
        # Run the eval, capture exit code
        f"python -m hal.cli "
        f"  --agent_name '{spec.agent}' "
        f"  --agent_function '{spec.agent_function}' "
        f"  --agent_dir '{spec.agent_dir}' "
        f"  --benchmark '{spec.benchmark}' "
        f"  --task_ids '{spec.task_id}' "
        f"  --run_id '{run_id}' "
        f"  -A 'model_name={spec.model}' ; "
        # Always run reporter regardless of eval exit code
        f"HAL_EXIT_CODE=$? && "
        f"export HAL_EXIT_CODE && "
        f"python prefect/sandbox_reporter.py"
    )


def run_eval_on_daytona(spec: DaytonaEvalSpec) -> dict:
    """Create a Daytona sandbox that runs eval + writes results to TiDB autonomously."""
    client = _daytona_client()
    task_key = _task_key(spec)
    run_id = _make_run_id(spec)
    submitted_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Record intent in TiDB before sandbox creation
    insert_task_metadata(spec, task_key, run_id, submitted_at)

    print(f"Creating sandbox | task_key={task_key}")
    sandbox = client.create(
        CreateSandboxFromImageParams(
            image="python:3.11-slim",
            language="python",
            env_vars=_sandbox_env_vars(spec, task_key, run_id),
            auto_stop_interval=0,
        ),
        timeout=300,
    )

    try:
        print(f"Sandbox created | id={sandbox.id} — uploading code...")
        zip_bytes = _zip_repo()
        sandbox.fs.upload_file(zip_bytes, "/home/daytona/hal-harness.zip")
        sandbox.process.exec(
            "python3 -m zipfile -e /home/daytona/hal-harness.zip /home/daytona/hal-harness",
            timeout=120,
        )

        print(f"Running eval + reporter | task_key={task_key}")
        cmd = _build_sandbox_command(spec, run_id)
        response = sandbox.process.exec(cmd, timeout=1800)

        stdout = response.result if response.result else ""
        exit_code = response.exit_code

        if exit_code != 0:
            print(f"Warning: sandbox command exited with code {exit_code}")

        # Results are already in TiDB — fetch them back for the Prefect artifact
        result = fetch_task_result(spec.job_id, task_key) or {}
        local_path = save_task_results_local(spec.job_id, task_key, result)
        print(f"Results in TiDB + local cache | local={local_path}")

        result["_stdout"] = stdout[:5000]
        return result

    finally:
        print(f"Deleting sandbox | id={sandbox.id}")
        client.delete(sandbox)
