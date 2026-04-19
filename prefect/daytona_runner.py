"""
Daytona sandbox runner — submit, execute, and collect results for eval tasks.

Each sandbox writes its own results directly to TiDB via sandbox_reporter.py,
so the pipeline is resilient to local machine crashes.
"""

import datetime
import io
import json
import re
import socket
import uuid
import zipfile
from pathlib import Path

from daytona_sdk import Daytona, DaytonaConfig, CreateSandboxFromImageParams
from daytona_sdk._sync.sandbox import Resources

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
    insert_task_metadata,
    insert_agent_trace,
    update_task_result,
    save_task_results_local,
)

_ZIP_EXCLUDES = {
    ".git",
    ".venv",
    "prefect",
    "results",
    "__pycache__",
    ".mypy_cache",
    # Large downloaded data — never needed inside sandboxes
    "irt_data/traces",
    "irt_data/test_traces",
    "irt_data/extracted",
    # Agents not used in this run
    "agents/auto-code-rover",
    "agents/Moatless",
    "agents/SWE-agent-v1.0",
    "agents/Enigma",
    "agents/USACO",
    # Benchmark data not needed for swebench
    "hal/benchmarks/corebench",
    "hal/benchmarks/USACO",
    "hal/benchmarks/appworld",
    "hal/benchmarks/scienceagentbench",
    "hal/benchmarks/taubench/taubench_setup.sh",
    "hal/benchmarks/colbench",
}


def _resolve_ip(hostname: str) -> str:
    return socket.gethostbyname(hostname) + "/32"


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
    reporter = Path(__file__).parent / "sandbox_reporter.py"
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
        # Inject sandbox_reporter.py from prefect/ (excluded above) at repo root level
        zf.write(reporter, "prefect/sandbox_reporter.py")
    size_mb = len(buf.getvalue()) / 1024 / 1024
    print(f"Repo zip size: {size_mb:.1f} MB")
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
    # Don't forward TIDB_SSL_CA — it's a local file path that won't exist in sandbox.
    # sandbox_reporter.py will use system CA bundle when TIDB_SSL_CA is unset.
    env.update({
        "HF_DATASETS_CACHE": "/home/daytona/hal-harness/hf_cache",
        "HF_DATASETS_OFFLINE": "1",
        "HAL_JOB_ID": spec.job_id,
        "HAL_TASK_KEY": task_key,
        "HAL_SCAFFOLD": spec.agent,
        "HAL_MODEL": spec.model,
        "HAL_BENCHMARK": spec.benchmark,
        "HAL_TASK_ID": spec.task_id,
        "HAL_RUN_ID": run_id,
    })
    return env


_SANDBOX_LOG = "/home/daytona/eval.log"
_SANDBOX_DONE = "/home/daytona/eval.done"


def _build_install_command(spec: DaytonaEvalSpec) -> str:
    """Install all system + Python deps. Run synchronously before eval."""
    return (
        f"echo '[eval] started' && "
        f"apt-get update -qq 2>&1 | tail -1 && apt-get install -y -q git docker.io 2>&1 | tail -1 && "
        f"(dockerd --host=unix:///var/run/docker.sock &) && sleep 3 && "
        f"echo '[eval] apt done' && "
        f"cd /home/daytona/hal-harness && "
        f"pip install --no-deps -e . && "
        f"pip install swebench python-dotenv docker tenacity cryptography && echo '[eval] hal installed' && "
        f"REQ={spec.agent_dir}/requirements_{spec.benchmark}.txt && "
        f"[ -f \"$REQ\" ] || REQ={spec.agent_dir}/requirements.txt && "
        f"pip install -r \"$REQ\" && echo '[eval] agent deps installed'"
    )


def _build_eval_command(spec: DaytonaEvalSpec, run_id: str) -> str:
    """Run eval + reporter in background; caller polls for reporter_output.json."""
    return (
        f"cd /home/daytona/hal-harness && "
        f"echo '[eval] launching hal.cli' && python -m hal.cli "
        f"  --agent_name '{spec.agent}' "
        f"  --agent_function '{spec.agent_function}' "
        f"  --agent_dir '{spec.agent_dir}' "
        f"  --benchmark '{spec.benchmark}' "
        f"  --task_ids '{spec.task_id}' "
        f"  --run_id '{run_id}' "
        f"  -A 'model_name={spec.model}' ; "
        f"export HAL_EXIT_CODE=$? ; "
        f"python /home/daytona/hal-harness/prefect/sandbox_reporter.py"
    )


_REPORTER_OUTPUT = "/home/daytona/reporter_output.json"


def _poll_for_reporter(sandbox, task_key: str, timeout_seconds: int = 3600) -> str:
    """Poll eval.log every 30s; return full log as soon as reporter_output.json appears."""
    import time
    deadline = time.time() + timeout_seconds
    lines_seen = 0
    while time.time() < deadline:
        time.sleep(30)
        # Stream new log lines
        log = sandbox.process.exec(f"cat {_SANDBOX_LOG} 2>/dev/null || true")
        full_log = log.result or ""
        lines = full_log.splitlines()
        new_lines = lines[lines_seen:]
        if new_lines:
            for line in new_lines:
                print(f"[sandbox:{task_key[:12]}] {line}")
            lines_seen = len(lines)

        # Exit as soon as reporter has written its output
        done = sandbox.process.exec(
            f"[ -f {_REPORTER_OUTPUT} ] && echo yes || echo no"
        )
        if (done.result or "").strip() == "yes":
            print(f"[sandbox:{task_key[:12]}] Reporter done — collecting results")
            return full_log

    raise TimeoutError(f"Sandbox task {task_key} exceeded {timeout_seconds}s timeout")


def _write_reporter_to_tidb(data: dict, job_id: str, task_key: str) -> None:
    """Write sandbox reporter output to TiDB from the local machine."""
    result = data.get("result_json") or {}
    update_task_result(
        job_id,
        task_key,
        status=data.get("status", "failed"),
        result=result,
        total_cost=data.get("total_cost"),
        stdout="",
        stderr="",
    )
    insert_agent_trace(
        job_id=job_id,
        task_key=task_key,
        scaffold=data["scaffold"],
        model=data["model"],
        benchmark=data["benchmark"],
        task_id=data["task_id"],
        run_id=data["run_id"],
        correct=data.get("correct"),
        trace_log=data.get("trace_log", ""),
        raw_submission=data.get("raw_submission"),
        wall_clock_seconds=data.get("wall_clock_seconds"),
        total_cost=data.get("total_cost"),
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
            resources=Resources(cpu=2, memory=8, disk=10),
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

        # Phase 1: install deps synchronously (safe to block — no eval running yet)
        print(f"Installing deps | task_key={task_key}")
        install_cmd = _build_install_command(spec)
        r = sandbox.process.exec(f"bash -c {repr(install_cmd)} 2>&1 | tee {_SANDBOX_LOG}", timeout=1800)
        print(f"Install done (exit={r.exit_code})")

        # Phase 2: launch eval + reporter in background, poll for reporter_output.json
        print(f"Launching eval | task_key={task_key}")
        eval_cmd = _build_eval_command(spec, run_id)
        eval_script = "/home/daytona/eval.sh"
        sandbox.fs.upload_file(eval_cmd.encode(), eval_script)
        sandbox.process.exec(
            f"nohup bash {eval_script} >> {_SANDBOX_LOG} 2>&1 &",
            timeout=10,
        )
        _poll_for_reporter(sandbox, task_key, timeout_seconds=3600)

        # Read reporter output and write to TiDB from local machine (sandbox can't reach port 4000)
        reporter_raw = sandbox.fs.download_file(_REPORTER_OUTPUT)
        reporter_data = json.loads(reporter_raw) if reporter_raw else {}
        print(f"[reporter] status={reporter_data.get('status')} correct={reporter_data.get('correct')}")
        _write_reporter_to_tidb(reporter_data, spec.job_id, task_key)

        result = reporter_data.get("result_json", {})
        log = sandbox.process.exec(f"tail -100 {_SANDBOX_LOG} 2>/dev/null || true")
        stdout = log.result or ""
        print(f"[stdout tail]\n{stdout}")
        local_path = save_task_results_local(spec.job_id, task_key, result)
        print(f"Results in TiDB + local cache | local={local_path}")
        result["_stdout"] = stdout
        return result

    finally:
        print(f"Deleting sandbox | id={sandbox.id}")
        client.delete(sandbox)
