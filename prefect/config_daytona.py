import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("SSL_CERT_FILE"):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass


@dataclass(frozen=True)
class DaytonaEvalSpec:
    """One (agent × benchmark × task × model) evaluation unit for Daytona."""

    agent: str
    agent_function: str
    agent_dir: str
    benchmark: str
    task_id: str
    model: str
    job_id: str
    benchmark_extra: str = "swebench"  # pyproject.toml optional-dep group to install


# ---------------------------------------------------------------------------
# Daytona
# ---------------------------------------------------------------------------
DAYTONA_API_KEY = os.getenv("DAYTONA_API_KEY", "")
DAYTONA_SERVER_URL = os.getenv("DAYTONA_SERVER_URL", "https://app.daytona.io/api")
DAYTONA_TARGET = os.getenv("DAYTONA_TARGET", "us")

# ---------------------------------------------------------------------------
# TiDB
# ---------------------------------------------------------------------------
TIDB_HOST = os.getenv("TIDB_HOST", "")
TIDB_PORT = int(os.getenv("TIDB_PORT", "4000"))
TIDB_USER = os.getenv("TIDB_USER", "")
TIDB_PASSWORD = os.getenv("TIDB_PASSWORD", "")
TIDB_DATABASE = os.getenv("TIDB_DATABASE", "hal_traces")
TIDB_SSL_CA = os.getenv("TIDB_SSL_CA", "")

# ---------------------------------------------------------------------------
# Environment variables forwarded into each Daytona sandbox
# ---------------------------------------------------------------------------
TASK_ENV_VARS: dict[str, str] = {
    k: v
    for k in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "WEAVE_API_KEY",
    ]
    if (v := os.getenv(k))
}

POLL_INTERVAL_SECONDS = 15

# ---------------------------------------------------------------------------
# Demo evaluation matrix — 1 agent × 1 model × 10 swebench-mini tasks
# ---------------------------------------------------------------------------
AGENTS = [
    {
        "name": "hal_generalist_agent",
        "function": "main.run",
        "dir": "agents/hal_generalist_agent",
    },
]

DEMO_TASK_IDS = [
    "django__django-11790",
    "django__django-11815",
    "django__django-11848",
    "django__django-11880",
    "django__django-11885",
    "django__django-11951",
    "django__django-11964",
    "django__django-11999",
    "django__django-12039",
    "django__django-12050",
]

BENCHMARK_TASKS: dict[str, list[str]] = {
    "swebench_verified_mini": DEMO_TASK_IDS,
}

MODELS = [
    "anthropic/claude-sonnet-4-5-20250929",
]
