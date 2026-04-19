# HAL-IRT: Adaptive Agent Evaluation

Extensions to the [HAL leaderboard](https://github.com/benediktstroebl/hal-harness) adding IRT-based adaptive task selection, cold-start predictions for novel scaffold × model pairs, and cloud-native evaluation infrastructure.

## What we add

### 1. Multidimensional IRT model (`irt/`)

A K=2 MIRT model fitted on the public HAL response matrix (13 harnesses × 36 models × 1796 tasks across 8 benchmarks). Each task gets a discrimination vector **a**_j and easiness scalar d_j; each agent gets a latent ability vector **θ**_i ∈ ℝ².

Agent features (model release date, parameter counts, MoE flag, 16 scaffold tool booleans, etc.) feed a projection network f_agent(·) that provides a cold-start **θ** prior for scaffold × model combinations never seen during training — no evaluation runs required.

Key files:
| File | Purpose |
|------|---------|
| `irt/features.py` | Feature extraction for models and scaffolds |
| `irt/data.py` | Response matrix encoding and train/val split |
| `irt/train.py` | MIRT training loop (binary cross-entropy + L2) |
| `irt/predict_new_agent.py` | Cold-start prediction for novel agents |
| `irt/FINDINGS.md` | K-factor sweep results and model selection rationale |
| `irt/GENERALIZATION_TEST.md` | Cold-start validation and tool ablation experiments |

### 2. Adaptive task subsets (`adaptive_task_subsets.json`)

Tasks ranked by discrimination magnitude ‖**a**_j‖; the 80% coverage threshold reduces evaluation cost by 25–44% per benchmark with minimal loss in ability estimation accuracy.

```python
import json
with open("irt_data/adaptive_task_subsets.json") as f:
    subsets = json.load(f)
task_ids = subsets["swebench_verified_mini"]["thresholds"]["80pct"]["task_ids"]
# 28 tasks instead of 50 — 44% cost reduction
```

### 3. Daytona sandbox runner (`../prefect/`)

Each evaluation task runs in an isolated [Daytona](https://www.daytona.io/) cloud sandbox. The harness zip (agents + dependencies) is uploaded once per job; tasks run in parallel up to a configurable worker limit.

```python
from prefect.config_daytona import DaytonaEvalSpec
from prefect.daytona_runner import run_eval_on_daytona

spec = DaytonaEvalSpec(
    agent="hal_generalist_agent",
    agent_function="main.run",
    agent_dir="agents/hal_generalist_agent",
    benchmark="swebench_verified_mini",
    task_id="django__django-11099",
    model="anthropic/claude-3-7-sonnet-20250219",
    job_id="my-run-001",
)
result = run_eval_on_daytona(spec)  # returns {"score": 0/1, ...}
```

Sandboxes are ephemeral and auto-deleted after each task. The `_ZIP_EXCLUDES` list in `daytona_runner.py` controls which agent directories are bundled.

### 4. TiDB trace logging

Agent traces (tool calls, model outputs, scores) are written to a TiDB Serverless instance for downstream analysis. Connection config is read from `.env` (`TIDB_HOST`, `TIDB_USER`, `TIDB_PASSWORD`, `TIDB_DB`). The response matrix used to fit the IRT model (`response_matrix.csv`) is extracted from these logs via `prepare_irt_data.py`.

### 5. Experiment runners

| Script | Experiment |
|--------|-----------|
| `prefect/run_irt_validation.py` | Generalization test: SWE-Agent × DeepSeek-R1 on 28 swebench tasks, predicted 38.4% |
| `prefect/run_tool_ablation.py` | Tool ablation: HAL Generalist −{web_search, page_browse, text_inspect, vision_query} on swebench, predicted +18pp |

Override any parameter via environment variable (`MODEL`, `BENCHMARK`, `DISABLE_TOOLS`, `JOB_ID`, `MAX_WORKERS`).

## Demo

Results are presented at `../docs/index.html` (served via GitHub Pages). The demo includes the K-factor sweep, adaptive subset cost reductions, a swebench model × harness heatmap with IRT-predicted fill-in, and cold-start predictions for novel scaffold × model pairs.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in DAYTONA_API_KEY, TIDB_*, ANTHROPIC_API_KEY

# Refit IRT model after new data
python3 -m irt_data.irt.run --k 2

# Regenerate adaptive subsets
python3 -m irt_data.irt.discriminate --k 2 --thresholds 0.7 0.8 0.9
```
