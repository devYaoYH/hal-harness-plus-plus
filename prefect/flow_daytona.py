"""
Daytona + TiDB evaluation flow — drop-in replacement for the Azure flow.

Usage:
    cd prefect && python flow_daytona.py          # run once
    cd prefect && python flow_daytona.py --serve   # serve as Prefect deployment
"""

import sys

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.futures import wait
from prefect.runtime import flow_run as current_flow_run
from prefect.tasks import exponential_backoff

from config_daytona import AGENTS, BENCHMARK_TASKS, MODELS, DaytonaEvalSpec
from daytona_runner import run_eval_on_daytona
from tidb_storage import ensure_schema


@task(
    task_run_name="{spec.model}/{spec.agent}/{spec.benchmark}/{spec.task_id}",
    retry_delay_seconds=exponential_backoff(backoff_factor=5),
    log_prints=True,
)
def run_eval_task(spec: DaytonaEvalSpec) -> dict:
    result = run_eval_on_daytona(spec)
    stdout = result.pop("_stdout", "")
    create_markdown_artifact(
        key="task-result",
        description=f"{spec.agent} / {spec.benchmark} / {spec.task_id} / {spec.model}",
        markdown=f"""# Eval Result

| Field | Value |
|:------|:------|
| Agent | `{spec.agent}` |
| Benchmark | `{spec.benchmark}` |
| Task | `{spec.task_id}` |
| Model | `{spec.model}` |

## Results
```json
{result}
```

## Sandbox stdout
```
{stdout[:5000]}
```
""",
    )
    return result


@flow(log_prints=True)
def evaluation_harness_daytona(
    agents: list[dict] = AGENTS,
    benchmark_tasks: dict[str, list[str]] = BENCHMARK_TASKS,
    models: list[str] = MODELS,
) -> None:
    """Submit all (agent × benchmark × model) evaluations via Daytona sandboxes."""
    job_id = str(current_flow_run.id)

    ensure_schema()
    print(f"Job started | job_id={job_id}")

    specs = [
        DaytonaEvalSpec(
            agent=agent["name"],
            agent_function=agent["function"],
            agent_dir=agent["dir"],
            benchmark=benchmark,
            task_id=task_id,
            model=model,
            job_id=job_id,
        )
        for agent in agents
        for benchmark, tasks in benchmark_tasks.items()
        for task_id in tasks
        for model in models
    ]

    print(f"Submitting {len(specs)} eval tasks")
    futures = [run_eval_task.submit(spec) for spec in specs]
    wait(futures)

    rows = []
    for spec, future in zip(specs, futures):
        try:
            run_id = future.result().get("run_id", "n/a")
            rows.append(
                f"| {spec.agent} | {spec.benchmark} | `{spec.task_id}` | {spec.model} | ✅ | `{run_id}` |"
            )
        except Exception as e:
            rows.append(
                f"| {spec.agent} | {spec.benchmark} | `{spec.task_id}` | {spec.model} | ❌ | {e} |"
            )

    table = "\n".join(rows)
    passed = sum(1 for r in rows if "✅" in r)
    failed = len(rows) - passed

    create_markdown_artifact(
        key="eval-summary",
        description="Daytona eval results",
        markdown=f"""# Evaluation Run Summary (Daytona + TiDB)

**{passed} passed · {failed} failed · {len(rows)} total**

| Agent | Benchmark | Task | Model | Status | Run ID |
|:------|:----------|:-----|:------|:------:|:-------|
{table}
""",
    )


if __name__ == "__main__":
    if "--serve" in sys.argv:
        evaluation_harness_daytona.serve(name="eval-harness-daytona")
    else:
        evaluation_harness_daytona()
