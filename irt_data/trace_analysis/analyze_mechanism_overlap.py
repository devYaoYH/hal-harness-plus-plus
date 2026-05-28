#!/usr/bin/env python3
"""Summarize trace signals for a task covered by multiple tool mechanisms."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from irt_data.prepare_irt_data import decrypt_upload_json  # noqa: E402

TARGET_BENCHMARK = "scienceagentbench"
TARGET_TASK_ID = "3"
OUT_DIR = ROOT / "irt_data" / "trace_analysis" / "mechanism_overlap"


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(flatten_text(v) for v in value)
    if isinstance(value, dict):
        if "text" in value:
            return flatten_text(value["text"])
        if "content" in value:
            return flatten_text(value["content"])
        return "\n".join(flatten_text(v) for v in value.values())
    return str(value)


def output_text(log: dict[str, Any]) -> str:
    choices = ((log.get("output") or {}).get("choices") or [])
    if not choices:
        return ""
    return flatten_text((choices[0].get("message") or {}).get("content"))


def input_text(log: dict[str, Any]) -> str:
    messages = (log.get("inputs") or {}).get("messages") or []
    return flatten_text(messages)


def final_history_text(agent_output: Any) -> str:
    if isinstance(agent_output, dict):
        history = agent_output.get("history")
        if history:
            return flatten_text(history)
    return flatten_text(agent_output)


def finish_reason(log: dict[str, Any]) -> str:
    choices = ((log.get("output") or {}).get("choices") or [])
    if not choices:
        return ""
    return str(choices[0].get("finish_reason") or "")


def classify_failure(log_info: Any, valid_program: Any, correct: int) -> str:
    text = flatten_text(log_info)
    low = text.lower()
    if int(correct) == 1:
        return "success"
    if "does not save its output correctly" in low:
        return "output_not_saved"
    if "traceback" in low or "keyerror" in low or "valueerror" in low:
        return "runtime_exception"
    if "func_correctness': false" in low or '"func_correctness": false' in low:
        return "functionally_incorrect"
    if str(valid_program) in {"0", "0.0"}:
        return "invalid_program"
    return "failed_other"


def task_logs(data: dict[str, Any], task_id: str) -> list[dict[str, Any]]:
    logs = data.get("raw_logging_results") or []
    selected = [
        log
        for log in logs
        if str((log.get("attributes") or {}).get("weave_task_id")) == task_id
    ]
    return sorted(selected, key=lambda log: log.get("started_at") or "")


def mechanism_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (benchmark, task_id), sub in df.groupby(["benchmark", "task_id"]):
        mechanisms = set(sub["tool_exposure_mechanism"])
        if len(mechanisms) < 2:
            continue
        rec = {
            "benchmark": benchmark,
            "task_id": task_id,
            "n": len(sub),
            "mechanisms": len(mechanisms),
            "harnesses": sub["normalized_harness"].nunique(),
            "models": sub["normalized_model"].nunique(),
            "pass_rate": sub["correct"].mean(),
            "mechanism_list": "; ".join(sorted(mechanisms)),
        }
        for mechanism, mech_sub in sub.groupby("tool_exposure_mechanism"):
            rec[f"{mechanism}_n"] = len(mech_sub)
            rec[f"{mechanism}_pass_rate"] = mech_sub["correct"].mean()
            rec[f"{mechanism}_successes"] = int(mech_sub["correct"].sum())
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if {"code_agent_tool_registry_pass_rate", "harness_side_augmentation_pass_rate"} <= set(
        out.columns
    ):
        out["code_vs_augmentation_gap"] = (
            out["code_agent_tool_registry_pass_rate"]
            - out["harness_side_augmentation_pass_rate"]
        ).abs()
    return out.sort_values(
        ["mechanisms", "code_vs_augmentation_gap", "n"],
        ascending=[False, False, False],
        na_position="last",
    )


def summarize_run(row: pd.Series) -> dict[str, Any]:
    zip_path = ROOT / "irt_data" / "traces" / row["source_file"]
    data = decrypt_upload_json(zip_path)
    if data is None:
        return {"agent_id": row["agent_id"], "trace_loaded": False}

    eval_result = (data.get("raw_eval_results") or {}).get("eval_result") or {}
    agent_outputs = (data.get("raw_eval_results") or {}).get("agent_output") or {}
    task_eval = eval_result.get(TARGET_TASK_ID, {})
    task_output = agent_outputs.get(TARGET_TASK_ID)
    logs = task_logs(data, TARGET_TASK_ID)

    inputs = "\n".join(input_text(log) for log in logs)
    outputs = "\n".join(output_text(log) for log in logs)
    final_text = final_history_text(task_output)
    all_text = "\n".join([inputs, outputs, final_text])
    finish_counts = Counter(finish_reason(log) for log in logs if finish_reason(log))
    tool_names = [
        "web_search",
        "page_browse",
        "python_exec",
        "execute_bash",
        "text_inspect",
        "file_edit",
        "file_search",
        "vision_query",
    ]
    log_info = task_eval.get("log_info")

    return {
        "agent_id": row["agent_id"],
        "trace_loaded": True,
        "benchmark": row["benchmark"],
        "task_id": TARGET_TASK_ID,
        "correct": row["correct"],
        "raw_scaffold": row["scaffold"],
        "normalized_harness": row["normalized_harness"],
        "tool_exposure_mechanism": row["tool_exposure_mechanism"],
        "tool_exposure_strength": row.get("tool_exposure_strength", ""),
        "raw_model": row["model"],
        "normalized_model": row["normalized_model"],
        "reasoning_effort_hint": row.get("reasoning_effort_hint", ""),
        "run_id": row["run_id"],
        "source_file": row["source_file"],
        "llm_call_count": len(logs),
        "finish_reasons": json.dumps(dict(finish_counts), sort_keys=True),
        "length_finish_count": finish_counts.get("length", 0),
        "output_code_block_count": len(re.findall(r"```(?:python|py)?", outputs)),
        "final_code_block_count": len(re.findall(r"```(?:python|py)?", final_text)),
        "observation_mentions": len(re.findall(r"\bObservation:", all_text)),
        "error_mentions": len(re.findall(r"Traceback|ImportError|Error:|exception", all_text, re.I)),
        "tool_registry_prompt": int("given access to a list of tools" in inputs),
        "self_debug_prompt": int("reported issues" in inputs or "error messages" in inputs),
        "complete_program_prompt": int("complete program" in inputs),
        "tool_name_mentions": sum(int(name in inputs) for name in tool_names),
        "exec_feedback_loop": int("Observation:" in all_text),
        "final_text_chars": len(final_text),
        "valid_program": task_eval.get("valid_program"),
        "success_rate": task_eval.get("success_rate"),
        "codebert_score": task_eval.get("codebert_score"),
        "failure_mode": classify_failure(log_info, task_eval.get("valid_program"), row["correct"]),
        "log_info": log_info,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    response = pd.read_csv(ROOT / "irt_data" / "response_matrix.csv")
    agents = pd.read_csv(ROOT / "irt_data" / "agents.csv")[
        ["agent_id", "source_file", "date", "total_cost", "n_tasks", "n_correct"]
    ]
    harness = pd.read_csv(ROOT / "irt_data" / "harness_metadata.csv")
    model = pd.read_csv(ROOT / "irt_data" / "model_metadata.csv")

    df = (
        response.merge(agents, on="agent_id", how="left")
        .merge(harness, left_on="scaffold", right_on="raw_scaffold", how="left")
        .merge(model, left_on="model", right_on="raw_model", how="left")
    )

    candidates = mechanism_candidates(df)
    candidates.to_csv(OUT_DIR / "candidate_tasks.csv", index=False)

    target = df[
        (df["benchmark"] == TARGET_BENCHMARK)
        & (df["task_id"].astype(str) == TARGET_TASK_ID)
    ].copy()
    summaries = pd.DataFrame([summarize_run(row) for _, row in target.iterrows()])
    summaries.to_csv(OUT_DIR / "scienceagentbench_task3_trace_signals.csv", index=False)

    by_mechanism = (
        summaries.groupby("tool_exposure_mechanism")
        .agg(
            runs=("agent_id", "count"),
            successes=("correct", "sum"),
            pass_rate=("correct", "mean"),
            avg_llm_calls=("llm_call_count", "mean"),
            avg_length_finishes=("length_finish_count", "mean"),
            avg_output_code_blocks=("output_code_block_count", "mean"),
            avg_observation_mentions=("observation_mentions", "mean"),
            avg_error_mentions=("error_mentions", "mean"),
            tool_registry_prompt_rate=("tool_registry_prompt", "mean"),
            self_debug_prompt_rate=("self_debug_prompt", "mean"),
            exec_feedback_loop_rate=("exec_feedback_loop", "mean"),
            avg_final_chars=("final_text_chars", "mean"),
        )
        .reset_index()
    )
    by_mechanism.to_csv(OUT_DIR / "scienceagentbench_task3_by_mechanism.csv", index=False)

    failure_modes = (
        summaries.groupby(["tool_exposure_mechanism", "failure_mode"])
        .size()
        .reset_index(name="runs")
        .sort_values(["tool_exposure_mechanism", "runs"], ascending=[True, False])
    )
    failure_modes.to_csv(OUT_DIR / "scienceagentbench_task3_failure_modes.csv", index=False)
    failure_text = "\n".join(
        f"- `{row.tool_exposure_mechanism}` / `{row.failure_mode}`: {row.runs}"
        for row in failure_modes.itertuples(index=False)
    )

    chosen = candidates[
        (candidates["benchmark"] == TARGET_BENCHMARK)
        & (candidates["task_id"].astype(str) == TARGET_TASK_ID)
    ].iloc[0]
    readme = f"""# Mechanism-overlap trace probe

Selected task: `{TARGET_BENCHMARK}:{TARGET_TASK_ID}`.

Why this task: it has {int(chosen['n'])} runs across {int(chosen['mechanisms'])}
tool-exposure mechanisms and {int(chosen['models'])} normalized models. The
overall pass rate is {chosen['pass_rate']:.2f}. The mechanism split is:

- `code_agent_tool_registry`: {int(chosen['code_agent_tool_registry_successes'])}/{int(chosen['code_agent_tool_registry_n'])}
  success, pass rate {chosen['code_agent_tool_registry_pass_rate']:.2f}.
- `harness_side_augmentation`: {int(chosen['harness_side_augmentation_successes'])}/{int(chosen['harness_side_augmentation_n'])}
  success, pass rate {chosen['harness_side_augmentation_pass_rate']:.2f}.

Files:

- `candidate_tasks.csv`: all benchmark-task pairs with at least two
  tool-exposure mechanisms, sorted to surface code-registry versus
  harness-augmentation gaps.
- `scienceagentbench_task3_trace_signals.csv`: one row per run on this task,
  with extracted trace-level process signals.
- `scienceagentbench_task3_by_mechanism.csv`: aggregate trace signals by
  mechanism.
- `scienceagentbench_task3_failure_modes.csv`: evaluator-visible success and
  failure categories by mechanism.

Initial trace read:

- The code-agent registry traces all contain the smolagents-style tool prompt
  and an execution feedback loop (`Observation:`). They average
  {by_mechanism.loc[by_mechanism.tool_exposure_mechanism == 'code_agent_tool_registry', 'avg_llm_calls'].iloc[0]:.1f}
  LLM calls and
  {by_mechanism.loc[by_mechanism.tool_exposure_mechanism == 'code_agent_tool_registry', 'avg_observation_mentions'].iloc[0]:.1f}
  observation mentions.
- The harness-side augmentation traces contain the ScienceAgentBench
  complete-program/self-debug prompt. They average
  {by_mechanism.loc[by_mechanism.tool_exposure_mechanism == 'harness_side_augmentation', 'avg_llm_calls'].iloc[0]:.1f}
  LLM calls and no in-model `Observation:` loop.

Failure modes:

{failure_text}
"""
    (OUT_DIR / "README.md").write_text(readme)

    print(f"Wrote {OUT_DIR}")
    print(by_mechanism.to_string(index=False))


if __name__ == "__main__":
    main()
