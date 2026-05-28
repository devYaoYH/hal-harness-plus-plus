#!/usr/bin/env python3
"""Analyze process trajectories for one ScienceAgentBench task."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from irt_data.prepare_irt_data import decrypt_upload_json  # noqa: E402


DEFAULT_BENCHMARK = "scienceagentbench"
DEFAULT_TASK_ID = "3"
DEFAULT_MAX_SUFFIX = 8


@dataclass
class TraceRun:
    run_id: str
    scaffold: str
    normalized_harness: str
    tool_exposure_mechanism: str
    model: str
    normalized_model: str
    source_file: str
    correct: int
    eval_result: dict[str, Any]
    agent_output: Any
    logs: list[dict[str, Any]]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def normalize_task_id(value: Any) -> str:
    return str(value)


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


def input_text(log: dict[str, Any]) -> str:
    return flatten_text((log.get("inputs") or {}).get("messages") or [])


def output_text(log: dict[str, Any]) -> str:
    choices = ((log.get("output") or {}).get("choices") or [])
    if not choices:
        return ""
    return flatten_text((choices[0].get("message") or {}).get("content"))


def finish_reason(log: dict[str, Any]) -> str:
    choices = ((log.get("output") or {}).get("choices") or [])
    if not choices:
        return ""
    return str(choices[0].get("finish_reason") or "")


def final_history_text(agent_output: Any) -> str:
    if isinstance(agent_output, dict) and agent_output.get("history"):
        return flatten_text(agent_output["history"])
    return flatten_text(agent_output)


def task_logs(data: dict[str, Any], task_id: str) -> list[dict[str, Any]]:
    logs = data.get("raw_logging_results") or []
    selected = [
        log
        for log in logs
        if str((log.get("attributes") or {}).get("weave_task_id")) == task_id
    ]
    return sorted(selected, key=lambda log: log.get("started_at") or "")


def classify_event(log: dict[str, Any], index: int) -> str:
    inp = input_text(log).lower()
    out = output_text(log)
    out_low = out.lower()
    finish = finish_reason(log)

    if "given access to a list of tools" in inp:
        prompt_kind = "registry"
    elif "reported issues" in inp or "complete program" in inp:
        prompt_kind = "self_debug"
    else:
        prompt_kind = "plain"

    if finish == "length":
        result = "truncated"
    elif out_low.lstrip().startswith("```python"):
        result = "direct_code"
    elif "thought:" in out_low and "code:" in out_low:
        result = "thought_code"
    elif "facts survey" in out_low or "plan" in out_low[:500]:
        result = "plan"
    elif "traceback" in out_low or "error" in out_low or "exception" in out_low:
        result = "error_response"
    elif "```python" in out_low or "```py" in out_low:
        result = "code_block"
    else:
        result = "text_response"

    if index > 0 and "observation:" in inp:
        return f"{prompt_kind}:observation_to_{result}"
    return f"{prompt_kind}:{result}"


def event_signature(log: dict[str, Any], index: int) -> str:
    event = classify_event(log, index)
    finish = finish_reason(log)
    code_blocks = len(re.findall(r"```(?:python|py)?", output_text(log)))
    if finish == "length":
        return f"{event}(finish=length)"
    if code_blocks:
        return f"{event}(code_blocks={code_blocks})"
    return event


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


def load_task_runs(benchmark: str, task_id: str) -> list[TraceRun]:
    response_rows = load_csv_rows(REPO_ROOT / "irt_data" / "response_matrix.csv")
    agent_rows = load_csv_rows(REPO_ROOT / "irt_data" / "agents.csv")
    harness_rows = load_csv_rows(REPO_ROOT / "irt_data" / "harness_metadata.csv")
    model_rows = load_csv_rows(REPO_ROOT / "irt_data" / "model_metadata.csv")
    agent_by_run = {row["run_id"]: row for row in agent_rows}
    harness_by_raw = {row["raw_scaffold"]: row for row in harness_rows}
    model_by_raw = {row["raw_model"]: row for row in model_rows}

    selected = [
        row
        for row in response_rows
        if row["benchmark"] == benchmark and normalize_task_id(row["task_id"]) == task_id
    ]
    if not selected:
        raise SystemExit(f"No rows found for benchmark={benchmark!r}, task_id={task_id!r}")

    runs: list[TraceRun] = []
    for row in selected:
        agent = agent_by_run[row["run_id"]]
        data = decrypt_upload_json(REPO_ROOT / "irt_data" / "traces" / agent["source_file"])
        if not data:
            continue
        raw_eval = data.get("raw_eval_results") or {}
        eval_result = (raw_eval.get("eval_result") or {}).get(task_id) or {}
        agent_output = (raw_eval.get("agent_output") or {}).get(task_id)
        harness = harness_by_raw[row["scaffold"]]
        model = model_by_raw[row["model"]]
        runs.append(
            TraceRun(
                run_id=row["run_id"],
                scaffold=row["scaffold"],
                normalized_harness=harness["normalized_harness"],
                tool_exposure_mechanism=harness["tool_exposure_mechanism"],
                model=row["model"],
                normalized_model=model["normalized_model"],
                source_file=agent["source_file"],
                correct=int(row["correct"]),
                eval_result=eval_result if isinstance(eval_result, dict) else {},
                agent_output=agent_output,
                logs=task_logs(data, task_id),
            )
        )
    return runs


def event_sequence(run: TraceRun, use_signatures: bool) -> list[str]:
    sequence = [
        event_signature(log, index) if use_signatures else classify_event(log, index)
        for index, log in enumerate(run.logs)
    ]
    if not sequence:
        sequence.append("no_llm_log")
    return sequence


def summarize_run(run: TraceRun) -> dict[str, Any]:
    sequence = event_sequence(run, use_signatures=False)
    signature_sequence = event_sequence(run, use_signatures=True)
    inputs = "\n".join(input_text(log) for log in run.logs)
    outputs = "\n".join(output_text(log) for log in run.logs)
    final_text = final_history_text(run.agent_output)
    all_text = "\n".join([inputs, outputs, final_text])
    finish_counts = Counter(finish_reason(log) for log in run.logs if finish_reason(log))
    log_info = run.eval_result.get("log_info")

    return {
        "run_id": run.run_id,
        "scaffold": run.scaffold,
        "normalized_harness": run.normalized_harness,
        "tool_exposure_mechanism": run.tool_exposure_mechanism,
        "model": run.model,
        "normalized_model": run.normalized_model,
        "source_file": run.source_file,
        "correct": run.correct,
        "valid_program": run.eval_result.get("valid_program"),
        "success_rate": run.eval_result.get("success_rate"),
        "codebert_score": run.eval_result.get("codebert_score"),
        "failure_mode": classify_failure(log_info, run.eval_result.get("valid_program"), run.correct),
        "llm_call_count": len(run.logs),
        "length_finish_count": finish_counts.get("length", 0),
        "stop_finish_count": finish_counts.get("stop", 0),
        "output_code_block_count": len(re.findall(r"```(?:python|py)?", outputs)),
        "final_code_block_count": len(re.findall(r"```(?:python|py)?", final_text)),
        "observation_mentions": len(re.findall(r"\bObservation:", all_text)),
        "error_mentions": len(re.findall(r"Traceback|ImportError|Error:|exception", all_text, re.I)),
        "tool_registry_prompt": int("given access to a list of tools" in inputs),
        "self_debug_prompt": int("reported issues" in inputs or "error messages" in inputs),
        "complete_program_prompt": int("complete program" in inputs),
        "final_text_chars": len(final_text),
        "final_event": sequence[-1] if sequence else "",
        "final_signature": signature_sequence[-1] if signature_sequence else "",
        "sequence": " -> ".join(sequence),
        "signature_sequence": " -> ".join(signature_sequence),
        "log_info": log_info,
    }


def suffix_records(
    runs: list[TraceRun],
    max_suffix: int,
    use_signatures: bool,
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, ...], Counter[int]] = defaultdict(Counter)
    mechanisms: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for run in runs:
        sequence = event_sequence(run, use_signatures=use_signatures)
        seen: set[tuple[str, ...]] = set()
        for size in range(1, min(max_suffix, len(sequence)) + 1):
            seen.add(tuple(sequence[-size:]))
        for suffix in seen:
            counts[suffix][run.correct] += 1
            mechanisms[suffix][run.tool_exposure_mechanism] += 1

    total_success = sum(run.correct for run in runs)
    total_failure = len(runs) - total_success
    records: list[dict[str, Any]] = []
    for suffix, counter in counts.items():
        successes = counter[1]
        failures = counter[0]
        support = successes + failures
        if support < 2:
            continue
        odds_in = (successes + 0.5) / (failures + 0.5)
        odds_out = (total_success - successes + 0.5) / (total_failure - failures + 0.5)
        records.append(
            {
                "suffix": " -> ".join(suffix),
                "length": len(suffix),
                "support": support,
                "successes": successes,
                "failures": failures,
                "success_rate": round(successes / support, 4),
                "log_odds_lift": round(math.log(odds_in / odds_out), 4),
                "mechanisms": "; ".join(f"{k}:{v}" for k, v in sorted(mechanisms[suffix].items())),
            }
        )
    return sorted(
        records,
        key=lambda r: (abs(r["log_odds_lift"]), r["support"], r["length"]),
        reverse=True,
    )


def feature_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = [
        "tool_exposure_mechanism",
        "normalized_harness",
        "failure_mode",
        "tool_registry_prompt",
        "self_debug_prompt",
        "complete_program_prompt",
        "length_finish_count",
        "llm_call_count",
        "final_event",
        "final_signature",
    ]
    records: list[dict[str, Any]] = []
    for feature in features:
        buckets: dict[str, Counter[int]] = defaultdict(Counter)
        for row in rows:
            value = str(row[feature])
            if feature == "llm_call_count":
                calls = int(row[feature])
                value = "1" if calls == 1 else "2-4" if calls <= 4 else "5-12" if calls <= 12 else "13+"
            if feature == "length_finish_count":
                value = "1+" if int(row[feature]) >= 1 else "0"
            buckets[value][int(row["correct"])] += 1
        for value, counter in buckets.items():
            support = counter[0] + counter[1]
            if support < 2:
                continue
            records.append(
                {
                    "feature": feature,
                    "value": value,
                    "support": support,
                    "successes": counter[1],
                    "failures": counter[0],
                    "success_rate": round(counter[1] / support, 4),
                }
            )
    return sorted(records, key=lambda r: (r["feature"], -r["support"], r["value"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    benchmark: str,
    task_id: str,
    runs: list[TraceRun],
    suffixes: list[dict[str, Any]],
    signature_suffixes: list[dict[str, Any]],
    features: list[dict[str, Any]],
) -> None:
    successes = sum(run.correct for run in runs)
    failures = len(runs) - successes
    by_mechanism = Counter(run.tool_exposure_mechanism for run in runs)
    success_by_mechanism: dict[str, int] = defaultdict(int)
    for run in runs:
        success_by_mechanism[run.tool_exposure_mechanism] += run.correct

    lines = [
        f"# Trace analysis: {benchmark} task {task_id}",
        "",
        f"Runs analyzed: {len(runs)} ({successes} success, {failures} failure).",
        "",
        "Mechanism split:",
        "",
    ]
    for mechanism, count in sorted(by_mechanism.items()):
        succ = success_by_mechanism[mechanism]
        lines.append(f"- `{mechanism}`: {succ}/{count} success")

    lines += [
        "",
        "## Strongest suffix separators",
        "",
        "| suffix | support | success rate | log-odds lift | mechanisms |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in suffixes[:12]:
        lines.append(
            f"| `{row['suffix']}` | {row['support']} | "
            f"{row['success_rate']:.2f} | {row['log_odds_lift']:.2f} | "
            f"{row['mechanisms']} |"
        )

    lines += [
        "",
        "## Strongest signature suffix separators",
        "",
        "| suffix | support | success rate | log-odds lift | mechanisms |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in signature_suffixes[:12]:
        lines.append(
            f"| `{row['suffix']}` | {row['support']} | "
            f"{row['success_rate']:.2f} | {row['log_odds_lift']:.2f} | "
            f"{row['mechanisms']} |"
        )

    lines += [
        "",
        "## Feature cuts",
        "",
        "| feature | value | support | success rate | successes | failures |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    interesting = sorted(
        features,
        key=lambda r: (abs(r["success_rate"] - 0.5), r["support"]),
        reverse=True,
    )
    for row in interesting[:18]:
        lines.append(
            f"| `{row['feature']}` | `{row['value']}` | {row['support']} | "
            f"{row['success_rate']:.2f} | {row['successes']} | {row['failures']} |"
        )

    lines += [
        "",
        "## Files",
        "",
        "- `task_trajectory_summary.csv`: one row per trace/run.",
        "- `suffix_patterns.csv`: suffix-trie counts over coarse process events.",
        "- `signature_suffix_patterns.csv`: suffix-trie counts over process events with code-block/finish annotations.",
        "- `feature_cuts.csv`: coarse success/failure cuts for interpretable factors.",
        "- `selected_task.json`: target task metadata.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "scienceagentbench_task3",
    )
    parser.add_argument("--max-suffix", type=int, default=DEFAULT_MAX_SUFFIX)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_task_runs(args.benchmark, normalize_task_id(args.task_id))
    rows = [summarize_run(run) for run in runs]
    suffixes = suffix_records(runs, args.max_suffix, use_signatures=False)
    signature_suffixes = suffix_records(runs, args.max_suffix, use_signatures=True)
    features = feature_records(rows)

    write_csv(args.out_dir / "task_trajectory_summary.csv", rows)
    write_csv(args.out_dir / "suffix_patterns.csv", suffixes)
    write_csv(args.out_dir / "signature_suffix_patterns.csv", signature_suffixes)
    write_csv(args.out_dir / "feature_cuts.csv", features)
    selected_task = {
        "benchmark": args.benchmark,
        "task_id": normalize_task_id(args.task_id),
        "n_runs": len(runs),
        "n_success": sum(run.correct for run in runs),
        "n_failure": sum(1 - run.correct for run in runs),
        "mechanisms": sorted({run.tool_exposure_mechanism for run in runs}),
    }
    (args.out_dir / "selected_task.json").write_text(json.dumps(selected_task, indent=2) + "\n")
    write_report(
        args.out_dir / "README.md",
        args.benchmark,
        normalize_task_id(args.task_id),
        runs,
        suffixes,
        signature_suffixes,
        features,
    )
    print(
        f"Analyzed {len(runs)} runs for {args.benchmark} task {args.task_id}; "
        f"wrote outputs to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
