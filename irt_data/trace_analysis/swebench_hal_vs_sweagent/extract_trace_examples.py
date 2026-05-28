"""Extract matched SWE-bench trace examples for appendix reporting.

The examples focus on matched GPT-5 runs where HAL Generalist failed and
SWE-Agent succeeded on the same SWE-bench Verified Mini task.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from irt_data.prepare_irt_data import decrypt_upload_json


TRACE_DIR = ROOT / "irt_data" / "traces"
OUT_DIR = ROOT / "irt_data" / "trace_analysis" / "swebench_hal_vs_sweagent"

HAL_RUN_ID = "swebench_verified_mini_hal_generalist_gpt520250807_1755463923"
SWE_RUN_ID = "swebench_verified_mini_sweagentgpt520250807_1754592641"
HAL_ZIP = TRACE_DIR / f"{HAL_RUN_ID}_UPLOAD.zip"
SWE_ZIP = TRACE_DIR / f"{SWE_RUN_ID}_UPLOAD.zip"

EXAMPLE_TASKS = [
    "django__django-12143",
    "django__django-12050",
    "sphinx-doc__sphinx-9698",
]

HAL_TOOL_NAMES = [
    "file_content_search",
    "edit_file",
    "execute_bash",
    "inspect_file_as_text",
    "python_interpreter",
    "final_answer",
]
SWE_TOOL_NAMES = ["bash", "str_replace_editor", "filemap", "submit"]


def iter_task_llm_calls(data: dict, task_id: str) -> list[dict]:
    calls = []
    for call in data.get("raw_logging_results", []):
        if (call.get("attributes") or {}).get("weave_task_id") != task_id:
            continue
        op_name = call.get("op_name", "")
        if "chat.completions" not in op_name:
            continue
        calls.append(call)
    return sorted(calls, key=lambda call: call.get("started_at") or "")


def message_text(call: dict) -> str:
    choices = (call.get("output") or {}).get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                chunks.append(str(item.get("text") or ""))
            else:
                chunks.append(str(item))
        content = "\n".join(chunks)
    return str(content)


def user_prompt(call: dict) -> str:
    for message in (call.get("inputs") or {}).get("messages") or []:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text") or item) if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content or "")
    return ""


def extract_pr_title(prompt: str) -> str:
    match = re.search(r"<pr_description>\s*(.*?)\n", prompt, flags=re.S)
    if not match:
        return ""
    return " ".join(match.group(1).strip().split())


def tool_counts(calls: list[dict], names: list[str]) -> Counter:
    counts: Counter = Counter()
    for call in calls:
        choices = (call.get("output") or {}).get("choices") or []
        message = choices[0].get("message") if choices else {}
        for tool_call in (message or {}).get("tool_calls") or []:
            function = tool_call.get("function") or {}
            name = function.get("name")
            if name:
                counts[name] += 1
        text = message_text(call)
        for name in names:
            counts[name] += text.count(name)
    return counts


def final_excerpt(calls: list[dict], max_chars: int = 220) -> str:
    for call in reversed(calls):
        text = " ".join(message_text(call).split())
        if text:
            return text[:max_chars]
    return ""


def trace_row(data: dict, task_id: str, harness: str, run_id: str, zip_path: Path) -> dict:
    calls = iter_task_llm_calls(data, task_id)
    prompt = user_prompt(calls[0]) if calls else ""
    tools = tool_counts(calls, HAL_TOOL_NAMES if harness == "HAL Generalist" else SWE_TOOL_NAMES)
    return {
        "task_id": task_id,
        "pr_description_title": extract_pr_title(prompt),
        "harness": harness,
        "run_id": run_id,
        "source_file": zip_path.name,
        "correct": int(task_id in data.get("results", {}).get("successful_tasks", [])),
        "llm_calls": len(calls),
        "trace_ids": ";".join(sorted({call.get("trace_id", "") for call in calls if call.get("trace_id")})),
        "tool_signal": "; ".join(f"{name}={count}" for name, count in sorted(tools.items()) if count),
        "final_message_excerpt": final_excerpt(calls),
    }


def main() -> None:
    responses = pd.read_csv(ROOT / "irt_data" / "response_matrix.csv")
    hal = responses[responses.run_id.eq(HAL_RUN_ID)][["task_id", "correct"]].rename(
        columns={"correct": "hal_correct"}
    )
    swe = responses[responses.run_id.eq(SWE_RUN_ID)][["task_id", "correct"]].rename(
        columns={"correct": "swe_correct"}
    )
    matched = hal.merge(swe, on="task_id")
    failures = matched[(matched.hal_correct == 0) & (matched.swe_correct == 1)].copy()
    failures.to_csv(OUT_DIR / "gpt5_hal_fail_swe_success_tasks.csv", index=False)

    hal_data = decrypt_upload_json(HAL_ZIP)
    swe_data = decrypt_upload_json(SWE_ZIP)
    rows = []
    for task_id in EXAMPLE_TASKS:
        rows.append(trace_row(hal_data, task_id, "HAL Generalist", HAL_RUN_ID, HAL_ZIP))
        rows.append(trace_row(swe_data, task_id, "SWE-Agent", SWE_RUN_ID, SWE_ZIP))

    with (OUT_DIR / "gpt5_trace_examples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "benchmark": "swebench_verified_mini",
        "normalized_model": "GPT-5",
        "hal_run_id": HAL_RUN_ID,
        "swe_run_id": SWE_RUN_ID,
        "matched_tasks": int(len(matched)),
        "hal_fail_swe_success_tasks": int(len(failures)),
        "example_tasks": EXAMPLE_TASKS,
    }
    (OUT_DIR / "README.md").write_text(
        "# SWE-bench HAL Generalist vs SWE-Agent trace examples\n\n"
        "Matched GPT-5 runs on SWE-bench Verified Mini. The main candidate set is "
        "`gpt5_hal_fail_swe_success_tasks.csv`; the appendix examples are "
        "`gpt5_trace_examples.csv`.\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
