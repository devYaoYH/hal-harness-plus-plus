"""
Analyze tool-call trajectories for one HAL task.

The default target is a mixed-success TauBench Airline task with enough runs to
make suffix patterns useful:

    python3 irt_data/trace_analysis/analyze_task_traces.py

Outputs are written beside this script by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from irt_data.prepare_irt_data import decrypt_upload_json  # noqa: E402


DEFAULT_BENCHMARK = "taubench_airline"
DEFAULT_TASK_ID = "20"
DEFAULT_MAX_SUFFIX = 8


@dataclass
class TraceRun:
    run_id: str
    scaffold: str
    model: str
    source_file: str
    correct: int
    reward: float | None
    task_instruction: str
    expected_actions: list[dict[str, Any]]
    raw_actions: list[dict[str, Any]]
    observed_actions: list[dict[str, Any]]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def normalize_task_id(value: Any) -> str:
    return str(value)


def action_name(action: dict[str, Any]) -> str:
    return str(action.get("name", "<unknown>"))


def action_signature(action: dict[str, Any]) -> str:
    """Return a compact action label that keeps high-value TauBench arguments."""
    name = action_name(action)
    kwargs = action.get("kwargs")
    if not isinstance(kwargs, dict):
        return name

    if name == "update_reservation_flights":
        flights = kwargs.get("flights")
        if isinstance(flights, list):
            numbers = [
                str(f.get("flight_number"))
                for f in flights
                if isinstance(f, dict) and f.get("flight_number")
            ]
            if numbers:
                return f"{name}({'+'.join(numbers)})"
    for key in ("reservation_id", "origin", "destination", "user_id"):
        if key in kwargs:
            return f"{name}({key}={kwargs[key]})"
    return name


def strip_trailing_oracle_actions(
    taken_actions: list[dict[str, Any]],
    expected_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """TauBench traces append the gold action list to taken_actions; remove it."""
    if not expected_actions or len(taken_actions) < len(expected_actions):
        return taken_actions

    tail = taken_actions[-len(expected_actions) :]
    if [action_name(a) for a in tail] != [action_name(a) for a in expected_actions]:
        return taken_actions

    # Names are enough for detection across traces, but keep this conservative:
    # exact kwargs equality is required where both sides have kwargs.
    for observed, expected in zip(tail, expected_actions):
        if "kwargs" in observed and "kwargs" in expected:
            if observed.get("kwargs") != expected.get("kwargs"):
                return taken_actions
    return taken_actions[: -len(expected_actions)]


def load_task_runs(benchmark: str, task_id: str) -> list[TraceRun]:
    response_rows = load_csv_rows(REPO_ROOT / "irt_data" / "response_matrix.csv")
    agent_rows = load_csv_rows(REPO_ROOT / "irt_data" / "agents.csv")
    agent_by_run = {row["run_id"]: row for row in agent_rows}

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
        source_file = agent["source_file"]
        trace_path = REPO_ROOT / "irt_data" / "traces" / source_file
        data = decrypt_upload_json(trace_path)
        if not data:
            continue

        result = data.get("raw_eval_results", {}).get(task_id)
        if not isinstance(result, dict):
            continue

        task = result.get("task", {})
        expected_actions = task.get("actions", [])
        raw_actions = result.get("taken_actions", [])
        if not isinstance(expected_actions, list):
            expected_actions = []
        if not isinstance(raw_actions, list):
            raw_actions = []

        observed_actions = strip_trailing_oracle_actions(raw_actions, expected_actions)
        reward = result.get("reward")
        runs.append(
            TraceRun(
                run_id=row["run_id"],
                scaffold=row["scaffold"],
                model=row["model"],
                source_file=source_file,
                correct=int(row["correct"]),
                reward=float(reward) if isinstance(reward, (int, float)) else None,
                task_instruction=str(task.get("instruction", "")),
                expected_actions=[
                    a for a in expected_actions if isinstance(a, dict)
                ],
                raw_actions=[a for a in raw_actions if isinstance(a, dict)],
                observed_actions=[
                    a for a in observed_actions if isinstance(a, dict)
                ],
            )
        )
    return runs


def contains_subsequence(sequence: list[str], pattern: list[str]) -> bool:
    if not pattern:
        return True
    pos = 0
    for item in sequence:
        if item == pattern[pos]:
            pos += 1
            if pos == len(pattern):
                return True
    return False


def update_flight_numbers(actions: list[dict[str, Any]]) -> list[str]:
    numbers: list[str] = []
    for action in actions:
        if action_name(action) != "update_reservation_flights":
            continue
        kwargs = action.get("kwargs")
        if not isinstance(kwargs, dict):
            continue
        for flight in kwargs.get("flights", []):
            if isinstance(flight, dict) and flight.get("flight_number"):
                numbers.append(str(flight["flight_number"]))
    return numbers


def first_update_signature(actions: list[dict[str, Any]]) -> str:
    for action in actions:
        if action_name(action) == "update_reservation_flights":
            return action_signature(action)
    return ""


def summarize_run(run: TraceRun) -> dict[str, Any]:
    names = [action_name(action) for action in run.observed_actions]
    signatures = [action_signature(action) for action in run.observed_actions]
    expected_names = [action_name(action) for action in run.expected_actions]
    expected_sigs = [action_signature(action) for action in run.expected_actions]
    updates = update_flight_numbers(run.observed_actions)
    expected_updates = set(update_flight_numbers(run.expected_actions))
    used_expected_update = bool(expected_updates.intersection(updates))
    used_non_expected_update = bool(set(updates) - expected_updates)
    expected_update_signatures = {
        action_signature(action)
        for action in run.expected_actions
        if action_name(action) == "update_reservation_flights"
    }
    update_signatures = [
        action_signature(action)
        for action in run.observed_actions
        if action_name(action) == "update_reservation_flights"
    ]

    return {
        "run_id": run.run_id,
        "scaffold": run.scaffold,
        "model": run.model,
        "source_file": run.source_file,
        "correct": run.correct,
        "reward": run.reward,
        "observed_n_actions": len(run.observed_actions),
        "raw_n_actions": len(run.raw_actions),
        "oracle_suffix_stripped": int(len(run.raw_actions) != len(run.observed_actions)),
        "n_respond": names.count("respond"),
        "n_search_direct_flight": names.count("search_direct_flight"),
        "n_update_reservation_flights": names.count("update_reservation_flights"),
        "has_update_reservation_baggages": int("update_reservation_baggages" in names),
        "has_send_certificate": int("send_certificate" in names),
        "has_transfer_to_human_agents": int("transfer_to_human_agents" in names),
        "final_action": names[-1] if names else "",
        "final_signature": signatures[-1] if signatures else "",
        "contains_expected_action_subsequence": int(
            contains_subsequence(names, expected_names)
        ),
        "contains_expected_signature_subsequence": int(
            contains_subsequence(signatures, expected_sigs)
        ),
        "used_expected_update_flight": int(used_expected_update),
        "used_non_expected_update_flight": int(used_non_expected_update),
        "first_update_is_expected": int(
            first_update_signature(run.observed_actions) in expected_update_signatures
        ),
        "all_updates_are_expected": int(
            bool(update_signatures)
            and all(sig in expected_update_signatures for sig in update_signatures)
        ),
        "update_flight_numbers": "|".join(updates),
        "sequence": " -> ".join(names),
        "signature_sequence": " -> ".join(signatures),
    }


def suffix_records(
    runs: list[TraceRun],
    max_suffix: int,
    use_signatures: bool,
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, ...], Counter[int]] = defaultdict(Counter)
    for run in runs:
        sequence = [
            action_signature(action) if use_signatures else action_name(action)
            for action in run.observed_actions
        ]
        seen: set[tuple[str, ...]] = set()
        for size in range(1, min(max_suffix, len(sequence)) + 1):
            suffix = tuple(sequence[-size:])
            seen.add(suffix)
        for suffix in seen:
            counts[suffix][run.correct] += 1

    total_success = sum(run.correct for run in runs)
    total_failure = len(runs) - total_success
    records: list[dict[str, Any]] = []
    for suffix, counter in counts.items():
        successes = counter[1]
        failures = counter[0]
        support = successes + failures
        if support < 2:
            continue
        success_rate = successes / support
        # Haldane-Anscombe smoothing keeps the score finite for pure suffixes.
        odds_in = (successes + 0.5) / (failures + 0.5)
        odds_out = (total_success - successes + 0.5) / (
            total_failure - failures + 0.5
        )
        records.append(
            {
                "suffix": " -> ".join(suffix),
                "length": len(suffix),
                "support": support,
                "successes": successes,
                "failures": failures,
                "success_rate": round(success_rate, 4),
                "log_odds_lift": round(math.log(odds_in / odds_out), 4),
            }
        )
    return sorted(
        records,
        key=lambda r: (abs(r["log_odds_lift"]), r["support"], r["length"]),
        reverse=True,
    )


def feature_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = [
        "contains_expected_action_subsequence",
        "contains_expected_signature_subsequence",
        "used_expected_update_flight",
        "used_non_expected_update_flight",
        "first_update_is_expected",
        "all_updates_are_expected",
        "has_update_reservation_baggages",
        "has_send_certificate",
        "has_transfer_to_human_agents",
        "n_search_direct_flight",
        "n_update_reservation_flights",
        "final_action",
        "final_signature",
    ]
    records: list[dict[str, Any]] = []
    for feature in features:
        buckets: dict[str, Counter[int]] = defaultdict(Counter)
        for row in rows:
            value = str(row[feature])
            if feature in {"n_update_reservation_flights", "n_search_direct_flight"}:
                value = "2+" if int(row[feature]) >= 2 else value
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
    rows: list[dict[str, Any]],
    suffixes: list[dict[str, Any]],
    signature_suffixes: list[dict[str, Any]],
    features: list[dict[str, Any]],
) -> None:
    successes = sum(run.correct for run in runs)
    failures = len(runs) - successes
    instruction = runs[0].task_instruction if runs else ""
    expected = [action_signature(a) for a in runs[0].expected_actions] if runs else []

    lines = [
        f"# Trace analysis: {benchmark} task {task_id}",
        "",
        f"Runs analyzed: {len(runs)} ({successes} success, {failures} failure).",
        "",
        "## Task",
        "",
        instruction,
        "",
        "Expected action signatures:",
        "",
        " -> ".join(expected),
        "",
        "## Strongest suffix separators",
        "",
        "| suffix | support | success rate | log-odds lift |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in suffixes[:12]:
        lines.append(
            f"| `{row['suffix']}` | {row['support']} | "
            f"{row['success_rate']:.2f} | {row['log_odds_lift']:.2f} |"
        )

    lines += [
        "",
        "## Strongest signature suffix separators",
        "",
        "| suffix | support | success rate | log-odds lift |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in signature_suffixes[:12]:
        lines.append(
            f"| `{row['suffix']}` | {row['support']} | "
            f"{row['success_rate']:.2f} | {row['log_odds_lift']:.2f} |"
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
        "- `suffix_patterns.csv`: suffix trie counts over action names.",
        "- `signature_suffix_patterns.csv`: suffix trie counts over action signatures.",
        "- `feature_cuts.csv`: coarse success/failure cuts for interpretable factors.",
        "- `selected_task.json`: target task metadata.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
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
        "instruction": runs[0].task_instruction if runs else "",
        "expected_actions": runs[0].expected_actions if runs else [],
    }
    (args.out_dir / "selected_task.json").write_text(
        json.dumps(selected_task, indent=2) + "\n"
    )
    write_report(
        args.out_dir / "README.md",
        args.benchmark,
        normalize_task_id(args.task_id),
        runs,
        rows,
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
