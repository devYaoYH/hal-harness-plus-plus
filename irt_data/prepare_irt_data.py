"""
Download, decrypt, and extract task-level results from HAL traces into
IRT-ready data structures.

Two-stage pipeline:
  Stage 1 (extract): Per-trace extraction into irt_data/extracted/<trace>.csv
                      Incremental — skips already-extracted traces. Parallel.
  Stage 2 (combine): Merges extracted CSVs into final outputs.

Outputs:
  irt_data/response_matrix.csv      — (agent_id, task_id, correct, benchmark, ...)
  irt_data/agents.csv               — agent metadata (name, model, scaffold, cost, ...)
  irt_data/tasks.csv                — task metadata (benchmark, task_id, pass_rate, ...)

Usage:
  python irt_data/prepare_irt_data.py                    # full download + extract + combine
  python irt_data/prepare_irt_data.py --skip-download    # reuse already-downloaded traces
  python irt_data/prepare_irt_data.py --combine-only     # skip extraction, just combine
  python irt_data/prepare_irt_data.py --benchmarks swebench_verified_mini taubench_airline
  python irt_data/prepare_irt_data.py --workers 8        # parallel extraction (default: 4)
"""

import argparse
import base64
import json
import logging
import os
import re
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hal.utils.json_encryption import JsonEncryption

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ENCRYPTION_PASSWORD = "hal1234"
HF_REPO = "agent-evals/hal_traces"
TRACES_DIR = Path(__file__).resolve().parent / "traces"
EXTRACTED_DIR = Path(__file__).resolve().parent / "extracted"
OUTPUT_DIR = Path(__file__).resolve().parent


def list_remote_files():
    from huggingface_hub import HfApi

    api = HfApi()
    return [
        f
        for f in api.list_repo_files(HF_REPO, repo_type="dataset")
        if f.endswith("_UPLOAD.zip")
    ]


def download_traces(files: list[str]):
    from huggingface_hub import hf_hub_download

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    for i, fname in enumerate(files):
        dest = TRACES_DIR / fname
        if dest.exists():
            continue
        logger.info(f"[{i+1}/{len(files)}] Downloading {fname}")
        try:
            hf_hub_download(
                HF_REPO, fname, repo_type="dataset", local_dir=str(TRACES_DIR)
            )
        except Exception as e:
            logger.warning(f"Failed to download {fname}: {e}")


def decrypt_upload_json(zip_path: Path) -> dict | None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if "_UPLOAD" in name and name.endswith(".encrypted"):
                    with zf.open(name) as f:
                        enc = json.load(f)
                    salt_bytes = base64.b64decode(enc["salt"].encode("utf-8"))
                    cipher = JsonEncryption(ENCRYPTION_PASSWORD, salt=salt_bytes)
                    encrypted_bytes = base64.b64decode(
                        enc["encrypted_data"].encode("utf-8")
                    )
                    decrypted = cipher.cipher.decrypt(encrypted_bytes)
                    return json.loads(decrypted.decode("utf-8"))
    except Exception as e:
        logger.warning(f"Failed to decrypt {zip_path.name}: {e}")
    return None


# --------------------------------------------------------------------------- #
#  Per-benchmark extractors: each returns list of (task_id, correct: 0|1)
# --------------------------------------------------------------------------- #

def _extract_from_successful_failed(data: dict) -> list[tuple[str, int]]:
    """Works for benchmarks that store successful_tasks/failed_tasks lists."""
    results = data.get("results", {})
    rows = []
    for tid in results.get("successful_tasks", []):
        rows.append((str(tid), 1))
    for tid in results.get("failed_tasks", []):
        rows.append((str(tid), 0))
    return rows


def _extract_taubench(data: dict) -> list[tuple[str, int]]:
    raw = data.get("raw_eval_results", {})
    rows = []
    for tid, val in raw.items():
        if isinstance(val, dict) and "reward" in val:
            rows.append((str(tid), 1 if val["reward"] > 0 else 0))
    return rows


def _extract_gaia(data: dict) -> list[tuple[str, int]]:
    raw = data.get("raw_eval_results", {})
    rows = []
    for tid, val in raw.items():
        if isinstance(val, dict) and "score" in val:
            rows.append((str(tid), 1 if int(val["score"]) > 0 else 0))
    return rows


def _extract_usaco(data: dict) -> list[tuple[str, int]]:
    raw = data.get("raw_eval_results", {})
    sdict = raw.get("sdict", {})
    if sdict:
        rows = []
        for tid, val in sdict.items():
            passed = val.get("passed", 0) if isinstance(val, dict) else 0
            total = val.get("total", 1) if isinstance(val, dict) else 1
            rows.append((str(tid), 1 if passed == total and total > 0 else 0))
        return rows
    return _extract_from_successful_failed(data)


def _extract_corebench(data: dict) -> list[tuple[str, int]]:
    results = data.get("results", {})
    successful = set(str(t) for t in results.get("successful_tasks", []))
    failed = set(str(t) for t in results.get("failed_tasks", []))
    if successful or failed:
        rows = [(tid, 1) for tid in successful] + [(tid, 0) for tid in failed]
        return rows
    raw = data.get("raw_eval_results", {})
    rows = []
    for tid, val in raw.items():
        if isinstance(val, dict):
            score = val.get("score", val.get("correct", 0))
            rows.append((str(tid), 1 if score > 0 else 0))
    return rows


def _extract_scienceagentbench(data: dict) -> list[tuple[str, int]]:
    raw = data.get("raw_eval_results", {})
    eval_result = raw.get("eval_result", {})
    if eval_result:
        rows = []
        for tid, val in eval_result.items():
            if isinstance(val, dict):
                score = val.get("score", val.get("success_rate", val.get("success", 0)))
                rows.append((str(tid), 1 if score > 0 else 0))
            elif isinstance(val, (int, float)):
                rows.append((str(tid), 1 if val > 0 else 0))
        if rows:
            return rows
    return _extract_from_successful_failed(data)


def _extract_scicode(data: dict) -> list[tuple[str, int]]:
    return _extract_from_successful_failed(data)


def _extract_assistantbench(data: dict) -> list[tuple[str, int]]:
    return _extract_from_successful_failed(data)


def _extract_mind2web(data: dict) -> list[tuple[str, int]]:
    return _extract_from_successful_failed(data)


EXTRACTORS = {
    "swebench_verified_mini": _extract_from_successful_failed,
    "swebench_verified": _extract_from_successful_failed,
    "taubench_airline": _extract_taubench,
    "taubench_retail": _extract_taubench,
    "gaia": _extract_gaia,
    "usaco": _extract_usaco,
    "corebench_hard": _extract_corebench,
    "corebench_medium": _extract_corebench,
    "corebench_easy": _extract_corebench,
    "scienceagentbench": _extract_scienceagentbench,
    "scicode": _extract_scicode,
    "scicode_hard": _extract_scicode,
    "assistantbench": _extract_assistantbench,
    "online_mind2web": _extract_mind2web,
}


def _detect_benchmark(data: dict, filename: str) -> str | None:
    bench = data.get("config", {}).get("benchmark_name", "")
    if bench:
        return bench
    for key in EXTRACTORS:
        if key in filename.lower():
            return key
    return None


def _parse_agent_model(data: dict) -> tuple[str, str]:
    config = data.get("config", {})
    agent_name = config.get("agent_name", "unknown")
    agent_args = config.get("agent_args", {})

    model = "unknown"
    for key in ["model", "agent.model.name", "model_name"]:
        if key in agent_args:
            model = agent_args[key]
            break

    # Try to extract model from agent_name if not in args
    if model == "unknown" and "(" in agent_name and ")" in agent_name:
        model = agent_name.split("(")[-1].rstrip(")")

    # Extract scaffold name (agent name without model info)
    scaffold = re.sub(r"\s*\(.*?\)\s*$", "", agent_name).strip()

    return scaffold, model


def _extract_single_trace(zip_path: Path) -> str | None:
    """Extract one trace to irt_data/extracted/<stem>.csv. Returns stem on success."""
    out_csv = EXTRACTED_DIR / f"{zip_path.stem}.csv"
    if out_csv.exists():
        return zip_path.stem

    data = decrypt_upload_json(zip_path)
    if data is None:
        return None

    benchmark = _detect_benchmark(data, zip_path.name)
    if benchmark is None:
        return None

    extractor = EXTRACTORS.get(benchmark)
    if extractor is None:
        for key, ext in EXTRACTORS.items():
            if benchmark.startswith(key) or key.startswith(benchmark):
                extractor = ext
                break
    if extractor is None:
        extractor = _extract_from_successful_failed

    task_results = extractor(data)
    if not task_results:
        return None

    scaffold, model = _parse_agent_model(data)
    run_id = data.get("config", {}).get("run_id", zip_path.stem)
    agent_id = f"{scaffold}__{model}__{run_id}"
    date = data.get("config", {}).get("date", "")
    total_cost = data.get("total_cost", data.get("results", {}).get("total_cost"))
    n_correct = sum(c for _, c in task_results)
    n_tasks = len(task_results)

    rows = []
    for task_id, correct in task_results:
        rows.append(
            {
                "agent_id": agent_id,
                "scaffold": scaffold,
                "model": model,
                "benchmark": benchmark,
                "task_id": task_id,
                "correct": correct,
                "run_id": run_id,
                "date": date,
                "total_cost": total_cost,
                "n_tasks": n_tasks,
                "n_correct": n_correct,
                "source_file": zip_path.name,
            }
        )

    df = pd.DataFrame(rows)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(out_csv)
    return zip_path.stem


def extract_all(benchmark_filter: list[str] | None = None, workers: int = 4):
    """Stage 1: extract each trace to its own CSV, in parallel."""
    zip_files = sorted(TRACES_DIR.glob("*.zip"))
    if benchmark_filter:
        zip_files = [
            zp for zp in zip_files
            if any(b in zp.name.lower() for b in benchmark_filter)
        ]
    logger.info(f"Found {len(zip_files)} trace files to process")

    already = sum(1 for zp in zip_files if (EXTRACTED_DIR / f"{zp.stem}.csv").exists())
    to_process = len(zip_files) - already
    logger.info(f"Already extracted: {already}, remaining: {to_process}")

    if to_process == 0:
        return

    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_single_trace, zp): zp for zp in zip_files}
        for future in as_completed(futures):
            result = future.result()
            if result:
                done += 1
            else:
                failed += 1
            total_done = done + failed
            if total_done % 50 == 0:
                logger.info(f"Extracted {done}/{total_done} ({failed} failed)")

    logger.info(f"Extraction complete: {done} succeeded, {failed} failed")


def combine_extracted(benchmark_filter: list[str] | None = None):
    """Stage 2: combine per-trace CSVs into final outputs."""
    csv_files = sorted(EXTRACTED_DIR.glob("*.csv"))
    if benchmark_filter:
        csv_files = [
            f for f in csv_files
            if any(b in f.name.lower() for b in benchmark_filter)
        ]
    logger.info(f"Combining {len(csv_files)} extracted CSVs")

    if not csv_files:
        logger.error("No extracted CSVs found. Run extraction first.")
        return pd.DataFrame(), pd.DataFrame()

    dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(dfs, ignore_index=True)

    response_cols = ["agent_id", "scaffold", "model", "benchmark", "task_id", "correct", "run_id"]
    response_df = combined[response_cols]

    agent_df = (
        combined.groupby("agent_id")
        .first()
        .reset_index()[["agent_id", "scaffold", "model", "benchmark",
                        "n_tasks", "n_correct", "total_cost", "date", "run_id", "source_file"]]
    )
    agent_df["accuracy"] = agent_df["n_correct"] / agent_df["n_tasks"]

    return response_df, agent_df


def build_task_summary(response_df: pd.DataFrame) -> pd.DataFrame:
    if response_df.empty:
        return pd.DataFrame()

    task_stats = (
        response_df.groupby(["benchmark", "task_id"])
        .agg(
            n_agents=("correct", "count"),
            n_correct=("correct", "sum"),
            pass_rate=("correct", "mean"),
        )
        .reset_index()
    )
    task_stats["n_correct"] = task_stats["n_correct"].astype(int)
    return task_stats


def main():
    parser = argparse.ArgumentParser(description="Prepare HAL data for IRT modeling")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--combine-only", action="store_true", help="Skip extraction, just combine")
    parser.add_argument("--benchmarks", nargs="*", help="Filter to specific benchmarks")
    parser.add_argument("--workers", type=int, default=4, help="Parallel extraction workers")
    args = parser.parse_args()

    if not args.skip_download and not args.combine_only:
        logger.info("Listing files on HuggingFace...")
        remote_files = list_remote_files()
        logger.info(f"Found {len(remote_files)} trace files on HuggingFace")

        if args.benchmarks:
            remote_files = [
                f
                for f in remote_files
                if any(b in f.lower() for b in args.benchmarks)
            ]
            logger.info(
                f"Filtered to {len(remote_files)} files for benchmarks: {args.benchmarks}"
            )

        download_traces(remote_files)

    if not args.combine_only:
        logger.info("Stage 1: Extracting task-level results from traces...")
        extract_all(args.benchmarks, workers=args.workers)

    logger.info("Stage 2: Combining extracted CSVs...")
    response_df, agents_df = combine_extracted(args.benchmarks)

    if response_df.empty:
        logger.error("No data extracted. Check trace files.")
        return

    tasks_df = build_task_summary(response_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    response_df.to_csv(OUTPUT_DIR / "response_matrix.csv", index=False)
    agents_df.to_csv(OUTPUT_DIR / "agents.csv", index=False)
    tasks_df.to_csv(OUTPUT_DIR / "tasks.csv", index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Response matrix: {len(response_df)} rows")
    logger.info(f"  Agents: {response_df['agent_id'].nunique()}")
    logger.info(f"  Tasks: {response_df['task_id'].nunique()}")
    logger.info(f"  Benchmarks: {response_df['benchmark'].nunique()}")
    logger.info(f"\nPer-benchmark summary:")
    for bench, grp in response_df.groupby("benchmark"):
        n_agents = grp["agent_id"].nunique()
        n_tasks = grp["task_id"].nunique()
        mean_acc = grp.groupby("agent_id")["correct"].mean().mean()
        logger.info(
            f"  {bench}: {n_agents} agents × {n_tasks} tasks, "
            f"mean accuracy={mean_acc:.2%}"
        )

    logger.info(f"\nIRT suitability diagnostics:")
    for bench, grp in tasks_df.groupby("benchmark"):
        uninformative = grp[(grp["pass_rate"] == 0) | (grp["pass_rate"] == 1)]
        logger.info(
            f"  {bench}: {len(uninformative)}/{len(grp)} tasks are uninformative "
            f"(pass_rate=0 or 1) — these give zero IRT information"
        )
        if len(grp) > 0:
            high_disc = grp[
                (grp["pass_rate"] > 0.2) & (grp["pass_rate"] < 0.8)
            ]
            logger.info(
                f"    {len(high_disc)} tasks in 0.2-0.8 pass_rate range "
                f"(high potential discrimination)"
            )

    logger.info(f"\nOutputs written to {OUTPUT_DIR}/")
    logger.info(f"  response_matrix.csv — for IRT fitting")
    logger.info(f"  agents.csv — agent metadata")
    logger.info(f"  tasks.csv — task difficulty estimates (pass_rate)")


if __name__ == "__main__":
    main()
