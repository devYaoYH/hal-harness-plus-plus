---
name: hal-data-prep
description: "Download, decrypt, and prepare HAL leaderboard trace data for analysis. Downloads encrypted agent traces from HuggingFace (agent-evals/hal_traces), decrypts them, and extracts task-level pass/fail results into IRT-ready CSVs. Use when you need HAL benchmark data, agent traces, or task-level results."
---

# HAL Data Prep

You are preparing HAL leaderboard trace data for analysis. The data pipeline is already built at `irt_data/prepare_irt_data.py`.

## What this does

Downloads encrypted agent evaluation traces from the `agent-evals/hal_traces` HuggingFace dataset, decrypts them (password: `hal1234`), and extracts task-level pass/fail results into structured CSVs.

## How to run

### Full download + extraction (all benchmarks)

```bash
python3 irt_data/prepare_irt_data.py
```

### Filter to specific benchmarks

```bash
python3 irt_data/prepare_irt_data.py --benchmarks swebench_verified_mini taubench_airline
```

### Re-extract from already-downloaded traces (skip download)

```bash
python3 irt_data/prepare_irt_data.py --skip-download
```

## Outputs (in `irt_data/`)

- **`response_matrix.csv`** — Core data: `(agent_id, scaffold, model, benchmark, task_id, correct, run_id)`. One row per agent×task. The `correct` column is binary (0/1).
- **`agents.csv`** — Agent metadata: scaffold name, model, benchmark, accuracy, n_tasks, total_cost, date, source file.
- **`tasks.csv`** — Task summary: benchmark, task_id, n_agents evaluated, pass_rate across agents.

## Data source details

- **HuggingFace repo**: `agent-evals/hal_traces` (public, no auth needed)
- **Encryption**: Fernet (PBKDF2-SHA256), password `hal1234`, implemented in `hal/utils/json_encryption.py`
- **Trace format**: Each `_UPLOAD.zip` contains an encrypted JSON with keys: `config`, `results` (with `successful_tasks`/`failed_tasks`), `raw_eval_results` (per-task details), `total_cost`, `total_usage`

## Supported benchmarks

swebench_verified_mini, swebench_verified, taubench_airline, taubench_retail, gaia, usaco, corebench_hard, corebench_medium, corebench_easy, scienceagentbench, scicode, scicode_hard, assistantbench, online_mind2web

## Dependencies

```bash
pip3 install huggingface_hub cryptography pandas
```

## Notes

- Downloads are cached in `irt_data/traces/` — rerunning skips already-downloaded files.
- Some trace files are large (up to 800MB). Full download of all ~381 traces takes significant time.
- The script prints IRT suitability diagnostics: how many tasks per benchmark fall in the informative 0.2–0.8 pass rate range vs. ceiling/floor tasks.
- To add support for a new benchmark, add an extractor function in `prepare_irt_data.py` and register it in the `EXTRACTORS` dict.
