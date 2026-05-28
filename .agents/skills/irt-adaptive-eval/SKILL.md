---
name: irt-adaptive-eval
description: "Run a new agent harness configuration on the IRT-selected minimum discriminating task subset, then compare against baseline. Use when proposing a new tool, scaffold change, or model switch and you want fast signal without running full benchmarks. Outputs per-benchmark accuracy and estimated latent ability θ."
---

# IRT Adaptive Evaluation

You are evaluating a new agent configuration (harness change, tool addition, model switch) against the minimum discriminating task subset identified by the MIRT model.

## Context

A K=2 MIRT model has been fitted on 273 agents × 1796 tasks across 8 HAL benchmarks. Each task has a fitted discrimination vector ‖a_j‖. Running only the high-discrimination tasks gives ~40-70% cost reduction with minimal loss in ability estimation accuracy.

Key files:
- **`irt_data/adaptive_task_subsets.json`** — task_id shortlists per benchmark × coverage threshold (50/70/80/90%)
- **`irt_data/task_discrimination.csv`** — all tasks ranked by discrimination, with a₀, a₁, easiness
- **`irt_data/irt/FINDINGS.md`** — full analysis report

## Workflow when user proposes a change

### Step 1 — Identify the relevant benchmarks

Ask (or infer from context) which benchmarks the change is most likely to affect. A new web tool affects GAIA/assistantbench/online_mind2web. A code tool change affects swebench/scicode/corebench_hard.

### Step 2 — Extract the adaptive task subset

```python
import json

with open("irt_data/adaptive_task_subsets.json") as f:
    subsets = json.load(f)

# Use 80% discrimination coverage by default
benchmark = "swebench_verified_mini"
task_ids = subsets[benchmark]["thresholds"]["80pct"]["task_ids"]
n_total = subsets[benchmark]["n_total"]
n_subset = subsets[benchmark]["thresholds"]["80pct"]["n_tasks"]
print(f"Running {n_subset}/{n_total} tasks ({1 - n_subset/n_total:.0%} cost reduction)")
```

### Step 3 — Run the new configuration on the subset

Use the HAL harness to run only the selected task_ids:

```bash
# Filter task_ids into a HAL-compatible task list, then run
python3 irt_data/prepare_irt_data.py --skip-download --benchmarks <benchmark>
```

Or if the harness supports task filtering, pass the task_id list directly.

### Step 4 — Compare results

```python
import pandas as pd, json

# Load baseline results for this benchmark from response_matrix
baseline = pd.read_csv("irt_data/response_matrix.csv")
baseline = baseline[baseline["benchmark"] == benchmark]

# Load new run results (same format: agent_id, task_id, correct)
new_run = pd.read_csv("new_run_results.csv")

# Accuracy on the subset tasks
subset_tasks = set(task_ids)
baseline_subset = baseline[baseline["task_id"].isin(subset_tasks)]
print("Baseline accuracy on subset:", baseline_subset.groupby("agent_id")["correct"].mean())
print("New run accuracy on subset:", new_run.groupby("agent_id")["correct"].mean())
```

### Step 5 — Estimate latent ability θ (optional)

Fit θ for the new agent using frozen MIRT parameters:

```python
from irt_data.irt.data import load_and_split
from irt_data.irt.train import TrainConfig, train
import torch

dataset = load_and_split()
config = TrainConfig(k=2, use_features=True)
model, _, _ = train(dataset, config)

# Freeze task params, optimize only θ for the new agent
# (MAP estimation on the subset responses)
```

## Task subset sizes (80% discrimination coverage)

| Benchmark | Full | Subset | Reduction |
|-----------|------|--------|-----------|
| swebench_verified_mini | 50 | ~28 | 44% |
| gaia | 165 | ~100 | 39% |
| corebench_hard | 45 | ~26 | 42% |
| scicode | 65 | ~50 | 23% |
| scienceagentbench | 102 | ~61 | 40% |
| online_mind2web | 336 | ~190 | 43% |
| assistantbench | 33 | ~23 | 30% |
| colbench_backend_programming | 1000 | ~571 | 43% |

## Regenerating subsets after new data

If the response matrix has been updated with new agent traces, refit and regenerate:

```bash
python3 -m irt_data.irt.discriminate --k 2 --thresholds 0.7 0.8 0.9
```

This overwrites `task_discrimination.csv` and `adaptive_task_subsets.json` with fresh IRT estimates.

## Adding a new scaffold to the feature taxonomy

When proposing a new harness, add its tool features to `irt_data/irt/features.py`:
1. Add entry to `_SCAFFOLD_ENTRIES` with all 16 boolean tool fields
2. Add any name variants to `_SCAFFOLD_ALIASES`
3. Refit: `python3 -m irt_data.irt.run --k 2`

## Arguments

The user may specify:
- **benchmark** — which benchmark(s) to focus on
- **threshold** — discrimination coverage (default 80%)
- **change description** — what scaffold/tool/model is being changed
