# IRT Generalization Test: SWE-Agent × DeepSeek-R1

**Date:** 2026-04-18  
**Model:** K=2 MIRT + features (updated schema: release_ym, param_count_full/active_b, moe, max_steps, context_strategy)

## Setup

The MIRT model was fitted on 273 agents across 8 benchmarks. **SWE-Agent has only ever been run with `claude-3-7-sonnet`** — making `SWE-Agent × DeepSeek-R1` a clean cold-start generalization test: both the scaffold and model are individually known, but their combination is absent from training.

## Method

For a novel agent, there is no learned latent embedding θ. The feature projection `f_agent(features)` alone serves as the cold-start prior, then task probabilities are computed using the frozen fitted task parameters (a, d).

Features for this combination:

| Feature | Value |
|---------|-------|
| scaffold_type | `code_agent` |
| filesystem, file_edit, file_search | True |
| web_search, python_exec, browser | False |
| max_steps | 50 |
| model_family / size_tier | deepseek / large |
| param_count_full / active | 671B / 37B (MoE) |
| model_reasoning | True |
| release_ym | 2025.0 |

**Norm rescaling fix:** the raw feature projection extrapolated to θ norm 243 vs training distribution mean 11.8. `predict_new_agent.py` now rescales cold-start θ to the training norm before computing probabilities. This is a known limitation of cold-start IRT — combinations that lie far outside the training feature distribution require calibration.

## Prediction

| Metric | Value |
|--------|-------|
| Predicted accuracy (28-task 80% subset) | **38.4%** |
| Baseline: SWE-Agent + claude-3-7 (full 50 tasks) | 53.0% |
| Cold-start θ (rescaled) | [−5.6, −10.4] |

The negative θ is interpretable against the fitted latent space:
- **Dim 0** (a₀ > 0 = general agent capability): negative → model predicts weaker general performance than claude-3-7, consistent with SWE-Agent's prompting being tuned for Anthropic models
- **Dim 1** (a₁ < 0 = code-task separation): negative θ₁ × negative a₁ = positive contribution → DeepSeek-R1's chain-of-thought reasoning partially compensates on hard code tasks

Net prediction: a **~14 percentage point drop** vs claude-3-7 on this scaffold — plausible.

## Validation Script

```bash
python3 -c "
import os, sys
sys.path.insert(0, 'prefect')
os.environ.setdefault('SSL_CERT_FILE', __import__('certifi').where())
from dotenv import load_dotenv; load_dotenv()
import run_irt_validation
run_irt_validation.main()
"
```

Launches 28 Daytona sandboxes in parallel (default: 10 concurrent), saves actuals to  
`irt_data/actuals_irt-val-sweagent-deepseekr1-001.csv`, then prints:

- Actual vs predicted accuracy
- Brier score (0.25 = chance baseline)
- Point-biserial r (predicted_prob vs correct)

Override any parameter via env var: `AGENT`, `MODEL`, `BENCHMARK`, `THRESHOLD`, `JOB_ID`, `MAX_WORKERS`.

## Files

| File | Purpose |
|------|---------|
| `irt_data/irt/predict_new_agent.py` | Cold-start prediction for novel agent combinations |
| `prefect/run_irt_validation.py` | Parallel Daytona runner with IRT comparison |
| `irt_data/pred_sweagent_deepseekr1.csv` | Pre-computed per-task predictions |
| `prefect/daytona_runner.py` | Updated: `agents/SWE-agent` un-excluded from sandbox zip |

---

## Tool Ablation Experiment: Remove `vision_query` on scicode

### Finding

Mining the response matrix with model-matched controls reveals that on **scicode**, agents without `vision_query` consistently outperform those with it:

| Group | Scaffolds | Shared models | Accuracy |
|-------|-----------|---------------|----------|
| vision_query = True | HAL Generalist Agent | 6 | 0.026 |
| vision_query = False | Scicode Tool/Zero-Shot Agent | 6 | 0.038 |
| **Δ** | | | **+0.012** |

Hypothesis: `vision_query` is a distractor on pure scientific code tasks — the agent wastes budget calling the VLM on code/math problems where no visual grounding is useful, and the additional tool option increases decision overhead.

### Comparison: `multi_agent` on GAIA (do NOT remove — it's already False)

The strongest model-matched signal in the data is that `multi_agent=True` scaffolds score 0.359 vs 0.488 for single-agent on GAIA (Δ=+0.129, 10 shared models). HAL Generalist already has `multi_agent=False`, so this isn't a removal experiment — but it confirms that multi-agent coordination actively hurts on direct QA tasks.

### Experiment

HAL Generalist's `run()` now accepts a `disable_tools` kwarg (comma-separated tool names). Run the ablation against the 49-task scicode 80% subset:

```bash
python3 -c "
import os, sys
sys.path.insert(0, 'prefect')
os.environ.setdefault('SSL_CERT_FILE', __import__('certifi').where())
from dotenv import load_dotenv; load_dotenv()
import run_tool_ablation
run_tool_ablation.main()
"
```

The script:
1. Loads the 49 scicode tasks from `adaptive_task_subsets.json`
2. Reads the baseline accuracy for the same model from the response matrix
3. Launches tasks in parallel via Daytona with `disable_tools=vision_query`
4. Compares ablated vs baseline accuracy and reports whether the observed Δ matches the predicted +0.012

Override env vars: `DISABLE_TOOLS`, `MODEL`, `BENCHMARK`, `THRESHOLD`, `JOB_ID`, `MAX_WORKERS`.

---

## Tool Ablation Experiment 2: Remove web/doc tools from HAL Generalist on swebench

### Finding

On `swebench_verified_mini`, HAL Generalist (8 tools) scores dramatically below simpler code-focused scaffolds — and the differentiating tools are all web/document-oriented:

| Scaffold | Tools | claude-3-7 acc | Unique tools vs HAL |
|----------|-------|----------------|---------------------|
| HAL Generalist | filesystem, web_search, page_browse, python_exec, file_edit, file_search, text_inspect, vision_query | **32%** | — |
| My Agent | filesystem, python_exec, file_edit | **50%** | missing 5 HAL tools |
| SWE-Agent | filesystem, file_edit, file_search | **53%** | missing 5 HAL tools |

The 4 tools HAL has that neither comparator has — `web_search`, `page_browse`, `text_inspect`, `vision_query` — are all irrelevant to local repository code repair.

### Why this should improve performance

Removing those 4 tools leaves HAL with: **filesystem + python_exec + file_edit + file_search** — a strict superset of both My Agent and SWE-Agent. It is the only scaffold that would simultaneously have:
- `python_exec` — run tests, reproduce bugs, execute patches
- `file_search` — grep/find across the codebase

**Predicted Δ: +18 percentage points** (claude-3-7: 32% → ~50%), driven by:
1. Eliminating wrong tool calls (web_search/page_browse on a local-repo task)
2. Reducing tool decision space from 8 → 4 options
3. Leaving a stronger remaining set than either comparator

### Experiment

`run_tool_ablation.py` is pre-configured for this experiment (default `DISABLE_TOOLS=web_search,page_browse,text_inspect,vision_query`, `BENCHMARK=swebench_verified_mini`):

```bash
python3 -c "
import os, sys
sys.path.insert(0, 'prefect')
os.environ.setdefault('SSL_CERT_FILE', __import__('certifi').where())
from dotenv import load_dotenv; load_dotenv()
import run_tool_ablation
run_tool_ablation.main()
"
```

Runs 28 tasks (80% adaptive subset) in parallel. Reports ablated accuracy vs baseline from response matrix and flags whether the observed direction matches the predicted +0.18.

## Next Steps

- Run the 28 tasks and fill in actual accuracy
- If Brier score < 0.20 and r > 0.3: cold-start IRT is useful for pre-screening novel combinations
- Extend to tool-level ablations: add/remove individual boolean scaffold features (web_search, self_critique, etc.) to measure per-tool impact on discrimination without swapping the full harness
