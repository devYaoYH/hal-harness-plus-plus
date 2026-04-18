# IRT Analysis Findings

K=2 Multidimensional IRT fitted on HAL leaderboard data (April 2026).

## Dataset

- **273 agents** (scaffold × model pairs) across **8 benchmarks**
- **1796 tasks**, **37,110 observations** (agent × task binary correct/incorrect)
- Train/val split: 20% of agents held out, 30% of their responses withheld for evaluation

## Model Sweep Results

Swept K ∈ {1, 2, 4, 8} × with/without scaffold+model features. Baseline (marginal pass rate): acc=0.703, AUC=0.500.

| K | Features | Val Loss | Val AUC | Val Acc | Cal Err |
|---|----------|----------|---------|---------|---------|
| 1 | no       | 0.483    | 0.814   | 0.775   | 0.053   |
| 1 | yes      | 0.468    | 0.821   | 0.782   | 0.055   |
| 2 | no       | 0.481    | 0.815   | 0.773   | 0.040   |
| 2 | yes      | 0.457    | 0.833   | 0.790   | 0.032   |
| 4 | no       | 0.475    | 0.823   | 0.774   | 0.053   |
| 4 | yes      | 0.459    | 0.833   | 0.790   | 0.037   |
| 8 | no       | **0.428**| **0.857**| **0.812**| 0.047 |
| 8 | yes      | 0.449    | 0.842   | 0.796   | 0.041   |

**K=8 latent-only** wins on raw AUC/accuracy. **K=2+features** is the best-calibrated model (cal_err=0.032) at much lower complexity — recommended for adaptive testing where calibration matters.

Features used: model provider/family/generation/size_tier/reasoning flag + 16 scaffold tool boolean features (filesystem, web_search, page_browse, full_browser, browser_vision, python_exec, file_edit, file_search, http_requests, wiki_search, text_inspect, vision_query, multi_agent, has_instructions, self_critique, has_skills).

## Extraction Bugs Fixed

Two extraction bugs were found and fixed in `prepare_irt_data.py` before fitting:

1. **scienceagentbench** (0% → 21.4% mean accuracy): Extractor looked for `score`/`success` keys but actual field is `success_rate` in `raw_eval_results.eval_result`.
2. **corebench_hard** (3.5% → 25.4% mean accuracy): Raw eval results use `correct_written_answers`/`total_written_questions` — no simple `score` key. Extractor found rows and returned all-zeros instead of falling through to `successful_tasks`/`failed_tasks`. Fixed to prefer the pre-computed pass/fail lists.

## Adaptive Task Subsets

Tasks ranked by discrimination magnitude ‖a_j‖ from the K=2+features model. Full ranked list in `task_discrimination.csv`. Task ID subsets at each coverage threshold in `adaptive_task_subsets.json`.

### Tasks needed for X% of total discrimination coverage

| Benchmark | Total | 50% | 70% | 80% | 90% | 80% reduction |
|-----------|-------|-----|-----|-----|-----|----------------|
| swebench_verified_mini | 50 | 13 | 22 | 28 | 36 | **44%** |
| gaia | 165 | 46 | 75 | 94 | 117 | **43%** |
| corebench_hard | 45 | 13 | 21 | 26 | 33 | **42%** |
| scienceagentbench | 102 | 32 | 49 | 61 | 76 | **40%** |
| online_mind2web | 336 | 90 | 152 | 190 | 241 | **43%** |
| assistantbench | 33 | 14 | 20 | 23 | 27 | 30% |
| scicode | 65 | 30 | 44 | 50 | 57 | 23% |
| colbench_backend_programming | 1000 | 312 | 458 | 571 | 722 | **43%** |

### Most discriminating tasks — swebench_verified_mini (top 12 for 50% coverage)

Top tasks are predominantly sphinx-doc and django issues, suggesting these repositories have the highest inter-agent variability. See `task_discrimination.csv` for full ranked list.

### Most discriminating tasks — corebench_hard (top 13 for 50% coverage)

Capsules: 9660931, 3821950, 8536428, 4252248, 5507257, 7716865, 5136217, 3449234, 7186268, 2804717, 4299879, 9911222, 0921079

### Latent Space Interpretation (K=2)

- **Dim 0 (a₀ > 0)**: Tasks where stronger general agents succeed — loaded positively by HAL Generalist, CORE-Agent on GAIA/swebench
- **Dim 1 (a₁ < 0)**: Tasks that specifically separate code-capable agents — swebench and corebench tasks cluster heavily in negative a₁

The 2D discrimination plot (`discrimination_2d.png`) shows benchmarks form loose clusters in latent space, confirming the K=2 model captures cross-benchmark structure.

## Outputs

| File | Description |
|------|-------------|
| `task_discrimination.csv` | All 1796 tasks with discrimination, a0, a1, easiness |
| `adaptive_task_subsets.json` | task_id lists per benchmark × threshold (50/70/80/90%) |
| `sweep_results.json` | Full sweep metrics |
| `discrimination_distributions.png` | Discrimination bar charts per benchmark |
| `discrimination_2d.png` | 2D scatter of task vectors colored by benchmark |
| `cost_reduction.png` | Cost reduction fractions per benchmark × threshold |
| `heatmap_by_model.png` | Agent accuracy heatmap grouped by model |
| `heatmap_by_scaffold.png` | Agent accuracy heatmap grouped by scaffold |
| `irt/scaffold_tools_report.md` | Verified tool configurations per scaffold |

## Next Steps

- **Verify adaptive subsets**: Run a new agent on both full benchmark and the 80% subset, compare ability estimates — expected correlation > 0.9 if IRT is well-calibrated
- **Computerized Adaptive Testing (CAT)**: After each task, update MAP estimate of θ_agent, pick next task maximizing Fisher information — should converge in even fewer tasks than the static subset
- **Model metadata**: Add param count, architecture (MoE vs dense), context window to `MODEL_REGISTRY` in `features.py` for richer agent features
- **More benchmarks**: taubench_airline/retail, swebench_verified (full) not yet included
