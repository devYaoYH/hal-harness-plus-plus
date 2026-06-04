# HAL-IRT data and analysis

This directory contains the data pipeline and analysis artifacts for fitting IRT
models on HAL leaderboard traces. The repository is a fork of the public
[HAL harness](https://github.com/benediktstroebl/hal-harness); the additions here
turn uploaded HAL traces into response matrices, train 2PL-style IRT models, and
produce adaptive-evaluation and trace-analysis artifacts.

The public-facing report is served from `docs/index.html` via GitHub Pages:
[HAL-IRT report](https://devyaoyh.github.io/hal-harness-plus-plus/).

## Start from a fresh clone

Clone this repository with submodules, then install the normal harness
dependencies from the repository root.

```bash
git clone --recursive git@github.com:devYaoYH/hal-harness-plus-plus.git
cd hal-harness-plus-plus
pip install -r requirements.txt
```

Most commands below are run from the repository root so Python can import both
`hal` and `irt_data`.

## A. Download trace data

HAL leaderboard traces are published as encrypted zip files in the Hugging Face
dataset `agent-evals/hal_traces`. The preparation script downloads those zips,
decrypts them with the public HAL trace password used by the harness, extracts
task-level pass/fail outcomes, and writes IRT-ready CSVs.

```bash
python irt_data/prepare_irt_data.py
```

Useful variants:

```bash
# Reuse zips already present in irt_data/traces/.
python irt_data/prepare_irt_data.py --skip-download

# Recombine existing per-trace CSV extracts without re-downloading or re-parsing.
python irt_data/prepare_irt_data.py --combine-only

# Limit work to specific benchmarks.
python irt_data/prepare_irt_data.py --benchmarks swebench_verified_mini taubench_airline

# Increase parallel extraction workers.
python irt_data/prepare_irt_data.py --workers 8
```

Primary outputs:

| File | Purpose |
| --- | --- |
| `response_matrix.csv` | One row per agent-task outcome; main input to IRT training. |
| `agents.csv` | Agent/run metadata derived from the traces. |
| `tasks.csv` | Task-level benchmark metadata and pass-rate summaries. |
| `extracted/*.csv` | Incremental per-upload extraction cache. |
| `traces/*.zip` | Downloaded encrypted HAL trace uploads; ignored by git when absent. |

## B. Train the IRT 2PL model

The main training entrypoint fits multidimensional 2PL-style models. A
one-dimensional run is the closest scalar 2PL configuration; `--k 2` trains the
default two-factor model used for most HAL-IRT analysis.

```bash
# Scalar 2PL-style model with feature-informed agent/task priors.
python -m irt_data.irt.run --k 1

# Two-dimensional 2PL/MIRT model.
python -m irt_data.irt.run --k 2

# Compare K in {1, 2, 4, 8}, with and without features.
python -m irt_data.irt.run --sweep
```

Training reads `irt_data/response_matrix.csv` by default. To train on another
matrix:

```bash
python -m irt_data.irt.run --k 2 --response-csv irt_data/eval_splits/response_matrix_train.csv
```

Training outputs include `training_curves_k*_feat.png`,
`training_curves_k*_nofeat.png`, and `sweep_results.json`.

## C. Use the fitted models

Generate discrimination-ranked task lists and adaptive subsets:

```bash
python -m irt_data.irt.discriminate --k 2 --thresholds 0.5 0.7 0.8 0.9
```

This writes:

| File | Purpose |
| --- | --- |
| `task_discrimination.csv` | All tasks ranked by fitted discrimination magnitude. |
| `adaptive_task_subsets.json` | Per-benchmark task shortlists at each coverage threshold. |

Use cold-start predictions for a new scaffold x model combination:

```bash
python -m irt_data.irt.predict_new_agent \
  --scaffold "SWE-Agent" \
  --model "deepseek-r1" \
  --benchmarks swebench_verified_mini \
  --threshold 80pct \
  --out irt_data/irt/validation/pred_sweagent_deepseekr1.csv
```

The prediction script refits the model from `response_matrix.csv`, estimates the
new agent's latent ability from feature projections, and scores the selected
adaptive task subset.

## Analysis artifacts

Use these files and folders when inspecting results or regenerating figures.

| Path | Contents |
| --- | --- |
| `irt/report/FINDINGS.md` | Model-selection notes, K sweep summary, and adaptive subset rationale. |
| `irt/report/GENERALIZATION_TEST.md` | Cold-start validation notes and tool-ablation discussion. |
| `irt/report/scaffold_tools_report.md` | Scaffold/tool feature summary. |
| `irt/plots/` | Heatmaps, discrimination plots, cost-reduction plots, and training curves. |
| `irt/experiments/` | Split-v1 sweeps, cost-reduction tables, feature attribution, and cold-start heatmaps. |
| `irt/validation/` | Prediction and actual-result CSVs for validation runs. |
| `eval_splits/` | Canonical train/holdout split files and split summary. |
| `gstudy/` | Generalizability-study data, residual summaries, and report. |
| `trace_analysis/` | Trace-level analyses of task trajectories, failure modes, and tool-use patterns. |
| `demo/index.html` | Local static demo entrypoint. |
| `../docs/index.html` | GitHub Pages report source. |

For the rendered public summary, use the
[GitHub Pages report](https://devyaoyh.github.io/hal-harness-plus-plus/). For
the original benchmark harness usage, agents, and leaderboard submission flow,
see the top-level `README.md` inherited from the HAL harness fork.
