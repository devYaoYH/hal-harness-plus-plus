# Evaluation Split

This directory stores the canonical train/holdout split for IRT evaluation.

The leakage unit is:

```text
benchmark x normalized_model x normalized_harness
```

All raw runs and task outcomes belonging to a selected semantic unit are held
out together. This prevents training on one raw alias, run id, or reasoning
variant of a semantic model-harness tuple and testing on another within the
same benchmark.

The holdout set is selected by `irt_data/create_eval_split.py`. For each
`benchmark x normalized_harness` cell, the script holds out one representative
normalized model: the model whose pass rate is closest to that cell's median
pass rate. This gives the holdout coverage across benchmarks and normalized
harness mechanisms while avoiding obvious cherry-picking of only best or worst
systems.

Files:

- `holdout_configurations.csv`: selected holdout semantic units and their raw
  aliases, run ids, outcome counts, and pass rates.
- `response_matrix_train.csv`: training rows with the same columns as
  `irt_data/response_matrix.csv`.
- `response_matrix_holdout.csv`: held-out rows with the same columns as
  `irt_data/response_matrix.csv`.
- `response_matrix_with_split.csv`: source response rows plus normalized
  metadata and split labels for auditing.
- `split_summary.json`: machine-readable row/config/unit counts and validation
  status.
- `split_summary_by_benchmark.csv`: train/holdout counts per benchmark.

Current split summary:

- Source rows: 45,069
- Training rows: 39,461
- Holdout rows: 5,608
- Holdout row fraction: 12.44%
- Source semantic selection units: 219
- Training semantic selection units: 197
- Holdout semantic selection units: 22
- Source observed agent configurations: 352
- Training observed agent configurations: 316
- Holdout observed agent configurations: 36
- Selection-unit overlap between train and holdout: none
