# HAL G-Study Variance Audit

This is a fast G-study-inspired audit over the fixed HAL response matrix. Rows are binary pass/fail observations; factors are estimated with regularized logistic fixed effects because the design is sparse and unbalanced.

## Binary row dataset

- Rows: 45,069
- Canonical models: 20
- Canonical scaffolds: 15
- Model × harness combinations: 165
- Benchmarks: 10
- Tasks nested in benchmark: 2,153
- Mean accuracy: 0.292

## Incremental logistic deviance audit

| model | n_features | log_loss | pseudo_r2_vs_null | deviance_drop_vs_benchmark_main_effects | deviance_drop_vs_task_nested_main_effects | incremental_deviance_drop | auc | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | 1 | 0.6036 | 0.0000 |  |  |  | 0.5000 | 0.7084 |
| benchmark | 10 | 0.5363 | 0.1114 |  |  | 6058.8260 | 0.6741 | 0.7084 |
| task_nested | 2153 | 0.3881 | 0.3569 |  | 0.0000 | 13357.4808 | 0.8926 | 0.8184 |
| task_nested + model | 2173 | 0.3764 | 0.3763 |  | 1058.1423 | 1058.1423 | 0.9004 | 0.8294 |
| task_nested + scaffold | 2168 | 0.3728 | 0.3823 |  | 1379.8298 | 321.6874 | 0.8976 | 0.8279 |
| task_nested + model + scaffold | 2188 | 0.3615 | 0.4011 |  | 2406.3039 | 1026.4742 | 0.9051 | 0.8380 |
| task_nested + model + scaffold + model:harness | 2353 | 0.3495 | 0.4210 |  | 3487.9185 | 1081.6146 | 0.9122 | 0.8447 |
| task_nested + model + scaffold + model:harness + model:benchmark + harness:benchmark | 2494 | 0.3430 | 0.4317 |  | 4070.2290 | 582.3105 | 0.9149 | 0.8482 |

## Staged G-study audit

| model | n_features | log_loss | pseudo_r2_vs_null | deviance_drop_vs_benchmark_main_effects | deviance_drop_vs_task_nested_main_effects | incremental_deviance_drop | auc | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | 1 | 0.6036 | 0.0000 |  |  |  | 0.5000 | 0.7084 |
| benchmark | 10 | 0.5363 | 0.1114 |  |  | 6058.8260 | 0.6741 | 0.7084 |
| benchmark + model | 30 | 0.5278 | 0.1255 |  |  | 771.4008 | 0.7064 | 0.7103 |
| benchmark + harness | 25 | 0.5296 | 0.1225 |  |  | -163.2698 | 0.6925 | 0.7120 |
| benchmark + model + harness | 45 | 0.5214 | 0.1361 | 0.0000 |  | 739.9771 | 0.7152 | 0.7217 |
| benchmark + model + harness + model:harness | 210 | 0.5130 | 0.1501 | 759.7584 |  | 759.7584 | 0.7332 | 0.7260 |
| benchmark + model + harness + benchmark:harness | 67 | 0.5214 | 0.1362 | 1.8257 |  | -757.9327 | 0.7152 | 0.7217 |
| benchmark + model + harness + model:harness + benchmark:harness | 232 | 0.5129 | 0.1501 | 761.4060 |  | 759.5803 | 0.7332 | 0.7260 |
| task_nested + model + harness | 2188 | 0.3615 | 0.4011 | 14415.6766 |  | 13654.2705 | 0.9051 | 0.8380 |
| task_nested + model + harness + model:harness | 2353 | 0.3495 | 0.4210 | 15497.2912 |  | 1081.6146 | 0.9122 | 0.8447 |
| task_nested + model + harness + benchmark:harness | 2210 | 0.3572 | 0.4082 | 14800.7236 |  | -696.5676 | 0.9060 | 0.8382 |
| task_nested + model + harness + model:harness + benchmark:harness | 2375 | 0.3456 | 0.4273 | 15841.1520 |  | 1040.4284 | 0.9132 | 0.8467 |
| task_nested + model + harness + task:harness | 5618 | 0.3314 | 0.4509 | 17120.9554 |  | 1279.8034 | 0.9205 | 0.8484 |

Interpretation: larger drops in log loss / deviance indicate factors that explain more structure in the binary outcomes. This is not a balanced random-effects G-study; it is the practical first pass we can run on the observed HAL matrix before deciding the IRT and cold-start model structure.

Residual summary CSVs are computed from the staged `benchmark + model + harness + model:harness + benchmark:harness` model. These residuals answer the requested question: what remains after accounting for benchmark, model, harness, model-harness compatibility, and benchmark-specific harness effects?

## Immediate read

- Task identity explains the largest single chunk, which supports item-level modeling rather than aggregate benchmark scores.
- Model and scaffold both add signal after task difficulty, so the object of measurement should remain model × harness.
- Model × harness and benchmark-specific interactions add further signal, which motivates cold-start features and targeted ablations.
