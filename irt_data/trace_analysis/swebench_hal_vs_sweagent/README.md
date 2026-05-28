# SWE-bench HAL Generalist vs SWE-Agent trace examples

Matched GPT-5 runs on SWE-bench Verified Mini. The main candidate set is `gpt5_hal_fail_swe_success_tasks.csv`; the appendix examples are `gpt5_trace_examples.csv`.

```json
{
  "benchmark": "swebench_verified_mini",
  "normalized_model": "GPT-5",
  "hal_run_id": "swebench_verified_mini_hal_generalist_gpt520250807_1755463923",
  "swe_run_id": "swebench_verified_mini_sweagentgpt520250807_1754592641",
  "matched_tasks": 50,
  "hal_fail_swe_success_tasks": 26,
  "example_tasks": [
    "django__django-12143",
    "django__django-12050",
    "sphinx-doc__sphinx-9698"
  ]
}
```
