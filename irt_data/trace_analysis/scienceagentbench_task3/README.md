# Trace analysis: scienceagentbench task 3

Runs analyzed: 25 (18 success, 7 failure).

Mechanism split:

- `code_agent_tool_registry`: 1/7 success
- `harness_side_augmentation`: 17/18 success

## Strongest suffix separators

| suffix | support | success rate | log-odds lift | mechanisms |
| --- | ---: | ---: | ---: | --- |
| `self_debug:direct_code` | 11 | 1.00 | 3.14 | harness_side_augmentation:11 |
| `plain:observation_to_direct_code` | 4 | 0.25 | -2.21 | code_agent_tool_registry:4 |
| `self_debug:direct_code -> self_debug:direct_code` | 6 | 1.00 | 2.05 | harness_side_augmentation:6 |
| `plain:observation_to_direct_code -> plain:observation_to_direct_code` | 3 | 0.33 | -1.67 | code_agent_tool_registry:3 |
| `self_debug:error_response` | 3 | 1.00 | 1.22 | harness_side_augmentation:3 |
| `registry:observation_to_direct_code -> registry:observation_to_direct_code -> registry:observation_to_direct_code -> registry:observation_to_direct_code -> plain:observation_to_direct_code -> plain:observation_to_direct_code` | 2 | 0.50 | -0.99 | code_agent_tool_registry:2 |
| `registry:observation_to_direct_code -> registry:observation_to_direct_code -> registry:observation_to_direct_code -> plain:observation_to_direct_code -> plain:observation_to_direct_code` | 2 | 0.50 | -0.99 | code_agent_tool_registry:2 |
| `registry:observation_to_direct_code -> registry:observation_to_direct_code -> plain:observation_to_direct_code -> plain:observation_to_direct_code` | 2 | 0.50 | -0.99 | code_agent_tool_registry:2 |
| `registry:observation_to_direct_code -> plain:observation_to_direct_code -> plain:observation_to_direct_code` | 2 | 0.50 | -0.99 | code_agent_tool_registry:2 |
| `self_debug:code_block` | 4 | 0.75 | -0.02 | harness_side_augmentation:4 |

## Strongest signature suffix separators

| suffix | support | success rate | log-odds lift | mechanisms |
| --- | ---: | ---: | ---: | --- |
| `self_debug:direct_code(code_blocks=2)` | 11 | 1.00 | 3.14 | harness_side_augmentation:11 |
| `plain:observation_to_direct_code(code_blocks=2)` | 4 | 0.25 | -2.21 | code_agent_tool_registry:4 |
| `self_debug:direct_code(code_blocks=2) -> self_debug:direct_code(code_blocks=2)` | 6 | 1.00 | 2.05 | harness_side_augmentation:6 |
| `plain:observation_to_direct_code(code_blocks=2) -> plain:observation_to_direct_code(code_blocks=2)` | 3 | 0.33 | -1.67 | code_agent_tool_registry:3 |
| `self_debug:error_response(code_blocks=2)` | 3 | 1.00 | 1.22 | harness_side_augmentation:3 |
| `registry:observation_to_direct_code(code_blocks=2) -> registry:observation_to_direct_code(code_blocks=2) -> registry:observation_to_direct_code(code_blocks=2) -> registry:observation_to_direct_code(code_blocks=2) -> plain:observation_to_direct_code(code_blocks=2) -> plain:observation_to_direct_code(code_blocks=2)` | 2 | 0.50 | -0.99 | code_agent_tool_registry:2 |
| `registry:observation_to_direct_code(code_blocks=2) -> registry:observation_to_direct_code(code_blocks=2) -> registry:observation_to_direct_code(code_blocks=2) -> plain:observation_to_direct_code(code_blocks=2) -> plain:observation_to_direct_code(code_blocks=2)` | 2 | 0.50 | -0.99 | code_agent_tool_registry:2 |
| `registry:observation_to_direct_code(code_blocks=2) -> registry:observation_to_direct_code(code_blocks=2) -> plain:observation_to_direct_code(code_blocks=2) -> plain:observation_to_direct_code(code_blocks=2)` | 2 | 0.50 | -0.99 | code_agent_tool_registry:2 |
| `registry:observation_to_direct_code(code_blocks=2) -> plain:observation_to_direct_code(code_blocks=2) -> plain:observation_to_direct_code(code_blocks=2)` | 2 | 0.50 | -0.99 | code_agent_tool_registry:2 |
| `self_debug:code_block(code_blocks=2)` | 4 | 0.75 | -0.02 | harness_side_augmentation:4 |

## Feature cuts

| feature | value | support | success rate | successes | failures |
| --- | --- | ---: | ---: | ---: | ---: |
| `failure_mode` | `success` | 18 | 1.00 | 18 | 0 |
| `final_event` | `self_debug:direct_code` | 11 | 1.00 | 11 | 0 |
| `final_signature` | `self_debug:direct_code(code_blocks=2)` | 11 | 1.00 | 11 | 0 |
| `llm_call_count` | `2-4` | 7 | 1.00 | 7 | 0 |
| `failure_mode` | `functionally_incorrect` | 4 | 0.00 | 0 | 4 |
| `final_event` | `self_debug:error_response` | 3 | 1.00 | 3 | 0 |
| `final_signature` | `self_debug:error_response(code_blocks=2)` | 3 | 1.00 | 3 | 0 |
| `failure_mode` | `runtime_exception` | 2 | 0.00 | 0 | 2 |
| `length_finish_count` | `1+` | 2 | 0.00 | 0 | 2 |
| `complete_program_prompt` | `1` | 18 | 0.94 | 17 | 1 |
| `normalized_harness` | `SAB Self-Debug` | 18 | 0.94 | 17 | 1 |
| `self_debug_prompt` | `1` | 18 | 0.94 | 17 | 1 |
| `tool_exposure_mechanism` | `harness_side_augmentation` | 18 | 0.94 | 17 | 1 |
| `tool_registry_prompt` | `0` | 18 | 0.94 | 17 | 1 |
| `llm_call_count` | `1` | 10 | 0.90 | 9 | 1 |
| `complete_program_prompt` | `0` | 7 | 0.14 | 1 | 6 |
| `normalized_harness` | `HAL Generalist Agent` | 7 | 0.14 | 1 | 6 |
| `self_debug_prompt` | `0` | 7 | 0.14 | 1 | 6 |

## Files

- `task_trajectory_summary.csv`: one row per trace/run.
- `suffix_patterns.csv`: suffix-trie counts over coarse process events.
- `signature_suffix_patterns.csv`: suffix-trie counts over process events with code-block/finish annotations.
- `feature_cuts.csv`: coarse success/failure cuts for interpretable factors.
- `selected_task.json`: target task metadata.
