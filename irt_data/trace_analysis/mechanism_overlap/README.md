# Mechanism-overlap trace probe

Selected task: `scienceagentbench:3`.

Why this task: it has 25 runs across 2
tool-exposure mechanisms and 12 normalized models. The
overall pass rate is 0.72. The mechanism split is:

- `code_agent_tool_registry`: 1/7
  success, pass rate 0.14.
- `harness_side_augmentation`: 17/18
  success, pass rate 0.94.

Files:

- `candidate_tasks.csv`: all benchmark-task pairs with at least two
  tool-exposure mechanisms, sorted to surface code-registry versus
  harness-augmentation gaps.
- `scienceagentbench_task3_trace_signals.csv`: one row per run on this task,
  with extracted trace-level process signals.
- `scienceagentbench_task3_by_mechanism.csv`: aggregate trace signals by
  mechanism.
- `scienceagentbench_task3_failure_modes.csv`: evaluator-visible success and
  failure categories by mechanism.

Initial trace read:

- The code-agent registry traces all contain the smolagents-style tool prompt
  and an execution feedback loop (`Observation:`). They average
  26.6
  LLM calls and
  184.7
  observation mentions.
- The harness-side augmentation traces contain the ScienceAgentBench
  complete-program/self-debug prompt. They average
  1.9
  LLM calls and no in-model `Observation:` loop.

Failure modes:

- `code_agent_tool_registry` / `functionally_incorrect`: 3
- `code_agent_tool_registry` / `runtime_exception`: 2
- `code_agent_tool_registry` / `output_not_saved`: 1
- `code_agent_tool_registry` / `success`: 1
- `harness_side_augmentation` / `success`: 17
- `harness_side_augmentation` / `functionally_incorrect`: 1
