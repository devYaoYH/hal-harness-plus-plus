# Scaffold Tool Configuration Report

Source: verified from `agents/` directory in hal-harness repo.

## HAL Generalist Agent
- **Source**: `agents/hal_generalist_agent/main.py:1260-1270`
- **Framework**: smolagents `CodeAgent`, planning_interval=4, max_steps=200
- **Tools**:
  - `web_search` (wrapped GoogleSearchTool, ignores year filter)
  - `VisitWebpageTool`
  - `PythonInterpreterTool`
  - `execute_bash` (custom)
  - `TextInspectorTool` (LLM-based, text_limit=5000)
  - `edit_file` (custom)
  - `file_content_search` (custom)
  - `query_vision_language_model` (custom)
- **Instructions**: Yes (system prompt with GAIA-specific instructions, perturbable)
- **Skills**: No

## CORE-Agent
- **Source**: `agents/core_agent/main.py:841-851`
- **Framework**: smolagents `CodeAgent`, planning_interval=4, max_steps=40
- **Tools**:
  - `DuckDuckGoSearchTool`
  - `VisitWebpageTool`
  - `PythonInterpreterTool`
  - `execute_bash` (custom)
  - `TextInspectorTool` (LLM-based, text_limit=5000)
  - `edit_file` (custom)
  - `file_content_search` (custom)
  - `query_vision_language_model` (custom)
  - `custom_final_answer_tool` (custom tool class)
- **Instructions**: Yes (system prompt)
- **Skills**: No

## HF Open Deep Research
- **Source**: `agents/open_deep_research/examples/open_deep_research/run.py:59-112`
- **Framework**: smolagents, multi-agent (manager CodeAgent + sub-agent ToolCallingAgent)
- **Manager Agent Tools** (CodeAgent, max_steps=12, planning_interval=4):
  - `visualizer`
  - `TextInspectorTool`
  - Managed sub-agent: `search_agent`
- **Sub-Agent Tools** (ToolCallingAgent "search_agent", max_steps=20, planning_interval=4):
  - `GoogleSearchTool` (serpapi)
  - `VisitTool` (text browser)
  - `PageUpTool` / `PageDownTool`
  - `FinderTool` / `FindNextTool`
  - `ArchiveSearchTool`
  - `TextInspectorTool`
- **Instructions**: Yes (sub-agent has detailed search instructions)
- **Skills**: No
- **Notable**: Multi-agent architecture with delegated web browsing

## SWE-Agent
- **Source**: `agents/SWE-agent/config/commands/`
- **Framework**: Custom command-based system (shell commands with YAML signatures)
- **Tools** (shell commands):
  - `open <path> [<line>]`
  - `goto <line_number>`
  - `scroll_up` / `scroll_down`
  - `create <filename>`
  - `edit <start_line>:<end_line>` (with linting)
  - `search_dir <term> [<dir>]`
  - `search_file <term> [<file>]`
  - `find_file <file_name> [<dir>]`
  - `filemap <file_path>`
  - `submit`
  - `set_cursors <start> <end>` (cursors variant)
- **Instructions**: Yes (detailed system template with demos)
- **Skills**: No

## Assistantbench Browser Agent
- **Source**: `agents/assistantbench_browser_agent/main.py:88-110`
- **Framework**: browser-use library `Agent` with `Browser`
- **Tools** (implicit via browser-use framework):
  - Browser navigation (click, type, scroll, navigate)
  - JavaScript execution
  - DOM interaction
- **Vision**: Yes (default), disabled for DeepSeek models
- **Instructions**: Yes (message_context with answer format instructions)
- **Skills**: No
- **Notable**: max_steps=20, headless browser

## Browser-Use
- **Source**: Same framework as Assistantbench Browser Agent (browser-use library)
- **Tools**: Same browser-use built-ins
- **Instructions**: Minimal
- **Skills**: No

## SciCode Tool Calling Agent
- **Source**: `agents/scicode_tool_calling_agent/agent.py:216-223`
- **Framework**: smolagents `CodeAgent`, planning_interval=3, max_steps=5
- **Tools**:
  - `RateLimitAwareDuckDuckGoSearchTool`
  - `PythonInterpreterTool`
  - `ModifiedWikipediaSearchTool`
  - `FinalAnswerTool` (custom description with scientific computing instructions)
- **Instructions**: Yes (detailed scientific coding constraints in FinalAnswerTool description)
- **Skills**: No

## SciCode Zero Shot Agent
- **Source**: `agents/scicode_zero_shot_agent/main.py`
- **Framework**: Direct API calls (no agent framework)
- **Tools**: None
- **Instructions**: Prompt template only
- **Skills**: No

## SAB Self-Debug (sab_example_agent)
- **Source**: `agents/sab_example_agent/main.py:20-46`
- **Framework**: `ScienceAgent` from `science_agent` library
- **Tools**: Wrapped by ScienceAgent (not explicitly listed in HAL harness)
  - Python code execution (inferred from self-debug loop)
  - File I/O for datasets and predictions
- **Config flags**: `use_self_debug`, `use_knowledge`, `context_cutoff=28000`
- **Instructions**: Via ScienceAgent library internals
- **Skills**: No

## Colbench Example Agent
- **Source**: `agents/colbench_example_agent/main.py`
- **Framework**: Direct API calls (OpenAI / Anthropic clients)
- **Tools**: None (conversational agent, no tool calling)
- **Interaction**: Multi-turn dialogue with simulated human (HumanInteractionEnv)
- **Instructions**: Yes (`code_agent_prompt.txt`, `html_agent_prompt.txt`)
- **Skills**: No
- **Notable**: No tool use — pure conversational code generation within 10 rounds

## Claude Code
- **Source**: Not in `agents/` directory (external tool, traces only)
- **Tools**: (known from Claude Code product)
  - File read/write/edit
  - Bash execution
  - Glob/Grep search
  - LSP integration
  - Sub-agent spawning
- **Instructions**: Yes (CLAUDE.md, skills)
- **Skills**: Yes

## SeeAct
- **Source**: Not in `agents/` directory (traces only)
- **Tools**: (known from SeeAct paper)
  - Browser navigation via visual grounding
  - Screenshot-based action selection
- **Instructions**: Yes
- **Skills**: No

## My Agent
- **Source**: Not in `agents/` directory (traces only)
- **Tools**: Unknown
- **Instructions**: Unknown
- **Skills**: Unknown

---

## Tool Category Summary

| Scaffold | web_search | page_browse | full_browser | python_exec | bash/shell | file_edit | file_search | vision_query | wiki | text_inspect | multi_agent | has_instructions | self_critique |
|----------|-----------|-------------|-------------|-------------|-----------|-----------|-------------|-------------|------|-------------|-------------|-----------------|---------------|
| HAL Generalist | google | visit_page | - | yes | yes | yes | yes | yes | - | yes | - | yes | - |
| CORE-Agent | ddg | visit_page | - | yes | yes | yes | yes | yes | - | yes | - | yes | - |
| HF Open Deep Research | google | yes+nav | - | yes | - | - | - | - | - | yes | yes (sub-agent) | yes | - |
| SWE-Agent | - | - | - | - | yes | yes | yes | - | - | - | - | yes | - |
| Assistantbench Browser | - | - | yes (DOM+JS) | - | - | - | - | vision | - | - | - | yes | - |
| Browser-Use | - | - | yes (DOM+JS) | - | - | - | - | vision | - | - | - | minimal | - |
| SciCode Tool Calling | ddg | - | - | yes | - | - | - | - | yes | - | - | yes | - |
| SciCode Zero Shot | - | - | - | - | - | - | - | - | - | - | - | yes | - |
| SAB Self-Debug | - | - | - | yes | yes | - | - | - | - | - | - | yes | yes |
| Colbench Example | - | - | - | - | - | - | - | - | - | - | - | yes | - |
| Claude Code | - | - | - | - | yes | yes | yes | - | - | - | yes (sub-agents) | yes | yes |
| SeeAct | - | - | yes (visual) | - | - | - | - | vision | - | - | - | yes | - |
| My Agent | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
