"""
Feature extraction for agents (model × scaffold) and tasks.

Taxonomy is defined declaratively — each feature has a name, type, and extractor.
Missing values are represented as NaN and handled downstream (the model uses
learned embeddings for categorical features and imputation-aware linear layers
for continuous features).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


IRT_DATA_DIR = Path(__file__).resolve().parents[1]


# ── Model taxonomy ──────────────────────────────────────────────────────────
#
# release_ym: year + (month - 1) / 12  (e.g. March 2025 → 2025.167)
# param_count_full_b: total parameter count in billions (MoE = sum of all experts)
# param_count_active_b: active parameters per forward pass (= full for dense models)
# moe: True if Mixture-of-Experts architecture
#
# Counts marked with approximate estimates where not officially published.

MODEL_REGISTRY: dict[str, dict] = {
    # Anthropic — dense architecture, param counts are estimates
    "claude-3-7-sonnet-20250219": {
        "provider": "anthropic", "family": "claude", "release_ym": 2025.083,
        "size_tier": "medium", "reasoning": False,
        "param_count_full_b": 100.0, "param_count_active_b": 100.0, "moe": False,
    },
    "claude-sonnet-4-20250514": {
        "provider": "anthropic", "family": "claude", "release_ym": 2025.333,
        "size_tier": "medium", "reasoning": False,
        "param_count_full_b": 100.0, "param_count_active_b": 100.0, "moe": False,
    },
    "claude-sonnet-4-5-20250929": {
        "provider": "anthropic", "family": "claude", "release_ym": 2025.667,
        "size_tier": "medium", "reasoning": True,
        "param_count_full_b": 100.0, "param_count_active_b": 100.0, "moe": False,
    },
    "claude-sonnet-4-5": {
        "provider": "anthropic", "family": "claude", "release_ym": 2025.667,
        "size_tier": "medium", "reasoning": True,
        "param_count_full_b": 100.0, "param_count_active_b": 100.0, "moe": False,
    },
    "claude-opus-4-20250514": {
        "provider": "anthropic", "family": "claude", "release_ym": 2025.333,
        "size_tier": "large", "reasoning": False,
        "param_count_full_b": 200.0, "param_count_active_b": 200.0, "moe": False,
    },
    "claude-opus-4-1": {
        "provider": "anthropic", "family": "claude", "release_ym": 2025.583,
        "size_tier": "large", "reasoning": False,
        "param_count_full_b": 200.0, "param_count_active_b": 200.0, "moe": False,
    },
    "claude-opus-4-1-20250805": {
        "provider": "anthropic", "family": "claude", "release_ym": 2025.583,
        "size_tier": "large", "reasoning": False,
        "param_count_full_b": 200.0, "param_count_active_b": 200.0, "moe": False,
    },
    "claude-opus-4-5": {
        "provider": "anthropic", "family": "claude", "release_ym": 2025.833,
        "size_tier": "large", "reasoning": True,
        "param_count_full_b": 200.0, "param_count_active_b": 200.0, "moe": False,
    },
    "claude-opus-4-5-20251101": {
        "provider": "anthropic", "family": "claude", "release_ym": 2025.833,
        "size_tier": "large", "reasoning": True,
        "param_count_full_b": 200.0, "param_count_active_b": 200.0, "moe": False,
    },
    "claude-haiku-4-5": {
        "provider": "anthropic", "family": "claude", "release_ym": 2025.75,
        "size_tier": "small", "reasoning": True,
        "param_count_full_b": 20.0, "param_count_active_b": 20.0, "moe": False,
    },
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic", "family": "claude", "release_ym": 2025.75,
        "size_tier": "small", "reasoning": True,
        "param_count_full_b": 20.0, "param_count_active_b": 20.0, "moe": False,
    },
    # OpenAI — dense unless noted; param counts are estimates
    "gpt-4o": {
        "provider": "openai", "family": "gpt", "release_ym": 2024.333,
        "size_tier": "medium", "reasoning": False,
        "param_count_full_b": 200.0, "param_count_active_b": 200.0, "moe": False,
    },
    "gpt-4o-2024-11-20": {
        "provider": "openai", "family": "gpt", "release_ym": 2024.833,
        "size_tier": "medium", "reasoning": False,
        "param_count_full_b": 200.0, "param_count_active_b": 200.0, "moe": False,
    },
    "gpt-4.1-2025-04-14": {
        "provider": "openai", "family": "gpt", "release_ym": 2025.25,
        "size_tier": "medium", "reasoning": False,
        "param_count_full_b": 200.0, "param_count_active_b": 200.0, "moe": False,
    },
    "gpt-5": {
        "provider": "openai", "family": "gpt", "release_ym": 2025.333,
        "size_tier": "large", "reasoning": True,
        # Standard/Main variant: 1.8T–3T total, 250B–500B active (midpoints)
        "param_count_full_b": 2400.0, "param_count_active_b": 375.0, "moe": True,
    },
    "gpt-5-2025-08-07": {
        "provider": "openai", "family": "gpt", "release_ym": 2025.583,
        "size_tier": "large", "reasoning": True,
        "param_count_full_b": 2400.0, "param_count_active_b": 375.0, "moe": True,
    },
    "o1": {
        "provider": "openai", "family": "o-series", "release_ym": 2024.667,
        "size_tier": "large", "reasoning": True,
        "param_count_full_b": 200.0, "param_count_active_b": 200.0, "moe": False,
    },
    "o3-2025-04-16": {
        "provider": "openai", "family": "o-series", "release_ym": 2025.25,
        "size_tier": "large", "reasoning": True,
        "param_count_full_b": 200.0, "param_count_active_b": 200.0, "moe": False,
    },
    "o3-mini-2025-01-31": {
        "provider": "openai", "family": "o-series", "release_ym": 2025.0,
        "size_tier": "small", "reasoning": True,
        "param_count_full_b": 50.0, "param_count_active_b": 50.0, "moe": False,
    },
    "o4-mini-2025-04-16": {
        "provider": "openai", "family": "o-series", "release_ym": 2025.25,
        "size_tier": "small", "reasoning": True,
        "param_count_full_b": 50.0, "param_count_active_b": 50.0, "moe": False,
    },
    "gpt-oss-120b": {
        "provider": "openai", "family": "gpt", "release_ym": np.nan,
        "size_tier": "large", "reasoning": False,
        # Sparse MoE: 117B total, 5.1B active per token
        "param_count_full_b": 117.0, "param_count_active_b": 5.1, "moe": True,
    },
    # Google — Gemini 2.5 Pro and later are MoE; param counts are estimates
    "gemini-2.0-flash": {
        "provider": "google", "family": "gemini", "release_ym": 2025.083,
        "size_tier": "small", "reasoning": False,
        "param_count_full_b": 8.0, "param_count_active_b": 8.0, "moe": False,
    },
    "gemini-2.5-pro-preview-03-25": {
        "provider": "google", "family": "gemini", "release_ym": 2025.167,
        "size_tier": "large", "reasoning": True,
        "param_count_full_b": 1000.0, "param_count_active_b": 100.0, "moe": True,
    },
    "gemini-3-pro-preview": {
        "provider": "google", "family": "gemini", "release_ym": 2026.0,
        "size_tier": "large", "reasoning": True,
        # Ultra-sparse MoE: 2T–4T total, 150B–300B active (midpoints)
        "param_count_full_b": 3000.0, "param_count_active_b": 225.0, "moe": True,
    },
    # DeepSeek — MoE with 671B total / 37B active (published)
    "DeepSeek-R1": {
        "provider": "deepseek", "family": "deepseek", "release_ym": 2025.0,
        "size_tier": "large", "reasoning": True,
        "param_count_full_b": 671.0, "param_count_active_b": 37.0, "moe": True,
    },
    "DeepSeek-V3": {
        "provider": "deepseek", "family": "deepseek", "release_ym": 2024.917,
        "size_tier": "large", "reasoning": False,
        "param_count_full_b": 671.0, "param_count_active_b": 37.0, "moe": True,
    },
    "deepseek-chat-v3-0324": {
        "provider": "deepseek", "family": "deepseek", "release_ym": 2025.167,
        "size_tier": "large", "reasoning": False,
        "param_count_full_b": 671.0, "param_count_active_b": 37.0, "moe": True,
    },
    "deepseek-r1": {
        "provider": "deepseek", "family": "deepseek", "release_ym": 2025.0,
        "size_tier": "large", "reasoning": True,
        "param_count_full_b": 671.0, "param_count_active_b": 37.0, "moe": True,
    },
    "deepseek-r1-0528": {
        "provider": "deepseek", "family": "deepseek", "release_ym": 2025.333,
        "size_tier": "large", "reasoning": True,
        "param_count_full_b": 671.0, "param_count_active_b": 37.0, "moe": True,
    },
    "deepseek-chat-v3.1": {
        "provider": "deepseek", "family": "deepseek", "release_ym": 2025.5,
        "size_tier": "large", "reasoning": False,
        "param_count_full_b": 671.0, "param_count_active_b": 37.0, "moe": True,
    },
}


# ── Scaffold taxonomy ───────────────────────────────────────────────────────
# Tool categories (all boolean):
#   filesystem      — can read/write files and run shell commands
#   web_search      — can query search engines (google, ddg, etc.)
#   page_browse     — can visit/read web pages (text-based)
#   full_browser    — full browser automation (DOM, click, JS, screenshots)
#   browser_vision  — uses screenshots / visual grounding for browser
#   python_exec     — has python interpreter tool
#   file_edit       — has dedicated file edit tool (not just write)
#   file_search     — has file/code search tool (grep, find, etc.)
#   http_requests   — can make raw HTTP/API calls
#   wiki_search     — has wikipedia search tool
#   text_inspect    — has LLM-based text inspection tool
#   vision_query    — can query a vision-language model
#   multi_agent     — can delegate to sub-agents
#   has_instructions — has system prompt / agents.md with domain instructions
#   self_critique   — has explicit critique/retry loop built into harness
#   has_skills      — has pre-defined skill definitions beyond base tools
#
# Continuous/categorical:
#   max_steps       — maximum agentic steps before forced termination (None = unknown)
#   context_strategy — how the agent manages context: "full" | "rag"

_SCAFFOLD_ENTRIES: dict[str, dict] = {
    "Claude Code": {
        "scaffold_type": "cli_agent",
        "filesystem": True, "web_search": False, "page_browse": False,
        "full_browser": False, "browser_vision": False, "python_exec": False,
        "file_edit": True, "file_search": True, "http_requests": False,
        "wiki_search": False, "text_inspect": False, "vision_query": False,
        "multi_agent": True, "has_instructions": True, "self_critique": True,
        "has_skills": True,
        "max_steps": 200, "context_strategy": "full",
    },
    "HAL Generalist Agent": {
        "scaffold_type": "generalist",
        "filesystem": True, "web_search": True, "page_browse": True,
        "full_browser": False, "browser_vision": False, "python_exec": True,
        "file_edit": True, "file_search": True, "http_requests": False,
        "wiki_search": False, "text_inspect": True, "vision_query": True,
        "multi_agent": False, "has_instructions": True, "self_critique": False,
        "has_skills": False,
        "max_steps": 100, "context_strategy": "full",
    },
    "CORE-Agent": {
        "scaffold_type": "generalist",
        "filesystem": True, "web_search": True, "page_browse": True,
        "full_browser": False, "browser_vision": False, "python_exec": True,
        "file_edit": True, "file_search": True, "http_requests": False,
        "wiki_search": False, "text_inspect": True, "vision_query": True,
        "multi_agent": False, "has_instructions": True, "self_critique": True,
        "has_skills": False,
        "max_steps": 100, "context_strategy": "full",
    },
    "HF Open Deep Research": {
        "scaffold_type": "research_agent",
        "filesystem": False, "web_search": True, "page_browse": True,
        "full_browser": False, "browser_vision": False, "python_exec": True,
        "file_edit": False, "file_search": False, "http_requests": False,
        "wiki_search": False, "text_inspect": True, "vision_query": False,
        "multi_agent": True, "has_instructions": True, "self_critique": False,
        "has_skills": False,
        "max_steps": 50, "context_strategy": "rag",
    },
    "SeeAct": {
        "scaffold_type": "web_agent",
        "filesystem": False, "web_search": False, "page_browse": False,
        "full_browser": True, "browser_vision": True, "python_exec": False,
        "file_edit": False, "file_search": False, "http_requests": False,
        "wiki_search": False, "text_inspect": False, "vision_query": False,
        "multi_agent": False, "has_instructions": True, "self_critique": False,
        "has_skills": False,
        "max_steps": 30, "context_strategy": "full",
    },
    "Browser-Use": {
        "scaffold_type": "web_agent",
        "filesystem": False, "web_search": False, "page_browse": False,
        "full_browser": True, "browser_vision": True, "python_exec": False,
        "file_edit": False, "file_search": False, "http_requests": False,
        "wiki_search": False, "text_inspect": False, "vision_query": False,
        "multi_agent": False, "has_instructions": False, "self_critique": False,
        "has_skills": False,
        "max_steps": 50, "context_strategy": "full",
    },
    "Assistantbench Browser Agent": {
        "scaffold_type": "web_agent",
        "filesystem": False, "web_search": False, "page_browse": False,
        "full_browser": True, "browser_vision": True, "python_exec": False,
        "file_edit": False, "file_search": False, "http_requests": False,
        "wiki_search": False, "text_inspect": False, "vision_query": False,
        "multi_agent": False, "has_instructions": True, "self_critique": False,
        "has_skills": False,
        "max_steps": 30, "context_strategy": "full",
    },
    "SWE-Agent": {
        "scaffold_type": "code_agent",
        "filesystem": True, "web_search": False, "page_browse": False,
        "full_browser": False, "browser_vision": False, "python_exec": False,
        "file_edit": True, "file_search": True, "http_requests": False,
        "wiki_search": False, "text_inspect": False, "vision_query": False,
        "multi_agent": False, "has_instructions": True, "self_critique": False,
        "has_skills": False,
        "max_steps": 50, "context_strategy": "full",
    },
    "Scicode Tool Calling Agent": {
        "scaffold_type": "code_agent",
        "filesystem": False, "web_search": True, "page_browse": False,
        "full_browser": False, "browser_vision": False, "python_exec": True,
        "file_edit": False, "file_search": False, "http_requests": False,
        "wiki_search": True, "text_inspect": False, "vision_query": False,
        "multi_agent": False, "has_instructions": True, "self_critique": False,
        "has_skills": False,
        "max_steps": 30, "context_strategy": "full",
    },
    "Scicode Zero Shot Agent": {
        "scaffold_type": "code_agent",
        "filesystem": False, "web_search": False, "page_browse": False,
        "full_browser": False, "browser_vision": False, "python_exec": False,
        "file_edit": False, "file_search": False, "http_requests": False,
        "wiki_search": False, "text_inspect": False, "vision_query": False,
        "multi_agent": False, "has_instructions": True, "self_critique": False,
        "has_skills": False,
        "max_steps": 1, "context_strategy": "full",
    },
    "SAB Self-Debug": {
        "scaffold_type": "code_agent",
        "filesystem": True, "web_search": False, "page_browse": False,
        "full_browser": False, "browser_vision": False, "python_exec": True,
        "file_edit": False, "file_search": False, "http_requests": False,
        "wiki_search": False, "text_inspect": False, "vision_query": False,
        "multi_agent": False, "has_instructions": True, "self_critique": True,
        "has_skills": False,
        "max_steps": 30, "context_strategy": "full",
    },
    "Colbench Example Agent": {
        "scaffold_type": "code_agent",
        "filesystem": False, "web_search": False, "page_browse": False,
        "full_browser": False, "browser_vision": False, "python_exec": False,
        "file_edit": False, "file_search": False, "http_requests": False,
        "wiki_search": False, "text_inspect": False, "vision_query": False,
        "multi_agent": False, "has_instructions": True, "self_critique": False,
        "has_skills": False,
        "max_steps": 20, "context_strategy": "full",
    },
    "My Agent": {
        "scaffold_type": "generalist",
        "filesystem": True, "web_search": False, "page_browse": False,
        "full_browser": False, "browser_vision": False, "python_exec": True,
        "file_edit": True, "file_search": False, "http_requests": False,
        "wiki_search": False, "text_inspect": False, "vision_query": False,
        "multi_agent": False, "has_instructions": True, "self_critique": False,
        "has_skills": False,
        "max_steps": 50, "context_strategy": "full",
    },
}

# Name aliases: map variant names to canonical scaffold entries
_SCAFFOLD_ALIASES: dict[str, str] = {
    # HAL Generalist variants
    "HAL Generalist": "HAL Generalist Agent",
    "Hal Generalist Agent": "HAL Generalist Agent",
    "HAL-Generalist-Agent": "HAL Generalist Agent",
    "Hal Generalist Agent Opus 4.5": "HAL Generalist Agent",
    "Hal Generalist Agent Opus 4.5 high": "HAL Generalist Agent",
    # CORE-Agent variants
    "Core Agent": "CORE-Agent",
    "Core-Agent": "CORE-Agent",
    "Coreagent": "CORE-Agent",
    "Core Agent Opus 4.5": "CORE-Agent",
    "Core Agent Opus 4.5 high": "CORE-Agent",
    # Browser-Use variants
    "Browser-Use_test": "Browser-Use",
    # SciCode variants
    "SciCode Tool Calling Agent": "Scicode Tool Calling Agent",
    # SAB Self-Debug variants (model name in scaffold name)
    "SAB Self-Debug Claude-3-7": "SAB Self-Debug",
    "SAB Self-Debug Claude-3-7 high": "SAB Self-Debug",
    "SAB Self-Debug Claude-3-7 low": "SAB Self-Debug",
    "SAB Self-Debug Claude-Haiku-4-5": "SAB Self-Debug",
    "SAB Self-Debug Claude-Haiku-4-5 High": "SAB Self-Debug",
    "SAB Self-Debug Claude-Opus-4-1": "SAB Self-Debug",
    "SAB Self-Debug Claude-Opus-4-1-High": "SAB Self-Debug",
    "SAB Self-Debug Claude-Sonnet-4-5": "SAB Self-Debug",
    "SAB Self-Debug Claude-Sonnet-4-5 High": "SAB Self-Debug",
    "SAB Self-Debug DS-V3": "SAB Self-Debug",
    "SAB Self-Debug GPT-5-Medium": "SAB Self-Debug",
    "SAB Self-Debug gemini-2-0-flash": "SAB Self-Debug",
    "SAB Self-Debug gemini-2-5-pro": "SAB Self-Debug",
    "SAB Self-Debug o3 medium": "SAB Self-Debug",
    "SAB Self-Debug o4-mini high": "SAB Self-Debug",
    "SAB Self-Debug o4-mini low": "SAB Self-Debug",
    # Colbench variants
    "colbench_backend_programming colbench_example_agent": "Colbench Example Agent",
    "colbench_example_agent_gpt41": "Colbench Example Agent",
    # Lowercase hal generalist variants (from older trace naming)
    "hal_generalist_agent_deepseekaideepseekr1": "HAL Generalist Agent",
    "hal_generalist_agent_deepseekaideepseekv3": "HAL Generalist Agent",
    "hal_generalist_agent_gemini20flash": "HAL Generalist Agent",
    "hal_generalist_agent_o3mini20250131_high": "HAL Generalist Agent",
    "hal_generalist_agent_o3mini20250131_low": "HAL Generalist Agent",
    # Lowercase sab variants
    "sab_selfdebug_dsr1": "SAB Self-Debug",
    "sab_selfdebug_gpt_41": "SAB Self-Debug",
    # Lowercase coreagent
    "coreagent": "CORE-Agent",
}

SCAFFOLD_TOOL_FIELDS = [
    "filesystem", "web_search", "page_browse", "full_browser", "browser_vision",
    "python_exec", "file_edit", "file_search", "http_requests", "wiki_search",
    "text_inspect", "vision_query", "multi_agent", "has_instructions",
    "self_critique", "has_skills",
]


def _resolve_scaffold(raw: str) -> tuple[str, dict]:
    """Resolve a scaffold name through aliases and return (canonical_name, entry)."""
    clean = re.sub(r"\s*\(.*?\)\s*$", "", raw, flags=re.DOTALL).strip()
    canonical = _SCAFFOLD_ALIASES.get(clean, clean)
    entry = _SCAFFOLD_ENTRIES.get(canonical, {})
    return canonical, entry


# ── Benchmark taxonomy ──────────────────────────────────────────────────────

BENCHMARK_REGISTRY: dict[str, dict] = {
    "swebench_verified_mini": {
        "domain": "software_engineering", "task_type": "bug_fix",
        "requires_code": True, "requires_web": False, "requires_reasoning": True,
    },
    "gaia": {
        "domain": "general", "task_type": "qa",
        "requires_code": False, "requires_web": True, "requires_reasoning": True,
    },
    "assistantbench": {
        "domain": "general", "task_type": "web_task",
        "requires_code": False, "requires_web": True, "requires_reasoning": False,
    },
    "scicode": {
        "domain": "science", "task_type": "code_generation",
        "requires_code": True, "requires_web": False, "requires_reasoning": True,
    },
    "scienceagentbench": {
        "domain": "science", "task_type": "experiment",
        "requires_code": True, "requires_web": False, "requires_reasoning": True,
    },
    "corebench_hard": {
        "domain": "science", "task_type": "reproducibility",
        "requires_code": True, "requires_web": False, "requires_reasoning": True,
    },
    "online_mind2web": {
        "domain": "web", "task_type": "web_navigation",
        "requires_code": False, "requires_web": True, "requires_reasoning": False,
    },
    "colbench_backend_programming": {
        "domain": "software_engineering", "task_type": "code_generation",
        "requires_code": True, "requires_web": False, "requires_reasoning": True,
    },
    "taubench_airline": {
        "domain": "customer_service", "task_type": "domain_tool_use",
        "requires_code": False, "requires_web": False, "requires_reasoning": True,
    },
    "usaco": {
        "domain": "software_engineering", "task_type": "competitive_programming",
        "requires_code": True, "requires_web": False, "requires_reasoning": True,
    },
}


def _normalize_model_name(raw: str) -> str:
    """Strip provider routing prefixes to get canonical model id."""
    for prefix in [
        "anthropic/", "openai/", "openrouter/anthropic/", "openrouter/openai/",
        "openrouter/deepseek/", "together_ai/deepseek-ai/", "together_ai/",
        "gemini/", "google/", "deepseek-ai/",
    ]:
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    return raw.rstrip(":thinking")


def extract_model_features(model_raw: str) -> dict:
    """Extract features for a model string. Returns dict with NaN for unknowns."""
    name = _normalize_model_name(model_raw)
    entry = MODEL_REGISTRY.get(name, {})
    return {
        "model_name": name,
        "model_provider": entry.get("provider", np.nan),
        "model_family": entry.get("family", np.nan),
        "model_release_ym": entry.get("release_ym", np.nan),
        "model_size_tier": entry.get("size_tier", np.nan),
        "model_reasoning": entry.get("reasoning", np.nan),
        "model_param_count_full_b": entry.get("param_count_full_b", np.nan),
        "model_param_count_active_b": entry.get("param_count_active_b", np.nan),
        "model_moe": entry.get("moe", np.nan),
    }


def extract_scaffold_features(scaffold: str) -> dict:
    """Extract features for a scaffold name."""
    canonical, entry = _resolve_scaffold(scaffold)
    result = {
        "scaffold_name": canonical,
        "scaffold_type": entry.get("scaffold_type", np.nan),
        "scaffold_context_strategy": entry.get("context_strategy", np.nan),
        "scaffold_max_steps": entry.get("max_steps", np.nan),
    }
    for f in SCAFFOLD_TOOL_FIELDS:
        result[f"scaffold_{f}"] = entry.get(f, np.nan)
    return result


def extract_benchmark_features(benchmark: str) -> dict:
    """Extract features for a benchmark name."""
    entry = BENCHMARK_REGISTRY.get(benchmark, {})
    return {
        "bench_domain": entry.get("domain", np.nan),
        "bench_task_type": entry.get("task_type", np.nan),
        "bench_requires_code": entry.get("requires_code", np.nan),
        "bench_requires_web": entry.get("requires_web", np.nan),
        "bench_requires_reasoning": entry.get("requires_reasoning", np.nan),
    }


def _load_csv_if_exists(filename: str) -> pd.DataFrame | None:
    path = IRT_DATA_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


@dataclass
class FeatureSchema:
    """Describes the full feature set. Used by the model to build input layers."""

    categorical_agent: list[str] = field(default_factory=list)
    continuous_agent: list[str] = field(default_factory=list)
    boolean_agent: list[str] = field(default_factory=list)
    categorical_task: list[str] = field(default_factory=list)
    continuous_task: list[str] = field(default_factory=list)
    boolean_task: list[str] = field(default_factory=list)
    vocab_sizes: dict[str, int] = field(default_factory=dict)


def build_feature_tables(response_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, FeatureSchema]:
    """
    Build agent and task feature tables from the response matrix.

    Returns:
        agent_features: DataFrame indexed by agent_id with model/scaffold features
        task_features: DataFrame indexed by (benchmark, task_id) with benchmark features
        schema: FeatureSchema describing the feature layout
    """
    agent_meta = (
        response_df.groupby("agent_id")[["scaffold", "model"]]
        .first()
        .reset_index()
    )

    model_metadata = _load_csv_if_exists("model_metadata.csv")
    harness_metadata = _load_csv_if_exists("harness_metadata.csv")
    harness_affordances = _load_csv_if_exists("harness_affordances.csv")

    if model_metadata is not None and {"raw_model", "normalized_model"}.issubset(model_metadata.columns):
        model_feats = (
            agent_meta[["model"]]
            .merge(model_metadata, left_on="model", right_on="raw_model", how="left")
            .rename(columns={
                "normalized_model": "model_name",
                "provider": "model_provider",
                "model_family": "model_family",
            })
        )
        # Keep the old numeric feature names for model compatibility. They are
        # optional and remain NaN when the normalized metadata does not provide
        # release/size estimates.
        for col in [
            "model_release_ym", "model_size_tier", "model_reasoning",
            "model_param_count_full_b", "model_param_count_active_b", "model_moe",
        ]:
            if col not in model_feats:
                model_feats[col] = np.nan
        model_feats = model_feats[[
            "model_name", "model_provider", "model_family", "model_release_ym",
            "model_size_tier", "model_reasoning", "model_param_count_full_b",
            "model_param_count_active_b", "model_moe",
        ]]
        # Fall back row-wise for any labels absent from the metadata file.
        missing = model_feats["model_name"].isna()
        if missing.any():
            fallback = pd.DataFrame([extract_model_features(m) for m in agent_meta.loc[missing, "model"]])
            model_feats.loc[missing, fallback.columns] = fallback.values
    else:
        model_feats = pd.DataFrame([extract_model_features(m) for m in agent_meta["model"]])

    if harness_metadata is not None and {"raw_scaffold", "normalized_harness"}.issubset(harness_metadata.columns):
        harness_feats = (
            agent_meta[["scaffold"]]
            .merge(harness_metadata, left_on="scaffold", right_on="raw_scaffold", how="left")
            .rename(columns={
                "normalized_harness": "scaffold_name",
                "tool_exposure_mechanism": "scaffold_tool_exposure_mechanism",
                "tool_exposure_strength": "scaffold_tool_exposure_strength",
            })
        )
        if harness_affordances is not None and "normalized_harness" in harness_affordances.columns:
            # Prefer canonical normalized-harness affordances over per-raw-label
            # copies in harness_metadata.
            harness_feats = harness_feats.drop(
                columns=[c for c in harness_feats.columns if c.startswith("affordance_")],
                errors="ignore",
            )
            harness_feats = harness_feats.merge(
                harness_affordances,
                left_on="scaffold_name",
                right_on="normalized_harness",
                how="left",
            )
            harness_feats = harness_feats.drop(columns=["normalized_harness"], errors="ignore")
        harness_feats["scaffold_type"] = harness_feats.get(
            "scaffold_tool_exposure_mechanism", np.nan
        )
        harness_feats["scaffold_context_strategy"] = np.nan
        harness_feats["scaffold_max_steps"] = np.nan
        for col in [
            "scaffold_name", "scaffold_type", "scaffold_tool_exposure_mechanism",
            "scaffold_tool_exposure_strength", "scaffold_context_strategy",
            "scaffold_max_steps",
        ]:
            if col not in harness_feats:
                harness_feats[col] = np.nan
        # Fall back row-wise for any labels absent from the metadata file.
        missing = harness_feats["scaffold_name"].isna()
        if missing.any():
            fallback = pd.DataFrame([extract_scaffold_features(s) for s in agent_meta.loc[missing, "scaffold"]])
            for col in fallback.columns:
                if col not in harness_feats:
                    harness_feats[col] = np.nan
                harness_feats.loc[missing, col] = fallback[col].values
    else:
        harness_feats = pd.DataFrame([extract_scaffold_features(s) for s in agent_meta["scaffold"]])

    # Backfill legacy scaffold_* affordance columns from the new taxonomy where
    # the mapping is direct enough to preserve old scripts.
    legacy_from_new = {
        "scaffold_python_exec": "affordance_code_execution",
        "scaffold_file_edit": "affordance_file_edit",
        "scaffold_file_search": "affordance_file_read",
        "scaffold_web_search": "affordance_web_search",
        "scaffold_page_browse": "affordance_web_page_visit",
        "scaffold_full_browser": "affordance_browser_control",
        "scaffold_browser_vision": "affordance_browser_control",
        "scaffold_vision_query": "affordance_vision",
        "scaffold_filesystem": "affordance_file_read",
        "scaffold_http_requests": "affordance_benchmark_api_tools",
        "scaffold_self_critique": "affordance_self_debug_feedback",
    }
    for legacy_col, source_col in legacy_from_new.items():
        if legacy_col not in harness_feats and source_col in harness_feats:
            harness_feats[legacy_col] = harness_feats[source_col]
    for legacy_col in [f"scaffold_{f}" for f in SCAFFOLD_TOOL_FIELDS]:
        if legacy_col not in harness_feats:
            harness_feats[legacy_col] = np.nan

    agent_features = pd.concat([agent_meta[["agent_id"]], model_feats, harness_feats], axis=1)
    agent_features = agent_features.set_index("agent_id")

    task_meta = (
        response_df.groupby(["benchmark", "task_id"])
        .size()
        .reset_index(name="n_agents")
    )
    bench_feats = pd.DataFrame([extract_benchmark_features(b) for b in task_meta["benchmark"]])
    task_features = pd.concat([task_meta, bench_feats], axis=1)
    task_features = task_features.set_index(["benchmark", "task_id"])

    scaffold_bool_fields = [f"scaffold_{f}" for f in SCAFFOLD_TOOL_FIELDS]
    affordance_bool_fields = sorted(
        c for c in agent_features.columns if c.startswith("affordance_")
    )

    schema = FeatureSchema(
        # model_name and scaffold_name kept as metadata in the DataFrame but excluded
        # from model inputs — they are identity columns that don't generalize to new agents
        categorical_agent=[
            "model_provider", "model_family", "model_size_tier",
            "scaffold_type", "scaffold_context_strategy",
            "scaffold_tool_exposure_mechanism",
        ],
        continuous_agent=[
            "model_release_ym", "model_param_count_full_b", "model_param_count_active_b",
            "scaffold_max_steps", "scaffold_tool_exposure_strength",
        ],
        boolean_agent=["model_reasoning", "model_moe"] + scaffold_bool_fields + affordance_bool_fields,
        categorical_task=["bench_domain", "bench_task_type"],
        boolean_task=["bench_requires_code", "bench_requires_web", "bench_requires_reasoning"],
    )

    for col in schema.categorical_agent:
        if col in agent_features.columns:
            schema.vocab_sizes[col] = agent_features[col].nunique() + 1
    for col in schema.categorical_task:
        if col in task_features.columns:
            schema.vocab_sizes[col] = task_features[col].nunique() + 1

    return agent_features, task_features, schema
