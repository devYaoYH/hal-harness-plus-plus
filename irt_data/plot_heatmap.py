"""Plot heatmaps of agent accuracy across benchmarks."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({"font.size": 9})

df = pd.read_csv("irt_data/response_matrix.csv")

BENCHMARKS = [
    "swebench_verified_mini", "gaia", "assistantbench", "scicode",
    "scienceagentbench", "corebench_hard", "online_mind2web",
    "colbench_backend_programming",
]
df = df[df["benchmark"].isin(BENCHMARKS)]

def normalize_model(m):
    m = str(m)
    for prefix in ["anthropic/", "openai/", "openrouter/anthropic/", "openrouter/openai/",
                    "openrouter/deepseek/", "together_ai/deepseek-ai/", "gemini/",
                    "google/", "together_ai/", "deepseek-ai/"]:
        if m.startswith(prefix):
            m = m[len(prefix):]
    return m

df["model_norm"] = df["model"].apply(normalize_model)

# --- Heatmap 1: All agents, accuracy per (agent_id, benchmark) ---
acc = df.groupby(["agent_id", "scaffold", "model_norm", "benchmark"])["correct"].mean().reset_index()
acc["agent_label"] = acc["scaffold"].str[:25] + " | " + acc["model_norm"].str[:22]

# For duplicates (same scaffold+model in same benchmark), take mean
pivot_all = acc.pivot_table(index="agent_label", columns="benchmark", values="correct", aggfunc="mean")
pivot_all = pivot_all.reindex(columns=BENCHMARKS)
pivot_all["_mean"] = pivot_all.mean(axis=1)
pivot_all = pivot_all.sort_values("_mean", ascending=False).drop(columns="_mean")

def plot_heatmap(pivot, title, filename, figwidth=14):
    fig, ax = plt.subplots(figsize=(figwidth, max(6, len(pivot) * 0.2)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in pivot.columns], rotation=0, ha="center", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.3 or val > 0.85 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=5, color=color)
    plt.colorbar(im, ax=ax, label="Accuracy", shrink=0.4)
    ax.set_title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename} ({len(pivot)} rows x {len(pivot.columns)} cols)")
    plt.close()

plot_heatmap(pivot_all, "HAL: Agent (scaffold × model) Accuracy by Benchmark",
             "irt_data/heatmap_all_agents.png")

# --- Heatmap 2: Grouped by model (averaging across scaffolds per benchmark) ---
model_acc = df.groupby(["model_norm", "benchmark"])["correct"].mean().reset_index()
pivot_model = model_acc.pivot_table(index="model_norm", columns="benchmark", values="correct")
pivot_model = pivot_model.reindex(columns=BENCHMARKS)
pivot_model["_mean"] = pivot_model.mean(axis=1)
pivot_model = pivot_model.sort_values("_mean", ascending=False).drop(columns="_mean")

plot_heatmap(pivot_model, "HAL: Model Accuracy by Benchmark (averaged across scaffolds)",
             "irt_data/heatmap_by_model.png")

# --- Heatmap 3: Grouped by scaffold (averaging across models per benchmark) ---
scaffold_acc = df.groupby(["scaffold", "benchmark"])["correct"].mean().reset_index()
pivot_scaffold = scaffold_acc.pivot_table(index="scaffold", columns="benchmark", values="correct")
pivot_scaffold = pivot_scaffold.reindex(columns=BENCHMARKS)
pivot_scaffold["_mean"] = pivot_scaffold.mean(axis=1)
pivot_scaffold = pivot_scaffold.sort_values("_mean", ascending=False).drop(columns="_mean")

plot_heatmap(pivot_scaffold, "HAL: Scaffold Accuracy by Benchmark (averaged across models)",
             "irt_data/heatmap_by_scaffold.png")
