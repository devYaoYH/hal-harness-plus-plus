"""Visualize task discrimination from fitted MIRT K=2 model."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irt_data.irt.data import load_and_split
from irt_data.irt.train import TrainConfig, train

import logging
logging.basicConfig(level=logging.WARNING)

print("Fitting K=2 + features model...")
dataset = load_and_split()
config = TrainConfig(k=2, use_features=True, epochs=200)
model, _, _ = train(dataset, config)

a = model.a.weight.detach().numpy()
d = model.d.weight.detach().numpy().squeeze()
disc = np.linalg.norm(a, axis=1)

idx_to_key = {v: k for k, v in dataset.task_key_to_idx.items()}
rows = []
for i in range(dataset.n_tasks):
    bench, tid = idx_to_key[i]
    rows.append({"benchmark": bench, "task_id": tid, "disc": disc[i],
                 "a0": a[i, 0], "a1": a[i, 1], "easiness": d[i]})
tp = pd.DataFrame(rows)

benchmarks = ["swebench_verified_mini", "gaia", "corebench_hard", "scicode",
              "scienceagentbench", "online_mind2web"]

# --- Fig 1: Discrimination distributions per benchmark ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, bench in zip(axes.flat, benchmarks):
    bg = tp[tp["benchmark"] == bench].sort_values("disc", ascending=False)
    ax.bar(range(len(bg)), bg["disc"].values, color="steelblue", alpha=0.8)
    ax.set_title(bench.replace("_", " "), fontsize=10)
    ax.set_xlabel("Task rank")
    ax.set_ylabel("‖a‖ (discrimination)")

    # Mark 80% cumulative threshold
    cumsum = bg["disc"].cumsum() / bg["disc"].sum()
    n80 = (cumsum.values <= 0.8).sum()
    ax.axvline(n80, color="red", linestyle="--", alpha=0.7)
    ax.text(n80 + 1, ax.get_ylim()[1] * 0.9,
            f"{n80}/{len(bg)} tasks\nfor 80% disc.",
            fontsize=7, color="red")

fig.suptitle("Task Discrimination Magnitude (K=2 MIRT)", fontsize=13)
plt.tight_layout()
plt.savefig("irt_data/discrimination_distributions.png", dpi=150, bbox_inches="tight")
print("Saved: irt_data/discrimination_distributions.png")
plt.close()

# --- Fig 2: 2D discrimination vectors (a0 vs a1) colored by benchmark ---
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.Set2(np.linspace(0, 1, len(benchmarks)))
for bench, color in zip(benchmarks, colors):
    bg = tp[tp["benchmark"] == bench]
    ax.scatter(bg["a0"], bg["a1"], c=[color], s=bg["disc"] * 500 + 10,
               alpha=0.6, label=f"{bench} ({len(bg)})", edgecolors="gray", linewidth=0.3)

ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.set_xlabel("a₀ (discrimination dim 0)")
ax.set_ylabel("a₁ (discrimination dim 1)")
ax.set_title("Task Discrimination Vectors (K=2 MIRT)\nSize = discrimination magnitude")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig("irt_data/discrimination_2d.png", dpi=150, bbox_inches="tight")
print("Saved: irt_data/discrimination_2d.png")
plt.close()

# --- Fig 3: Cost reduction potential ---
fig, ax = plt.subplots(figsize=(10, 5))
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
bar_data = []
for bench in benchmarks:
    bg = tp[tp["benchmark"] == bench].sort_values("disc", ascending=False)
    n_total = len(bg)
    cumsum = bg["disc"].cumsum() / bg["disc"].sum()
    for thresh in thresholds:
        n_needed = (cumsum.values <= thresh).sum() + 1
        bar_data.append({"benchmark": bench, "threshold": f"{thresh:.0%}",
                         "fraction_needed": n_needed / n_total,
                         "n_needed": n_needed, "n_total": n_total})

bdf = pd.DataFrame(bar_data)
x = np.arange(len(benchmarks))
width = 0.13
for i, thresh in enumerate(thresholds):
    subset = bdf[bdf["threshold"] == f"{thresh:.0%}"]
    bars = ax.bar(x + i * width, subset["fraction_needed"].values, width,
                  label=f"{thresh:.0%}", alpha=0.85)
    for bar, row in zip(bars, subset.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{row.n_needed}", ha="center", fontsize=6)

ax.set_xticks(x + width * 2.5)
ax.set_xticklabels([b.replace("_", "\n") for b in benchmarks], fontsize=8)
ax.set_ylabel("Fraction of tasks needed")
ax.set_title("Cost Reduction: Tasks Needed for X% of Total Discrimination")
ax.legend(title="Disc. coverage", fontsize=7)
ax.set_ylim(0, 1.15)
plt.tight_layout()
plt.savefig("irt_data/cost_reduction.png", dpi=150, bbox_inches="tight")
print("Saved: irt_data/cost_reduction.png")
plt.close()
