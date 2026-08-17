#!/usr/bin/env python3
"""Grammar-FT effect figure: pooled correct% and unparseable% (all 777
problems), base vs fine-tuned, per model and arm. Dumbbell per row; data
read from the eval summary.json files, nothing hand-entered.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FT = Path(__file__).resolve().parents[2] / "ft-experiments" / "runs"
BASE_C, FT_C, MUTED, INK = "#0072B2", "#D55E00", "#9ca3af", "#1f2937"
ROWS = [
    ("8B", "story"), ("8B", "literal"), ("8B", "two-stage"),
    ("32B", "story"), ("32B", "literal"), ("32B", "two-stage"),
]
DIRS = {"8B": "8b", "32B": "qwen3-32b"}


def pooled(tag, model, arm):
    s = json.loads((FT / tag / f"{DIRS[model]}-{arm}" / "summary.json").read_text())
    n = sum(t["n"] for t in s.values())
    return (100 * sum(t["correct"] for t in s.values()) / n,
            100 * sum(t["unparseable"] for t in s.values()) / n)


data = {(m, a): (pooled("base-v1", m, a), pooled("ft-v1", m, a)) for m, a in ROWS}

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=150, sharey=True)
ys = range(len(ROWS))[::-1]
for ax, metric, title in ((axes[0], 0, "correct %"), (axes[1], 1, "unparseable %")):
    for y, (m, a) in zip(ys, ROWS):
        b, f = data[(m, a)][0][metric], data[(m, a)][1][metric]
        ax.plot([b, f], [y, y], color=MUTED, linewidth=1.4, zorder=1)
        ax.scatter([b], [y], color=BASE_C, s=46, zorder=2)
        ax.scatter([f], [y], color=FT_C, s=46, zorder=2)
        ax.annotate(f"{b:.0f}", (b, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5, color=BASE_C)
        ax.annotate(f"{f:.0f}", (f, y), textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=7.5, color=FT_C)
    ax.set_title(title, color=INK, fontsize=10)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6)
    ax.set_xlim(-4, 60)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
axes[0].set_yticks(list(ys))
axes[0].set_yticklabels([f"{m} {a}" for m, a in ROWS], fontsize=9, color=INK)
axes[0].scatter([], [], color=BASE_C, label="base")
axes[0].scatter([], [], color=FT_C, label="grammar-FT")
axes[0].legend(loc="lower right", frameon=False, fontsize=9)
fig.suptitle("Grammar-only fine-tuning: pooled effect on eval_v1 (n=777)",
             color=INK, fontsize=11)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "ft_effect.png"
fig.savefig(out)
print(f"-> {out}")
for (m, a), ((bc, bu), (fc, fu)) in data.items():
    print(f"{m:3s} {a:9s} correct {bc:5.1f} -> {fc:4.1f}   unparse {bu:5.1f} -> {fu:4.1f}")
