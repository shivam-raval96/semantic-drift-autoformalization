#!/usr/bin/env python3
"""Headline figure: per-layer probe AUROC (mean site, linear estimator) for
each probed model, x normalized to fractional depth so different stack
heights overlay. Law-disjoint results marked at each model's best layer;
chance and the law-disjoint lexical floor as reference lines.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("Qwen3-32B", "probe-32b", "#0072B2"),
    ("Llama-3.1-8B", "probe-v1", "#D55E00"),
]
BOW_LAWCC = 0.503  # char TF-IDF under the law-disjoint split (chance)
INK, MUTED = "#1f2937", "#6b7280"

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
for label, run_dir, color in RUNS:
    r = json.loads((ROOT / "runs" / run_dir / "probe_results.json").read_text())
    s = r["sites"]["mean"]
    n = len(s["per_layer"])
    xs = [p["layer"] / (n - 1) for p in s["per_layer"]]
    ys = [p["auroc_oof"] for p in s["per_layer"]]
    ax.plot(xs, ys, color=color, linewidth=2, label=label)
    bx = s["best_layer"] / (n - 1)
    rob = s["lawcc_robustness"]["auroc_oof"]
    ax.scatter([bx], [rob], color=color, marker="X", s=70, zorder=5)
    ax.annotate(f"law-disjoint {rob:.2f}", (bx, rob), textcoords="offset points",
                xytext=(-10, -13), ha="right", fontsize=8, color=color)

ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
ax.annotate("chance 0.50 = lexical floor under law-disjoint (0.503)",
            (1.02, 0.501), fontsize=8, color=MUTED, va="bottom")

ax.set_xlim(0, 1.32)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("fractional depth (layer / n_layers)", color=INK)
ax.set_ylabel("AUROC (out-of-fold)", color=INK)
ax.set_title("Correctness probe by depth: capable 32B vs incapable 8B",
             color=INK, fontsize=11)
ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=9)
fig.tight_layout()
out = ROOT / "auroc_by_layer.png"
fig.savefig(out)
print(f"-> {out}")
