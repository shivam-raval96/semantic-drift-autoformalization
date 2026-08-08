#!/usr/bin/env python3
"""Render the headline figure from runs/probe-v1/probe_results.json:
per-layer out-of-fold AUROC for both sites, with the chance and lexical-floor
reference lines and the law-disjoint result marked at each site's best layer.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / "runs" / "probe-v1" / "probe_results.json").read_text())

COLORS = {"mean": "#0072B2", "last": "#D55E00"}  # validated CVD-safe pair
INK, MUTED = "#1f2937", "#6b7280"

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
for site in ("mean", "last"):
    s = r["sites"][site]
    xs = [p["layer"] for p in s["per_layer"]]
    ys = [p["auroc_oof"] for p in s["per_layer"]]
    ax.plot(xs, ys, color=COLORS[site], linewidth=2, label=f"{site} site")
    bL, rob = s["best_layer"], s["lawcc_robustness"]["auroc_oof"]
    ax.scatter([bL], [rob], color=COLORS[site], marker="X", s=70, zorder=5)
    off = (-10, -13) if site == "mean" else (10, 6)
    ax.annotate(f"law-disjoint {rob:.2f}", (bL, rob), textcoords="offset points",
                xytext=off, ha="right" if site == "mean" else "left",
                fontsize=8, color=COLORS[site])

bow = r["baselines"]["bow_char_tfidf"]["auroc_oof"]
ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
ax.axhline(bow, color=MUTED, linewidth=1, linestyle=":")
ax.annotate("chance 0.50", (33.2, 0.501), fontsize=8, color=MUTED, va="bottom")
ax.annotate(f"bag-of-words {bow:.2f}", (33.2, bow + 0.001), fontsize=8, color=MUTED,
            va="bottom")

ax.set_xlim(0, 39)
ax.set_xticks(range(0, 33, 4))
ax.set_xlabel("layer", color=INK)
ax.set_ylabel("AUROC (out-of-fold)", color=INK)
ax.set_title("Correctness probe by layer: in-distribution vs law-disjoint",
             color=INK, fontsize=11)
ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=9)
fig.tight_layout()
out = ROOT / "runs" / "probe-v1" / "auroc_by_layer.png"
fig.savefig(out)
print(f"-> {out}")
