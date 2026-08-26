#!/usr/bin/env python3
"""Figure: where correctness enters the model's yes/no channel, and why.

Three curves on the same axis, all n=2000, all the same items:
  reader   - bare text, no question, read at the end of the RG
  placebo  - chat prompt, unrelated yes/no question, read at answer position
  asked    - chat prompt, the real verification question, same position

The gap between `asked` and `placebo` isolates the question-specific effect;
the placebo's own rise isolates the question-independent answer-position
effect. They separate cleanly by depth.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A_C, P_C, R_C, INK, MUTED = "#0072B2", "#D55E00", "#6b7280", "#1f2937", "#9ca3af"

asked = json.loads((ROOT / "runs" / "lens-v1" / "asked_last_full.json").read_text())
plac = json.loads((ROOT / "runs" / "lens-v1" / "margin_lens_placebo.json").read_text())
read = json.loads((ROOT / "runs" / "lens-v1" / "margin_lens.json").read_text())

a = np.array(asked["lens_auroc_by_layer"])
p = np.array(plac["lens_auroc_by_layer"]["last"])
r = np.array(read["lens_auroc_by_layer"]["last"])
x = np.arange(len(a))

fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=150)
ax.fill_between(x, p, a, where=(a > p), color=A_C, alpha=0.12, lw=0)
ax.plot(x, a, color=A_C, lw=2, label='asked: "is this formalization correct?"')
ax.plot(x, p, color=P_C, lw=2, label='placebo: "is the story written in English?"')
ax.plot(x, r, color=R_C, lw=1.8, ls="--", label="reader: bare text, no question")
ax.axhline(0.5, color=MUTED, lw=0.9, ls=":")
ax.annotate("chance", (len(a) + 0.6, 0.5), fontsize=8, color=MUTED, va="center")

gap = a - p
gi = int(np.argmax(gap))
ax.annotate(f"question-specific routing\nmax gap +{gap[gi]:.3f} at block {gi-1}",
            (gi + 1, (a[gi] + p[gi]) / 2), xytext=(37, 0.755),
            fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
ax.annotate("answer-position effect\n(question-independent)",
            (len(a) - 2, p[-1] - 0.008), xytext=(14, 0.625),
            fontsize=8.5, color=P_C, ha="left",
            arrowprops=dict(arrowstyle="->", color=P_C, lw=0.9,
                            connectionstyle="arc3,rad=-0.12"))

ax.set_xlim(0, len(a) + 7)
ax.set_ylim(0.45, 0.79)
ax.set_xlabel("capture row (row R = output of decoder block R−1)", color=INK)
ax.set_ylabel("correctness AUROC from the model's own yes/no axis", color=INK)
ax.set_title("Where correctness enters the verbal channel (Qwen3-32B, n=2000 each)",
             color=INK, fontsize=11)
ax.grid(axis="y", color="#e5e7eb", lw=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=8.5)
fig.tight_layout()
out = ROOT / "routing_by_depth.png"
fig.savefig(out)
print(f"-> {out}  (max gap {gap[gi]:.4f} at row {gi})")
