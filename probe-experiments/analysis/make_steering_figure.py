#!/usr/bin/env python3
"""Figure: the steering dose-response with its positive control.

Left panel  - mean yes/no margin vs injection strength, three directions.
Right panel - what each direction reads out as in vocabulary space (max
              J-lens readout score), the same three directions plus the
              random null.

The point of the figure: the harness demonstrably moves the model (J-lens
direction, +24 logits, yes-rate 6%->100%), while the supervised correctness
direction moves it less than norm-matched random noise.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
J_C, P_C, R_C, INK, MUTED = "#0072B2", "#D55E00", "#9ca3af", "#1f2937", "#6b7280"

steer = {}
for src, path in (("probe", "qwen3-32b.json"), ("jlens", "qwen3-32b-jlens.json")):
    p = ROOT / "runs" / "steer-v1" / path
    if p.exists():
        steer[src] = json.loads(p.read_text())["conditions"]
if "probe" not in steer or "jlens" not in steer:
    raise SystemExit(f"need both steering runs; have {list(steer)}")


def series(conds, kind):
    pts = []
    for k, v in conds.items():
        name, alpha = k.split("@")
        if name == kind:
            pts.append((float(alpha), v["mean_margin"]))
    return sorted(pts)


fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=150,
                         gridspec_kw={"width_ratios": [1.35, 1]})

ax = axes[0]
for src, color, label in (("jlens", J_C, "J-lens direction (positive control)"),
                          ("probe", P_C, "our correctness direction")):
    xs, ys = zip(*series(steer[src], "direction"))
    ax.plot(xs, ys, "o-", color=color, linewidth=2, markersize=5, label=label)
xs, ys = zip(*series(steer["probe"], "random"))
ax.plot(xs, ys, "s--", color=R_C, linewidth=1.6, markersize=4,
        label="random, norm-matched")
ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
ax.set_xlabel("injection strength α (× mean residual norm)", color=INK)
ax.set_ylabel("mean yes/no margin (logits)", color=INK)
ax.set_title("Steering the same site, three directions", color=INK, fontsize=10)
ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=8)

vocab_p = ROOT / "runs" / "lens-v1" / "direction_vocabulary.json"
ax2 = axes[1]
if vocab_p.exists():
    v = json.loads(vocab_p.read_text())["readouts"]
    order = [("jlens_yesno_POSITIVE_CONTROL", "J-lens yes/no\n(positive control)", J_C),
             ("probe", "our correctness\ndirection", P_C),
             ("random", "random direction\n(null)", R_C)]
    names = [o[1] for o in order if o[0] in v]
    vals = [v[o[0]]["score_max"] for o in order if o[0] in v]
    cols = [o[2] for o in order if o[0] in v]
    bars = ax2.bar(range(len(vals)), vals, color=cols, width=0.6)
    for b, val in zip(bars, vals):
        ax2.annotate(f"{val:.3f}", (b.get_x() + b.get_width() / 2, val),
                     ha="center", va="bottom", fontsize=8, color=INK)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=7.5, color=INK)
    ax2.set_ylabel("max vocabulary readout", color=INK)
    ax2.set_title("What each direction is disposed to say", color=INK, fontsize=10)
    ax2.set_ylim(0, 1.12)
    ax2.grid(axis="y", color="#e5e7eb", linewidth=0.6)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

fig.suptitle("Correctness is represented in a direction the model cannot say",
             color=INK, fontsize=11)
fig.tight_layout()
out = ROOT / "steering_mechanism.png"
fig.savefig(out)
print(f"-> {out}")
