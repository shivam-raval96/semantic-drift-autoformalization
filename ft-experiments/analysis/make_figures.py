#!/usr/bin/env python3
"""Figures for the v2 writeup.

fig 1  transfer by grammar distance: base vs FT, per model, both input arms.
       The gradient A ~ B-near >> B-far is the experiment's answer, so the
       x-axis is ordered by structural distance from the trained grammar.
fig 2  checkpoint dynamics: trained-grammar skill and never-trained-grammar
       transfer on one axis — tests lock-in (transfer would fall as skill
       rises) against skill-transfer (they rise together).

Reads only committed run dirs; every number traces to a summary.json.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
INK, MUTED, GRID = "#1f2937", "#9ca3af", "#e5e7eb"
BASE_C, FT_C = "#9ca3af", "#0072B2"
MODELS = [("8b", "Llama-3.1-8B"), ("ministral-14b", "Ministral-3-14B"),
          ("qwen3-32b", "Qwen3-32B")]
GRAMMARS = [("", "A\n(trained)"), ("-bnear", "B-near\n(re-skin)"),
            ("-bfar", "B-far\n(new structure)")]
BASE_A = {"8b": "base-v1/8b", "qwen3-32b": "base-v1/qwen3-32b",
          "ministral-14b": "base-v2/ministral-14b"}


def pooled(path):
    p = RUNS / path / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    n = sum(d["n"] for d in s.values())
    return 100 * sum(d["correct"] for d in s.values()) / n


def cell(model, arm, grammar, ft):
    """correct% for one (model, input arm, grammar) cell, base or FT."""
    full = f"{arm}{grammar}"
    if ft:
        return pooled(f"ft-v2/{model}-{full}")
    stem = BASE_A[model] if not grammar else f"base-v2/{model}"
    return pooled(f"{stem}-{full}" if grammar else f"{stem}-{arm}")


def fig_transfer(out):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), dpi=150, sharey=True)
    x = np.arange(len(GRAMMARS))
    for ax, (key, label) in zip(axes, MODELS):
        for arm, mark, ls in (("story", "o", "-"), ("literal", "s", "--")):
            b = [cell(key, arm, g, False) for g, _ in GRAMMARS]
            f = [cell(key, arm, g, True) for g, _ in GRAMMARS]
            ax.plot(x, b, mark, ls=ls, color=BASE_C, ms=6, lw=1.6,
                    label=f"base, {arm}" if key == "8b" else None)
            ax.plot(x, f, mark, ls=ls, color=FT_C, ms=6, lw=2,
                    label=f"after FT, {arm}" if key == "8b" else None)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in GRAMMARS], fontsize=8.5)
        ax.set_title(label, color=INK, fontsize=10)
        ax.set_ylim(-4, 104)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("correct (%), n=777 per point", color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="center left")
    fig.suptitle("Task-pair fine-tuning transfers to a re-skinned grammar, "
                 "partially to a restructured one", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out)
    print(f"-> {out}")


def curve_points(run_name, model, arm):
    steps, vals = [], []
    for d in sorted(RUNS.glob(f"curve-v2-{run_name}-step-*")):
        st = int(d.name.rsplit("step-", 1)[1])
        v = pooled(f"{d.name}/{model}-{arm}-limit200")
        if v is not None:
            steps.append(st)
            vals.append(v)
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(vals)[order]


def fig_dynamics(out):
    specs = [("v2-8b-s0", "8b", "Llama-3.1-8B"),
             ("v2-ministral-14b-s0", "ministral-14b", "Ministral-3-14B"),
             ("v2-qwen3-32b-s0", "qwen3-32b", "Qwen3-32B")]
    avail = [(r, m, l) for r, m, l in specs if len(curve_points(r, m, "story")[0]) > 1]
    if not avail:
        print("no curve data yet")
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(4.1 * len(avail), 3.8),
                             dpi=150, sharey=True, squeeze=False)
    for ax, (run, model, label) in zip(axes[0], avail):
        for arm, color, lab in (("story", FT_C, "grammar A (trained)"),
                                ("story-bfar", "#D55E00", "B-far (never trained)")):
            s, v = curve_points(run, model, arm)
            ax.plot(s, v, "o-", color=color, ms=5, lw=2, label=lab)
        ax.set_title(label, color=INK, fontsize=10)
        ax.set_xlabel("training step", color=INK, fontsize=9)
        ax.set_ylim(-4, 104)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("correct (%), n=200 per point", color=INK, fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="center right")
    fig.suptitle("Transfer rises with skill — no format lock-in",
                 color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out)
    print(f"-> {out}")


if __name__ == "__main__":
    fig_transfer(ROOT / "assets" / "v2_transfer.png")
    fig_dynamics(ROOT / "assets" / "v2_dynamics.png")
