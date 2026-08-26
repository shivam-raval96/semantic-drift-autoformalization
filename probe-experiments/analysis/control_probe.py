#!/usr/bin/env python3
"""Is the capability<->representation correlation SPECIFIC to correctness?

Alternative explanation for co-emergence: bigger/newer models simply have
richer representations of everything, so any probe scores higher on them. If
so, a probe for a task-IRRELEVANT property should scale with capability just
as correctness does.

Control label: the story's THEME (paint / tea / graft / signal). It is
present in the text, trivially decodable in principle, and has nothing to do
with whether the candidate formalization is correct. Both texts of a problem
share a theme, so it is a problem-level label like correctness.

Reports, per model, at the same layers and under the same law-disjoint
grouping: correctness AUROC vs theme accuracy (4-way, chance 0.25). If theme
accuracy is uniformly high and flat across models while correctness rises,
the co-emergence result is specific rather than generic.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SITE = "mean"
MODELS = [
    ("Llama-3.1-8B", "acts"),
    ("Qwen3.5-4B", "acts-q35-4b"),
    ("Qwen3-32B", "acts-qwen3-32b"),
    ("Qwen3.6-27B", "acts-q36"),
]

rows_meta = {json.loads(l)["problem_id"]: json.loads(l)
             for l in (ROOT / "contrast_v1" / "contrast.jsonl").open()}

out = {"site": SITE, "control_label": "story theme (4-way, chance 0.25)",
       "cv": "GroupKFold(5) by law component", "models": []}

for label, acts_dir in MODELS:
    d = ROOT / "capture" / acts_dir
    if not (d / f"acts-{SITE}.npz").exists():
        print(f"[skip] {label}")
        continue
    meta = json.loads((d / "meta.json").read_text())
    npz = np.load(d / f"acts-{SITE}.npz")
    ids = [i for i in meta["ids"] if i in npz]
    y = np.array([meta["labels"][i] for i in ids])
    pids = [i.split("::")[0] for i in ids]
    groups = np.array([rows_meta[p]["group_lawcc"] for p in pids])
    themes = np.array([rows_meta[p]["theme"] for p in pids])
    acts = np.stack([npz[i] for i in ids]).astype(np.float32)
    n_layers = acts.shape[1]
    # evaluate at the same fractional depths for architectures of different
    # depth, so no model gets more chances to find a maximum
    fracs = [0.5, 0.65, 0.8, 0.95]
    best_corr, best_theme = 0.0, 0.0
    per_depth = []
    for f in fracs:
        L = min(n_layers - 1, int(round(f * (n_layers - 1))))
        h = acts[:, L, :]
        sc = np.zeros(len(y))
        th_pred = np.empty(len(y), dtype=object)
        for tr, te in GroupKFold(5).split(h, y, groups):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=1000))
            m.fit(h[tr], y[tr]); sc[te] = m.decision_function(h[te])
            mt = make_pipeline(StandardScaler(),
                               LogisticRegression(C=1.0, max_iter=1000))
            mt.fit(h[tr], themes[tr]); th_pred[te] = mt.predict(h[te])
        c = float(roc_auc_score(y, sc))
        t = float(accuracy_score(themes, th_pred))
        per_depth.append({"frac_depth": f, "layer": L,
                          "correctness_auroc": round(c, 4),
                          "theme_acc": round(t, 4)})
        best_corr, best_theme = max(best_corr, c), max(best_theme, t)
    out["models"].append({"model": label, "n_layers": n_layers,
                          "per_depth": per_depth,
                          "best_correctness_auroc": round(best_corr, 4),
                          "best_theme_acc": round(best_theme, 4)})
    print(f"{label:14s} correctness(law-disj) {best_corr:.4f} | "
          f"theme acc {best_theme:.4f}")

o = ROOT / "runs" / "analysis-v1"
o.mkdir(parents=True, exist_ok=True)
(o / "control_probe.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"-> {o / 'control_probe.json'}")
