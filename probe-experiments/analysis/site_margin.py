#!/usr/bin/env python3
"""Margin lens on ONE site file, so a partially-downloaded capture is still
usable. Also reports the law-disjoint probe at the same layers for contrast.

    .venv/bin/python analysis/site_margin.py <acts-dir> <site> <out-name> [label]
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ACTS, SITE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LABEL = sys.argv[4] if len(sys.argv) > 4 else ACTS
EPS = 1e-6

d = np.load(ROOT / "capture" / "qwen3-32b-head-rows.npz")
norm_w = d["norm_weight"].astype(np.float32)
yes_rows = np.stack([d[f"row_{i}"] for i in d["yes_ids"]]).astype(np.float32)
no_rows = np.stack([d[f"row_{i}"] for i in d["no_ids"]]).astype(np.float32)

meta = json.loads((ROOT / "capture" / ACTS / "meta.json").read_text())
npz = np.load(ROOT / "capture" / ACTS / f"acts-{SITE}.npz")
ids = [i for i in meta["ids"] if i in npz]
y = np.array([meta["labels"][i] for i in ids])
rows_meta = {json.loads(l)["problem_id"]: json.loads(l)
             for l in (ROOT / "contrast_v1" / "contrast.jsonl").open()}
groups = np.array([rows_meta[i.split("::")[0]]["group_lawcc"] for i in ids])
acts = np.stack([npz[i] for i in ids]).astype(np.float32)
n_layers = acts.shape[1]

lens_auc, probe_pts = [], []
for L in range(n_layers):
    h = acts[:, L, :]
    hn = h if L == n_layers - 1 else (
        h / np.sqrt((h * h).mean(axis=1, keepdims=True) + EPS) * norm_w)
    mg = (hn @ yes_rows.T).max(axis=1) - (hn @ no_rows.T).max(axis=1)
    lens_auc.append(round(float(roc_auc_score(y, mg)), 4))
    if L in (int(0.5 * n_layers), int(0.75 * n_layers), int(0.9 * n_layers),
             n_layers - 1):
        sc = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(h, y, groups):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=1000))
            m.fit(h[tr], y[tr]); sc[te] = m.decision_function(h[te])
        probe_pts.append((L, round(float(roc_auc_score(y, sc)), 4)))

best = int(np.argmax(lens_auc))
out = {"label": LABEL, "acts": ACTS, "site": SITE, "n_texts": len(ids),
       "prompt_file": meta.get("config", {}).get("prompt_file"),
       "text_template": meta.get("text_template"),
       "lens_auroc_by_layer": lens_auc,
       "lens_best": {"layer": best, "auroc": lens_auc[best]},
       "lens_final_layer": lens_auc[-1],
       "probe_lawdisjoint_sampled": probe_pts}
o = ROOT / "runs" / "lens-v1"
o.mkdir(parents=True, exist_ok=True)
(o / OUT).write_text(json.dumps(out, indent=2) + "\n")
print(f"{LABEL:28s} n={len(ids):4d} | lens best {lens_auc[best]:.4f} @L{best} | "
      f"final {lens_auc[-1]:.4f} | probe(law-disj) {probe_pts}")
print(f"-> {o / OUT}")
