#!/usr/bin/env python3
"""Two follow-ups on the Qwen3-32B result, both local.

1. Direction stability: probes fit independently on each law-disjoint fold
   (mean site, best layer) - pairwise cosine of their weight vectors. One
   stable direction supports the steering premise; near-orthogonal fold
   directions would refute it.
2. Readout alignment: Spearman correlation between the probe's out-of-fold
   score and the model's own yes/no logit margin on the 300 gate texts -
   does the model's readout consult what the probe reads?

Writes runs/probe-32b/direction_analysis.json and the full-data direction
(runs/probe-32b/direction_L61_mean.npz) for the steering stage.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
LAYER, SITE = 61, "mean"

meta = json.loads((ROOT / "capture" / "acts-qwen3-32b" / "meta.json").read_text())
ids = meta["ids"]
y = np.array([meta["labels"][i] for i in ids])
rows = {json.loads(l)["problem_id"]: json.loads(l)
        for l in (ROOT / "contrast_v1" / "contrast.jsonl").open()}
problems = np.array([i.split("::")[0] for i in ids])
lawcc = np.array([rows[p]["group_lawcc"] for p in problems])

npz = np.load(ROOT / "capture" / "acts-qwen3-32b" / f"acts-{SITE}.npz")
X = np.stack([npz[i][LAYER] for i in ids]).astype(np.float32)

def fit(idx):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    m.fit(X[idx], y[idx])
    return m

# 1. Fold directions under the law-disjoint split.
dirs = []
for tr, _ in GroupKFold(5).split(X, y, lawcc):
    w = fit(tr)[-1].coef_[0]
    dirs.append(w / np.linalg.norm(w))
cos = np.array([[float(a @ b) for b in dirs] for a in dirs])
off = cos[np.triu_indices(5, k=1)]

# Full-data direction for steering (raw-activation space: w / sigma).
m_full = fit(np.arange(len(y)))
w_std = m_full[-1].coef_[0] / m_full[0].scale_
w_unit = w_std / np.linalg.norm(w_std)
np.savez(ROOT / "runs" / "probe-32b" / "direction_L61_mean.npz",
         direction=w_unit.astype(np.float32), layer=LAYER, site=SITE)

# 2. Readout alignment on the gate sample (OOF probe scores by problem split).
scores = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, problems):
    scores[te] = fit(tr).decision_function(X[te])
margins = {r["id"]: r["margin"] for r in json.loads(
    (ROOT / "runs" / "verify-v1" / "qwen3-32b-margin.json").read_text())["rows"]}
common = [i for i in ids if i in margins]
a = np.array([scores[ids.index(i)] for i in common])
b = np.array([margins[i] for i in common])
ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
spearman = float(np.corrcoef(ra, rb)[0, 1])

out = {
    "layer": LAYER, "site": SITE,
    "fold_direction_cosines": {"mean": round(float(off.mean()), 4),
                               "min": round(float(off.min()), 4),
                               "max": round(float(off.max()), 4)},
    "readout_alignment": {"n_texts": len(common),
                          "spearman_probe_vs_margin": round(spearman, 4)},
}
(ROOT / "runs" / "probe-32b" / "direction_analysis.json").write_text(
    json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
