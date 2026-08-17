#!/usr/bin/env python3
"""Replicate the orthogonality result on a second model.

Fits the correctness direction from that model's own activations at its best
law-disjoint layer, then measures cosine against the model's J-lens (and
R-lens, if present) yes/no direction, with a random-direction baseline.

    .venv/bin/python analysis/replicate_alignment.py acts-q36 qwen3.6-27b

Caveat to carry into any writeup: the workspace-lenses J/R pair for
Qwen3.6-27B is fitted on n=25 prompts, versus n=1000 for the Neuronpedia
Qwen3-32B lens. It is a weaker instrument, so a null here is weaker evidence
than the 32B result.
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
ACTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "acts-q36"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "qwen3.6-27b"
SITE = "mean"

d = ROOT / "capture" / ACTS_DIR
meta = json.loads((d / "meta.json").read_text())
npz = np.load(d / f"acts-{SITE}.npz")
ids = [i for i in meta["ids"] if i in npz]
y = np.array([meta["labels"][i] for i in ids])
rows_meta = {json.loads(l)["problem_id"]: json.loads(l)
             for l in (ROOT / "contrast_v1" / "contrast.jsonl").open()}
groups = np.array([rows_meta[i.split("::")[0]]["group_lawcc"] for i in ids])
acts = np.stack([npz[i] for i in ids]).astype(np.float32)

lens = np.load(ROOT / "capture" / f"{TARGET}-lens-dirs.npz")
lens_names = [k[:-5] for k in lens.files if k.endswith("_dirs")]

# choose the layer by law-disjoint probe strength, evaluated on a coarse grid
best = (None, 0.0)
n_layers = acts.shape[1]
for L in range(int(0.5 * n_layers), n_layers, 2):
    h = acts[:, L, :]
    sc = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(h, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        m.fit(h[tr], y[tr]); sc[te] = m.decision_function(h[te])
    a = roc_auc_score(y, sc)
    if a > best[1]:
        best = (L, float(a))
L_cap, auc = best
print(f"best law-disjoint layer: capture row {L_cap}, AUROC {auc:.4f}")

m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
m.fit(acts[:, L_cap, :], y)
w = m[-1].coef_[0] / m[0].scale_
p_dir = (w / np.linalg.norm(w)).astype(np.float32)

rng = np.random.RandomState(0)
R = rng.standard_normal((2000, len(p_dir))).astype(np.float32)
R /= np.linalg.norm(R, axis=1, keepdims=True)

out = {"model_target": TARGET, "acts": ACTS_DIR, "site": SITE,
       "n_texts": len(ids), "probe_layer_capture_row": L_cap,
       "probe_lawdisjoint_auroc": round(auc, 4), "lenses": {}}
block = L_cap - 1
for name in lens_names:
    layers = list(lens[f"{name}_layers"])
    if block not in layers:
        block_use = min(layers, key=lambda b: abs(b - block))
    else:
        block_use = block
    v = lens[f"{name}_dirs"][layers.index(block_use)].astype(np.float32)
    v = v / np.linalg.norm(v)
    cos_matched = float(p_dir @ v)
    all_cos = []
    for i, b in enumerate(layers):
        u = lens[f"{name}_dirs"][i].astype(np.float32)
        all_cos.append(abs(float(p_dir @ (u / np.linalg.norm(u)))))
    rb = float(np.percentile(np.abs(R @ v), 95))
    out["lenses"][name] = {
        "matched_block": int(block_use),
        "cosine": round(cos_matched, 4),
        "abs_cosine_mean_all_blocks": round(float(np.mean(all_cos)), 4),
        "abs_cosine_max_all_blocks": round(float(np.max(all_cos)), 4),
        "random_abs_cosine_p95": round(rb, 5),
        "blocks_exceeding_random_p95": int(sum(c > rb for c in all_cos)),
        "n_blocks": len(layers),
    }
    r = out["lenses"][name]
    print(f"[{name}] cos {r['cosine']:+.4f} | all-block |cos| mean "
          f"{r['abs_cosine_mean_all_blocks']} max {r['abs_cosine_max_all_blocks']} "
          f"| exceeding random p95: {r['blocks_exceeding_random_p95']}/{r['n_blocks']}")

o = ROOT / "runs" / "lens-v1"
o.mkdir(parents=True, exist_ok=True)
(o / f"replicate_alignment_{TARGET}.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"-> {o / f'replicate_alignment_{TARGET}.json'}")
