#!/usr/bin/env python3
"""Is our supervised correctness direction aligned with the model's
verbalization pathway?

J-lens direction at block L: v_L = (mean W_U[yes] - mean W_U[no]) @ J_L, the
residual direction that most raises the model's disposition to eventually say
"yes" over "no". Our probe direction: the supervised correctness readout.

Index mapping (explicit, this is easy to get wrong):
  capture row R  =  output of decoder block R-1   (row 0 = embeddings)
  J-lens block L =  decoder block L
  so capture row R corresponds to J-lens block L = R-1.

Three measurements:
  1. cosine(probe direction, J direction) at the matched layer, and across
     layers. Near-zero => the probe reads something the verbalization pathway
     does not use (workspace explanation for the steering null).
  2. The J direction used AS a probe: project activations onto it and score
     AUROC vs labels, reader vs asked. Tests whether the verbalization
     direction itself carries correctness information.
  3. How much of the probe direction is explained by the J direction
     (projection fraction).
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
L_CAPTURE, SITE = 61, "mean"
L_BLOCK = L_CAPTURE - 1

lens = np.load(ROOT / "capture" / "qwen3-32b-lens-dirs.npz")
jl_layers = list(lens["jlens_layers"])
jl_dirs = lens["jlens_dirs"].astype(np.float32)
assert L_BLOCK in jl_layers, f"block {L_BLOCK} not in lens layers {jl_layers[:3]}..."
j_at = {int(L): jl_dirs[i] / np.linalg.norm(jl_dirs[i])
        for i, L in enumerate(jl_layers)}

probe = np.load(ROOT / "runs" / "probe-32b" / "direction_L61_mean.npz")
p_dir = probe["direction"].astype(np.float32)
p_dir = p_dir / np.linalg.norm(p_dir)

cos_matched = float(p_dir @ j_at[L_BLOCK])
cos_all = {int(L): round(float(p_dir @ j_at[int(L)]), 4) for L in jl_layers}
best_L = max(cos_all, key=lambda k: abs(cos_all[k]))

# random-direction baseline for |cosine| in 5120 dims
rng = np.random.RandomState(0)
R = rng.standard_normal((2000, len(p_dir))).astype(np.float32)
R /= np.linalg.norm(R, axis=1, keepdims=True)
rand_cos = np.abs(R @ j_at[L_BLOCK])

out = {
    "capture_row": L_CAPTURE, "jlens_block": L_BLOCK, "site": SITE,
    "cosine_probe_vs_jlens_matched_layer": round(cos_matched, 4),
    "cosine_by_block": cos_all,
    "max_abs_cosine": {"block": best_L, "cos": cos_all[best_L]},
    "random_baseline_abs_cosine": {
        "mean": round(float(rand_cos.mean()), 5),
        "p95": round(float(np.percentile(rand_cos, 95)), 5),
        "max": round(float(rand_cos.max()), 5),
    },
    "probe_variance_along_j": round(cos_matched ** 2, 6),
}

# 2. J direction used as a classifier, reader vs asked
for mode, acts_dir in (("reader", "acts-qwen3-32b"), ("asked", "acts-asked-32b")):
    d = ROOT / "capture" / acts_dir
    if not (d / "meta.json").exists():
        continue
    meta = json.loads((d / "meta.json").read_text())
    ids = meta["ids"]
    y = np.array([meta["labels"][i] for i in ids])
    npz = np.load(d / f"acts-{SITE}.npz")
    acts = np.stack([npz[i] for i in ids]).astype(np.float32)
    j_auc, p_auc = [], []
    for R_ in range(acts.shape[1]):
        L = R_ - 1
        h = acts[:, R_, :]
        if L in j_at:
            j_auc.append(round(float(roc_auc_score(y, h @ j_at[L])), 4))
        else:
            j_auc.append(None)
        p_auc.append(round(float(roc_auc_score(y, h @ p_dir)), 4))
    valid = [(i, v) for i, v in enumerate(j_auc) if v is not None]
    out[f"{mode}_j_direction_auroc"] = j_auc
    out[f"{mode}_probe_direction_auroc"] = p_auc
    out[f"{mode}_summary"] = {
        "n_texts": meta["n_texts"],
        "j_dir_best": max(valid, key=lambda t: abs(t[1] - 0.5)),
        "probe_dir_best": max(enumerate(p_auc), key=lambda t: abs(t[1] - 0.5)),
    }
    print(f"[{mode}] J-direction best {out[f'{mode}_summary']['j_dir_best']} | "
          f"probe-direction best {out[f'{mode}_summary']['probe_dir_best']}")

o = ROOT / "runs" / "lens-v1"
o.mkdir(parents=True, exist_ok=True)
(o / "jspace_alignment.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"cosine(probe, J) at matched layer = {cos_matched:.4f} "
      f"(random |cos| p95 = {out['random_baseline_abs_cosine']['p95']})")
print(f"-> {o / 'jspace_alignment.json'}")
