#!/usr/bin/env python3
"""Depth-resolved margin lens: at every capture row, apply the model's OWN
readout (final RMSNorm + yes/no lm_head rows, logit-lens style) to the saved
reader-mode activations and score margin AUROC vs labels. Overlaid with the
supervised probe curve, this layer-resolves the represented-vs-read gap.

Caveats stated in the record: reader-mode captures (no verification prompt),
raw logit lens on a Qwen-class model (basis-drift risk, early layers
unreliable) - the probe curve is the recommended cross-check.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import sys

ROOT = Path(__file__).resolve().parents[1]
ACTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "capture" / "acts-qwen3-32b"
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "margin_lens.json"
EPS = 1e-6

d = np.load(ROOT / "capture" / "qwen3-32b-head-rows.npz")
norm_w = d["norm_weight"].astype(np.float32)
yes_rows = np.stack([d[f"row_{i}"] for i in d["yes_ids"]]).astype(np.float32)
no_rows = np.stack([d[f"row_{i}"] for i in d["no_ids"]]).astype(np.float32)

meta = json.loads((ACTS_DIR / "meta.json").read_text())
ids = meta["ids"]
y = np.array([meta["labels"][i] for i in ids])

results = {}
for site in ("last", "mean"):
    npz = np.load(ACTS_DIR / f"acts-{site}.npz")
    acts = np.stack([npz[i] for i in ids]).astype(np.float32)  # (2000, 65, 5120)
    per_layer = []
    for L in range(acts.shape[1]):
        h = acts[:, L, :]
        # HF appends the LAST hidden state after the final norm, so that row
        # is already normalized (verified: runs/lens-v1/validate_lens_*.json).
        # Normalizing it again corrupts the readout.
        if L == acts.shape[1] - 1:
            hn = h
        else:
            hn = h / np.sqrt((h * h).mean(axis=1, keepdims=True) + EPS) * norm_w
        margin = (hn @ yes_rows.T).max(axis=1) - (hn @ no_rows.T).max(axis=1)
        per_layer.append(round(float(roc_auc_score(y, margin)), 4))
    results[site] = per_layer
    print(f"{site}: L0 {per_layer[0]}  L32 {per_layer[32]}  L48 {per_layer[48]}  "
          f"L61 {per_layer[61]}  L64 {per_layer[64]}  max {max(per_layer)} "
          f"@L{per_layer.index(max(per_layer))}")

probe = json.loads((ROOT / "runs" / "probe-32b" / "probe_results.json").read_text())
record = {
    "stage": "margin-lens",
    "model": meta["model"],
    "n_texts": meta["n_texts"],
    "method": "RMSNorm(h)*w @ (yes_rows - no_rows), max over 4 token variants each",
    "caveats": ["reader-mode captures, no verification prompt",
                "raw logit lens on Qwen: early layers unreliable (basis drift)"],
    "lens_auroc_by_layer": results,
    "probe_auroc_by_layer_mean_site": [p["auroc_oof"] for p in probe["sites"]["mean"]["per_layer"]],
    "reference": {"behavioral_margin_gate_auroc": 0.6694,
                  "probe_best_lawcc": 0.5985},
}
out = ROOT / "runs" / "lens-v1"
out.mkdir(parents=True, exist_ok=True)
(out / OUT_NAME).write_text(json.dumps(record, indent=2) + "\n")
print(f"-> {out / OUT_NAME}")
