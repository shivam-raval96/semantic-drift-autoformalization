#!/usr/bin/env python3
"""Are the reader-mode and asked-mode correctness directions the same?

The reader-mode direction is verbally silent (J-lens readout indistinguishable
from random) yet carries correctness. Under questioning the model answers at
0.67. Either it recruits that same direction into the output pathway, or it
computes the verdict in a different representation. The cosine between the two
fitted directions distinguishes these.

Fits a probe on each activation set at its own best layer (full data, no CV:
we want the direction itself, not a generalization estimate), then reports
cosine, a random baseline, and cross-application AUROC (does each direction
read correctness in the other set's activations?).
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SITE = "mean"


def load(acts_dir):
    d = ROOT / "capture" / acts_dir
    meta = json.loads((d / "meta.json").read_text())
    site = SITE if (d / f"acts-{SITE}.npz").exists() else "last"
    npz = np.load(d / f"acts-{site}.npz")
    ids = [i for i in meta["ids"] if i in npz]
    y = np.array([meta["labels"][i] for i in ids])
    return meta, ids, y, np.stack([npz[i] for i in ids]).astype(np.float32)


def fit_dir(X, y):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    m.fit(X, y)
    w = m[-1].coef_[0] / m[0].scale_          # back to raw activation space
    return w / np.linalg.norm(w)


sets = {}
_asked = ("acts-ansend-32b"
          if (ROOT / "capture" / "acts-ansend-32b" / "acts-last.npz").exists()
          else "acts-asked-32b")
for mode, acts in (("reader", "acts-qwen3-32b"), ("asked", _asked)):
    if (ROOT / "capture" / acts / "meta.json").exists():
        sets[mode] = load(acts)

probe = json.loads((ROOT / "runs" / "probe-32b" / "probe_results.json").read_text())
best_reader = probe["sites"][SITE]["best_layer"]
pl = ROOT / "runs" / "lens-v1" / "asked_last_full.json"
if pl.exists():
    best_asked = max(json.loads(pl.read_text())["probe_lawdisjoint_sampled"],
                     key=lambda t: t[1])[0]
else:
    kv = json.loads((ROOT / "runs" / "lens-v1" / "knows_vs_says.json").read_text())
    best_asked = int(np.argmax(kv["sites"][SITE]["probe_auroc_lawdisjoint"]))

out = {"site": SITE, "layers": {"reader": best_reader, "asked": best_asked},
       "n_texts": {m: int(sets[m][0]["n_texts"]) for m in sets}}
dirs = {}
for mode, L in (("reader", best_reader), ("asked", best_asked)):
    if mode in sets:
        _, _, y, acts = sets[mode]
        dirs[mode] = fit_dir(acts[:, L, :], y)

if len(dirs) == 2:
    cos = float(dirs["reader"] @ dirs["asked"])
    rng = np.random.RandomState(0)
    R = rng.standard_normal((2000, len(dirs["reader"]))).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    out["cosine_reader_vs_asked"] = round(cos, 4)
    out["random_abs_cosine_p95"] = round(
        float(np.percentile(np.abs(R @ dirs["asked"]), 95)), 5)
    for src in dirs:
        for tgt in sets:
            _, _, y_t, acts_t = sets[tgt]
            auc = roc_auc_score(y_t, acts_t[:, out["layers"][tgt], :] @ dirs[src])
            out[f"{src}_dir_on_{tgt}_acts"] = round(float(auc), 4)
    np.savez(ROOT / "runs" / "probe-32b" / "direction_asked.npz",
             direction=dirs["asked"].astype(np.float32),
             layer=best_asked, site=SITE)
    print(f"cosine(reader_dir, asked_dir) = {cos:.4f} "
          f"(random |cos| p95 {out['random_abs_cosine_p95']})")
    for k, v in out.items():
        if k.endswith("_acts"):
            print(f"  {k}: {v}")

o = ROOT / "runs" / "lens-v1"
o.mkdir(parents=True, exist_ok=True)
(o / "two_directions.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"-> {o / 'two_directions.json'}")
