#!/usr/bin/env python3
"""Honest uncertainty for the law-disjoint probe numbers.

An audit showed the law-disjoint GroupKFold is pathologically imbalanced (one
fold holds ~1112 texts, three hold 74), so its fold SD is NOT an estimator
noise estimate - it is dominated by between-tier heterogeneity. This script
replaces it with two defensible quantities, computed per model on identical
items:

  1. paired clustered bootstrap over problems -> 95% CI on the AUROC, and
     paired CIs on adjacent model differences (same problems for both models).
  2. within-pair label-flip permutation null under the SAME law-disjoint
     grouping -> is the AUROC above chance at all, and by how many sigma.

Also reports the metric under a FIXED site (pre-registered `mean`) rather than
best-of-sites, since best-of-sites is selection on the reported metric.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SITE = "mean"          # pre-registered; `last` reported as robustness
N_BOOT, N_PERM, SEED = 400, 40, 0

MODELS = [
    ("Llama-3.1-8B", "acts", "probe-v1"),
    ("Qwen3.5-4B", "acts-q35-4b", "probe-q35-4b"),
    ("Qwen3-32B", "acts-qwen3-32b", "probe-32b"),
    ("Qwen3.6-27B", "acts-q36", "probe-q36"),
    ("Gemma-3-27B", "acts-gemma3-27b", "probe-gemma3-27b"),
    ("Llama-3.3-70B", "acts-llama33-70b", "probe-llama33-70b"),
]

rows_meta = {json.loads(l)["problem_id"]: json.loads(l)
             for l in (ROOT / "contrast_v1" / "contrast.jsonl").open()}


def oof_scores(X, y, groups, rng=None):
    sc = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        y_tr = y[tr]
        if rng is not None:            # within-pair label flip on train only
            flip = rng.random(len(tr) // 2) < 0.5
            y_tr = y_tr.copy()
            for k, f in enumerate(flip):
                if f:
                    i, j = 2 * k, 2 * k + 1
                    if j < len(y_tr):
                        y_tr[i], y_tr[j] = y_tr[j], y_tr[i]
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=1000))
        m.fit(X[tr], y_tr)
        sc[te] = m.decision_function(X[te])
    return sc


out = {"site": SITE, "n_boot": N_BOOT, "n_perm": N_PERM, "models": {}}
oof_by_model, ids_ref = {}, None
for label, acts_dir, probe_dir in MODELS:
    p = ROOT / "capture" / acts_dir / f"acts-{SITE}.npz"
    if not p.exists():
        print(f"[skip] {label}")
        continue
    meta = json.loads((ROOT / "capture" / acts_dir / "meta.json").read_text())
    npz = np.load(p)
    ids = [i for i in meta["ids"] if i in npz]
    y = np.array([meta["labels"][i] for i in ids])
    pids = np.array([i.split("::")[0] for i in ids])
    groups = np.array([rows_meta[q]["group_lawcc"] for q in pids])
    acts = np.stack([npz[i] for i in ids]).astype(np.float32)
    pr = json.loads((ROOT / "runs" / probe_dir / "probe_results.json").read_text())
    L = pr["sites"][SITE]["best_layer"]
    X = acts[:, L, :]

    sc = oof_scores(X, y, groups)
    auc = float(roc_auc_score(y, sc))
    rng = np.random.RandomState(SEED)
    uniq = np.unique(pids)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(pids == q)[0] for q in pick])
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], sc[idx]))
    perm = [float(roc_auc_score(y, oof_scores(X, y, groups,
                                              np.random.RandomState(SEED + k))))
            for k in range(N_PERM)]
    out["models"][label] = {
        "layer": L, "n_texts": len(ids), "auroc": round(auc, 4),
        "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                 round(float(np.percentile(boots, 97.5)), 4)],
        "perm_null_mean": round(float(np.mean(perm)), 4),
        "perm_null_sd": round(float(np.std(perm)), 4),
        "sigma_above_null": round(float((auc - np.mean(perm)) / (np.std(perm) + 1e-9)), 2),
    }
    oof_by_model[label] = (sc, y, pids)
    r = out["models"][label]
    print(f"{label:14s} L{L:2d} AUROC {auc:.4f} CI {r['ci95']} | "
          f"null {r['perm_null_mean']}±{r['perm_null_sd']} "
          f"({r['sigma_above_null']}σ)")

# paired differences on shared problems
labels = list(oof_by_model)
out["paired_deltas"] = {}
for a, b in zip(labels, labels[1:]):
    (sa, ya, pa), (sb, yb, pb) = oof_by_model[a], oof_by_model[b]
    common = np.intersect1d(pa, pb)
    ia = np.concatenate([np.where(pa == q)[0] for q in common])
    ib = np.concatenate([np.where(pb == q)[0] for q in common])
    rng = np.random.RandomState(SEED)
    ds = []
    for _ in range(N_BOOT):
        pick = rng.choice(common, len(common), replace=True)
        ja = np.concatenate([np.where(pa == q)[0] for q in pick])
        jb = np.concatenate([np.where(pb == q)[0] for q in pick])
        if len(np.unique(ya[ja])) == 2 and len(np.unique(yb[jb])) == 2:
            ds.append(roc_auc_score(yb[jb], sb[jb]) - roc_auc_score(ya[ja], sa[ja]))
    out["paired_deltas"][f"{b} - {a}"] = {
        "delta": round(float(roc_auc_score(yb[ib], sb[ib])
                             - roc_auc_score(ya[ia], sa[ia])), 4),
        "ci95": [round(float(np.percentile(ds, 2.5)), 4),
                 round(float(np.percentile(ds, 97.5)), 4)],
        "p_gt_0": round(float(np.mean(np.array(ds) > 0)), 3),
    }
    d = out["paired_deltas"][f"{b} - {a}"]
    print(f"  Δ {b} - {a}: {d['delta']:+.4f} CI {d['ci95']} P(>0)={d['p_gt_0']}")

o = ROOT / "runs" / "analysis-v1"
o.mkdir(parents=True, exist_ok=True)
(o / "probe_uncertainty.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"-> {o / 'probe_uncertainty.json'}")
