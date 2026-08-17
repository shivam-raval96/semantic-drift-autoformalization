#!/usr/bin/env python3
"""Supervised readout vs the model's own readout, layer by layer, on the
SAME asked-mode activations.

Question: the margin lens on asked-mode acts peaks mid-late and declines by
the output layer. Is there more correctness information in the stream than
the model's own yes/no channel expresses?

probe_auroc[L]  = supervised linear probe (grouped 5-fold CV by problem)
margin_auroc[L] = model's own yes/no readout (RMSNorm + lm_head rows)
gap[L]          = probe - margin  (information present but not expressed)
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
ACTS = ROOT / "capture" / "acts-asked-32b"
EPS = 1e-6

meta = json.loads((ACTS / "meta.json").read_text())
ids = meta["ids"]
y = np.array([meta["labels"][i] for i in ids])
groups = np.array([i.split("::")[0] for i in ids])
rows_meta = {json.loads(l)["problem_id"]: json.loads(l)
             for l in (ROOT / "contrast_v1" / "contrast.jsonl").open()}
groups_law = np.array([rows_meta[g]["group_lawcc"] for g in groups])

d = np.load(ROOT / "capture" / "qwen3-32b-head-rows.npz")
norm_w = d["norm_weight"].astype(np.float32)
yes_rows = np.stack([d[f"row_{i}"] for i in d["yes_ids"]]).astype(np.float32)
no_rows = np.stack([d[f"row_{i}"] for i in d["no_ids"]]).astype(np.float32)

out = {}
for site in ("last", "mean"):
    npz = np.load(ACTS / f"acts-{site}.npz")
    acts = np.stack([npz[i] for i in ids]).astype(np.float32)
    probe_auc, probe_auc_law, margin_auc = [], [], []
    for L in range(acts.shape[1]):
        h = acts[:, L, :]
        for grp, sink in ((groups, probe_auc), (groups_law, probe_auc_law)):
            scores = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(h, y, grp):
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=1.0, max_iter=1000))
                m.fit(h[tr], y[tr])
                scores[te] = m.decision_function(h[te])
            sink.append(round(float(roc_auc_score(y, scores)), 4))
        if L == acts.shape[1] - 1:
            hn = h          # already normed by HF (validated)
        else:
            hn = h / np.sqrt((h * h).mean(axis=1, keepdims=True) + EPS) * norm_w
        mg = (hn @ yes_rows.T).max(axis=1) - (hn @ no_rows.T).max(axis=1)
        margin_auc.append(round(float(roc_auc_score(y, mg)), 4))
    gaps = [round(p - m, 4) for p, m in zip(probe_auc, margin_auc)]
    gaps_law = [round(p - m, 4) for p, m in zip(probe_auc_law, margin_auc)]
    best_probe = int(np.argmax(probe_auc))
    best_margin = int(np.argmax(margin_auc))
    out[site] = {
        "probe_auroc": probe_auc, "probe_auroc_lawdisjoint": probe_auc_law,
        "margin_auroc": margin_auc, "gap": gaps, "gap_lawdisjoint": gaps_law,
        "max_gap_lawdisjoint": {"layer": int(np.argmax(gaps_law)),
                                "gap": max(gaps_law)},
        "best_probe": {"layer": best_probe, "auroc": probe_auc[best_probe]},
        "best_margin": {"layer": best_margin, "auroc": margin_auc[best_margin]},
        "final_layer_margin": margin_auc[-1],
        "max_gap": {"layer": int(np.argmax(gaps)), "gap": max(gaps)},
    }
    print(f"[{site}] probe(problem-CV) {probe_auc[best_probe]} @L{best_probe} | "
          f"probe(law-disjoint) max {max(probe_auc_law)} | "
          f"margin peak {margin_auc[best_margin]} @L{best_margin} | "
          f"margin final {margin_auc[-1]} | gap_law max {max(gaps_law)}")

record = {
    "stage": "knows-vs-says", "model": meta["model"], "mode": "asked (verification prompt)",
    "n_texts": meta["n_texts"], "n_problems": len(set(groups)),
    "cv": "GroupKFold(5), reported both by problem_id and by law component", "reference_behavioral_margin": 0.6694,
    "sites": out,
}
o = ROOT / "runs" / "lens-v1"
o.mkdir(parents=True, exist_ok=True)
(o / "knows_vs_says.json").write_text(json.dumps(record, indent=2) + "\n")
print(f"-> {o / 'knows_vs_says.json'}")
