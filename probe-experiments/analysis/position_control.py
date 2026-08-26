#!/usr/bin/env python3
"""The 2x2 that decides the 'recruitment' claim.

The reader-vs-asked comparison confounded two things: whether the question is
present, and WHERE the activation was read. Reader mode reads at the end of
the RG answer (mid-document); asked mode read at the generation position.
This script completes the grid using the position-matched `ansend` site: the
chat-formatted verification prompt captured at the last token of the RG.

                    | end-of-RG position | generation position
  reader (no Q)     | have               | n/a
  asked  (Q)        | NEW                | have

Reading:
  asked@end-of-RG ~ 0.50  => the effect is positional/licensing, not the
                            question. The routing claim dies.
  asked@end-of-RG ~ 0.68  => the question really does write correctness into
                            the yes/no channel while the answer is still
                            being read. The claim survives this control.

Also runs a supervised probe at the same site, so we can separate "the
information is not there" from "the information is there but not on the
yes/no axis".
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
EPS = 1e-6

d = np.load(ROOT / "capture" / "qwen3-32b-head-rows.npz")
norm_w = d["norm_weight"].astype(np.float32)
yes_rows = np.stack([d[f"row_{i}"] for i in d["yes_ids"]]).astype(np.float32)
no_rows = np.stack([d[f"row_{i}"] for i in d["no_ids"]]).astype(np.float32)

rows_meta = {json.loads(l)["problem_id"]: json.loads(l)
             for l in (ROOT / "contrast_v1" / "contrast.jsonl").open()}

CELLS = [
    ("reader_at_answer_end", "acts-qwen3-32b", "last"),
    ("asked_at_answer_end", "acts-ansend-32b", "ansend"),
    ("asked_at_generation_pos", "acts-ansend-32b", "last"),
]

out = {"model": "Qwen/Qwen3-32B", "cells": {}}
for name, acts_dir, site in CELLS:
    p = ROOT / "capture" / acts_dir / f"acts-{site}.npz"
    if not p.exists():
        print(f"[skip] {name}: {p.name} missing")
        continue
    meta = json.loads((ROOT / "capture" / acts_dir / "meta.json").read_text())
    npz = np.load(p)
    ids = [i for i in meta["ids"] if i in npz]
    y = np.array([meta["labels"][i] for i in ids])
    groups = np.array([rows_meta[i.split("::")[0]]["group_lawcc"] for i in ids])
    acts = np.stack([npz[i] for i in ids]).astype(np.float32)

    lens_auc, probe_auc = [], []
    for L in range(acts.shape[1]):
        h = acts[:, L, :]
        hn = h if L == acts.shape[1] - 1 else (
            h / np.sqrt((h * h).mean(axis=1, keepdims=True) + EPS) * norm_w)
        mg = (hn @ yes_rows.T).max(axis=1) - (hn @ no_rows.T).max(axis=1)
        lens_auc.append(round(float(roc_auc_score(y, mg)), 4))
        if L % 8 == 0 or L >= acts.shape[1] - 5:      # probe a subset (cost)
            sc = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(h, y, groups):
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=1.0, max_iter=1000))
                m.fit(h[tr], y[tr])
                sc[te] = m.decision_function(h[te])
            probe_auc.append((L, round(float(roc_auc_score(y, sc)), 4)))
    best = int(np.argmax(lens_auc))
    out["cells"][name] = {
        "acts": acts_dir, "site": site, "n_texts": len(ids),
        "lens_auroc_by_layer": lens_auc,
        "lens_best": {"layer": best, "auroc": lens_auc[best]},
        "lens_final": lens_auc[-1],
        "probe_auroc_lawdisjoint_sampled": probe_auc,
        "probe_best_sampled": max(probe_auc, key=lambda t: t[1]) if probe_auc else None,
    }
    print(f"{name:26s} n={len(ids):4d} lens best {lens_auc[best]:.4f} @L{best:2d} "
          f"final {lens_auc[-1]:.4f} | probe(law-disj) max "
          f"{max(probe_auc, key=lambda t: t[1]) if probe_auc else 'n/a'}")

o = ROOT / "runs" / "lens-v1"
o.mkdir(parents=True, exist_ok=True)
(o / "position_control.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"-> {o / 'position_control.json'}")
