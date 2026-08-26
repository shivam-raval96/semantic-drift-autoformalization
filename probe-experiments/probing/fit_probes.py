#!/usr/bin/env python3
"""Fit per-layer linear probes: is translation correctness linearly readable?

For each site (last / mean) and each of the 33 layers: standardize the
4096-dim activations and fit logistic regression, correct vs wrong, under
5-fold GroupKFold grouped by problem — a story's correct/wrong twins never
straddle a split. Reported AUROC is always out-of-fold (the probe never
scores a problem it trained on).

Controls, per PLAN.md:
  - bag-of-chars TF-IDF on the answer text alone (the lexical floor),
    same grouped CV;
  - answer token-length single-feature probe (the length tell);
  - layer 0 = raw embeddings (in the curve itself);
  - shuffled-train-labels probes at the best layer (must land ~0.5);
  - law-disjoint robustness split (GroupKFold by group_lawcc) at the best
    layer — one component holds 556/1000 problems (recorded caveat).

Reads capture/acts/{acts-last,acts-mean,meta.json} + contrast_v1/contrast.jsonl.
Writes runs/probe-v1/probe_results.json. CPU only.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
N_FOLDS = 5
SHUFFLE_SEEDS = (0, 1, 2)

ESTIMATORS = {
    "linear": lambda: LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
    "mlp": lambda: MLPClassifier(
        hidden_layer_sizes=(128,), early_stopping=True, n_iter_no_change=10,
        max_iter=300, random_state=0,
    ),
}


def make_probe(estimator: str):
    return lambda: make_pipeline(StandardScaler(), ESTIMATORS[estimator]())


def decision(m, X):
    return (
        m.decision_function(X)
        if hasattr(m[-1], "decision_function")
        else m.predict_proba(X)[:, 1]
    )


def oof_scores(X, y, groups, model_fn, shuffle_seed=None):
    """Out-of-fold decision scores under GroupKFold; optionally fit on
    shuffled train labels (the leakage control)."""
    scores = np.zeros(len(y))
    fold_aurocs = []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        y_tr = y[tr]
        if shuffle_seed is not None:
            y_tr = np.random.RandomState(shuffle_seed).permutation(y_tr)
        m = model_fn()
        m.fit(X[tr], y_tr)
        s = decision(m, X[te])
        scores[te] = s
        fold_aurocs.append(roc_auc_score(y[te], s))
    return scores, fold_aurocs


def main():
    cli = argparse.ArgumentParser(description="Fit per-layer probes.")
    cli.add_argument("--estimator", choices=ESTIMATORS, default="linear")
    cli.add_argument("--acts", type=Path, default=ROOT / "capture" / "acts")
    cli.add_argument("--run-dir", default="probe-v1")
    args = cli.parse_args()
    ACTS, fn = args.acts, make_probe(args.estimator)

    t0 = time.time()
    meta = json.loads((ACTS / "meta.json").read_text())
    ids = meta["ids"]
    y = np.array([meta["labels"][i] for i in ids])
    problems = np.array([i.split("::")[0] for i in ids])

    rows = {
        r["problem_id"]: r
        for r in map(json.loads, (ROOT / "contrast_v1" / "contrast.jsonl").open())
    }
    lawcc = np.array([rows[p]["group_lawcc"] for p in problems])
    tier = np.array([rows[p]["tier"] for p in problems])
    ptype = np.array([rows[p]["perturbation"]["type"] for p in problems])
    answers = [
        rows[i.split("::")[0]][("correct_rg" if i.endswith("correct") else "wrong_rg")]
        for i in ids
    ]

    results = {"sites": {}, "baselines": {}}
    for site, fname in (("last", "acts-last.npz"), ("mean", "acts-mean.npz")):
        t1 = time.time()
        npz = np.load(ACTS / fname)
        acts = np.stack([npz[i] for i in ids])  # (2000, 33, 4096) f16
        n_layers = acts.shape[1]
        per_layer, oof_by_layer = [], {}
        for L in range(n_layers):
            X = acts[:, L, :].astype(np.float32)
            scores, folds = oof_scores(X, y, problems, fn)
            oof_by_layer[L] = scores
            per_layer.append(
                {
                    "layer": L,
                    "auroc_oof": round(roc_auc_score(y, scores), 4),
                    "auroc_folds_mean": round(float(np.mean(folds)), 4),
                    "auroc_folds_sd": round(float(np.std(folds)), 4),
                }
            )
            print(f"[{site}] layer {L:2d}  AUROC {per_layer[-1]['auroc_oof']:.4f}",
                  flush=True)
        best = max(per_layer, key=lambda r: r["auroc_oof"])
        bL = best["layer"]
        X_best = acts[:, bL, :].astype(np.float32)

        # Law-disjoint robustness at the best layer.
        rob_scores, rob_folds = oof_scores(X_best, y, lawcc, fn)
        # Shuffled-label control at the best layer.
        shuffled = [
            round(roc_auc_score(y, oof_scores(X_best, y, problems, fn, seed)[0]), 4)
            for seed in SHUFFLE_SEEDS
        ]
        # Subgroup AUROCs from the primary OOF scores at the best layer.
        by_tier = {
            t: round(roc_auc_score(y[tier == t], oof_by_layer[bL][tier == t]), 4)
            for t in sorted(set(tier))
        }
        by_ptype = {
            p: round(roc_auc_score(y[ptype == p], oof_by_layer[bL][ptype == p]), 4)
            for p in sorted(set(ptype))
        }
        results["sites"][site] = {
            "per_layer": per_layer,
            "best_layer": bL,
            "best_auroc_oof": best["auroc_oof"],
            "lawcc_robustness": {
                "auroc_oof": round(roc_auc_score(y, rob_scores), 4),
                "fold_aurocs": [round(a, 4) for a in rob_folds],
                "caveat": "largest law component holds 556/1000 problems",
            },
            "shuffled_label_auroc": shuffled,
            "breakdown_at_best": {"by_tier": by_tier, "by_perturbation": by_ptype},
            "seconds": round(time.time() - t1, 1),
        }

    # Lexical floor: char TF-IDF on the answer text alone, same grouped CV.
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=20000)
    Xb = vec.fit_transform(answers).toarray().astype(np.float32)
    s, folds = oof_scores(Xb, y, problems, fn)
    results["baselines"]["bow_char_tfidf"] = {
        "auroc_oof": round(roc_auc_score(y, s), 4),
        "auroc_folds_mean": round(float(np.mean(folds)), 4),
    }
    # Length tell: answer character length as the only feature.
    Xl = np.array([[len(a)] for a in answers], dtype=np.float32)
    s, _ = oof_scores(Xl, y, problems, fn)
    results["baselines"]["answer_length"] = round(roc_auc_score(y, s), 4)
    results["baselines"]["majority"] = 0.5

    record = {
        "stage": "probing",
        "capture_tag": meta["config"]["tag"],
        "contrast_sha256": meta["config"]["contrast_sha256"],
        "model": meta["model"],
        "n_texts": meta["n_texts"],
        "cv": {
            "primary": f"GroupKFold({N_FOLDS}) grouped by problem_id",
            "robustness": f"GroupKFold({N_FOLDS}) grouped by group_lawcc",
        },
        "estimator": args.estimator,
        "estimator_spec": repr(ESTIMATORS[args.estimator]()),
        **results,
        "runtime_s": round(time.time() - t0, 1),
    }
    out = ROOT / "runs" / args.run_dir
    out.mkdir(parents=True, exist_ok=True)
    name = "probe_results.json" if args.estimator == "linear" else f"probe_results_{args.estimator}.json"
    (out / name).write_text(json.dumps(record, indent=2) + "\n")

    print("\n=== summary ===")
    for site, r in results["sites"].items():
        print(f"{site:4s} best layer {r['best_layer']:2d}  AUROC {r['best_auroc_oof']:.4f}  "
              f"lawcc {r['lawcc_robustness']['auroc_oof']:.4f}  "
              f"shuffled {r['shuffled_label_auroc']}")
    print(f"baselines: {results['baselines']}")
    print(f"-> {out / name}  ({record['runtime_s']}s)")


if __name__ == "__main__":
    main()
