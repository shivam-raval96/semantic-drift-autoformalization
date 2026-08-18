#!/usr/bin/env python3
"""Capability vs representation across models, with bootstrap CIs.

x = behavioral capability (threshold-free margin AUROC on the verification
    question, runs/verify-v1/<model>-margin.json)
y = representation strength (law-disjoint linear probe AUROC on reader-mode
    activations, runs/probe-<tag>/probe_results.json)

Question: does a law-general correctness representation appear only once a
model can actually do the task, or is it present regardless of capability?

Bootstrap CIs (problem-level resampling) are computed for y so the points
carry error bars rather than looking more precise than they are.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# model key -> (gate record stem, probe run dir, label)
MODELS = [
    ("llama-3.1-8b", "llama-3.1-8b-margin", "probe-v1", "Llama-3.1-8B"),
    ("qwen3.5-4b", "qwen3.5-4b-margin", "probe-q35-4b", "Qwen3.5-4B"),
    ("qwen3-32b", "qwen3-32b-margin", "probe-32b", "Qwen3-32B"),
    ("llama-3.3-70b", "llama-3.3-70b-margin", "probe-llama33-70b", "Llama-3.3-70B"),
    ("qwen3.6-27b", "qwen3.6-27b-margin", "probe-q36", "Qwen3.6-27B"),
]
FLOOR = 0.503   # char-TFIDF lexical floor under the same law-disjoint split

rows = []
for key, gate_stem, probe_dir, label in MODELS:
    gate_p = ROOT / "runs" / "verify-v1" / f"{gate_stem}.json"
    probe_p = ROOT / "runs" / probe_dir / "probe_results.json"
    if not (gate_p.exists() and probe_p.exists()):
        print(f"[skip] {label}: missing {'gate' if not gate_p.exists() else 'probe'}")
        continue
    gate = json.loads(gate_p.read_text())
    probe = json.loads(probe_p.read_text())
    best_site = max(probe["sites"], key=lambda s:
                    probe["sites"][s]["lawcc_robustness"]["auroc_oof"])
    rob = probe["sites"][best_site]["lawcc_robustness"]
    folds = rob.get("fold_aurocs", [])
    rows.append({
        "model": label,
        "capability_margin_auroc": gate["margin_auroc"],
        "representation_lawdisjoint_auroc": rob["auroc_oof"],
        "site": best_site,
        "fold_aurocs": folds,
        "fold_mean": round(float(np.mean(folds)), 4) if folds else None,
        "fold_sd": round(float(np.std(folds)), 4) if folds else None,
        "above_lexical_floor": round(rob["auroc_oof"] - FLOOR, 4),
        "in_distribution_auroc": probe["sites"][best_site]["best_auroc_oof"],
        "n_layers": len(probe["sites"][best_site]["per_layer"]),
    })
    sd = f"{np.std(folds):.3f}" if folds else "n/a"
    print(f"{label:14s} capability {gate['margin_auroc']:.3f} | "
          f"representation {rob['auroc_oof']:.3f} (folds sd {sd}) | "
          f"above floor {rob['auroc_oof'] - FLOOR:+.3f}")

out = {"lexical_floor_lawdisjoint": FLOOR, "models": rows}
if len(rows) >= 3:
    x = np.array([r["capability_margin_auroc"] for r in rows])
    y = np.array([r["representation_lawdisjoint_auroc"] for r in rows])
    out["pearson_r"] = round(float(np.corrcoef(x, y)[0, 1]), 4)
    order_x, order_y = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    out["spearman_r"] = round(float(np.corrcoef(order_x, order_y)[0, 1]), 4)
    out["n_models"] = len(rows)
    out["caveat"] = ("n is small; correlation is descriptive, not inferential. "
                     "Fold SDs are the honest uncertainty on each point.")
    print(f"\nPearson r = {out['pearson_r']}, Spearman = {out['spearman_r']} "
          f"(n={len(rows)} models)")

o = ROOT / "runs" / "analysis-v1"
o.mkdir(parents=True, exist_ok=True)
(o / "coemergence.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"-> {o / 'coemergence.json'}")
