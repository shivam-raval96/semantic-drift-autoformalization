#!/usr/bin/env python3
"""Fit probes on Modal, where the activations already live.

Downloading multi-GB activation archives repeatedly proved fragile. The probe
itself is cheap CPU work, so run it next to the data and return only the
results (a few KB).

    modal run probing/remote_probe.py --tag llama-3.3-70b-full --out probe-llama33-70b
"""

import json
from pathlib import Path

import modal

STAGE = Path(__file__).resolve().parent
PX_ROOT = STAGE.parent

app = modal.App("harsh-remote-probe")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "scikit-learn")
)
acts_vol = modal.Volume.from_name("harsh-probe-activations", create_if_missing=True)


@app.function(image=image, volumes={"/acts": acts_vol}, timeout=3600,
              cpu=8, memory=65536, retries=0)
def probe(tag: str, rows: list) -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    base = f"/acts/contrast_v1/{tag}"
    meta = json.loads(open(f"{base}/meta.json").read())
    lawcc = {r["problem_id"]: r["group_lawcc"] for r in rows}

    out = {"model": meta["model"], "tag": tag, "n_texts": meta["n_texts"],
           "layers": meta["layers"], "d_model": meta["d_model"], "sites": {}}
    for site in ("last", "mean"):
        npz = np.load(f"{base}/acts-{site}.npz")
        ids = [i for i in meta["ids"] if i in npz]
        y = np.array([meta["labels"][i] for i in ids])
        pid = np.array([i.split("::")[0] for i in ids])
        g_prob, g_law = pid, np.array([lawcc[p] for p in pid])
        acts = np.stack([npz[i] for i in ids]).astype(np.float32)
        n_bad = int(np.isinf(acts).sum() + np.isnan(acts).sum())
        if n_bad:
            out["sites"][site] = {"error": f"{n_bad} non-finite values "
                                           "(float16 overflow at capture)"}
            continue

        def run(X, groups):
            sc = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(X, y, groups):
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=1.0, max_iter=1000))
                m.fit(X[tr], y[tr]); sc[te] = m.decision_function(X[te])
            return float(roc_auc_score(y, sc))

        per_layer = []
        for L in range(acts.shape[1]):
            per_layer.append({"layer": L,
                              "auroc_oof": round(run(acts[:, L, :], g_prob), 4)})
        best = max(per_layer, key=lambda r: r["auroc_oof"])
        bL = best["layer"]
        law = run(acts[:, bL, :], g_law)
        shuf = []
        for seed in (0, 1, 2):
            rng = np.random.RandomState(seed)
            yp = rng.permutation(y)
            sc = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(acts[:, bL, :], y, g_prob):
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=1.0, max_iter=1000))
                m.fit(acts[tr][:, bL, :], yp[tr])
                sc[te] = m.decision_function(acts[te][:, bL, :])
            shuf.append(round(float(roc_auc_score(y, sc)), 4))
        out["sites"][site] = {
            "per_layer": per_layer, "best_layer": bL,
            "best_auroc_oof": best["auroc_oof"],
            "lawcc_robustness": {"auroc_oof": round(law, 4)},
            "shuffled_label_auroc": shuf,
        }
        print(f"[{site}] best {best['auroc_oof']} @L{bL} | lawcc {law:.4f} | "
              f"shuffled {shuf}", flush=True)
    return out


@app.local_entrypoint()
def main(tag: str = "llama-3.3-70b-full", out: str = "probe-llama33-70b"):
    rows = [json.loads(l) for l in
            (PX_ROOT / "contrast_v1" / "contrast.jsonl").open()]
    slim = [{"problem_id": r["problem_id"], "group_lawcc": r["group_lawcc"]}
            for r in rows]
    res = probe.remote(tag, slim)
    d = PX_ROOT / "runs" / out
    d.mkdir(parents=True, exist_ok=True)
    (d / "probe_results.json").write_text(json.dumps(res, indent=2) + "\n")
    for site, r in res["sites"].items():
        if "error" in r:
            print(f"[{site}] ERROR: {r['error']}")
        else:
            print(f"[{site}] best {r['best_auroc_oof']} @L{r['best_layer']} | "
                  f"lawcc {r['lawcc_robustness']['auroc_oof']} | "
                  f"shuffled {r['shuffled_label_auroc']}")
    print(f"-> {d / 'probe_results.json'}")
