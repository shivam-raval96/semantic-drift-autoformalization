"""PCA structure of the certified ETP task data (v5).

Two objects:
  1. Law-level: fingerprint vectors (row+column of the certified implication
     table, +1/-1/0) -> PCA. Colored by n_ops, depth, out-degree, family.
     This is the space fp3 charts (first 3 PCs).
  2. Pair-level: concat(fp_premise, fp_conclusion) -> PCA. Colored by truth
     value, pair ops, max depth. Question: do True pairs cluster?

Quantification, not just pictures:
  - explained variance per PC
  - silhouette score of each labeling in PC space
  - kNN label agreement (k=5) vs shuffled-label null
  - linear probe (logistic on top-k PCs) AUC for pair truth, vs label-shuffle

Outputs: PNGs (2D panels + 3D views) and pca_coords.json (for the interactive
artifact) under causalab-integration/analysis/dataset_pca/.

Run: ../causalab/.venv/bin/python causalab-integration/scripts/dataset_pca.py
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "tasks", "etp_implication", "data", "etp_pairs.json")
OUT = os.path.join(HERE, "..", "analysis", "dataset_pca")
os.makedirs(OUT, exist_ok=True)

RNG = np.random.default_rng(2026)

# palette: categorical (fixed order), sequential handled by cmap
C_TRUE = "#1f77b4"
C_FALSE = "#d62728"
SEQ_CMAP = "viridis"


def load():
    with open(DATA) as f:
        d = json.load(f)
    laws = d["laws"]
    pairs = d["pairs"]
    names = sorted(laws.keys())
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    # implication table M[i,j] = +1 if i implies j (certified), -1 if certified non-implication
    M = np.zeros((n, n), dtype=np.float32)
    for k, v in pairs.items():
        p, c = k.split("|")
        M[idx[p], idx[c]] = 1.0 if v else -1.0
    # fingerprint = row + column (what it implies, what implies it)
    F = np.concatenate([M, M.T], axis=1)  # n x 2n
    return d, laws, pairs, names, idx, M, F


def pca(X, k=10):
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = (S**2) / (S**2).sum()
    Z = Xc @ Vt[:k].T
    # deterministic sign: largest-|loading| positive
    for j in range(Z.shape[1]):
        v = Vt[j]
        if v[np.argmax(np.abs(v))] < 0:
            Z[:, j] *= -1
    return Z, ev[:k], mu, Vt[:k]


def silhouette(Z, labels):
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(Z, labels))


def knn_agreement(Z, labels, k=5, n_null=200):
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, ind = nn.kneighbors(Z)
    ind = ind[:, 1:]  # drop self
    agree = (labels[ind] == labels[:, None]).mean()
    nulls = []
    for _ in range(n_null):
        sh = RNG.permutation(labels)
        nulls.append((sh[ind] == sh[:, None]).mean())
    nulls = np.array(nulls)
    p = float((nulls >= agree).mean())
    return float(agree), float(nulls.mean()), p


def scatter2d(ax, Z, c, title, cmap=None, cat_colors=None, labels=None, dims=(0, 1)):
    a, b = dims
    if cat_colors is not None:
        for val, col, lab in cat_colors:
            m = c == val
            ax.scatter(Z[m, a], Z[m, b], s=14, c=col, alpha=0.75, label=lab,
                       edgecolors="white", linewidths=0.3)
        ax.legend(fontsize=7, framealpha=0.9)
    else:
        sc = ax.scatter(Z[:, a], Z[:, b], s=14, c=c, cmap=cmap, alpha=0.8,
                        edgecolors="white", linewidths=0.3)
        plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_xlabel(f"PC{a+1}")
    ax.set_ylabel(f"PC{b+1}")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2)


def scatter3d(fig, pos, Z, c, title, cmap=None, cat_colors=None, azim=-60):
    ax = fig.add_subplot(pos, projection="3d")
    if cat_colors is not None:
        for val, col, lab in cat_colors:
            m = c == val
            ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], s=10, c=col, alpha=0.75, label=lab)
        ax.legend(fontsize=7)
    else:
        sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=10, c=c, cmap=cmap, alpha=0.8)
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=18, azim=azim)
    return ax


def family_of(name):
    for stem in ("assoc_sw", "assoc", "central", "syn"):
        if name.startswith(stem):
            return stem
    return name.split("_")[0]


def main():
    d, laws, pairs, names, idx, M, F = load()
    n = len(names)
    print(f"laws={n} pairs={len(pairs)} true={int((M==1).sum())}")

    report = {"law_level": {}, "pair_level": {}}

    # ---------------- law level ----------------
    Z, ev, mu, comps = pca(F, k=10)
    print("law PCA explained variance:", np.round(ev[:6], 4).tolist(),
          "cum3=", round(float(ev[:3].sum()), 4))
    report["law_level"]["explained_variance"] = ev.tolist()

    n_ops = np.array([laws[nm]["n_ops"] for nm in names])
    depth = np.array([laws[nm]["depth"] for nm in names])
    outdeg = (M == 1).sum(axis=1)
    indeg = (M == 1).sum(axis=0)
    fam = np.array([family_of(nm) for nm in names])
    fam_vals = sorted(set(fam.tolist()))
    fam_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f"]

    # consistency check vs stored fp3
    fp3_stored = np.array([laws[nm]["fp3"] for nm in names])
    corr = [abs(np.corrcoef(Z[:, j], fp3_stored[:, j])[0, 1]) for j in range(3)]
    print("corr(recomputed PC, stored fp3):", np.round(corr, 4).tolist())
    report["law_level"]["fp3_consistency_abs_corr"] = corr

    metrics = {}
    for lab_name, lab in [("n_ops", n_ops), ("depth", depth), ("family", fam)]:
        sil = silhouette(Z[:, :3], lab)
        agr, null_m, p = knn_agreement(Z[:, :3], lab)
        metrics[lab_name] = dict(silhouette=sil, knn_agree=agr, knn_null=null_m, p=p)
        print(f"law/{lab_name}: silhouette={sil:.3f} knn={agr:.3f} null={null_m:.3f} p={p:.3f}")
    # out-degree is continuous: correlate with PCs
    for j in range(3):
        r = np.corrcoef(Z[:, j], outdeg)[0, 1]
        metrics.setdefault("outdeg_corr", []).append(round(float(r), 4))
    print("law/outdeg corr with PC1-3:", metrics["outdeg_corr"])
    report["law_level"]["metrics"] = metrics

    # 2D panel figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    scatter2d(axes[0, 0], Z, n_ops, "laws: n_ops (complexity)", cmap=SEQ_CMAP)
    scatter2d(axes[0, 1], Z, depth, "laws: nesting depth", cmap=SEQ_CMAP)
    scatter2d(axes[0, 2], Z, outdeg, "laws: out-degree (# implied)", cmap="magma")
    scatter2d(axes[1, 0], Z, fam,
              "laws: family",
              cat_colors=[(v, fam_palette[i % 8], v) for i, v in enumerate(fam_vals)])
    scatter2d(axes[1, 1], Z, n_ops, "laws: n_ops (PC1 vs PC3)", cmap=SEQ_CMAP, dims=(0, 2))
    scatter2d(axes[1, 2], Z, n_ops, "laws: n_ops (PC2 vs PC3)", cmap=SEQ_CMAP, dims=(1, 2))
    fig.suptitle(f"Law fingerprint PCA (247 laws, certified table; "
                 f"PC1-3 = {ev[:3].sum()*100:.1f}% var)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "laws_2d.png"), dpi=150)
    plt.close(fig)

    # 3D figure, two views x two colorings
    fig = plt.figure(figsize=(14, 10))
    scatter3d(fig, 221, Z, n_ops, "laws 3D: n_ops", cmap=SEQ_CMAP, azim=-60)
    scatter3d(fig, 222, Z, n_ops, "laws 3D: n_ops (rotated)", cmap=SEQ_CMAP, azim=30)
    scatter3d(fig, 223, Z, depth, "laws 3D: depth", cmap=SEQ_CMAP, azim=-60)
    scatter3d(fig, 224, Z, outdeg, "laws 3D: out-degree", cmap="magma", azim=-60)
    fig.suptitle("Law fingerprint PCA — 3D (the fp3 chart)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "laws_3d.png"), dpi=150)
    plt.close(fig)

    # ---------------- pair level ----------------
    keys = list(pairs.keys())
    y = np.array([1 if pairs[k] else 0 for k in keys])
    prem = np.array([idx[k.split("|")[0]] for k in keys])
    conc = np.array([idx[k.split("|")[1]] for k in keys])

    # balanced sample for PCA + plots: all true + equal false
    ti = np.where(y == 1)[0]
    fi = RNG.choice(np.where(y == 0)[0], size=len(ti), replace=False)
    sel = np.concatenate([ti, fi])
    RNG.shuffle(sel)
    Xp = np.concatenate([F[prem[sel]], F[conc[sel]]], axis=1)
    yp = y[sel]
    pair_ops = n_ops[prem[sel]] + n_ops[conc[sel]]
    pair_depth = np.maximum(depth[prem[sel]], depth[conc[sel]])

    Zp, evp, _, _ = pca(Xp, k=10)
    print("pair PCA explained variance:", np.round(evp[:6], 4).tolist())
    report["pair_level"]["explained_variance"] = evp.tolist()
    report["pair_level"]["n"] = int(len(sel))

    pm = {}
    for lab_name, lab in [("truth", yp), ("pair_ops", pair_ops), ("max_depth", pair_depth)]:
        sil = silhouette(Zp[:, :3], lab)
        agr, null_m, p = knn_agreement(Zp[:, :3], lab)
        pm[lab_name] = dict(silhouette=sil, knn_agree=agr, knn_null=null_m, p=p)
        print(f"pair/{lab_name}: silhouette={sil:.3f} knn={agr:.3f} null={null_m:.3f} p={p:.3f}")

    # linear probe: truth from top-k PCs (balanced set, 5-fold)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    for kk in (3, 10):
        auc = cross_val_score(LogisticRegression(max_iter=2000), Zp[:, :kk], yp,
                              cv=5, scoring="roc_auc")
        pm[f"truth_auc_pc{kk}"] = dict(mean=float(auc.mean()), std=float(auc.std()))
        print(f"pair truth AUC from top-{kk} PCs: {auc.mean():.3f} +- {auc.std():.3f}")
    report["pair_level"]["metrics"] = pm

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    tv = [(1, C_TRUE, "True (certified implies)"), (0, C_FALSE, "False (certified non)")]
    scatter2d(axes[0, 0], Zp, yp, "pairs: truth value", cat_colors=tv)
    scatter2d(axes[0, 1], Zp, pair_ops, "pairs: total ops (complexity)", cmap=SEQ_CMAP)
    scatter2d(axes[0, 2], Zp, pair_depth, "pairs: max nesting depth", cmap=SEQ_CMAP)
    scatter2d(axes[1, 0], Zp, yp, "pairs: truth (PC1 vs PC3)", cat_colors=tv, dims=(0, 2))
    scatter2d(axes[1, 1], Zp, yp, "pairs: truth (PC2 vs PC3)", cat_colors=tv, dims=(1, 2))
    scatter2d(axes[1, 2], Zp, pair_ops, "pairs: ops (PC2 vs PC3)", cmap=SEQ_CMAP, dims=(1, 2))
    fig.suptitle(f"Pair PCA (concat premise+conclusion fingerprints; balanced "
                 f"{len(sel)} pairs; PC1-3 = {evp[:3].sum()*100:.1f}% var)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pairs_2d.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(14, 10))
    scatter3d(fig, 221, Zp, yp, "pairs 3D: truth", cat_colors=tv, azim=-60)
    scatter3d(fig, 222, Zp, yp, "pairs 3D: truth (rotated)", cat_colors=tv, azim=45)
    scatter3d(fig, 223, Zp, pair_ops, "pairs 3D: total ops", cmap=SEQ_CMAP, azim=-60)
    scatter3d(fig, 224, Zp, pair_depth, "pairs 3D: max depth", cmap=SEQ_CMAP, azim=-60)
    fig.suptitle("Pair PCA — 3D", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pairs_3d.png"), dpi=150)
    plt.close(fig)

    # coords for interactive artifact
    coords = {
        "laws": {
            "pc": Z[:, :3].round(4).tolist(),
            "explained_variance": ev[:3].tolist(),
            "n_ops": n_ops.tolist(),
            "depth": depth.tolist(),
            "outdeg": outdeg.tolist(),
            "family": fam.tolist(),
            "names": names,
        },
        "pairs": {
            "pc": Zp[:, :3].round(4).tolist(),
            "explained_variance": evp[:3].tolist(),
            "truth": yp.tolist(),
            "pair_ops": pair_ops.tolist(),
            "max_depth": pair_depth.tolist(),
            "keys": [keys[i] for i in sel],
        },
    }
    with open(os.path.join(OUT, "pca_coords.json"), "w") as f:
        json.dump(coords, f)
    with open(os.path.join(OUT, "pca_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
