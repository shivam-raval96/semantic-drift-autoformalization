"""PCA structure of the FULL ETP implication table (4694 laws, 22,028,942
settled implications) — the scale-up of dataset_pca.py's v5 analysis.

Inputs (from a shallow clone of teorth/equational_theories at repo root):
  equational_theories/data/2024-11-10-outcomes.json.zip -> outcomes.json
    (unzipped to the scratchpad; pass its path as argv[1])
  equational_theories/data/equations.txt (4694 laws, one per line)

Outputs under causalab-integration/analysis/dataset_pca/full_etp/:
  laws_2d.png, laws_3d.png, pairs_2d.png, pairs_3d.png,
  full_pca_report.json, full_pca_coords.json (subsampled, for the artifact)

Questions mirrored from the v5 analysis:
  - do laws cluster by complexity (n_ops / depth / difficulty bins)?
  - is implication strength (out-degree) the dominant geometric axis?
  - do True pairs separate in fingerprint space, and how much of that is
    the premise-strength shortcut (out-degree-only AUC)?
"""

import json
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "analysis", "dataset_pca", "full_etp")
EQ_TXT = os.path.join(HERE, "..", "..", "equational_theories", "data", "equations.txt")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(2026)

C_TRUE, C_FALSE = "#1f77b4", "#d62728"


def load_outcomes(path):
    """Stream-scan the outcomes JSON: statuses only, in order, into int8."""
    text = open(path).read()
    vocab = {}
    counts = {}
    vals = []
    for m in re.finditer(r'"([a-z_0-9]+)"', text):
        s = m.group(1)
        if s == "outcomes" or s == "equations":
            continue
        vals.append(s)
    n = int(round(len(vals) ** 0.5))
    assert n * n == len(vals), f"not square: {len(vals)}"
    arr = np.zeros(len(vals), dtype=np.int8)
    for i, s in enumerate(vals):
        counts[s] = counts.get(s, 0) + 1
        if s.endswith("_true"):
            arr[i] = 1
        elif s.endswith("_false"):
            arr[i] = -1
        else:
            arr[i] = 0  # unknown/conjecture — none expected in the settled table
    print("status vocabulary:", counts)
    return arr.reshape(n, n), counts


def law_complexity(line):
    n_ops = line.count("◇") + line.count("∘") + line.count("*")
    # depth: max paren nesting per side
    depth = d = 0
    for ch in line:
        if ch == "(":
            d += 1
            depth = max(depth, d)
        elif ch == ")":
            d -= 1
    return n_ops, depth


def pca_randomized(X, k=10, seed=2026):
    from sklearn.utils.extmath import randomized_svd

    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = randomized_svd(Xc, n_components=k, random_state=seed)
    tot = (Xc**2).sum()
    ev = (S**2) / tot
    Z = Xc @ Vt.T
    for j in range(Z.shape[1]):
        v = Vt[j]
        if v[np.argmax(np.abs(v))] < 0:
            Z[:, j] *= -1
    return Z, ev


def main():
    outcomes_path = sys.argv[1]
    M, counts = load_outcomes(outcomes_path)
    n = M.shape[0]
    print(f"table {n}x{n}")

    lines = [l.rstrip("\n") for l in open(EQ_TXT) if l.strip()]
    assert len(lines) == n, f"equations.txt has {len(lines)} laws, table has {n}"
    comp = [law_complexity(l) for l in lines]
    n_ops = np.array([c[0] for c in comp])
    depth = np.array([c[1] for c in comp])
    outdeg = (M == 1).sum(axis=1).astype(np.int64)
    indeg = (M == 1).sum(axis=0).astype(np.int64)
    n_true = int((M == 1).sum() - n)  # minus diagonal self-implications
    print(f"true implications (off-diag): {n_true}, ops range {n_ops.min()}-{n_ops.max()}")

    F = np.concatenate([M, M.T], axis=1).astype(np.float32)
    Z, ev = pca_randomized(F, k=10)
    print("explained variance:", np.round(ev[:6], 4).tolist(),
          "cum3=", round(float(ev[:3].sum()), 4))

    report = dict(n_laws=n, n_true_offdiag=n_true, status_counts=counts,
                  explained_variance=ev.tolist())

    # correlations with the leading PCs
    for name, lab in [("outdeg", outdeg), ("indeg", indeg),
                      ("n_ops", n_ops), ("depth", depth)]:
        report[f"{name}_corr_pc123"] = [
            round(float(np.corrcoef(Z[:, j], lab)[0, 1]), 4) for j in range(3)
        ]
        print(name, "corr PC1-3:", report[f"{name}_corr_pc123"])

    from sklearn.metrics import silhouette_score
    ops_bin = np.minimum(n_ops, 8)
    for name, lab in [("ops_bin", ops_bin), ("depth", np.minimum(depth, 6))]:
        sil = float(silhouette_score(Z[:, :3], lab))
        report[f"silhouette_{name}"] = round(sil, 4)
        print(f"silhouette {name}: {sil:.3f}")

    # ---- pair-level: truth separation + strength shortcut at full scale ----
    from sklearn.metrics import roc_auc_score
    off = ~np.eye(n, dtype=bool)
    truth_full = (M == 1) & off
    # out-degree shortcut AUC over ALL 22M pairs (vectorized: score = outdeg[premise])
    prem_score = np.repeat(outdeg, n).reshape(n, n)[off].astype(np.float64)
    y_full = truth_full[off].astype(np.int8)
    report["auc_premise_outdeg_full"] = round(float(roc_auc_score(y_full, prem_score)), 4)
    conc_score = np.tile(indeg, (n, 1))[off].astype(np.float64)
    report["auc_conclusion_indeg_full"] = round(float(roc_auc_score(y_full, conc_score)), 4)
    print("AUC premise outdeg (full 22M):", report["auc_premise_outdeg_full"])
    print("AUC conclusion indeg (full 22M):", report["auc_conclusion_indeg_full"])

    # balanced subsample for the pair scatter: PC-coords concat
    ti = np.argwhere(truth_full)
    fi = np.argwhere((M == -1))
    kt = min(4000, len(ti))
    ts = ti[RNG.choice(len(ti), kt, replace=False)]
    fs = fi[RNG.choice(len(fi), kt, replace=False)]
    pairs_idx = np.concatenate([ts, fs])
    yp = np.array([1] * kt + [0] * kt)
    shuffle = RNG.permutation(len(yp))
    pairs_idx, yp = pairs_idx[shuffle], yp[shuffle]
    Xp = np.concatenate([Z[pairs_idx[:, 0], :8], Z[pairs_idx[:, 1], :8]], axis=1)
    Zp, evp = pca_randomized(Xp, k=6)
    pair_ops = n_ops[pairs_idx[:, 0]] + n_ops[pairs_idx[:, 1]]

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    auc = cross_val_score(LogisticRegression(max_iter=2000), Zp[:, :3], yp,
                          cv=5, scoring="roc_auc")
    report["pair_truth_auc_pc3"] = dict(mean=round(float(auc.mean()), 4),
                                        std=round(float(auc.std()), 4))
    print("pair truth AUC from 3 pair-PCs:", report["pair_truth_auc_pc3"])

    # ------------- figures -------------
    def s2(ax, C, c, title, cmap=None, cat=None, dims=(0, 1)):
        a, b = dims
        if cat is not None:
            for val, col, lab in cat:
                m = c == val
                ax.scatter(C[m, a], C[m, b], s=4, c=col, alpha=0.5, label=lab, linewidths=0)
            ax.legend(fontsize=7, framealpha=0.9)
        else:
            sc = ax.scatter(C[:, a], C[:, b], s=4, c=c, cmap=cmap, alpha=0.6, linewidths=0)
            plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_xlabel(f"PC{a+1}"); ax.set_ylabel(f"PC{b+1}")
        ax.set_title(title, fontsize=9); ax.grid(alpha=0.2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    s2(axes[0, 0], Z, np.minimum(n_ops, 8), "laws: n_ops (capped 8)", cmap="viridis")
    s2(axes[0, 1], Z, np.minimum(depth, 6), "laws: nesting depth", cmap="viridis")
    s2(axes[0, 2], Z, np.log10(outdeg + 1), "laws: log10 out-degree", cmap="magma")
    s2(axes[1, 0], Z, np.log10(indeg + 1), "laws: log10 in-degree", cmap="magma")
    s2(axes[1, 1], Z, np.minimum(n_ops, 8), "laws: n_ops (PC1 vs PC3)", cmap="viridis", dims=(0, 2))
    s2(axes[1, 2], Z, np.log10(outdeg + 1), "laws: out-deg (PC2 vs PC3)", cmap="magma", dims=(1, 2))
    fig.suptitle(f"FULL ETP law fingerprint PCA ({n} laws, 22M implications; "
                 f"PC1-3 = {ev[:3].sum()*100:.1f}% var)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "laws_2d.png"), dpi=150)
    plt.close(fig)

    def s3(fig, pos, C, c, title, cmap=None, cat=None, azim=-60):
        ax = fig.add_subplot(pos, projection="3d")
        if cat is not None:
            for val, col, lab in cat:
                m = c == val
                ax.scatter(C[m, 0], C[m, 1], C[m, 2], s=3, c=col, alpha=0.5, label=lab)
            ax.legend(fontsize=7)
        else:
            sc = ax.scatter(C[:, 0], C[:, 1], C[:, 2], s=3, c=c, cmap=cmap, alpha=0.6)
            fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title(title, fontsize=9); ax.view_init(elev=18, azim=azim)

    fig = plt.figure(figsize=(14, 10))
    s3(fig, 221, Z, np.minimum(n_ops, 8), "laws 3D: n_ops", cmap="viridis")
    s3(fig, 222, Z, np.minimum(n_ops, 8), "laws 3D: n_ops (rotated)", cmap="viridis", azim=30)
    s3(fig, 223, Z, np.log10(outdeg + 1), "laws 3D: log out-degree", cmap="magma")
    s3(fig, 224, Z, np.minimum(depth, 6), "laws 3D: depth", cmap="viridis")
    fig.suptitle("FULL ETP law fingerprint PCA — 3D", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "laws_3d.png"), dpi=150)
    plt.close(fig)

    tv = [(1, C_TRUE, "True"), (0, C_FALSE, "False")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    s2(axes[0, 0], Zp, yp, "pairs: truth", cat=tv)
    s2(axes[0, 1], Zp, np.minimum(pair_ops, 16), "pairs: total ops", cmap="viridis")
    s2(axes[0, 2], Zp, yp, "pairs: truth (PC1 vs PC3)", cat=tv, dims=(0, 2))
    s2(axes[1, 0], Zp, yp, "pairs: truth (PC2 vs PC3)", cat=tv, dims=(1, 2))
    s2(axes[1, 1], Zp, np.minimum(pair_ops, 16), "pairs: ops (PC1 vs PC3)", cmap="viridis", dims=(0, 2))
    s2(axes[1, 2], Zp, np.minimum(pair_ops, 16), "pairs: ops (PC2 vs PC3)", cmap="viridis", dims=(1, 2))
    fig.suptitle(f"FULL ETP pair PCA (balanced {2*kt} of 22M pairs; law-PC concat)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pairs_2d.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(14, 10))
    s3(fig, 221, Zp, yp, "pairs 3D: truth", cat=tv)
    s3(fig, 222, Zp, yp, "pairs 3D: truth (rotated)", cat=tv, azim=45)
    s3(fig, 223, Zp, np.minimum(pair_ops, 16), "pairs 3D: total ops", cmap="viridis")
    s3(fig, 224, Zp, np.minimum(pair_ops, 16), "pairs 3D: ops (rotated)", cmap="viridis", azim=45)
    fig.suptitle("FULL ETP pair PCA — 3D", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pairs_3d.png"), dpi=150)
    plt.close(fig)

    # subsampled coords for the interactive artifact (2000 laws + all pairs)
    li = RNG.choice(n, size=min(2000, n), replace=False)
    coords = {
        "laws": dict(pc=Z[li, :3].round(3).tolist(),
                     n_ops=n_ops[li].tolist(), depth=depth[li].tolist(),
                     outdeg_log10=np.log10(outdeg[li] + 1).round(3).tolist(),
                     explained_variance=ev[:3].tolist(),
                     eq_index=li.tolist()),
        "pairs": dict(pc=Zp[:, :3].round(3).tolist(), truth=yp.tolist(),
                      pair_ops=pair_ops.tolist(),
                      explained_variance=evp[:3].tolist()),
    }
    with open(os.path.join(OUT, "full_pca_coords.json"), "w") as f:
        json.dump(coords, f)
    with open(os.path.join(OUT, "full_pca_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
