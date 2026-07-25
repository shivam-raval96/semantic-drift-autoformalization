"""Figures for the manifold checks (C1/C2/C4) from cached law centroids.

Per model (each cached centroids_k*.npz under analysis/causal/<tag>/):
  <tag>_loo_scatter.png  C2 as a picture: LOO-predicted centroid positions
                         (ridge from the certified chart) vs actual, drawn in
                         the centroids' own top-2 PC plane, with error
                         segments; right panel = same under a permuted chart.
  <tag>_checks.png       C1 spectrum bars + C4 strength-ablation bars.

Offline, deterministic. Outputs to analysis/causal/figures/.
Run: ../causalab/.venv/bin/python causalab-integration/scripts/manifold_viz.py
"""

import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "tasks", "etp_implication", "data", "etp_pairs.json")
CAUSAL = os.path.join(HERE, "..", "analysis", "causal")
FIGS = os.path.join(CAUSAL, "figures")
os.makedirs(FIGS, exist_ok=True)
RNG = np.random.default_rng(2026)


def loo_ridge_preds(X, Y, alpha=1.0):
    n, k = X.shape
    preds = np.zeros_like(Y)
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        Xt, Yt = X[tr], Y[tr]
        mu_x, mu_y = Xt.mean(0), Yt.mean(0)
        Xc, Yc = Xt - mu_x, Yt - mu_y
        W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(k), Xc.T @ Yc)
        preds[i] = (X[i] - mu_x) @ W + mu_y
    return preds


def r2(Y, P):
    return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def main():
    d = json.load(open(DATA))
    laws = d["laws"]
    names = sorted(laws)
    FP3 = np.stack([np.array(laws[n]["fp3"]) for n in names])

    for cache in sorted(glob.glob(os.path.join(CAUSAL, "*", "centroids_k*.npz"))):
        tag = os.path.basename(os.path.dirname(cache))
        z = np.load(cache)
        common = [n for n in names if n in z.files]
        Y = np.stack([z[n] for n in common]).astype(np.float64)
        X = np.stack([FP3[names.index(n)] for n in common])
        Xs = (X - X.mean(0)) / X.std(0)

        # centroid top-2 PC plane for drawing
        Yc = Y - Y.mean(0)
        U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
        P2 = Vt[:2].T

        preds_true = loo_ridge_preds(Xs, Y)
        Xp = Xs[RNG.permutation(len(Xs))]
        preds_perm = loo_ridge_preds(Xp, Y)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
        for ax, preds, title in (
            (axes[0], preds_true, f"certified chart  (LOO R² = {r2(Y, preds_true):.3f})"),
            (axes[1], preds_perm, f"permuted chart  (LOO R² = {r2(Y, preds_perm):.3f})"),
        ):
            A, B = Yc @ P2, (preds - Y.mean(0)) @ P2
            for a, b in zip(A, B):
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#b0b0b0",
                        lw=0.6, alpha=0.6, zorder=1)
            ax.scatter(A[:, 0], A[:, 1], s=16, c="#1f77b4", label="actual centroid",
                       zorder=3, edgecolors="white", linewidths=0.3)
            ax.scatter(B[:, 0], B[:, 1], s=12, c="#d62728", marker="x",
                       label="LOO prediction from chart", zorder=2)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("centroid PC1")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("centroid PC2")
        axes[0].legend(fontsize=9, loc="lower right")
        fig.suptitle(f"{tag}: held-out law centroids predicted from the certified chart "
                     f"(each x = a law never seen by the fit)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"{tag}_loo_scatter.png"), dpi=150)
        plt.close(fig)

        # C1 spectrum + C4 ablations from the saved report
        rep = json.load(open(os.path.join(os.path.dirname(cache), "manifold_checks.json")))
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        spec = rep["C1_pca_spectrum_top10"]
        axes[0].bar(range(1, 11), [100 * v for v in spec], color="#1f77b4")
        axes[0].set_title(f"C1: centroid PCA spectrum — participation ratio "
                          f"{rep['C1_participation_ratio']}", fontsize=11)
        axes[0].set_xlabel("component")
        axes[0].set_ylabel("% variance")
        axes[0].grid(alpha=0.2, axis="y")
        c4 = rep["C4_strength"]
        bars = [
            ("full chart", rep["C2_loo_r2_ridge"]["r2"], "#2ca02c"),
            ("chart ⊥ out-degree", c4["loo_r2_chart_residualized_on_outdeg"], "#2ca02c"),
            ("PC2–3 only", c4["loo_r2_pc23_only"], "#1f77b4"),
            ("PC1 only", c4["loo_r2_pc1_only"], "#1f77b4"),
            ("out-degree only", c4["loo_r2_outdeg_only"], "#d62728"),
            ("perm null (mean)", rep["C2_loo_r2_ridge"]["null_mean"], "#7f7f7f"),
        ]
        ypos = np.arange(len(bars))[::-1]
        axes[1].barh(ypos, [b[1] for b in bars], color=[b[2] for b in bars])
        axes[1].set_yticks(ypos, [b[0] for b in bars], fontsize=9)
        axes[1].axvline(0, color="#555", lw=1)
        axes[1].set_title("C2/C4: held-out (LOO) R² by predictor", fontsize=11)
        axes[1].set_xlabel("LOO R² on unseen-law centroids")
        axes[1].grid(alpha=0.2, axis="x")
        fig.suptitle(f"{tag}: manifold checks", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"{tag}_checks.png"), dpi=150)
        plt.close(fig)
        print("wrote figures for", tag)


if __name__ == "__main__":
    main()
