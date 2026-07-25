"""Manifold checks: is the "law manifold" claim methodologically sound?

The isometry result says: a thin-plate spline maps the certified fp3 chart
onto law centroids better than shuffled charts (in-sample). That leaves four
methodological questions the spline fit itself cannot answer:

  C1  Effective dimensionality. Do the 247 law centroids actually occupy a
      low-dimensional structure (participation ratio, PCA spectrum), or is
      "3D chart" an arbitrary choice?
  C2  Held-out generalization. Does the chart -> activation map predict
      UNSEEN laws (leave-one-law-out), or does the spline memorize 136-247
      anchor points? LOO R^2 vs a permuted-chart LOO null is the real
      isometry test.
  C3  Linear vs nonlinear. Does a kernel map beat ridge regression out of
      sample? If not, the structure is a linear 3D subspace and "manifold"
      language is overclaiming (the Othello lesson).
  C4  Strength confound (gap G2). fp3's PC1 largely encodes implication
      strength (out-degree). How much of the alignment survives when the
      strength coordinate is dropped or residualized?

Inputs: the per-law centroid caches written by causal_patch.py
(analysis/causal/<model>_L<layer>/centroids_k*.npz) + fp3 from the task data.
Offline, deterministic. Writes manifold_checks.json next to each cache.

Run: ../causalab/.venv/bin/python causalab-integration/scripts/manifold_checks.py
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "tasks", "etp_implication", "data", "etp_pairs.json")
CAUSAL = os.path.join(HERE, "..", "analysis", "causal")
RNG = np.random.default_rng(2026)
N_PERM = 50


def load_task():
    d = json.load(open(DATA))
    laws, pairs = d["laws"], d["pairs"]
    fp3 = {k: np.array(v["fp3"]) for k, v in laws.items()}
    names = sorted(fp3)
    idx = {n: i for i, n in enumerate(names)}
    M = np.zeros((len(names), len(names)))
    for k, v in pairs.items():
        p, c = k.split("|")
        M[idx[p], idx[c]] = 1 if v else -1
    outdeg = (M == 1).sum(1)
    return names, np.stack([fp3[n] for n in names]), outdeg


def participation_ratio(Y):
    Yc = Y - Y.mean(0)
    ev = np.linalg.svd(Yc, compute_uv=False) ** 2
    ev = ev / ev.sum()
    return float(1.0 / (ev**2).sum()), ev[:10].tolist()


def loo_r2(X, Y, kind="ridge", gamma=None):
    """Leave-one-out R^2 of predicting centroids from chart coords."""
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import Ridge

    n = len(X)
    preds = np.zeros_like(Y)
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        if kind == "ridge":
            m = Ridge(alpha=1.0)
        else:
            m = KernelRidge(alpha=1.0, kernel="rbf", gamma=gamma)
        m.fit(X[tr], Y[tr])
        preds[i] = m.predict(X[i : i + 1])[0]
    ss_res = ((Y - preds) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    return 1 - ss_res / ss_tot


def loo_with_null(X, Y, kind, gamma=None, n_perm=N_PERM):
    r2 = loo_r2(X, Y, kind, gamma)
    nulls = []
    for _ in range(n_perm):
        Xp = X[RNG.permutation(len(X))]
        nulls.append(loo_r2(Xp, Y, kind, gamma))
    nulls = np.array(nulls)
    p = float(((nulls >= r2).sum() + 1) / (n_perm + 1))
    return float(r2), float(nulls.mean()), float(nulls.max()), p


def knn_overlap(A, B, k=10, n_null=500):
    from sklearn.neighbors import NearestNeighbors

    def nn(X):
        m = NearestNeighbors(n_neighbors=k + 1).fit(X)
        return m.kneighbors(X)[1][:, 1:]

    na, nb = nn(A), nn(B)
    n = len(A)
    obs = np.mean([len(set(na[i]) & set(nb[i])) for i in range(n)]) / k
    nulls = []
    for _ in range(n_null):
        perm = RNG.permutation(n)
        inv = np.empty(n, dtype=int)
        inv[perm] = np.arange(n)
        nulls.append(np.mean([len(set(na[i]) & set(inv[nb[perm[i]]])) for i in range(n)]) / k)
    nulls = np.array(nulls)
    return float(obs), float(nulls.mean()), float(((nulls >= obs).sum() + 1) / (n_null + 1))


def main():
    names, FP3, outdeg = load_task()
    for cache in sorted(glob.glob(os.path.join(CAUSAL, "*", "centroids_k*.npz"))):
        run_dir = os.path.dirname(cache)
        tag = os.path.basename(run_dir)
        z = np.load(cache)
        common = [n for n in names if n in z.files]
        Y = np.stack([z[n] for n in common]).astype(np.float64)
        X = np.stack([FP3[names.index(n)] for n in common])
        od = np.array([outdeg[names.index(n)] for n in common])
        # standardize chart coords; median-heuristic gamma for the RBF
        Xs = (X - X.mean(0)) / X.std(0)
        d2 = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1)
        gamma = 1.0 / np.median(d2[d2 > 0])
        rep = {"model_layer": tag, "n_laws": len(common)}

        pr, spectrum = participation_ratio(Y)
        rep["C1_participation_ratio"] = round(pr, 2)
        rep["C1_pca_spectrum_top10"] = [round(v, 4) for v in spectrum]

        for kind in ("ridge", "rbf"):
            r2, nm, nx, p = loo_with_null(Xs, Y, kind, gamma)
            rep[f"C2_loo_r2_{kind}"] = dict(
                r2=round(r2, 4), null_mean=round(nm, 4), null_max=round(nx, 4), p=round(p, 4)
            )
        rep["C3_nonlinear_gain"] = round(
            rep["C2_loo_r2_rbf"]["r2"] - rep["C2_loo_r2_ridge"]["r2"], 4
        )

        ov, ovn, ovp = knn_overlap(Xs, Y)
        rep["C2_knn_overlap_k10"] = dict(obs=round(ov, 4), null=round(ovn, 4), p=round(ovp, 4))

        # C4: strength ablations
        r2_pc1 = loo_r2(Xs[:, :1], Y, "ridge")
        r2_pc23 = loo_r2(Xs[:, 1:], Y, "ridge")
        r2_od = loo_r2(((od - od.mean()) / od.std()).reshape(-1, 1), Y, "ridge")
        # residualize chart on outdeg, refit
        odz = (od - od.mean()) / od.std()
        beta = np.linalg.lstsq(odz.reshape(-1, 1), Xs, rcond=None)[0]
        Xres = Xs - np.outer(odz, beta[0])
        r2_res, nm_res, _, p_res = loo_with_null(Xres, Y, "ridge")
        rep["C4_strength"] = dict(
            loo_r2_pc1_only=round(float(r2_pc1), 4),
            loo_r2_pc23_only=round(float(r2_pc23), 4),
            loo_r2_outdeg_only=round(float(r2_od), 4),
            loo_r2_chart_residualized_on_outdeg=round(float(r2_res), 4),
            residualized_null_mean=round(nm_res, 4),
            residualized_p=round(p_res, 4),
        )

        out = os.path.join(run_dir, "manifold_checks.json")
        json.dump(rep, open(out, "w"), indent=1)
        print(f"== {tag} (n={len(common)}) ==")
        print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
