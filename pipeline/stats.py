"""Confirmatory statistics: cluster bootstrap CIs, permutation tests, Holm.

DEFENSE-PLAN Phase 0 item 3. Design rules:
- Every CI is a percentile CLUSTER bootstrap: resample clusters (items/laws),
  not rows, so per-item candidate multiplicity and per-law surface multiplicity
  cannot fake precision. Pass clusters=None only when rows ARE the units.
- Permutation tests are exact-null, two-sided unless stated.
- No test here decides anything alone: kill conditions live in the prereg;
  this module only computes the numbers they reference.
Self-verifying: test_defense.py checks known-truth synthetic cases (analytic
AUROC, shuffled-label nulls, Holm textbook example, cluster-vs-naive widening).
"""
import numpy as np


def _cluster_ids(n, clusters):
    if clusters is None:
        return np.arange(n)
    c = np.asarray(clusters)
    assert len(c) == n, 'clusters must align with rows'
    return c


def bootstrap_ci(rows_stat, n_rows, clusters=None, n_boot=2000, alpha=0.05, seed=0):
    """Generic percentile cluster bootstrap.
    rows_stat: callable(row_indices) -> float, computed on a resample.
    Returns (point, lo, hi). NaN resamples (degenerate draws) are dropped."""
    rng = np.random.default_rng(seed)
    ids = _cluster_ids(n_rows, clusters)
    uniq = np.unique(ids)
    by_cluster = {u: np.where(ids == u)[0] for u in uniq}
    point = rows_stat(np.arange(n_rows))
    stats = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_cluster[u] for u in draw])
        s = rows_stat(idx)
        if not np.isnan(s):
            stats.append(s)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def auroc(y, score):
    """Rank-based AUROC (no sklearn dependency for the hot path)."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(score, dtype=float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # midranks for ties
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_ci(y, score, clusters=None, **kw):
    y = np.asarray(y)
    s = np.asarray(score)
    return bootstrap_ci(lambda idx: auroc(y[idx], s[idx]), len(y), clusters, **kw)


def delta_auroc_ci(y, score_a, score_b, clusters=None, **kw):
    """CI on AUROC(a) - AUROC(b), computed on SHARED resamples (paired)."""
    y = np.asarray(y)
    a, b = np.asarray(score_a), np.asarray(score_b)
    return bootstrap_ci(lambda idx: auroc(y[idx], a[idx]) - auroc(y[idx], b[idx]),
                        len(y), clusters, **kw)


def rate_ci(successes, clusters=None, **kw):
    x = np.asarray(successes, dtype=float)
    return bootstrap_ci(lambda idx: x[idx].mean(), len(x), clusters, **kw)


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind='mergesort')
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(1, len(v) + 1)
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den else float('nan')


def perm_test(x, y, stat=spearman, n_perm=10000, seed=0, sided='two'):
    """Permutation p-value for association between x and y under shuffling y."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    obs = stat(x, y)
    null = np.array([stat(x, rng.permutation(y)) for _ in range(n_perm)])
    if sided == 'two':
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    elif sided == 'greater':
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(null <= obs) + 1) / (n_perm + 1)
    return float(obs), float(p)


def paired_perm(a, b, n_perm=10000, seed=0):
    """Sign-flip permutation test for mean(a-b) != 0 on paired data."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    obs = d.mean()
    null = np.array([(d * rng.choice([-1, 1], size=len(d))).mean()
                     for _ in range(n_perm)])
    return float(obs), float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
