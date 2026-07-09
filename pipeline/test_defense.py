"""Self-tests for the defense layer: stats.py against known truths, ledger.py's
auto-downgrade rule. All offline, all synthetic. python test_defense.py"""
import json, math, os, tempfile
import numpy as np
import stats, ledger


def test_auroc_analytic():
    # two unit normals separated by d=1 -> AUROC = Phi(1/sqrt(2)) ~ 0.7602
    rng = np.random.default_rng(0)
    n = 20000
    y = np.r_[np.zeros(n), np.ones(n)]
    s = np.r_[rng.normal(0, 1, n), rng.normal(1, 1, n)]
    a = stats.auroc(y, s)
    assert abs(a - 0.7602) < 0.01, a


def test_auroc_ties_midrank():
    assert stats.auroc([0, 1], [0.5, 0.5]) == 0.5


def test_shuffled_labels_ci_spans_half():
    rng = np.random.default_rng(1)
    y = rng.permutation(np.r_[np.zeros(150), np.ones(150)])
    s = rng.normal(0, 1, 300)  # independent of y
    _, lo, hi = stats.auroc_ci(y, s, n_boot=500, seed=1)
    assert lo < 0.5 < hi, (lo, hi)


def test_delta_auroc_identical_scores_is_zero():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(100), np.ones(100)]
    s = rng.normal(y, 1)
    d, lo, hi = stats.delta_auroc_ci(y, s, s, n_boot=300)
    assert d == 0 and lo == 0 and hi == 0


def test_cluster_bootstrap_resists_pseudoreplication():
    # 40 independent Bernoulli items, each duplicated 10x (Family-A style).
    # Naive CI shrinks with the fake n; cluster CI must stay ~ as wide as the
    # CI of the 40 originals.
    rng = np.random.default_rng(3)
    base = (rng.random(40) < 0.5).astype(float)
    dup = np.repeat(base, 10)
    clusters = np.repeat(np.arange(40), 10)
    _, lo_n, hi_n = stats.rate_ci(dup, clusters=None, n_boot=800, seed=3)
    _, lo_c, hi_c = stats.rate_ci(dup, clusters=clusters, n_boot=800, seed=3)
    _, lo_o, hi_o = stats.rate_ci(base, clusters=None, n_boot=800, seed=3)
    assert (hi_c - lo_c) > 2.5 * (hi_n - lo_n), 'cluster CI failed to widen'
    assert abs((hi_c - lo_c) - (hi_o - lo_o)) < 0.06, 'cluster CI != original-scale CI'


def test_holm_textbook():
    adj = stats.holm([0.01, 0.04, 0.03, 0.005])
    assert [round(a, 4) for a in adj] == [0.03, 0.06, 0.06, 0.02], adj


def test_spearman_and_perm():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [2, 1, 4, 3, 6, 5, 8, 7]  # noisy monotone
    rho, p = stats.perm_test(x, y, n_perm=4000, seed=0, sided='greater')
    assert rho > 0.8 and p < 0.01, (rho, p)
    rho0 = stats.spearman([1, 2, 3], [3, 2, 1])
    assert abs(rho0 + 1.0) < 1e-9


def test_paired_perm_detects_shift():
    rng = np.random.default_rng(4)
    a = rng.normal(0.6, 0.1, 30)
    b = a - 0.15 + rng.normal(0, 0.02, 30)
    d, p = stats.paired_perm(a, b, n_perm=4000, seed=4)
    assert d > 0.1 and p < 0.01, (d, p)


def test_ledger_downgrade_rule():
    with tempfile.TemporaryDirectory() as tmp:
        led = os.path.join(tmp, 'ledger.jsonl')
        frozen = os.path.join(tmp, 'frozen.json')
        cfg = {'model': 'llama-3.1-8b', 'layer': 3, 'site': 'mean', 'C': 0.5}
        data = os.path.join(tmp, 'data-fresh.jsonl')
        open(data, 'w').write('{"x":1}\n')
        # (1) no frozen config -> downgrade (this IS the two-tier rule)
        r = ledger.register_run('probe', cfg, mode='confirmatory', hypothesis='H1',
                                data_files=[data], ledger_path=led, frozen_path=frozen)
        assert r['mode'] == 'exploratory' and 'downgraded' in str(r)
        # (2) frozen config matching -> granted
        json.dump({'hypotheses': {'H1': {
            'config_hash': ledger.hash_obj(cfg),
            'data_hashes': {'data-fresh.jsonl': ledger.hash_file(data)}}}},
            open(frozen, 'w'))
        r = ledger.register_run('probe', cfg, mode='confirmatory', hypothesis='H1',
                                data_files=[data], ledger_path=led, frozen_path=frozen)
        assert r['mode'] == 'confirmatory', r
        # (3) ANY config perturbation -> downgrade
        bad = dict(cfg, layer=4)
        r = ledger.register_run('probe', bad, mode='confirmatory', hypothesis='H1',
                                data_files=[data], ledger_path=led, frozen_path=frozen)
        assert r['mode'] == 'exploratory' and 'mismatch' in r['downgraded_from_confirmatory']
        # (4) data file edited -> downgrade
        open(data, 'a').write('{"x":2}\n')
        r = ledger.register_run('probe', cfg, mode='confirmatory', hypothesis='H1',
                                data_files=[data], ledger_path=led, frozen_path=frozen)
        assert r['mode'] == 'exploratory' and 'data hash' in r['downgraded_from_confirmatory']
        # ledger is append-only and complete
        assert sum(1 for _ in open(led)) == 4


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'  ok {fn.__name__}')
    print(f'{len(fns)}/{len(fns)} defense tests pass')


if __name__ == '__main__':
    main()
