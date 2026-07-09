"""Run: python test_detectors.py — offline, deterministic."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import (DETECTOR_FIELDS, auroc, canonical, detector_view,
                       evaluate, run_mock_seeds, self_inconsistency)

def test_auroc_basics():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0   # perfect
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0   # perfectly wrong
    assert auroc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5   # ties -> chance
    assert auroc([0.1, 0.2], [1, 1]) is None                  # degenerate

def test_detector_view_blocks_gold():
    row = {'item_id': 'x', 'task': 'translation', 'surface': 's', 'register': 'r',
           'family': 'A', 'model_output': 'x * y = y * x',
           'verdict': 'drift: weaker', 'intended_law': 'SECRET', 'drift_law': 'SECRET',
           'drift_move': 'weakening', 'gold': 'SECRET', 'evidence': {}, 'tclass': 'c'}
    view = detector_view(row)
    assert set(view) <= set(DETECTOR_FIELDS)
    for forbidden in ('verdict', 'intended_law', 'drift_law', 'drift_move', 'gold',
                      'evidence', 'tclass'):
        assert forbidden not in view

def test_canonicalization():
    assert canonical('x*y = y*x') == canonical('x * y   =   (y * x)')
    assert canonical('x * (y =') is None

def test_self_inconsistency_shape():
    a = {'model_output': 'x * y = y * x', 'surface': ''}
    b = {'model_output': 'x*y = y*x', 'surface': ''}
    c = {'model_output': 'x = x', 'surface': ''}
    assert self_inconsistency(a, [b, b]) == 0.0      # canonical agreement
    assert self_inconsistency(a, [c, c]) == 1.0      # total disagreement
    assert self_inconsistency(a, [b, c]) == 0.5

def test_consistency_detectors_beat_chance_on_mock():
    # mock drift is per-seed stochastic, so cross-seed disagreement must carry
    # signal; deterministic seeds make this a stable regression test
    results = evaluate(run_mock_seeds(5))
    assert results['self_inconsistency'][0] > 0.65, results
    assert results['modal_deviation'][0] > 0.65, results

if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
