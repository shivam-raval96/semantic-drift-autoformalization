"""Cheap-signal detector baselines for VBU (valid-but-unfaithful) translations.

Research question 'Prediction': can unfaithful-but-valid outputs be flagged by cheap
signals, before/alongside full verification? Each detector maps ONE translation
attempt to a suspicion score (higher = more likely VBU) using only deployment-time
information: the NL surface, the model's own output, and sibling outputs from other
seeds. Gold labels, intended laws, drift targets and certificates are structurally
unavailable to detectors: features are built from a whitelisted view (detector_view).

Offline demo: k mock-backend seeds stand in for k samples of a real model.
    python baselines.py [k]
For real models: pipeline/pilot.py writes results-<model>-s<k>.jsonl per seed; load
those instead of the mock runs (same schema, same code path).
"""
import os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pipeline'))
from laws import ParseError, count_ops, parse_law_str, render, variables_of  # noqa: E402

# The ONLY fields a detector may see. 'verdict' etc. are used by the EVALUATOR only.
DETECTOR_FIELDS = ('item_id', 'family', 'task', 'surface', 'register', 'model_output')

def detector_view(row):
    return {k: row[k] for k in DETECTOR_FIELDS if k in row}

def canonical(output):
    """Canonical rendering of a parsable output, else None (invalid)."""
    try:
        lhs, rhs = parse_law_str(output)
        return f"{render(lhs)} = {render(rhs)}"
    except ParseError:
        return None

# ---------- detectors: view (+ sibling views from other seeds) -> score ----------

def self_inconsistency(view, sibling_views):
    """1 - share of sibling seeds whose canonical output matches this one.
    Rationale: temperature/seed variance concentrates on items the model is unsure
    about; disagreement is a gold-free instability signal."""
    mine = canonical(view['model_output'])
    sibs = [canonical(v['model_output']) for v in sibling_views]
    if mine is None or not sibs:
        return 1.0
    return 1.0 - sum(1 for s in sibs if s == mine) / len(sibs)

def modal_deviation(view, sibling_views):
    """1 if this output differs from the modal canonical output across all seeds
    (majority vote as the reference), else 0."""
    outs = [canonical(view['model_output'])] + [canonical(v['model_output']) for v in sibling_views]
    outs = [o for o in outs if o is not None]
    if not outs:
        return 1.0
    modal, _ = Counter(outs).most_common(1)[0]
    return 0.0 if canonical(view['model_output']) == modal else 1.0

def complexity_mismatch(view, sibling_views):
    """|ops in output - entities hinted by the surface| (weak, single-attempt signal:
    counts single-letter entity mentions in the surface as a proxy for expected size)."""
    c = canonical(view['model_output'])
    if c is None:
        return 10.0
    lhs, rhs = parse_law_str(c)
    ops = count_ops(lhs) + count_ops(rhs)
    entities = len({w for w in view['surface'].replace(',', ' ').split() if len(w) == 1})
    return abs(ops - 2 * max(entities - 1, 1))

DETECTORS = {'self_inconsistency': self_inconsistency,
             'modal_deviation': modal_deviation,
             'complexity_mismatch': complexity_mismatch}

# ---------- evaluation (gold used HERE only) ----------

def auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney, ties averaged)."""
    pairs = sorted(zip(scores, labels))
    ranks, i = {}, 0
    ranked = []
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average rank for the tie block
        ranked += [(avg_rank, lab) for _, lab in pairs[i:j]]
        i = j
    pos = [r for r, lab in ranked if lab == 1]
    neg = [r for r, lab in ranked if lab == 0]
    if not pos or not neg:
        return None
    return (sum(pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def evaluate(runs):
    """runs: list of scored results lists (one per seed), pipeline schema.
    Scores every valid translation attempt with every detector; AUROC vs VBU gold."""
    by_item = defaultdict(dict)
    for s, rows in enumerate(runs):
        for r in rows:
            if r['task'] in ('translation', 'transcription'):
                by_item[r['item_id']][s] = r
    results = {}
    for name, det in DETECTORS.items():
        scores, labels = [], []
        for item_id, per_seed in by_item.items():
            for s, row in per_seed.items():
                if row['verdict'].startswith('syntax'):
                    continue  # detectors target VBU within the valid column
                sibs = [detector_view(v) for t, v in per_seed.items() if t != s]
                scores.append(det(detector_view(row), sibs))
                labels.append(1 if row['verdict'].startswith('drift') else 0)
        results[name] = (auroc(scores, labels), len(labels), sum(labels))
    return results

def run_mock_seeds(k=5):
    from dataset import generate
    from pilot import MockBackend, run, score
    data = generate(0)
    return [score(run(data, MockBackend(seed))) for seed in range(k)]

def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"mock backend, {k} seeds (pipeline-validation numbers, not findings)")
    results = evaluate(run_mock_seeds(k))
    print(f"{'detector':22s} {'AUROC':>6s} {'n':>6s} {'VBU':>5s}")
    for name, (a, n, pos) in sorted(results.items(), key=lambda kv: -(kv[1][0] or 0)):
        print(f"{name:22s} {a:6.3f} {n:6d} {pos:5d}")

if __name__ == '__main__':
    main()
