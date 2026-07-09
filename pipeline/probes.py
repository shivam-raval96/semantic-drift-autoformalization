"""Probe pair + geometry diagnostics on residual-stream activations.

Locked design rules (Goodfire Neural Geometry series, S4/S5):
- RULE 1: every probe target is evaluated with a LINEAR probe (logistic regression)
  AND a NONLINEAR probe (small MLP) on identical splits. A linear null is never
  reported as 'not represented'; linear-fails/nonlinear-succeeds is flagged as
  curvature evidence.
- RULE 2: no elicited self-report anywhere in this file. Labels come from the
  certificate checker; features come from activations. (Self-report is measured
  separately, as a foil — selfreport.py.)

Targets:
  monitor  — will this translation drift? (valid outputs only; binary, AUROC)
  intent   — which law was intended? (multiclass; ELK-lite: decodable even when
             the output drifted?)
  register — control task to calibrate probe power on this n.

Split: pre-registered template split (dataset.probe_split) — register held out for
A/B/D items; hash-by-law for family E. Run: python probes.py [acts.npz results.jsonl]
"""
import glob, json, sys
import numpy as np

def load(acts_path, results_path):
    acts = np.load(acts_path)
    rows = [json.loads(l) for l in open(results_path, encoding='utf-8')]
    rows = [r for r in rows if r['item_id'] in acts.files]
    return acts, rows

def _xy(acts, rows, layer, target):
    X = np.stack([acts[r['item_id']][layer].astype(np.float32) for r in rows])
    if target == 'monitor':
        y = np.array([1 if r['verdict'].startswith('drift') else 0 for r in rows])
    elif target == 'intent':
        labels = sorted({r['intended_lid'] for r in rows})
        y = np.array([labels.index(r['intended_lid']) for r in rows])
    else:
        labels = sorted({r['register'] for r in rows})
        y = np.array([labels.index(r['register']) for r in rows])
    return X, y

def probe_pair(Xtr, ytr, Xte, yte, binary):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    out = {}
    for name, clf in (('linear', LogisticRegression(max_iter=4000, C=0.5)),
                      ('nonlinear', MLPClassifier(hidden_layer_sizes=(64,), max_iter=3000,
                                                  random_state=0))):
        sc = StandardScaler().fit(Xtr)
        clf.fit(sc.transform(Xtr), ytr)
        Xte_s = sc.transform(Xte)
        if binary and len(set(yte)) == 2:
            out[name] = roc_auc_score(yte, clf.predict_proba(Xte_s)[:, 1])
        else:
            out[name] = float((clf.predict(Xte_s) == yte).mean())
    return out

def split_rows(rows, mode='mixed'):
    """Template split. 'mixed': register-held-out for A/B/D + law-hash for E (fine
    for binary targets). Multiclass targets need the split matched to the target:
    intent needs 'register' (every class must appear in train — a held-out LAW is
    an unseen class, 0% by construction); the register control needs 'law'."""
    from dataset import probe_split
    if mode == 'register':
        return probe_split([r for r in rows if r['family'] != 'E'],
                           mode='register', held_out='paraphrase')
    if mode == 'law':
        return probe_split(rows, mode='law', seed=0)
    abd = [r for r in rows if r['family'] != 'E']
    e = [r for r in rows if r['family'] == 'E']
    tr1, te1 = probe_split(abd, mode='register', held_out='paraphrase') if abd else ([], [])
    tr2, te2 = probe_split(e, mode='law', seed=0) if e else ([], [])
    return tr1 + tr2, te1 + te2

def evaluate(acts, rows):
    n_layers = acts[rows[0]['item_id']].shape[0]
    valid = [r for r in rows if not r['verdict'].startswith('syntax')]
    report = {}
    for target, pool, binary, mode in (
            ('monitor', valid, True, 'mixed'),
            ('intent', [r for r in rows if r['family'] != 'E'], False, 'register'),
            ('register', [r for r in rows if r['family'] != 'E'], False, 'law')):
        tr, te = split_rows(pool, mode)
        if len(te) < 8 or len({_label(r, target) for r in te}) < 2:
            report[target] = {'skipped': f'n_test={len(te)}, too few/degenerate'}
            continue
        per_layer = {}
        for layer in range(0, n_layers, max(1, n_layers // 12)):
            Xtr, ytr = _xy(acts, tr, layer, target)
            Xte, yte = _xy(acts, te, layer, target)
            per_layer[layer] = probe_pair(Xtr, ytr, Xte, yte, binary)
        report[target] = {'n_train': len(tr), 'n_test': len(te), 'layers': per_layer}
    return report

def _label(r, target):
    return (r['verdict'].startswith('drift') if target == 'monitor'
            else r['intended_lid'] if target == 'intent' else r['register'])

def geometry(acts, rows):
    """Which Goodfire-world: do intended laws form recoverable clusters?
    Silhouette of law-identity clusters per layer (linear-view diagnostic)."""
    from sklearn.metrics import silhouette_score
    from collections import Counter
    counts = Counter(r['intended_lid'] for r in rows)
    keep = [r for r in rows if counts[r['intended_lid']] >= 3]
    labels = [r['intended_lid'] for r in keep]
    n_layers = acts[rows[0]['item_id']].shape[0]
    out = {}
    for layer in range(0, n_layers, max(1, n_layers // 12)):
        X = np.stack([acts[r['item_id']][layer].astype(np.float32) for r in keep])
        out[layer] = float(silhouette_score(X, labels))
    return out

def main():
    acts_path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob('acts-hf-*.npz'))[-1]
    res_path = sys.argv[2] if len(sys.argv) > 2 else acts_path.replace('acts-', 'results-').replace('.npz', '.jsonl')
    acts, rows = load(acts_path, res_path)
    print(f"{len(rows)} items with activations from {acts_path}")
    rep = evaluate(acts, rows)
    lines = [f"# Probe report — {res_path}", "",
             "Rule 1 in force: linear+nonlinear pairs; divergence = curvature evidence, "
             "a linear null alone is never 'not represented'.", ""]
    for target, r in rep.items():
        if 'skipped' in r:
            lines += [f"## {target} — SKIPPED ({r['skipped']})", ""]
            continue
        metric = 'AUROC' if target == 'monitor' else 'acc'
        lines += [f"## {target}  (train {r['n_train']} / test {r['n_test']}, {metric})",
                  "| layer | linear | nonlinear | note |", "|---|---|---|---|"]
        for layer, pair in r['layers'].items():
            note = 'CURVATURE?' if pair['nonlinear'] - pair['linear'] > 0.1 else ''
            lines.append(f"| {layer} | {pair['linear']:.3f} | {pair['nonlinear']:.3f} | {note} |")
        lines.append("")
    geo = geometry(acts, rows)
    lines += ["## Geometry: law-identity cluster silhouette by layer",
              "| layer | silhouette |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in geo.items()]
    text = "\n".join(lines) + "\n"
    out = res_path.replace('results-', 'probe-report-').replace('.jsonl', '.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    print(f"wrote {out}")

if __name__ == '__main__':
    main()
