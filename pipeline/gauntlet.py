"""Baseline gauntlet: try to kill the drift-monitor probe with input-side features.

The probe reads the residual stream at the final PROMPT token — a deterministic
function of the input. If input-observable features (register, bag-of-words, or
ANOTHER model's encoding of the same prompt) predict drift as well as the target
model's own activations, the probe is a difficulty detector, not a monitor.
Every contender uses the same split as probes.py; probe layer is chosen by
TRAIN-ONLY cross-validation (no test peeking).

    python gauntlet.py [target-acts.npz] [target-results.jsonl]
Writes gauntlet-report.md.
"""
import glob, json, sys
import numpy as np
from probes import load, split_rows
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer

def y_of(rows):
    return np.array([1 if r['verdict'].startswith('drift') else 0 for r in rows])

def fit_auc(Xtr, ytr, Xte, yte):
    sc = StandardScaler(with_mean=False) if hasattr(Xtr, 'tocsr') else StandardScaler()
    clf = LogisticRegression(max_iter=4000, C=0.5)
    clf.fit(sc.fit_transform(Xtr), ytr)
    return roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])

def acts_X(acts, rows, layer):
    return np.stack([acts[r['item_id']][layer].astype(np.float32) for r in rows])

def pick_layer_cv(acts, rows, y):
    """Train-only layer selection: 5-fold CV AUROC per layer."""
    best, best_score = None, -1
    n_layers = acts[rows[0]['item_id']].shape[0]
    for layer in range(2, n_layers, 2):
        X = StandardScaler().fit_transform(acts_X(acts, rows, layer))
        s = cross_val_score(LogisticRegression(max_iter=2000, C=0.5), X, y,
                            cv=5, scoring='roc_auc').mean()
        if s > best_score:
            best, best_score = layer, s
    return best, best_score

def main():
    tacts = sys.argv[1] if len(sys.argv) > 1 else 'acts-hf-meta-llama-Llama-3.1-8B-Instruct.npz'
    tres = sys.argv[2] if len(sys.argv) > 2 else tacts.replace('acts-', 'results-').replace('.npz', '.jsonl')
    acts, rows = load(tacts, tres)
    valid = [r for r in rows if not r['verdict'].startswith('syntax')]
    tr, te = split_rows(valid, 'mixed')
    ytr, yte = y_of(tr), y_of(te)
    lines = [f"# Gauntlet — can input-side features match the probe? ({tres})", "",
             f"train {len(tr)} (drift {ytr.sum()}) / test {len(te)} (drift {yte.sum()}), "
             "identical split for every contender.", "",
             "| contender | internals used | AUROC |", "|---|---|---|"]
    results = {}

    # 1. register one-hot (the confound suspect)
    regs = sorted({r['register'] for r in valid})
    def reg_X(rs):
        return np.array([[1.0 if r['register'] == g else 0.0 for g in regs] for r in rs])
    results['register one-hot'] = ('none', fit_auc(reg_X(tr), ytr, reg_X(te), yte))

    # 2. bag-of-words on the surface
    cv = CountVectorizer(max_features=300, binary=True, ngram_range=(1, 2))
    Xtr_b = cv.fit_transform([r['surface'] for r in tr])
    Xte_b = cv.transform([r['surface'] for r in te])
    results['bag-of-words (1-2gr)'] = ('none', fit_auc(Xtr_b, ytr, Xte_b, yte))

    # 3. cheap surface stats
    def stats_X(rs):
        return np.array([[len(r['surface']), r['surface'].count('('),
                          r['surface'].count(','), r.get('n_ops') or 0,
                          len(r['surface'].split())] for r in rs], dtype=float)
    results['surface stats'] = ('none', fit_auc(stats_X(tr), ytr, stats_X(te), yte))

    # 4. cross-model encoder: predict the TARGET's drift from ANOTHER model's
    #    activations on the same prompt (input encoding, no target internals)
    for other in sorted(glob.glob('acts-hf-Qwen-Qwen2.5-*.npz')):
        oacts = np.load(other)
        keep_tr = [r for r in tr if r['item_id'] in oacts.files]
        keep_te = [r for r in te if r['item_id'] in oacts.files]
        layer, _ = pick_layer_cv(oacts, keep_tr, y_of(keep_tr))
        auc = fit_auc(acts_X(oacts, keep_tr, layer), y_of(keep_tr),
                      acts_X(oacts, keep_te, layer), y_of(keep_te))
        name = other.split('acts-hf-')[1].replace('.npz', '')
        results[f'encoder: {name} L{layer}'] = ('other model', auc)

    # 5. the probe itself, layer chosen by train-only CV
    layer, cv_score = pick_layer_cv(acts, tr, ytr)
    probe_auc = fit_auc(acts_X(acts, tr, layer), ytr, acts_X(acts, te, layer), yte)
    results[f'TARGET probe L{layer} (CV-picked)'] = ('target model', probe_auc)

    # 6. probe + register stacked (does register ADD anything to the probe?)
    Xtr_s = np.hstack([acts_X(acts, tr, layer), reg_X(tr)])
    Xte_s = np.hstack([acts_X(acts, te, layer), reg_X(te)])
    results['probe + register'] = ('target model', fit_auc(Xtr_s, ytr, Xte_s, yte))

    for name, (kind, auc) in results.items():
        lines.append(f"| {name} | {kind} | {auc:.3f} |")

    # 7. within-register: register constant -> its confound is removed by design
    lines += ["", "## Within-register AUROC (register held constant; small n, directional)",
              "| register | n(test) | drift | probe | bag-of-words |", "|---|---|---|---|---|"]
    for g in regs:
        sub_tr = [r for r in tr if r['register'] == g]
        sub_te = [r for r in te if r['register'] == g]
        pool = sub_tr + sub_te
        if len(pool) < 20 or len(set(y_of(pool))) < 2:
            continue
        # small n: 5-fold CV within the register pool (no template split possible here)
        yp = y_of(pool)
        Xp = StandardScaler().fit_transform(acts_X(acts, pool, layer))
        pa = cross_val_score(LogisticRegression(max_iter=2000, C=0.5), Xp, yp,
                             cv=5, scoring='roc_auc').mean()
        Xb = CountVectorizer(max_features=300, binary=True).fit_transform(
            [r['surface'] for r in pool]).toarray()
        ba = cross_val_score(LogisticRegression(max_iter=2000, C=0.5), Xb, yp,
                             cv=5, scoring='roc_auc').mean()
        lines.append(f"| {g} | {len(pool)} | {yp.sum()} | {pa:.3f} | {ba:.3f} |")

    probe = probe_auc
    best_base = max(a for k, (kind, a) in results.items() if kind != 'target model')
    verdict = ('SURVIVES: probe beats best input-side baseline by '
               if probe - best_base > 0.03 else
               'DOES NOT CLEARLY SURVIVE: probe within noise of input-side baseline, delta ')
    lines += ["", f"## Verdict: {verdict}{probe - best_base:+.3f}",
              "(threshold 0.03 is informal; bootstrap before quoting anywhere)"]
    open('gauntlet-report.md', 'w', encoding='utf-8').write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == '__main__':
    main()
