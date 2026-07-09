"""Certified Semantic Entropy (CSE) — the pre-hoc ambiguity instrument.

Semantic entropy (Kuhn/Gal/Farquhar) with certified-equivalence clustering
replacing NLI: sample k formalizations of each surface from a translator
ENSEMBLE (input-side property, not the tested model's), partition outputs by
certified equivalence, take entropy over the partition. Unparsable outputs form
their own class ('L'). CSE(surface)=0 -> everyone converges on one meaning.

Validates the ambiguity ladder: CSE must be ~0 at A0-A2 and rise at A3
(grouping-implicit; enumerable readings) or the ladder's ordering fails.

    python cse.py sample   # ensemble sampling over data-ladder.jsonl (~$1)
    python cse.py analyze  # cluster + entropy + ladder verdict -> cse-report.md
"""
import json, math, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from laws import Law, ParseError, parse_law_str, render
from magma import classify_relation
from pilot import PROMPTS, ApiBackend, extract_equation

ENSEMBLE = ['meta-llama/llama-3.3-70b-instruct', 'openai/gpt-4o-mini',
            'anthropic/claude-sonnet-4.6']
K = 3          # samples per model per surface
TEMP = 0.8

def canon(s):
    try:
        lhs, rhs = parse_law_str(s)
        return f"{render(lhs)} = {render(rhs)}"
    except ParseError:
        return None

_EQ_MEMO = {}
def equivalent(a, b):
    key = (min(a, b), max(a, b))
    if key not in _EQ_MEMO:
        la = Law('a', 'a', *parse_law_str(a))
        lb = Law('b', 'b', *parse_law_str(b))
        _EQ_MEMO[key] = classify_relation(la, lb)[0].startswith('equivalent')
    return _EQ_MEMO[key]

def sample(variant='hot'):
    """v0.1: A0 uses the TRANSCRIPTION prompt (its design); 'hot' = temp 0.8 k=3
    (includes decoding noise); 't0' = deterministic, 1/model (pure inter-model
    disagreement = input ambiguity without decoding randomness)."""
    items = [json.loads(l) for l in open('data-ladder.jsonl', encoding='utf-8')]
    temp, k = (TEMP, K) if variant == 'hot' else (None, 1)
    backends = [ApiBackend(m, temperature=temp) for m in ENSEMBLE]
    def one(args):
        it, b, kk = args
        tmpl = 'transcription' if it['rung'] == 0 else 'translation'
        prompt = PROMPTS[tmpl].format(surface=it['surface'])
        return {'item_id': it['item_id'], 'rung': it['rung'],
                'intended_law': it['intended_law'], 'model': b.model, 'k': kk,
                'output': extract_equation(b.transport(prompt))}
    jobs = [(it, b, kk) for it in items for b in backends for kk in range(k)]
    with ThreadPoolExecutor(max_workers=12) as ex:
        out = list(ex.map(one, jobs))
    path = f'cse-samples-v01-{variant}.jsonl'
    with open(path, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o) + '\n')
    print(f'wrote {path} ({len(out)} samples)')

def analyze(variant='hot'):
    rows = [json.loads(l) for l in open(f'cse-samples-v01-{variant}.jsonl', encoding='utf-8')]
    by_item = defaultdict(list)
    for r in rows:
        by_item[r['item_id']].append(r)
    results = []
    for item_id, rs in by_item.items():
        canons = [canon(r['output']) for r in rs]
        classes = []            # list of (representative_canon or 'L', count)
        for c in canons:
            if c is None:
                for cl in classes:
                    if cl[0] == 'L':
                        cl[1] += 1
                        break
                else:
                    classes.append(['L', 1])
                continue
            for cl in classes:
                if cl[0] != 'L' and (cl[0] == c or equivalent(cl[0], c)):
                    cl[1] += 1
                    break
            else:
                classes.append([c, 1])
        n_all = sum(ct for _, ct in classes)
        l_ct = sum(ct for c, ct in classes if c == 'L')
        sem = [(c, ct) for c, ct in classes if c != 'L']
        n = sum(ct for _, ct in sem)
        cse = (-sum((ct / n) * math.log2(ct / n) for _, ct in sem)) if n else float('nan')
        modal = (max(ct for _, ct in sem) / n) if n else 0.0
        results.append({'item_id': item_id, 'rung': rs[0]['rung'],
                        'intended_law': rs[0]['intended_law'], 'n': n_all,
                        'l_rate': round(l_ct / n_all, 3),
                        'n_classes': len(sem), 'cse': round(cse, 4),
                        'disagreement': round(1 - modal, 4),
                        'classes': [[c, ct] for c, ct in classes]})
    with open('cse-results.jsonl', 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    import numpy as np
    means = {}
    lines = [f"# CSE v0.1 report ({variant}) — L-class excluded from entropy", "",
             "| rung | mean CSE | CSE_excess (vs A1) | L-rate | mean #sem-classes |",
             "|---|---|---|---|---|"]
    for rung in range(4):
        rs = [r for r in results if r['rung'] == rung]
        means[rung] = float(np.nanmean([r['cse'] for r in rs]))
    floor = means[1]
    for rung in range(4):
        rs = [r for r in results if r['rung'] == rung]
        lines.append(f"| A{rung} | {means[rung]:.3f} | {means[rung]-floor:+.3f} | "
                     f"{np.mean([r['l_rate'] for r in rs]):.3f} | "
                     f"{np.nanmean([r['n_classes'] for r in rs]):.2f} |")
    mono = means[0] <= means[1] <= means[3] and means[2] <= means[3]
    from scipy import stats as st
    rho = st.spearmanr([r['rung'] for r in results],
                       [r['cse'] for r in results]).statistic
    lines += ["", f"Ladder verdict: CSE monotone A0<=A1<=A3 (A2 lexical-noise rung "
              f"compared separately): {'PASS' if mono else 'FAIL'}; "
              f"Spearman(rung, CSE) = {rho:.3f}.",
              f"Ensemble: {ENSEMBLE}, k={K}/model, temp={TEMP}; clustering by "
              "certified equivalence (unparsable = own class)."]
    open('cse-report.md', 'w', encoding='utf-8').write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == '__main__':
    variant = 't0' if 't0' in sys.argv else 'hot'
    sample(variant) if 'sample' in sys.argv else analyze(variant)
