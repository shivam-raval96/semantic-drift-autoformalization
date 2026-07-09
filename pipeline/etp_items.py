"""Family E: unnamed ETP equations, rendered into English compositionally.

The named-law benchmark confounds translation with name recall (FINDINGS caveat B).
ETP's 4694 equations are name-free: a model can only succeed by parsing structure.
Surfaces are generated mechanically from the term tree (the 'instance' register
generalized), so they are unambiguous by construction — every parenthesis is
spelled out in words.

    python etp_items.py [n_items] [seed]   -> data-etp.jsonl
Then: python pilot.py run --backend api --model <id> --data data-etp.jsonl
"""
import json, random, sys
from laws import LAWS, parse_law_str, render, variables_of
from verify_etp import ETP_FILES, fetch, load_etp_equations, parse_etp, same_law

VAR_NAMES = {'x': 'a', 'y': 'b', 'z': 'c', 'w': 'd'}

def term_nl(t):
    if t[0] == 'var':
        return VAR_NAMES[t[1]]
    return f"({term_nl(t[1])} combined with {term_nl(t[2])})"

def law_nl(lhs, rhs):
    """'for any a and b, (a combined with b) combined with b equals a' — outermost
    parentheses dropped, every inner grouping spelled explicitly."""
    vs = sorted(variables_of(lhs) | variables_of(rhs))
    names = [VAR_NAMES[v] for v in vs]
    quant = ('for any ' + ', '.join(names[:-1]) + ' and ' + names[-1]
             if len(names) > 1 else f'for any {names[0]}' if names else 'always')
    def side(t):
        s = term_nl(t)
        return s[1:-1] if s.startswith('(') else s
    return f"{quant}, {side(lhs)} equals {side(rhs)}"

def build(n_items=150, seed=0, cache='etp_cache'):
    fetch(cache)
    eqs = load_etp_equations(cache)
    lib = [(law.lhs, law.rhs) for law in LAWS.values()]
    pool = []
    for num, s in eqs.items():
        lhs, rhs = parse_etp(s)
        if not (variables_of(lhs) | variables_of(rhs)) <= set(VAR_NAMES):
            continue  # frozen translation prompt promises variables x,y,z,w only
        if any(same_law((lhs, rhs), l) for l in lib):
            continue  # exclude library laws: those are the named arm
        pool.append((num, lhs, rhs))
    rng = random.Random(seed)
    # stratify by op count so difficulty spans the range
    by_ops = {}
    from laws import count_ops
    for num, lhs, rhs in pool:
        by_ops.setdefault(count_ops(lhs) + count_ops(rhs), []).append((num, lhs, rhs))
    strata = sorted(by_ops)
    per = max(1, n_items // len(strata))
    items = []
    for k in strata:
        rng.shuffle(by_ops[k])
        for num, lhs, rhs in by_ops[k][:per]:
            law_str = f"{render(lhs)} = {render(rhs)}"
            items.append({
                'family': 'E', 'task': 'translation',
                'item_id': f"E-eq{num}",
                'surface': law_nl(lhs, rhs), 'register': 'instance',
                'intended_lid': f'etp{num}', 'intended_law': law_str,
                'etp_node': f'Eq{num}', 'tclass': 'unnamed',
                'drift_lid': None, 'drift_law': None, 'drift_move': None,
                'n_ops': k,
            })
    rng.shuffle(items)
    return items[:n_items]

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    items = build(n, seed)
    with open('data-etp.jsonl', 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    from collections import Counter
    print(f"wrote data-etp.jsonl ({len(items)} items), by n_ops:",
          dict(Counter(i['n_ops'] for i in items)))
    print("example:", items[0]['surface'], '->', items[0]['intended_law'])
