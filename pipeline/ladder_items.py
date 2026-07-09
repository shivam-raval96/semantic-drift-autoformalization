"""Ambiguity-ladder items, mechanical rungs A0-A3 (A4/A5 await template sign-off).

Same ETP equation rendered at four explicitness levels — the ordered variable for
the days-of-week-style geometry test, and the properly-powered intent-probe set
(each equation = one class with 4 surfaces; hold out a rung, decode law identity).

  A0 formal string itself            (no ambiguity)
  A1 parenthesis-spelled instance    (explicit grouping; = family E renderer)
  A2 entity-fuzzified, still explicit (Luiza's fuzzification; lexical noise only)
  A3 grouping-implicit               (parentheses deleted -> scope ambiguity)

    python ladder_items.py [n_equations] [seed]   -> data-ladder.jsonl
"""
import json, sys
from laws import render, variables_of
from etp_items import VAR_NAMES, law_nl, term_nl
from verify_etp import fetch, load_etp_equations, parse_etp, same_law
from laws import LAWS, count_ops

ENTITY = {'x': "Ana's piece", 'y': "Bo's piece", 'z': "Cy's piece", 'w': "Dee's piece"}

def term_entity(t):
    if t[0] == 'var':
        return ENTITY[t[1]]
    return f"({term_entity(t[1])} merged with {term_entity(t[2])})"

def rung_surfaces(lhs, rhs):
    law_str = f"{render(lhs)} = {render(rhs)}"
    def side(f, t):
        s = f(t)
        return s[1:-1] if s.startswith('(') else s
    a2 = f"{side(term_entity, lhs)} equals {side(term_entity, rhs)}"
    a3 = law_nl(lhs, rhs).replace('(', '').replace(')', '')
    return {0: law_str, 1: law_nl(lhs, rhs), 2: a2, 3: a3}

def build(n_eq=60, seed=1, cache='etp_cache'):
    import random
    fetch(cache)
    eqs = load_etp_equations(cache)
    used = set()
    try:
        used = {json.loads(l)['etp_node'] for l in open('data-etp.jsonl', encoding='utf-8')}
    except FileNotFoundError:
        pass
    lib = [(law.lhs, law.rhs) for law in LAWS.values()]
    pool = []
    for num, s in eqs.items():
        if f'Eq{num}' in used:
            continue
        lhs, rhs = parse_etp(s)
        if not (variables_of(lhs) | variables_of(rhs)) <= set(VAR_NAMES):
            continue
        if any(same_law((lhs, rhs), l) for l in lib):
            continue
        ops = count_ops(lhs) + count_ops(rhs)
        if not 2 <= ops <= 4:
            continue  # A3 needs enough structure for grouping to be ambiguous
        pool.append((num, lhs, rhs, ops))
    rng = random.Random(seed)
    rng.shuffle(pool)
    items = []
    for num, lhs, rhs, ops in pool[:n_eq]:
        for rung, surface in rung_surfaces(lhs, rhs).items():
            items.append({
                'family': 'L', 'task': 'translation',
                'item_id': f'L-eq{num}-r{rung}', 'rung': rung,
                'surface': surface, 'register': f'ladder{rung}',
                'intended_lid': f'etp{num}',
                'intended_law': f"{render(lhs)} = {render(rhs)}",
                'etp_node': f'Eq{num}', 'tclass': 'ladder',
                'drift_lid': None, 'drift_law': None, 'drift_move': None,
                'n_ops': ops,
            })
    return items

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    items = build(n, seed)
    with open('data-ladder.jsonl', 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    print(f'wrote data-ladder.jsonl ({len(items)} items, {len(items)//4} equations x 4 rungs)')
    for it in items[:4]:
        print(f"  r{it['rung']}: {it['surface'][:90]}")
