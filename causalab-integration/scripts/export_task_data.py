"""Export certified ETP implication data for the causalab etp_implication task.
Provenance: laws.py/magma.py/etp_items.py(law_nl) from semantic-drift-autoformalization
branch certificate-pipeline. Certified routes only; degenerate laws excluded."""
import json, hashlib, random
from laws import LAWS, render_law, variables_of, apply_substitution
from magma import certify_all
import laws as laws_mod

VAR_NAMES = {'x': 'a', 'y': 'b', 'z': 'c', 'w': 'd'}

def term_nl(t):
    if t[0] == 'var':
        return VAR_NAMES[t[1]]
    return f"({term_nl(t[1])} combined with {term_nl(t[2])})"

def law_nl(lhs, rhs):
    vs = sorted(variables_of(lhs) | variables_of(rhs))
    names = [VAR_NAMES[v] for v in vs]
    quant = ('for any ' + ', '.join(names[:-1]) + ' and ' + names[-1]
             if len(names) > 1 else f'for any {names[0]}' if names else 'always')
    def side(t):
        s = term_nl(t)
        return s[1:-1] if s.startswith('(') else s
    return f"{quant}, {side(lhs)} equals {side(rhs)}"


def _canon(lhs, rhs):
    """Canonical key: first-appearance variable renaming, minimum over side order."""
    from laws import render
    def side_key(a, b):
        order = []
        def walk(t):
            if t[0] == 'var':
                if t[1] not in order: order.append(t[1])
            else:
                walk(t[1]); walk(t[2])
        walk(a); walk(b)
        ren = {v: n for v, n in zip(order, 'xyzwuv')}
        def rn(t):
            return ('var', ren[t[1]]) if t[0] == 'var' else ('op', rn(t[1]), rn(t[2]))
        return f"{render(rn(a))} = {render(rn(b))}"
    return min(side_key(lhs, rhs), side_key(rhs, lhs))


def _depth(t):
    return 0 if t[0] == 'var' else 1 + max(_depth(t[1]), _depth(t[2]))


def _catalan(n):
    from math import comb
    return comb(2 * n, n) // (n + 1)


def _rand_tree(m, rng, leaves):
    """Uniform binary tree with m ops (Catalan-weighted splits); leaves are
    appended placeholders to be filled with variables afterwards."""
    if m == 0:
        leaves.append(None)
        idx = len(leaves) - 1
        return ('leaf', idx)
    weights = [_catalan(k) * _catalan(m - 1 - k) for k in range(m)]
    r = rng.random() * sum(weights)
    for k, w in enumerate(weights):
        r -= w
        if r <= 0:
            break
    return ('op', _rand_tree(k, rng, leaves), _rand_tree(m - 1 - k, rng, leaves))


def _fill_vars(t, leaves, assignment):
    if t[0] == 'leaf':
        return ('var', assignment[t[1]])
    return ('op', _fill_vars(t[1], leaves, assignment), _fill_vars(t[2], leaves, assignment))


def synth_laws(existing_canon, bins, per_bin, seed):
    """Random novel laws with a chosen total op count (genform-style: uniform
    tree shapes via Catalan-weighted splits, first-appearance variables from
    x,y,z,w). Deduped against existing canonical forms; degenerate redrawn."""
    from laws import Law
    rng = random.Random(seed)
    out = []
    letters = ['x', 'y', 'z', 'w']
    for m in bins:
        made = 0
        attempts = 0
        while made < per_bin and attempts < 2000:
            attempts += 1
            lm = rng.randint(0, m)
            leaves_l, leaves_r = [], []
            tl = _rand_tree(lm, rng, leaves_l)
            tr = _rand_tree(m - lm, rng, leaves_r)
            n_leaves = len(leaves_l) + len(leaves_r)
            used = []
            assignment = []
            for _ in range(n_leaves):
                fresh = [v for v in letters if v not in used]
                choices = used + (fresh[:1] if fresh else [])
                v = rng.choice(choices)
                if v not in used:
                    used.append(v)
                assignment.append(v)
            lhs = _fill_vars(tl, leaves_l, assignment[:len(leaves_l)])
            rhs = _fill_vars(tr, leaves_r, assignment[len(leaves_l):])
            if lhs == rhs:
                continue
            key = _canon(lhs, rhs)
            if key in existing_canon:
                continue
            existing_canon.add(key)
            out.append(Law(f"syn{m}_{made}", f"synthetic law ({m} ops)", lhs, rhs, None, 'synthetic'))
            made += 1
    return out


def derive_instance_laws(base):
    """Novel laws from single-variable identifications of base laws. Each derived
    law is a substitution instance of its parent, so parent -> child implications
    certify via the substitution route in certify_all. Dedup by canonical form
    against base and each other; collapses to trivial equations are dropped."""
    from laws import Law
    existing = {_canon(l.lhs, l.rhs) for l in base}
    out = []
    for l in base:
        vs = sorted(variables_of(l.lhs) | variables_of(l.rhs))
        for a in vs:
            for b in vs:
                if a == b:
                    continue
                sig = {a: ('var', b)}
                lhs, rhs = apply_substitution(l.lhs, sig), apply_substitution(l.rhs, sig)
                if lhs == rhs:
                    continue
                key = _canon(lhs, rhs)
                if key in existing:
                    continue
                existing.add(key)
                out.append(Law(f"{l.lid}_i_{a}{b}", f"{l.name} instance ({a}:={b})",
                               lhs, rhs, None, l.tclass))
    return out


BASE = [l for l in LAWS.values() if l.tclass != 'degenerate']
_canon_seen = {_canon(l.lhs, l.rhs) for l in BASE}
SYNTH = synth_laws(_canon_seen, bins=range(1, 9), per_bin=4, seed=2026)
DERIVED = derive_instance_laws(BASE + SYNTH)
pool = BASE + SYNTH + DERIVED
NL = laws_mod.NL_TEMPLATES

laws_out = {}
for l in pool:
    t = NL.get(l.lid, {})
    entry = {
        "eq": render_law(l),
        "formal": render_law(l),
        "instance": t.get('instance') or law_nl(l.lhs, l.rhs),
        "instance_source": "handwritten" if t.get('instance') else "mechanical",
        "tclass": l.tclass,
        "etp": l.etp_node,
        "n_ops": l.n_ops,
        "depth": max(_depth(l.lhs), _depth(l.rhs)),
        "derived": "_i_" in l.lid,
        "synthetic": l.lid.startswith("syn"),
    }
    if t.get('paraphrase'): entry["paraphrase"] = t['paraphrase']
    if t.get('canonical'):  entry["named"] = t['canonical']
    laws_out[l.lid] = entry

status = certify_all(pool, n4_samples=4000)
pairs = {}
for (p, c), (label, cert) in status.items():
    if p == c: continue
    if label == 'implies (known)':
        pairs[f"{p}|{c}"] = True
    elif label == 'non-implication':
        pairs[f"{p}|{c}"] = False

true_n = sum(1 for v in pairs.values() if v)
n = len(pool)
print(f"laws: {n}; certified pairs: {len(pairs)} (implies: {true_n}, non: {len(pairs)-true_n}, excluded: {n*(n-1)-len(pairs)})")
data = {"registers": ["formal", "instance", "paraphrase", "named"],
        "laws": laws_out, "pairs": pairs,
        "provenance": "v3 (adds per-op-bin synthetic laws bins 1-8, depth metadata, n4_samples=4000; v2 added single-variable-identification instance laws); certificate-pipeline pipeline/{laws,magma}.py + etp_items.law_nl; certify_all exhaustive<=3 sampled n=4; degenerate excluded; self-pairs excluded; uncertified excluded"}
blob = json.dumps(data, sort_keys=True, indent=1)
print("sha256:", hashlib.sha256(blob.encode()).hexdigest()[:16])
open("etp_pairs.json", "w").write(blob)
