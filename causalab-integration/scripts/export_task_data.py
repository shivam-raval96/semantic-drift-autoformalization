"""Export certified ETP implication data for the causalab etp_implication task.
Provenance: laws.py/magma.py/etp_items.py(law_nl) from semantic-drift-autoformalization
branch certificate-pipeline. Certified routes only; degenerate laws excluded."""
import json, hashlib
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
DERIVED = derive_instance_laws(BASE)
pool = BASE + DERIVED
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
        "derived": "_i_" in l.lid,
    }
    if t.get('paraphrase'): entry["paraphrase"] = t['paraphrase']
    if t.get('canonical'):  entry["named"] = t['canonical']
    laws_out[l.lid] = entry

status = certify_all(pool)
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
        "provenance": "v2 (adds single-variable-identification instance laws); certificate-pipeline pipeline/{laws,magma}.py + etp_items.law_nl; certify_all exhaustive<=3 sampled n=4; degenerate excluded; self-pairs excluded; uncertified excluded"}
blob = json.dumps(data, sort_keys=True, indent=1)
print("sha256:", hashlib.sha256(blob.encode()).hexdigest()[:16])
open("etp_pairs.json", "w").write(blob)
