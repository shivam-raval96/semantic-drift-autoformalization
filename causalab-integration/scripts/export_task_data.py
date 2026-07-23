"""Export certified ETP implication data for the causalab etp_implication task.
Provenance: laws.py/magma.py/etp_items.py(law_nl) from semantic-drift-autoformalization
branch certificate-pipeline. Certified routes only; degenerate laws excluded."""
import json, hashlib
from laws import LAWS, render_law, variables_of
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

pool = [l for l in LAWS.values() if l.tclass != 'degenerate']
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
        "provenance": "certificate-pipeline pipeline/{laws,magma}.py + etp_items.law_nl; certify_all exhaustive<=3 sampled n=4; degenerate excluded; self-pairs excluded; uncertified excluded"}
blob = json.dumps(data, sort_keys=True, indent=1)
print("sha256:", hashlib.sha256(blob.encode()).hexdigest()[:16])
open("etp_pairs.json", "w").write(blob)
