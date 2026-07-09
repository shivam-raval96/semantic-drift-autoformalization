"""Dataset generation per the Dataset Construction Spec (file 12).
Families: A (matched-register faithful/drifted pairs), B (template rotation),
D (syntax-only controls), plus implication-judgment items with certified gold.
"""
import json, random
from laws import LAWS, NL_TEMPLATES, drift_moves, render_law
from magma import certify_all

# laws with full 4-register NL template sets; only these enter Families A and B.
# All other laws still enter Family D (transcription) and Family I (implication).
CORE = ['idem', 'comm', 'assoc', 'lproj', 'labsorb',
        'rproj', 'unipot', 'rabsorb', 'lselfdist', 'medial']

_CERT_CACHE = None

def certified_pairs():
    """All ordered law pairs with certified-or-excluded implication status, computed
    once per process (pure function of the law library)."""
    global _CERT_CACHE
    if _CERT_CACHE is None:
        _CERT_CACHE = certify_all(LAWS.values())
    return _CERT_CACHE

def family_A(rng):
    items = []
    for lid in CORE:
        intended = LAWS[lid]
        for reg, surface in NL_TEMPLATES[lid].items():
            if reg == 'distractor':
                continue
            for drifted, move in drift_moves(intended):
                items.append({
                    'family': 'A', 'task': 'translation',
                    'item_id': f"A-{lid}-{reg}-{move}",
                    'surface': surface, 'register': reg,
                    'intended_lid': lid, 'intended_law': render_law(intended),
                    'etp_node': intended.etp_node, 'tclass': intended.tclass,
                    'drift_lid': drifted.lid, 'drift_law': render_law(drifted),
                    'drift_move': move, 'n_ops': intended.n_ops,
                })
    return items

def family_B(rng):
    items = []
    for lid in CORE:
        intended = LAWS[lid]
        for reg, surface in NL_TEMPLATES[lid].items():
            items.append({
                'family': 'B', 'task': 'translation',
                'item_id': f"B-{lid}-{reg}",
                'surface': surface, 'register': reg,
                'intended_lid': lid, 'intended_law': render_law(intended),
                'etp_node': intended.etp_node, 'tclass': intended.tclass,
                'drift_lid': None, 'drift_law': None, 'drift_move': None,
                'n_ops': intended.n_ops,
            })
    return items

def family_D(rng):
    """Syntax-only controls: transcribe an explicit formal law into the same formal syntax.
    Semantics held at identity; any failure is an L-code (syntax/capability) failure.
    Every law gets one item; CORE laws get a second, whitespace-mangled variant, keeping
    the control ratio near the spec's ~1 control per 4 experimental items."""
    items = []
    def add(lid, variant, surface_law):
        law = LAWS[lid]
        items.append({
            'family': 'D', 'task': 'transcription',
            'item_id': f"D-{lid}{variant}",
            'surface': f"Transcribe exactly, preserving structure: {surface_law}",
            'register': 'formal',
            'intended_lid': lid, 'intended_law': render_law(law),
            'etp_node': law.etp_node, 'tclass': law.tclass,
            'drift_lid': None, 'drift_law': None, 'drift_move': None,
            'n_ops': law.n_ops,
        })
    for lid in LAWS:
        add(lid, '', render_law(LAWS[lid]))
    for lid in CORE:  # same law, whitespace-free surface; target rendering unchanged
        add(lid, '-dense', render_law(LAWS[lid]).replace(' ', ''))
    return items

def implication_items(rng):
    """Balanced implication-judgment items. Gold only from certified routes:
    construction-known / substitution-instance implications, or explicit finite
    countermodels (stored inline). Uncertified pairs are excluded, per spec."""
    items, pool = [], list(LAWS.values())
    certs = certified_pairs()
    for p in pool:
        for c in pool:
            if p.lid == c.lid and p.lid in ('refl', 'triv'):
                continue
            label, cert = certs[(p.lid, c.lid)]
            if label == 'not refuted (<=4)':
                continue  # not certified either way at pilot scale: excluded, per spec
            gold = 'implies' if label.startswith('implies') else 'does_not_imply'
            items.append({
                'family': 'I', 'task': 'implication',
                'item_id': f"I-{p.lid}-{c.lid}",
                'premise_lid': p.lid, 'premise': render_law(p),
                'conclusion_lid': c.lid, 'conclusion': render_law(c),
                'gold': gold, 'certificate': cert,
                'n_ops': max(p.n_ops, c.n_ops),
            })
    implies = [i for i in items if i['gold'] == 'implies']
    non = [i for i in items if i['gold'] == 'does_not_imply']
    rng.shuffle(non)
    k = min(len(implies), len(non))
    balanced = implies[:k] + non[:max(k, min(len(non), 3 * k))]
    rng.shuffle(balanced)
    return balanced

def probe_split(rows, mode='register', held_out='paraphrase', seed=0):
    """Pre-registration gate: probes train/test on DISJOINT surfaces.
    mode='register': hold out one register entirely (systematicity split).
    mode='law': hold out ~25% of intended laws (hash-based, seed-stable).
    Returns (train, test); raises if any surface leaks across the boundary."""
    import hashlib
    train, test = [], []
    for r in rows:
        if mode == 'register':
            is_test = r.get('register') == held_out
        else:
            h = hashlib.sha256(f"{seed}:{r['intended_lid']}".encode()).digest()[0]
            is_test = h % 4 == 0
        (test if is_test else train).append(r)
    leaked = {r['surface'] for r in train} & {r['surface'] for r in test}
    if leaked:
        raise ValueError(f"surface leakage across split: {sorted(leaked)[:3]}")
    return train, test

def generate(seed=0):
    rng = random.Random(seed)
    data = family_A(rng) + family_B(rng) + family_D(rng) + implication_items(rng)
    return data

def generate_v2(seed=0):
    """Confirmatory dataset from the v2 template bank (handcheck-2026-07-05):
    dropped registers excluded, obscure-name items carry stratum tags, distractor
    register is well-posed (gold = intended). A/B translation families only —
    D and I are template-independent and unchanged."""
    from templates_v2 import bank
    b = bank()
    items = []
    for lid, regs in b.items():
        intended = LAWS[lid]
        for reg, (surface, status) in regs.items():
            base = {
                'surface': surface, 'register': reg, 'template_version': 'v2',
                'stratum': status,
                'intended_lid': lid, 'intended_law': render_law(intended),
                'etp_node': intended.etp_node, 'tclass': intended.tclass,
                'n_ops': intended.n_ops,
            }
            items.append({'family': 'B', 'task': 'translation',
                          'item_id': f"B2-{lid}-{reg}",
                          'drift_lid': None, 'drift_law': None, 'drift_move': None,
                          **base})
            if reg != 'distractor':
                for drifted, move in drift_moves(intended):
                    items.append({'family': 'A', 'task': 'translation',
                                  'item_id': f"A2-{lid}-{reg}-{move}",
                                  'drift_lid': drifted.lid,
                                  'drift_law': render_law(drifted),
                                  'drift_move': move, **base})
    rng = random.Random(seed)
    data = items + family_D(rng) + implication_items(rng)
    return data

if __name__ == '__main__':
    data = generate()
    with open('data.jsonl', 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    from collections import Counter
    print(Counter(d['family'] for d in data), 'total:', len(data))
