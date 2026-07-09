"""Telephone experiment: semantic persistence under iterated translation.

Chain: law_0 -> NL -> law_1 -> NL -> ... (k hops). Each hop is one informalization
(formal -> English, name-free by instruction) followed by one formalization (the
pipeline's fixed translation prompt). The certificate checker classifies every
law_i against law_0 (survival) and against law_{i-1} (the hop kernel), so meaning
half-life, drift direction and attractor states are all machine-certified.

Fixed prompt templates (never vary across configurations), temperature 0, one
deterministic chain per (origin law, model).

Usage:
    python telephone.py --model meta-llama/llama-3.3-70b-instruct --hops 6
    python telephone.py --report          # aggregate all telephone-*.jsonl
Offline test: test_pilot.py::test_telephone_offline (fake transport).
"""
import argparse, glob, json, sys
from collections import Counter, defaultdict
from laws import LAWS, Law, ParseError, parse_law_str, render, render_law
from magma import classify_relation
from pilot import PROMPTS, ApiBackend, _alpha_equal, extract_equation, _slug

INFORMALIZE_PROMPT = (
    "Describe the meaning of the following equational law about a single binary "
    "operation * in one plain-English sentence. Do not use any standard law names "
    "(such as 'commutativity' or 'associativity'); describe the structure itself. "
    "Law: {law}\nOutput only the sentence.")

def canonical(s):
    lhs, rhs = parse_law_str(s)
    return f"{render(lhs)} = {render(rhs)}"

def _law(canon, tag):
    lhs, rhs = parse_law_str(canon)
    return Law(tag, tag, lhs, rhs)

_REL_MEMO = {}

def relation(reference, current):
    """Certified relation of `current` RELATIVE TO `reference` (memoized).
    'weaker' = current is weaker than the reference (constraint lost);
    'stronger' = current is stronger (constraint added). Argument order matters:
    classify_relation(intended, output) direction-labels the OUTPUT."""
    key = (reference, current)
    if key not in _REL_MEMO:
        if reference == current:
            _REL_MEMO[key] = 'equivalent'
        else:
            rel = classify_relation(_law(reference, 'ref'), _law(current, 'cur'))[0]
            if rel == 'equivalent*' and _alpha_equal(_law(reference, 'ref'), _law(current, 'cur')):
                rel = 'equivalent'  # certified: alpha-equivalence (bijective renaming)
            _REL_MEMO[key] = rel
    return _REL_MEMO[key]

def run_chain(transport, origin_lid, origin_canon, hops):
    """One telephone chain. Returns hop records (classification filled in later)."""
    records, current = [], origin_canon
    for hop in range(1, hops + 1):
        sentence = transport(INFORMALIZE_PROMPT.format(law=current)).strip()
        out = extract_equation(transport(PROMPTS['translation'].format(surface=sentence)))
        rec = {'origin_lid': origin_lid, 'origin_law': origin_canon, 'hop': hop,
               'sentence': sentence, 'raw_output': out}
        try:
            rec['law'] = canonical(out)
        except ParseError as e:
            rec['law'], rec['death'] = None, f'unparsable: {e}'
            records.append(rec)
            break  # chain dies
        records.append(rec)
        current = rec['law']
    return records

def classify_records(records):
    for chain_key, recs in _group(records).items():
        prev = recs[0]['origin_law']
        for r in sorted(recs, key=lambda r: r['hop']):
            if r['law'] is None:
                r['rel_to_origin'] = r['rel_to_prev'] = 'dead (L)'
                continue
            r['rel_to_origin'] = relation(r['origin_law'], r['law'])
            r['rel_to_prev'] = relation(prev, r['law'])
            # alive = PROVEN equivalent only; 'equivalent*' (unrefuted at <=4) is
            # unresolved, pooled into neither (three-way verdict convention)
            r['alive'] = r['rel_to_origin'] == 'equivalent'
            r['unresolved'] = r['rel_to_origin'] == 'equivalent*'
            prev = r['law']
    return records

def _group(records):
    g = defaultdict(list)
    for r in records:
        g[(r.get('backend', '?'), r['origin_lid'])].append(r)
    return g

def run_experiment(model, hops=6, workers=8, origins=None):
    from concurrent.futures import ThreadPoolExecutor
    backend = ApiBackend(model)
    origins = origins or list(LAWS)
    def one(lid):
        return run_chain(backend.transport, lid, canonical(render_law(LAWS[lid])), hops)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        chains = list(ex.map(one, origins))
    records = [r for chain in chains for r in chain]
    for r in records:
        r['backend'] = backend.name
    classify_records(records)  # serial: CPU-bound, memoized
    path = f'telephone-{_slug(backend.name)}.jsonl'
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    print(f"wrote {path} ({len(records)} hop records, {len(chains)} chains)")
    return records

def report(paths=None):
    paths = paths or sorted(glob.glob('telephone-*.jsonl'))
    rows = [json.loads(l) for p in paths for l in open(p, encoding='utf-8')]
    lines = ["# Telephone report — semantic persistence under iterated translation", "",
             f"Sources: {', '.join(paths)}", ""]
    by_model = defaultdict(list)
    for r in rows:
        by_model[r['backend']].append(r)
    max_hop = max(r['hop'] for r in rows)
    lines += ["## Survival: share of chains still equivalent to origin at hop k",
              "| model | " + " | ".join(f"k={k}" for k in range(1, max_hop + 1)) + " |",
              "|---|" + "---|" * max_hop]
    for m, rs in sorted(by_model.items()):
        n_chains = len({r['origin_lid'] for r in rs})
        cells = []
        for k in range(1, max_hop + 1):
            alive = sum(1 for r in rs if r['hop'] == k and r.get('alive'))
            cells.append(f"{100*alive/n_chains:.0f}%")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "## Hop kernel — what one hop does to meaning",
              "| model | equivalent | weaker | stronger | incomparable | dead |", "|---|---|---|---|---|---|"]
    for m, rs in sorted(by_model.items()):
        c = Counter()
        for r in rs:
            rel = r.get('rel_to_prev', 'dead (L)')
            c['equivalent' if rel.startswith('equivalent') else
              'weaker' if rel.startswith('weaker') else
              'stronger' if rel.startswith('stronger') else
              'incomparable' if rel == 'incomparable' else 'dead'] += 1
        n = sum(c.values())
        lines.append(f"| {m} | " + " | ".join(
            f"{100*c[k]/n:.0f}%" for k in ('equivalent', 'weaker', 'stronger', 'incomparable', 'dead')) + " |")
    lines += ["", "## Final states (attractors): where chains end up",
              "| model | top terminal laws (count) |", "|---|---|"]
    lib_canon = {canonical(render_law(l)): lid for lid, l in LAWS.items()}
    for m, rs in sorted(by_model.items()):
        finals = {}
        for r in rs:  # last record per chain
            key = r['origin_lid']
            if key not in finals or r['hop'] > finals[key]['hop']:
                finals[key] = r
        c = Counter(lib_canon.get(f['law'], f['law'] if f['law'] else 'DEAD')
                    for f in finals.values())
        top = ", ".join(f"`{k}`×{v}" for k, v in c.most_common(6))
        lines.append(f"| {m} | {top} |")
    ex = next((r for r in rows if r.get('rel_to_origin', '').startswith(('incomparable', 'weaker', 'stronger'))), None)
    if ex:
        lines += ["", "## Example drift hop",
                  f"origin `{ex['origin_law']}` -> \"{ex['sentence'][:140]}\" -> "
                  f"`{ex['law']}` ({ex['rel_to_origin']}, {ex['backend']}, hop {ex['hop']})"]
    text = "\n".join(lines) + "\n"
    with open('telephone-report.md', 'w', encoding='utf-8') as f:
        f.write(text)
    print("wrote telephone-report.md")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model')
    p.add_argument('--hops', type=int, default=6)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--report', action='store_true')
    args = p.parse_args()
    if args.model:
        run_experiment(args.model, args.hops, args.workers)
    if args.report or not args.model:
        report()

if __name__ == '__main__':
    main()
