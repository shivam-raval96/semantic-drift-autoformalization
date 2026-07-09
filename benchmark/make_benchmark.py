"""Build the quadrant-labeled benchmark from pipeline results.

Reads one or more results*.jsonl files produced by pipeline/pilot.py and emits
benchmark.jsonl where every translation/transcription attempt carries its cell in
the paper's 2x2 (faithful x valid), plus handcheck.csv — the Family A rows queued
for human spot-checking (benchmark v0 target: ~100s hand-checked).

Validity here is v0 = parses as a single equation over * (see ROADMAP: v1 = Lean
elaboration via the verified ETP node ids).

Usage: python make_benchmark.py [results.jsonl ...]   (default: ../pipeline/results.jsonl)
"""
import csv, glob, json, os, sys

QUADRANTS = {'FV': 'faithful-valid', 'VBU': 'unfaithful-valid (target)',
             'INV': 'invalid (L-code)'}

def quadrant(row):
    v = row['verdict']
    if v == 'faithful':
        return 'FV'
    if v.startswith('drift'):
        return 'VBU'
    return 'INV'

def build(paths):
    items, counts = [], {}
    for path in paths:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if r['task'] not in ('translation', 'transcription'):
                continue
            q = quadrant(r)
            counts[q] = counts.get(q, 0) + 1
            items.append({
                'item_id': r['item_id'], 'family': r['family'],
                'backend': r.get('backend', 'unknown'),
                'surface': r['surface'], 'register': r['register'],
                'model_output': r['model_output'],
                'quadrant': q,
                # gold/diagnostics (never detector features):
                'verdict': r['verdict'], 'intended_law': r['intended_law'],
                'drift_move': r.get('drift_move'), 'tclass': r.get('tclass'),
                'evidence': r.get('evidence'),
            })
    return items, counts

def main():
    paths = sys.argv[1:] or sorted(glob.glob(
        os.path.join(os.path.dirname(__file__), '..', 'pipeline', 'results*.jsonl')))
    if not paths:
        sys.exit('no results files; run `python pilot.py demo` in pipeline/ first')
    items, counts = build(paths)
    out = os.path.join(os.path.dirname(__file__), 'benchmark.jsonl')
    with open(out, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    hc = os.path.join(os.path.dirname(__file__), 'handcheck.csv')
    with open(hc, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['item_id', 'surface', 'model_output', 'machine_verdict',
                    'quadrant', 'human_agrees(Y/N)', 'notes'])
        for it in items:
            if it['family'] == 'A':
                w.writerow([it['item_id'], it['surface'], it['model_output'],
                            it['verdict'], it['quadrant'], '', ''])
    n = len(items)
    print(f"wrote {out} ({n} attempts from {len(paths)} results file(s))")
    for q, label in QUADRANTS.items():
        print(f"  {q:4s} {counts.get(q, 0):5d}  {label}")
    print(f"wrote {hc} (hand-check queue: Family A rows)")

if __name__ == '__main__':
    main()
