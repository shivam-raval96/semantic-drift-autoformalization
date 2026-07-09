"""Audit repair (2026-07-07): rescore every stored translation result under the
three-way verdict. No API calls — raw model_output strings were stored, so the
entire historical record re-derives locally.

Per row: verdict recomputed (faithful = proven only: exact form, certified
alpha-equivalence, or countermodel-verified equivalence; equivalent* ->
'unresolved (<=4)'); the old verdict preserved as verdict_v1 (first rescore
only). Also counts strict-parser impact: outputs that the lenient parser read
left-assoc but strict mode would reject (quantifies the interpretive-convention
exposure). Telephone chains: fully reclassified via telephone.classify_records
(alive = proven equivalent only; equivalent* hops flagged 'unresolved').
Idempotent: rerunning changes nothing after the first pass.

    python rescore.py    -> rescore-report.md (per-file deltas), files updated in place
"""
import glob, json
from laws import LAWS, Law, ParseError, parse_law_str, parse_term, render_law
from magma import classify_relation
from pilot import _alpha_equal, verdict_from_relation

MEMO = {}

def classify_memo(intended, out_law):
    key = (render_law(intended), render_law(out_law))
    if key not in MEMO:
        rel, ev = classify_relation(intended, out_law)
        MEMO[key] = (verdict_from_relation(rel), ev)
    return MEMO[key]

def rescore_row(r):
    """Returns (new_verdict, new_evidence, strict_flag) for a translation row."""
    out = r['model_output']
    strict_flag = 0
    try:
        lhs, rhs = parse_law_str(out)
    except ParseError as e:
        return 'syntax_failure (L)', {'error': str(e)}, 0
    try:
        parse_law_str_strict(out)
    except ParseError:
        strict_flag = 1
    out_law = Law('out', 'model output', lhs, rhs)
    intended = LAWS.get(r['intended_lid'])
    if intended is None:
        il, ir = parse_law_str(r['intended_law'])
        intended = Law(r['intended_lid'], 'etp item', il, ir)
    if render_law(out_law) == r['intended_law']:
        return 'faithful', {'route': 'exact form'}, strict_flag
    if _alpha_equal(out_law, intended):
        # certified route (mirrors pilot.score): bijective renaming + '=' symmetry
        return 'faithful', {'route': 'alpha-equivalent'}, strict_flag
    v, ev = classify_memo(intended, out_law)
    return v, ev, strict_flag

def parse_law_str_strict(s):
    l, r = s.split('=')
    parse_term(l, strict=True)
    parse_term(r, strict=True)

def main():
    lines = ["# Rescore report — three-way verdict (audit repair 2026-07-07)", "",
             "| file | rows | faithful v1->v2 | new unresolved | drift v1->v2 | strict-affected |",
             "|---|---|---|---|---|---|"]
    for path in sorted(glob.glob('results-*.jsonl')):
        rows = [json.loads(l) for l in open(path, encoding='utf-8')]
        ch_f = [0, 0]; ch_d = [0, 0]; unres = 0; strictn = 0; touched = False
        for r in rows:
            if r.get('task') not in ('translation', 'transcription'):
                continue
            v2, ev2, sflag = rescore_row(r)
            strictn += sflag
            v1 = r.get('verdict_v1', r['verdict'])
            ch_f[0] += v1 == 'faithful'; ch_f[1] += v2 == 'faithful'
            ch_d[0] += v1.startswith('drift'); ch_d[1] += v2.startswith('drift')
            unres += v2.startswith('unresolved')
            if v2 != r['verdict']:
                if 'verdict_v1' not in r:
                    r['verdict_v1'] = r['verdict']
                r['verdict'], r['evidence'] = v2, ev2
                touched = True
            r['strict_reject'] = sflag
        if touched or unres or strictn:
            with open(path, 'w', encoding='utf-8') as f:
                for r in rows:
                    f.write(json.dumps(r) + '\n')
        lines.append(f"| {path[8:48]} | {len(rows)} | {ch_f[0]}->{ch_f[1]} | {unres} | "
                     f"{ch_d[0]}->{ch_d[1]} | {strictn} |")
    # telephone chains: full recompute via telephone.classify_records (carries the
    # certified alpha-equivalence upgrade in relation(); rerun-safe)
    from telephone import classify_records
    for path in sorted(glob.glob('telephone-api-*.jsonl')):
        recs = [json.loads(l) for l in open(path, encoding='utf-8')]
        for r in recs:
            r.setdefault('alive_v1', r.get('alive'))
        n_alive_v1 = sum(1 for r in recs if r.get('alive_v1'))
        classify_records(recs)
        n_alive_v2 = sum(1 for r in recs if r.get('alive'))
        with open(path, 'w', encoding='utf-8') as f:
            for r in recs:
                f.write(json.dumps(r) + '\n')
        lines.append(f"| {path[:48]} | {len(recs)} | alive {n_alive_v1}->{n_alive_v2} | "
                     f"{sum(1 for r in recs if r.get('unresolved'))} | - | - |")
    text = "\n".join(lines) + "\n"
    open('rescore-report.md', 'w', encoding='utf-8').write(text)
    print(text)

if __name__ == '__main__':
    main()
