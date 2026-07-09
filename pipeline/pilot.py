"""Backends, runner, scoring, report. CLI at bottom.

MockBackend exists to validate the PIPELINE, not to produce findings: it simulates a
model with drift rates that differ by drift move, plus a syntax-failure floor, so every
downstream table exercises its real code path. Swap in ApiBackend for real results.
"""
import json, random, sys, os
from collections import defaultdict
from laws import LAWS, ParseError, parse_law_str, render_law, Law
from magma import classify_relation

# ---------- backends ----------

class MockBackend:
    """Deterministic simulated model. Rates chosen to make structure visible in the
    report so the pipeline's stratified tables can be eyeballed for correctness."""
    name = 'mock'
    DRIFT_P = {'weakening': 0.35, 'strengthening': 0.10,
               'neighbor_confusion': 0.45, 'variable_role_swap': 0.25}
    B_DRIFT_P = 0.15
    SYNTAX_P = 0.08
    IMPL_ERR = {1: 0.05, 2: 0.15, 3: 0.30, 4: 0.35}

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def translate(self, item):
        if self.rng.random() < self.SYNTAX_P:
            return "x * (y ="  # unparsable: L-code failure
        if item['family'] == 'A' and self.rng.random() < self.DRIFT_P[item['drift_move']]:
            return item['drift_law']
        if item['family'] == 'B' and self.rng.random() < self.B_DRIFT_P:
            others = [l for l in LAWS.values() if l.lid != item['intended_lid']]
            return render_law(self.rng.choice(others))
        return item['intended_law']

    def judge_implication(self, item):
        err = self.IMPL_ERR.get(item['n_ops'], 0.3)
        gold = item['gold']
        if self.rng.random() < err:
            return 'implies' if gold == 'does_not_imply' else 'does_not_imply'
        return gold

# ONE fixed prompt template per task type (pre-registration section 4). Never vary
# these across configurations. Built ONLY from surface/premise/conclusion — gold labels,
# drift targets, and graph distances must never reach a prompt (guardrail, tested).
PROMPTS = {
    'translation': ("Express the following as a single equational law over one binary "
                    "operation *, variables x,y,z,w, output only the equation: {surface}"),
    'transcription': "{surface}\nOutput only the equation.",
    'implication': ("Over magmas (a set with one binary operation *), does the first "
                    "equational law imply the second for all elements?\n"
                    "Law 1: {premise}\nLaw 2: {conclusion}\n"
                    "Answer with exactly one word: implies or does_not_imply"),
}

def build_prompt(item):
    if item['task'] == 'implication':
        return PROMPTS['implication'].format(premise=item['premise'],
                                             conclusion=item['conclusion'])
    return PROMPTS[item['task']].format(surface=item['surface'])

def extract_equation(text):
    """Minimal cleanup only: markdown fences/backticks and surrounding blank lines.
    Anything beyond this would mask L-code (syntax) failures we want to measure."""
    lines = [l for l in text.strip().splitlines() if not l.strip().startswith('```')]
    for line in lines:
        line = line.strip().strip('`').strip()
        if line:
            return line
    return text.strip()

# These models reject sampling params entirely (temperature 400s), so temp-0 determinism
# is unavailable: run >=3 seeds there. Older models accept temperature=0 -> 1 seed is fine.
NO_TEMPERATURE_PREFIXES = ('claude-opus-4-7', 'claude-opus-4-8', 'claude-sonnet-5',
                           'claude-fable', 'claude-mythos')

def detect_provider(model):
    """(provider, deterministic). OpenRouter model ids are 'vendor/model' (one key,
    many models); bare ids route to the native provider."""
    base = model.rsplit('/', 1)[-1]
    deterministic = not base.startswith(NO_TEMPERATURE_PREFIXES)
    if '/' in model:
        return 'openrouter', deterministic
    if model.startswith(('gpt', 'o1', 'o3', 'o4')):
        return 'openai', deterministic
    return 'anthropic', deterministic

class ApiBackend:
    """Real-model backend. Keys from env (ANTHROPIC_API_KEY / OPENAI_API_KEY /
    OPENROUTER_API_KEY — OpenRouter is selected by 'vendor/model' ids). The model's
    raw law string is returned as-is after fence-stripping; the scorer parses it.
    `transport` is injectable for offline tests: callable(prompt: str) -> str."""

    def __init__(self, model='claude-opus-4-8', transport=None, max_output_tokens=300,
                 temperature=None):  # None = frozen default (0 where accepted)
        self.model = model
        self.name = f'api:{model}'
        self.provider, self.deterministic = detect_provider(model)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.transport = transport or self._make_transport()

    def _make_transport(self):
        if self.provider == 'anthropic':
            if not os.environ.get('ANTHROPIC_API_KEY'):
                raise RuntimeError('ANTHROPIC_API_KEY not set')
            import anthropic
            client = anthropic.Anthropic()  # SDK retries 429/5xx itself

            def call(prompt):
                kwargs = {}
                if self.temperature is not None:
                    kwargs['temperature'] = self.temperature
                elif self.deterministic:
                    kwargs['temperature'] = 0.0  # newer models reject sampling params
                resp = client.messages.create(
                    model=self.model, max_tokens=self.max_output_tokens,
                    messages=[{'role': 'user', 'content': prompt}], **kwargs)
                if resp.stop_reason == 'refusal':
                    return ''  # scored as syntax_failure; flagged in results
                return ''.join(b.text for b in resp.content if b.type == 'text')
            return call
        # openai and openrouter both speak the OpenAI protocol
        import openai
        if self.provider == 'openrouter':
            if not os.environ.get('OPENROUTER_API_KEY'):
                raise RuntimeError('OPENROUTER_API_KEY not set (https://openrouter.ai/keys)')
            try:  # soft budget guard: warn (never block) past 80% of the key's limit
                import urllib.request
                req = urllib.request.Request(
                    'https://openrouter.ai/api/v1/auth/key',
                    headers={'Authorization': f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
                d = json.loads(urllib.request.urlopen(req, timeout=10).read())['data']
                if d.get('limit') and d.get('usage', 0) > 0.8 * d['limit']:
                    print(f"WARNING: OpenRouter spend ${d['usage']:.2f} of ${d['limit']} limit")
            except Exception:
                pass
            client = openai.OpenAI(base_url='https://openrouter.ai/api/v1',
                                   api_key=os.environ['OPENROUTER_API_KEY'])
        else:
            if not os.environ.get('OPENAI_API_KEY'):
                raise RuntimeError('OPENAI_API_KEY not set')
            client = openai.OpenAI()

        # OpenRouter expects max_tokens; native OpenAI's newer models want
        # max_completion_tokens
        tok_param = ('max_tokens' if self.provider == 'openrouter'
                     else 'max_completion_tokens')

        def call(prompt):
            kwargs = {tok_param: self.max_output_tokens}
            if self.temperature is not None:
                kwargs['temperature'] = self.temperature
            elif self.deterministic:
                kwargs['temperature'] = 0.0
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}], **kwargs)
            return resp.choices[0].message.content or ''
        return call

    def translate(self, item):
        return extract_equation(self.transport(build_prompt(item)))

    def judge_implication(self, item):
        text = self.transport(build_prompt(item)).strip().lower()
        # normalize to the two labels; anything else is an invalid judgment
        if 'does_not_imply' in text or 'does not imply' in text:
            return 'does_not_imply'
        if 'implies' in text:
            return 'implies'
        return f'invalid: {text[:80]}'

# ---------- runner ----------

def run(data, backend, workers=1):
    def one(item):
        r = dict(item)
        r['backend'] = backend.name
        if item['task'] in ('translation', 'transcription'):
            r['model_output'] = backend.translate(item)
        else:
            r['model_output'] = backend.judge_implication(item)
        return r
    if workers <= 1:
        return [one(it) for it in data]
    # api items are independent; MockBackend must stay serial (shared rng)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, data))

# ---------- scoring ----------

def _law_from_string(s):
    lhs, rhs = parse_law_str(s)
    return Law('out', 'model output', lhs, rhs)

def verdict_from_relation(rel):
    """Three-way mapping (audit repair 2026-07-07). 'equivalent*' means NO
    countermodel found either direction at <=4 — absence of refutation, not proof.
    Honest-silence: that is 'unresolved (<=4)', never 'faithful'. Proven faithful =
    exact canonical form only (certified equivalence routes may extend this later)."""
    if rel == 'equivalent':
        return 'faithful'
    if rel == 'equivalent*':
        return 'unresolved (<=4)'
    return f'drift: {rel}'

def _alpha_equal(a, b):
    from verify_etp import same_law
    return same_law((a.lhs, a.rhs), (b.lhs, b.rhs))

def score(results):
    scored = []
    memo = {}  # (intended_lid, canonical output) -> (verdict, evidence); pure function of the pair
    for r in results:
        s = dict(r)
        if r['task'] in ('translation', 'transcription'):
            try:
                out_law = _law_from_string(r['model_output'])
                intended = LAWS.get(r['intended_lid'])
                if intended is None:  # family E: intended law outside the library
                    lhs, rhs = parse_law_str(r['intended_law'])
                    intended = Law(r['intended_lid'], 'etp item', lhs, rhs)
                # identical-law fast path via canonical rendering
                if render_law(out_law) == r['intended_law']:
                    s['verdict'], s['evidence'] = 'faithful', {'route': 'exact form'}
                elif _alpha_equal(out_law, intended):
                    # certified: bijective renaming + '=' symmetry preserve semantics
                    s['verdict'], s['evidence'] = 'faithful', {'route': 'alpha-equivalent'}
                else:
                    key = (r['intended_lid'], render_law(out_law))
                    if key not in memo:
                        rel, ev = classify_relation(intended, out_law)
                        memo[key] = (verdict_from_relation(rel), ev)
                    s['verdict'], s['evidence'] = memo[key]
            except ParseError as e:
                s['verdict'], s['evidence'] = 'syntax_failure (L)', {'error': str(e)}
        else:
            s['verdict'] = 'correct' if r['model_output'] == r['gold'] else 'incorrect'
            s['evidence'] = {'gold_certificate_route': r['certificate']['route']}
        scored.append(s)
    return scored

# ---------- report ----------

def _pct(part, whole):
    return f"{100*part/whole:.0f}% ({part}/{whole})" if whole else "-"

def boot_ci(flags, reps=2000, seed=0):
    """95% percentile bootstrap CI on the mean of 0/1 flags. Deterministic."""
    if not flags:
        return None
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(sum(rng.choices(flags, k=n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps)]

def _pct_ci(flags):
    """'62% [51, 72] (33/53)' — rate, bootstrap CI, raw counts."""
    if not flags:
        return '-'
    lo, hi = boot_ci(flags)
    return (f"{100*sum(flags)/len(flags):.0f}% [{100*lo:.0f}, {100*hi:.0f}] "
            f"({sum(flags)}/{len(flags)})")

def _csv_row(table, stratum, flags):
    lo, hi = boot_ci(flags) or (0.0, 0.0)
    return {'table': table, 'stratum': stratum, 'n': len(flags), 'k': sum(flags),
            'rate': round(sum(flags)/len(flags), 4) if flags else '',
            'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4)}

def report_csv(scored, path):
    """Summary CSV mirroring every report table (one row per stratum)."""
    import csv
    rows = []
    A = [s for s in scored if s['family'] == 'A']
    B = [s for s in scored if s['family'] == 'B']
    D = [s for s in scored if s['family'] == 'D']
    I = [s for s in scored if s['family'] == 'I']
    rows.append(_csv_row('D_syntax_fail', 'all',
                         [int(s['verdict'] != 'faithful') for s in D]))
    for key, group in (('A_faithful_by_move', 'drift_move'),
                       ('A_faithful_by_tclass', 'tclass'),):
        strata = defaultdict(list)
        for s in A:
            strata[s[group]].append(int(s['verdict'] == 'faithful'))
        for stratum, flags in sorted(strata.items()):
            rows.append(_csv_row(key, stratum, flags))
    strata = defaultdict(list)
    for s in B:
        strata[s['register']].append(int(s['verdict'] == 'faithful'))
    for stratum, flags in sorted(strata.items()):
        rows.append(_csv_row('B_faithful_by_register', stratum, flags))
    strata = defaultdict(list)
    for s in I:
        strata[s['n_ops']].append(int(s['verdict'] == 'correct'))
    for stratum, flags in sorted(strata.items()):
        rows.append(_csv_row('I_accuracy_by_n_ops', str(stratum), flags))
    rows.append(_csv_row('I_accuracy', 'all', [int(s['verdict'] == 'correct') for s in I]))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def dedupe_by_surface(scored):
    """One row per unique surface. Families A and B share surfaces, and A repeats
    each surface once per drift move — counting copies overstates n ~3x (identical
    outputs at temp 0). Rates belong on the deduped rows; Family A copies exist for
    move attribution only."""
    seen, out = set(), []
    for s in scored:
        if s['task'] not in ('translation', 'transcription'):
            continue
        if s['surface'] not in seen:
            seen.add(s['surface'])
            out.append(s)
    return out

def report_md(scored, backend_name):
    lines = [f"# Pilot report — backend: {backend_name}", ""]
    A = [s for s in scored if s['family'] == 'A']
    B = [s for s in scored if s['family'] == 'B']
    D = [s for s in scored if s['family'] == 'D']
    I = [s for s in scored if s['family'] == 'I']

    # handcheck-2026-07-05: obscure-name-stratum items are NEVER pooled into
    # faithful-rate denominators; reported as their own stratum below
    obscure = [s for s in scored if s.get('stratum') == 'obscure-name-stratum']
    scored = [s for s in scored if s.get('stratum') != 'obscure-name-stratum']
    A = [s for s in A if s.get('stratum') != 'obscure-name-stratum']
    B = [s for s in B if s.get('stratum') != 'obscure-name-stratum']
    uniq = [s for s in dedupe_by_surface(scored) if s['family'] in 'ABE']
    if obscure:
        ob = [int(s['verdict'] == 'faithful') for s in dedupe_by_surface(obscure)]
        lines += [f"*Obscure-name stratum (excluded from all rates): "
                  f"{_pct_ci(ob)} faithful on {len(ob)} unique surfaces.*", ""]
    if uniq:
        f = [int(s['verdict'] == 'faithful') for s in uniq]
        v = [int(s['verdict'].startswith('drift')) for s in uniq]
        l = sum(1 for s in uniq if s['verdict'].startswith('syntax'))
        u = sum(1 for s in uniq if s['verdict'].startswith('unresolved'))
        lines += ["## Unique-surface rates (the honest headline numbers)",
                  f"{len(uniq)} unique translation surfaces: PROVEN faithful **{_pct_ci(f)}**, "
                  f"certified drift (VBU) **{_pct_ci(v)}**, unresolved(<=4) {u}, syntax (L) {l}. "
                  "Per-move/per-register tables below reuse surfaces and are for "
                  "attribution, not rates.", ""]

    dfail = [int(s['verdict'] != 'faithful') for s in D]
    lines += [f"## Family D — syntax floor",
              f"Syntax/transcription failure floor: **{_pct_ci(dfail)}**. "
              f"Subtract this narratively — never silently — when reading drift rates below.", ""]

    lines += ["## Family A — drift by move (register held constant)",
              "| drift move | items | faithful [95% CI] | drifted | syntax (L) |", "|---|---|---|---|---|"]
    by_move = defaultdict(list)
    for s in A:
        by_move[s['drift_move']].append(s)
    for move, items in sorted(by_move.items()):
        flags = [int(s['verdict'] == 'faithful') for s in items]
        l = sum(1 for s in items if s['verdict'].startswith('syntax'))
        d = len(items) - sum(flags) - l
        lines.append(f"| {move} | {len(items)} | {_pct_ci(flags)} | {_pct(d, len(items))} | {l} |")

    lines += ["", "### Family A — faithfulness by theory class",
              "| theory class | items | faithful [95% CI] |", "|---|---|---|"]
    by_tc = defaultdict(list)
    for s in A:
        by_tc[s['tclass']].append(int(s['verdict'] == 'faithful'))
    for tc, flags in sorted(by_tc.items()):
        lines.append(f"| {tc} | {len(flags)} | {_pct_ci(flags)} |")

    lines += ["", "### Drift direction (semantic diff signal)",
              "| direction | count |", "|---|---|"]
    dirs = defaultdict(int)
    for s in A:
        if s['verdict'].startswith('drift'):
            dirs[s['verdict'].removeprefix('drift: ')] += 1
    for d, c in sorted(dirs.items()):
        lines.append(f"| {d} | {c} |")

    lines += ["", "## Family B — faithfulness by register (template rotation)",
              "| register | items | faithful [95% CI] |", "|---|---|---|"]
    by_reg = defaultdict(list)
    for s in B:
        by_reg[s['register']].append(int(s['verdict'] == 'faithful'))
    for reg, flags in sorted(by_reg.items()):
        lines.append(f"| {reg} | {len(flags)} | {_pct_ci(flags)} |")

    lines += ["", "## Implication judgment — accuracy by operation count",
              "| n_ops | items | accuracy [95% CI] |", "|---|---|---|"]
    by_ops = defaultdict(list)
    for s in I:
        by_ops[s['n_ops']].append(int(s['verdict'] == 'correct'))
    for n, flags in sorted(by_ops.items()):
        lines.append(f"| {n} | {len(flags)} | {_pct_ci(flags)} |")
    # handcheck-2026-07-05: stratify by certificate route (substitution-certified
    # implications reported separately from construction-known and countermodels)
    lines += ["", "### by certificate route", "| route | items | accuracy [95% CI] |", "|---|---|---|"]
    by_route = defaultdict(list)
    for s in I:
        by_route[s['certificate']['route']].append(int(s['verdict'] == 'correct'))
    for route, flags in sorted(by_route.items()):
        lines.append(f"| {route} | {len(flags)} | {_pct_ci(flags)} |")

    ex = next((s for s in A if s['verdict'].startswith('drift')), None)
    if ex:
        lines += ["", "## Example certified failure",
                  f"Item `{ex['item_id']}` — surface: \"{ex['surface']}\"",
                  f"- intended: `{ex['intended_law']}`",
                  f"- model output: `{ex['model_output']}`",
                  f"- verdict: **{ex['verdict']}**",
                  f"- certificate: `{json.dumps(ex['evidence'])[:400]}`"]
    lines += ["", "*Mock backend validates the pipeline, not any hypothesis. "
              "Real rates require ApiBackend + >=3 seeds.*"]
    return "\n".join(lines) + "\n"

# ---------- CLI ----------

def _slug(name):
    return ''.join(c if c.isalnum() or c in '.-' else '-' for c in name)

def log_run(kind, detail):
    """Append-only audit trail: every run that writes results files records what,
    when, with which instrument. Overwrites stop being silent."""
    import datetime, hashlib
    entry = {'ts': datetime.datetime.now().isoformat(timespec='seconds'),
             'kind': kind,
             'prompts_sha': hashlib.sha256(
                 json.dumps(PROMPTS, sort_keys=True, ensure_ascii=False).encode()
             ).hexdigest()[:12],
             **detail}
    with open('runs.log', 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')

def compare_md(paths):
    """Per-model comparison across results files (seeds pooled per backend)."""
    by_backend = defaultdict(list)
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            by_backend[r.get('backend', 'unknown')].append(r)
    lines = ["# Model comparison (translation rates on UNIQUE surfaces)", "",
             f"Pooled from: {', '.join(paths)}", "",
             "| backend | faithful [CI] | VBU [CI] | D syntax floor [CI] | I accuracy [CI] |",
             "|---|---|---|---|---|"]
    for name, rows in sorted(by_backend.items()):
        u = [s for s in dedupe_by_surface(rows) if s['family'] in 'ABE']
        uf = [int(s['verdict'] == 'faithful') for s in u]
        uv = [int(s['verdict'].startswith('drift')) for s in u]
        d = [int(s['verdict'] != 'faithful') for s in rows if s['family'] == 'D']
        i = [int(s['verdict'] == 'correct') for s in rows if s['family'] == 'I']
        lines.append(f"| {name} | {_pct_ci(uf)} | {_pct_ci(uv)} | {_pct_ci(d)} | {_pct_ci(i)} |")
    lines += ["", "*Mock rows, if present, are pipeline-validation fictions — never findings.*"]
    return "\n".join(lines) + "\n"

def main():
    import argparse, glob
    p = argparse.ArgumentParser(description='semdiff pilot runner')
    p.add_argument('cmd', nargs='?', default='demo', choices=['gen', 'run', 'demo', 'compare'])
    p.add_argument('--backend', default='mock', choices=['mock', 'api'])
    p.add_argument('--model', default='claude-opus-4-8',
                   help='model id for --backend api (per-model results files)')
    p.add_argument('--seeds', type=int, default=None,
                   help='independent runs; default 1, or 3 where sampling cannot be disabled')
    p.add_argument('--limit', type=int, default=None,
                   help='SMOKE ONLY: run ~N items, family mixture preserved; not a real run')
    p.add_argument('--workers', type=int, default=8,
                   help='parallel requests for --backend api (mock is always serial)')
    p.add_argument('--data', default='data.jsonl',
                   help='items file (e.g. data-etp.jsonl); results files get its stem as suffix')
    args = p.parse_args()
    seed = int(os.environ.get('SEED', '0'))
    from dataset import generate
    if args.cmd in ('gen', 'demo'):
        data = generate(seed)
        with open('data.jsonl', 'w') as f:
            for it in data:
                f.write(json.dumps(it) + '\n')
        print(f"wrote data.jsonl ({len(data)} items)")
        from dataset import generate_v2
        with open('data-v2.jsonl', 'w') as f:
            for it in generate_v2(seed):
                f.write(json.dumps(it) + '\n')
        print("wrote data-v2.jsonl (confirmatory bank)")
    if args.cmd in ('run', 'demo'):
        data = [json.loads(l) for l in open(args.data)]
        suffix = '' if args.data == 'data.jsonl' else '-' + args.data.removeprefix('data-').removesuffix('.jsonl')
        if args.limit and args.limit < len(data):
            data = data[::max(1, len(data) // args.limit)][:args.limit]
            print(f"SMOKE RUN: {len(data)} items (every-kth subsample; keep out of findings)")
        if args.backend == 'mock':
            backends = [(MockBackend(seed), 'results.jsonl', 'report.md')]
        else:
            b = ApiBackend(args.model)
            n_seeds = args.seeds if args.seeds is not None else (1 if b.deterministic else 3)
            tag = _slug(b.name) + suffix
            backends = [(ApiBackend(args.model) if k else b,
                         f'results-{tag}-s{k}.jsonl', f'report-{tag}-s{k}.md')
                        for k in range(n_seeds)]
        for backend, res_path, rep_path in backends:
            workers = args.workers if args.backend == 'api' else 1
            scored = score(run(data, backend, workers=workers))
            with open(res_path, 'w') as f:
                for s in scored:
                    f.write(json.dumps(s) + '\n')
            with open(rep_path, 'w', encoding='utf-8') as f:
                f.write(report_md(scored, backend.name))
            csv_path = rep_path.removesuffix('.md') + '.csv'
            report_csv(scored, csv_path)
            from collections import Counter
            log_run('run', {'backend': backend.name, 'data': args.data,
                            'files': [res_path, rep_path], 'n': len(scored),
                            'limit': args.limit,
                            'verdicts': dict(Counter(
                                s['verdict'].split(':')[0].split(' (')[0] for s in scored))})
            print(f"wrote {res_path}, {rep_path}, {csv_path}")
    if args.cmd == 'compare':
        paths = sorted(glob.glob('results*.jsonl'))
        if not paths:
            print('no results*.jsonl files found')
            return
        with open('comparison.md', 'w', encoding='utf-8') as f:
            f.write(compare_md(paths))
        print(f"wrote comparison.md from {len(paths)} results file(s)")

if __name__ == '__main__':
    main()
