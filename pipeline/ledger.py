"""Run ledger: every experiment registers itself; the confirmatory tag is earned.

DEFENSE-PLAN §4, operational core of variation tracking:
- register_run() appends one record per run to ledger.jsonl (append-only).
- mode='confirmatory' is a REQUEST, not a status: it is granted only if the run's
  config hash-matches prereg/frozen-config.json for its hypothesis AND every named
  data file matches its frozen hash. Any mismatch executes the run as exploratory
  and records why (auto-downgrade). The lab tree carries no frozen-config, so the
  two-tier rule (confirmatory only from the MARS tree) enforces itself.
- backfill() reconstructs exploratory history from existing artifacts so the
  registry covers everything that happened before the ledger existed.
- report() renders the variation tree: per hypothesis/kind, every configuration
  ever tried — the disclosure number for the methods section.

    python ledger.py backfill        # one-time reconstruction (idempotent)
    python ledger.py report          # -> ledger-report.md
Offline tests: test_defense.py (including the auto-downgrade rule).
"""
import glob, hashlib, json, os, subprocess, time

LEDGER = 'ledger.jsonl'
FROZEN = os.path.join('prereg', 'frozen-config.json')


def _git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def hash_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                          .encode()).hexdigest()[:16]


def hash_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def _prereg_check(hypothesis, config, data_files, frozen_path):
    """Returns (granted: bool, reason: str). Grant requires an exact match."""
    if not os.path.exists(frozen_path):
        return False, f'no frozen config at {frozen_path} (lab tree?)'
    with open(frozen_path, encoding='utf-8') as f:
        frozen = json.load(f)
    spec = frozen.get('hypotheses', {}).get(hypothesis or '')
    if spec is None:
        return False, f'hypothesis {hypothesis!r} not preregistered'
    if hash_obj(config) != spec.get('config_hash'):
        return False, 'config hash mismatch with frozen-config'
    want = spec.get('data_hashes', {})
    for p in data_files:
        base = os.path.basename(p)
        if base not in want:
            return False, f'data file not preregistered: {base}'
        if hash_file(p) != want[base]:
            return False, f'data hash mismatch: {base}'
    return True, ''


def register_run(kind, config, mode='exploratory', hypothesis=None,
                 data_files=(), notes='', ledger_path=LEDGER,
                 frozen_path=FROZEN, extra=None):
    """Register one run. Returns the record (with final, possibly downgraded, mode)."""
    assert mode in ('exploratory', 'confirmatory'), mode
    rec = {'run_id': f"{time.strftime('%Y%m%dT%H%M%S')}-{hash_obj(config)[:6]}",
           'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'kind': kind, 'mode': mode, 'hypothesis': hypothesis,
           'code_commit': _git_commit(),
           'config': config, 'config_hash': hash_obj(config),
           'data_files': {os.path.basename(p): hash_file(p) for p in data_files
                          if os.path.exists(p)},
           'notes': notes}
    if extra:
        rec.update(extra)
    if mode == 'confirmatory':
        granted, why = _prereg_check(hypothesis, config, data_files, frozen_path)
        if not granted:
            rec['mode'] = 'exploratory'
            rec['downgraded_from_confirmatory'] = why
    with open(ledger_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return rec


# ---------- backfill: reconstruct pre-ledger history from artifacts ----------

# (glob pattern, kind, hypothesis-family the artifact informs)
BACKFILL = [
    ('results-api-*-s0.jsonl',        'api_run',        'H2-exploratory'),
    ('results-hf-*.jsonl',            'hf_probe_run',   'H1-exploratory'),
    ('results-geo.jsonl',             'geometry_run',   'H1/H3-exploratory'),
    ('telephone-*.jsonl',             'telephone',      'H5-exploratory'),
    ('selfreport-*.jsonl',            'selfreport_foil', None),
    ('judge-labels-*.jsonl',          'judge_labels',   'H4-exploratory'),
    ('steer-results.jsonl',           'steering',       None),
    ('steer-random-results.jsonl',    'steering_control', None),
    ('overopt-candidates.jsonl',      'overopt_sampling', None),
    ('overopt-p64-candidates.jsonl',  'overopt_pressure', None),
    ('acts-*.npz',                    'activation_capture', None),
]


def backfill(ledger_path=LEDGER):
    """Idempotent: skips artifacts already ledgered (matched by artifact name)."""
    seen = set()
    if os.path.exists(ledger_path):
        for line in open(ledger_path, encoding='utf-8'):
            seen.add(json.loads(line).get('artifact'))
    n = 0
    for pattern, kind, hyp in BACKFILL:
        for path in sorted(glob.glob(pattern)):
            if path in seen:
                continue
            rec = {'run_id': f'backfill-{hash_file(path)[:8]}',
                   'ts': time.strftime('%Y-%m-%dT%H:%M:%S',
                                       time.localtime(os.path.getmtime(path))),
                   'kind': kind, 'mode': 'exploratory', 'hypothesis': hyp,
                   'code_commit': _git_commit(),
                   'config': {'source': 'backfill', 'artifact': path},
                   'config_hash': None, 'artifact': path,
                   'data_files': {os.path.basename(path): hash_file(path)},
                   'notes': 'reconstructed from artifact; pre-ledger run. '
                            'Known label caveat for library-trained probes: '
                            'distractor items mislabeled (handcheck-2026-07-05.md).'}
            with open(ledger_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n += 1
    print(f'backfilled {n} artifacts into {ledger_path}')


def report(ledger_path=LEDGER, out='ledger-report.md'):
    rows = [json.loads(l) for l in open(ledger_path, encoding='utf-8')]
    from collections import defaultdict
    by_h = defaultdict(list)
    for r in rows:
        by_h[r.get('hypothesis') or '(none)'].append(r)
    lines = [f"# Ledger report — {len(rows)} registered runs", "",
             "| hypothesis | runs | unique configs | confirmatory | downgraded |",
             "|---|---|---|---|---|"]
    for h, rs in sorted(by_h.items()):
        confs = {r['config_hash'] for r in rs if r['config_hash']}
        n_conf = sum(1 for r in rs if r['mode'] == 'confirmatory')
        n_down = sum(1 for r in rs if 'downgraded_from_confirmatory' in r)
        lines.append(f"| {h} | {len(rs)} | {len(confs)} | {n_conf} | {n_down} |")
    lines += ["", "## All runs", "| run_id | ts | kind | mode | hypothesis | notes |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['run_id']} | {r['ts']} | {r['kind']} | {r['mode']} | "
                     f"{r.get('hypothesis') or ''} | {(r.get('notes') or '')[:60]} |")
    text = "\n".join(lines) + "\n"
    open(out, 'w', encoding='utf-8').write(text)
    print(f'wrote {out}')


if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    {'backfill': backfill, 'report': report}[cmd]()
