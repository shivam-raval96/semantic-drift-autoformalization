"""Two-tier sync (defense plan): CODE+SPECS+CARDS sync to pipeline/; exploratory FINDINGS/reports are LAB-ONLY and are REMOVED from pipeline if found.

Guard against silent divergence between semdiff_pilot and the MARS repo's
pipeline/ copy (ROADMAP rule: never edit both).

    python sync_check.py          # report differing/missing files, exit 1 if any
    python sync_check.py --push   # copy tracked files pilot -> pipeline
"""
import filecmp, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, '..', 'semantic-drift-autoformalization', 'pipeline')
TRACKED = ['laws.py', 'magma.py', 'dataset.py', 'pilot.py', 'test_pilot.py',
           'verify_etp.py', 'z3_check.py', 'etp_items.py', 'telephone.py',
           'hf_backend.py', 'probes.py', 'selfreport.py', 'preflight.py',
           'lambda_driver.py', 'sync_check.py', 'LAMBDA.md',
           'gauntlet.py', 'steer.py', 'overopt.py', 'overopt_eval.py',
           'judge_labels.py', 'judgeswap.py', 'overopt_pressure.py',
           'ladder_items.py', 'geometry_run.py', 'geometry2.py',
           'templates_v2.py', 'mad.py', 'mad-spec.md', 'grumeter-cards.md',
           'gao_repro.py',
           # defense layer + rescoring instrument (added audit repair 2026-07-08)
           'rescore.py', 'ledger.py', 'stats.py', 'test_defense.py',
           'cse.py', 'mad_v01.py', 'DEFENSE-PLAN.md']

def tracked_files():
    """TRACKED + every grumeter card (cards/*.md) — CLAUDE.md's code+specs+cards."""
    import glob as g
    cards = [os.path.relpath(p, HERE)
             for p in g.glob(os.path.join(HERE, 'cards', '*.md'))]
    return TRACKED + sorted(cards)

LAB_ONLY_GLOBS = ['FINDINGS-*.md', '*-report*.md', 'report-*.md']

def purge_lab_only():
    import glob as g
    removed = []
    for pat in LAB_ONLY_GLOBS:
        for f in g.glob(os.path.join(PIPE, pat)):
            base = os.path.basename(f)
            if base == 'mad-spec.md':
                continue
            os.remove(f)
            removed.append(base)
    if removed:
        print(f'purged {len(removed)} exploratory artifacts from pipeline/ (two-tier rule)')

def main():
    push = '--push' in sys.argv
    if push:
        purge_lab_only()
    bad = []
    files = tracked_files()
    for f in files:
        src, dst = os.path.join(HERE, f), os.path.join(PIPE, f)
        if not os.path.exists(src):
            bad.append((f, 'missing in pilot'))
        elif not os.path.exists(dst) or not filecmp.cmp(src, dst, shallow=False):
            if push:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                print(f'pushed {f}')
            else:
                bad.append((f, 'differs/missing in pipeline'))
    if bad:
        for f, why in bad:
            print(f'DIVERGED: {f} ({why})')
        sys.exit(1)
    print(f'in sync ({len(files)} files)')

if __name__ == '__main__':
    main()
