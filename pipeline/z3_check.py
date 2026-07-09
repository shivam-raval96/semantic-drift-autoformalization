"""Z3 countermodel search: mirrors find_countermodel's interface, reaches past size 4.

Optional (task queue item 5): brute force suffices at pilot scale, so nothing in the
dataset path imports this module. Its one job is extending certificate reach for pairs
the <=4 search left 'not refuted'. Trust story is unchanged: Z3 only PROPOSES a table;
we re-verify every proposal with magma.satisfies before returning it, so a certificate
from here is exactly as checkable as one from brute force.

Usage:
    python z3_check.py            # sweep pairs excluded at <=4, sizes 5..6
    python z3_check.py 7          # ... up to size 7
Requires: pip install z3-solver  (import is deferred; module loads without it)
"""
import itertools, sys
from laws import LAWS, render_law
from magma import satisfies, implication_status

def z3_available():
    try:
        import z3  # noqa: F401
        return True
    except ImportError:
        return False

def _eval(term, env, O):
    if term[0] == 'var':
        return env[term[1]]
    return O(_eval(term[1], env, O), _eval(term[2], env, O))

def find_countermodel_z3(premise, conclusion, sizes=range(2, 7), timeout_ms=60000):
    """Search for a finite magma satisfying premise but violating conclusion.
    Returns (n, table) or None — same contract as magma.find_countermodel, but the
    size range is a parameter and each size is decided (sat/unsat) rather than sampled.
    Every model is re-verified with magma.satisfies before being returned."""
    import z3
    for n in sizes:
        O = z3.Function('op', z3.IntSort(), z3.IntSort(), z3.IntSort())
        s = z3.Solver()
        s.set('timeout', timeout_ms)
        for i, j in itertools.product(range(n), repeat=2):
            s.add(O(i, j) >= 0, O(i, j) < n)
        pv, cv = premise.vars, conclusion.vars
        for assignment in itertools.product(range(n), repeat=len(pv)):
            env = dict(zip(pv, assignment))
            s.add(_eval(premise.lhs, env, O) == _eval(premise.rhs, env, O))
        s.add(z3.Or([_eval(conclusion.lhs, dict(zip(cv, a)), O) !=
                     _eval(conclusion.rhs, dict(zip(cv, a)), O)
                     for a in itertools.product(range(n), repeat=len(cv))]))
        if s.check() == z3.sat:
            m = s.model()
            table = [[m.eval(O(i, j)).as_long() for j in range(n)] for i in range(n)]
            # certificate-level: never trust the solver, re-check with our own engine
            assert satisfies(table, premise, n) and not satisfies(table, conclusion, n)
            return (n, table)
    return None

def sweep_excluded(max_size=6):
    """Re-examine every pair the <=4 search left uncertified; report any pair that a
    larger countermodel certifies as a non-implication."""
    from dataset import certified_pairs
    certs = certified_pairs()
    excluded = [(p, c) for (p, c), (label, _) in certs.items()
                if label == 'not refuted (<=4)']
    print(f"{len(excluded)} pairs excluded at <=4; trying sizes 5..{max_size} with Z3")
    resolved = []
    for p_lid, c_lid in excluded:
        p, c = LAWS[p_lid], LAWS[c_lid]
        cm = find_countermodel_z3(p, c, sizes=range(5, max_size + 1))
        if cm:
            n, table = cm
            resolved.append((p_lid, c_lid, n))
            print(f"NEW non-implication: {p_lid} -/-> {c_lid} (countermodel size {n})")
        else:
            print(f"still not refuted (<={max_size}): {p_lid} -> {c_lid}")
    print(f"\nresolved {len(resolved)}/{len(excluded)} previously-excluded pairs")
    return resolved

if __name__ == '__main__':
    if not z3_available():
        sys.exit('z3-solver not installed: pip install z3-solver')
    max_size = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    sweep_excluded(max_size)
