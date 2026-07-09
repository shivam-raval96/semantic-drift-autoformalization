"""Finite magma engine: law satisfaction, countermodel search, semantic diff classification.

Checkable-not-decidable, operationalized:
- a NON-implication is certified by an explicit finite countermodel (a concrete table),
- an implication is never 'decided' here; absence of a countermodel up to size bound n
  yields the honest label 'not refuted (<= n)', unless construction-known gold exists.
"""
import itertools, random
from laws import Law, LAWS, known_implication, substitution_instance

def eval_term(t, env, table):
    if t[0] == 'var':
        return env[t[1]]
    return table[eval_term(t[1], env, table)][eval_term(t[2], env, table)]

def satisfies(table, law: Law, n: int) -> bool:
    vs = law.vars
    for assignment in itertools.product(range(n), repeat=len(vs)):
        env = dict(zip(vs, assignment))
        if eval_term(law.lhs, env, table) != eval_term(law.rhs, env, table):
            return False
    return True

def all_tables(n):
    cells = n * n
    for flat in itertools.product(range(n), repeat=cells):
        yield [list(flat[i*n:(i+1)*n]) for i in range(n)]

def random_tables(n, count, rng):
    for _ in range(count):
        yield [[rng.randrange(n) for _ in range(n)] for _ in range(n)]

def find_countermodel(premise: Law, conclusion: Law, max_exhaustive=3, n4_samples=20000, seed=0):
    """Search for a finite magma satisfying `premise` but violating `conclusion`.
    Returns (n, table) or None. Exhaustive for n<=max_exhaustive, sampled at n=4."""
    for n in range(2, max_exhaustive + 1):
        for table in all_tables(n):
            if satisfies(table, premise, n) and not satisfies(table, conclusion, n):
                return (n, table)
    rng = random.Random(seed)
    for table in random_tables(4, n4_samples, rng):
        if satisfies(table, premise, 4) and not satisfies(table, conclusion, 4):
            return (4, table)
    return None

def _known_or_instance(premise: Law, conclusion: Law):
    """The two certified 'implies' routes: construction-known, or conclusion being a
    substitution instance of premise (the substitution is the certificate)."""
    if known_implication(premise, conclusion):
        return 'implies (known)', {'route': 'construction-known'}
    sub = substitution_instance(premise, conclusion)
    if sub is not None:
        return 'implies (known)', {'route': 'substitution instance', 'substitution': sub}
    return None

def implication_status(premise: Law, conclusion: Law, **kw):
    """Returns (label, certificate).
    labels: 'implies (known)', 'non-implication', 'not refuted (<=bound)'."""
    k = _known_or_instance(premise, conclusion)
    if k:
        return k
    cm = find_countermodel(premise, conclusion, **kw)
    if cm:
        n, table = cm
        return 'non-implication', {'route': 'finite countermodel', 'size': n, 'table': table}
    return 'not refuted (<=4)', {'route': 'search exhausted (exhaustive <=3, sampled n=4)'}

def certify_all(laws, max_exhaustive=3, n4_samples=20000, seed=0):
    """Batch implication_status over all ordered pairs of `laws`: identical labels and
    certificate shapes, but one shared table enumeration resolves every unresolved
    pair, instead of one enumeration per pair (the expanded library would otherwise
    re-walk ~40k tables hundreds of times)."""
    laws = list(laws)
    out, unresolved = {}, []
    for p in laws:
        for c in laws:
            k = _known_or_instance(p, c)
            if k:
                out[(p.lid, c.lid)] = k
            else:
                unresolved.append((p, c))

    def sweep(tables, n):
        for table in tables:
            if not unresolved:
                return
            sat = {}
            def holds(law):
                if law.lid not in sat:
                    sat[law.lid] = satisfies(table, law, n)
                return sat[law.lid]
            still = []
            for p, c in unresolved:
                if holds(p) and not holds(c):
                    out[(p.lid, c.lid)] = ('non-implication', {
                        'route': 'finite countermodel', 'size': n,
                        'table': [row[:] for row in table]})
                else:
                    still.append((p, c))
            unresolved[:] = still

    for n in range(2, max_exhaustive + 1):
        sweep(all_tables(n), n)
    sweep(random_tables(4, n4_samples, random.Random(seed)), 4)
    for p, c in unresolved:
        out[(p.lid, c.lid)] = ('not refuted (<=4)',
                               {'route': 'search exhausted (exhaustive <=3, sampled n=4)'})
    return out

def classify_relation(intended: Law, output: Law, **kw):
    """Semantic diff with direction, mirroring the SPS table.
    output->intended has no countermodel and intended->output does  => output STRONGER (over-constrained)
    intended->output has no countermodel and output->intended does  => output WEAKER  (under-constrained)
    countermodels both ways                                          => incomparable
    neither way (and/or known equivalence)                           => equivalent*
    """
    if intended.lid == output.lid:
        return 'equivalent', {'route': 'identical law'}
    out_implies_int, c1 = implication_status(output, intended, **kw)
    int_implies_out, c2 = implication_status(intended, output, **kw)
    oi = out_implies_int.startswith(('implies', 'not refuted'))
    io = int_implies_out.startswith(('implies', 'not refuted'))
    if oi and io:
        return 'equivalent*', {'note': 'no countermodel either direction within bound',
                               'fwd': out_implies_int, 'bwd': int_implies_out}
    if oi and not io:
        return 'stronger (over-constrained)', {'fwd': out_implies_int, 'bwd_countermodel': c2}
    if io and not oi:
        return 'weaker (under-constrained)', {'bwd': int_implies_out, 'fwd_countermodel': c1}
    return 'incomparable', {'fwd_countermodel': c1, 'bwd_countermodel': c2}

if __name__ == '__main__':
    # sanity: NAND is commutative but not associative -> comm does not imply assoc
    nand = [[1, 1], [1, 0]]
    assert satisfies(nand, LAWS['comm'], 2)
    assert not satisfies(nand, LAWS['assoc'], 2)
    # left projection is associative but not commutative
    lp = [[0, 0], [1, 1]]
    assert satisfies(lp, LAWS['lproj'], 2)
    assert satisfies(lp, LAWS['assoc'], 2)
    assert not satisfies(lp, LAWS['comm'], 2)
    print('comm -> assoc:', implication_status(LAWS['comm'], LAWS['assoc'])[0])
    print('assoc -> comm:', implication_status(LAWS['assoc'], LAWS['comm'])[0])
    print('triv -> medial:', implication_status(LAWS['triv'], LAWS['medial'])[0])
    print('intended=comm, output=refl ->', classify_relation(LAWS['comm'], LAWS['refl'])[0])
    print('intended=refl, output=triv ->', classify_relation(LAWS['refl'], LAWS['triv'])[0])
    print('intended=comm, output=assoc ->', classify_relation(LAWS['comm'], LAWS['assoc'])[0])
    print('sanity OK')
