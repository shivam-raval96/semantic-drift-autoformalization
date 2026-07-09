"""Equational laws over magmas: terms, parsing, law library, NL surfaces, drift moves.

A term is ('var', name) or ('op', left, right). A law is lhs = rhs over the free magma.
The operation is written * in ASCII, rendered as <> in NL-adjacent output.
"""
from dataclasses import dataclass, field
from typing import Optional

# ---------- terms ----------

def variables_of(t):
    if t[0] == 'var':
        return {t[1]}
    return variables_of(t[1]) | variables_of(t[2])

def render(t):
    if t[0] == 'var':
        return t[1]
    return f"({render(t[1])} * {render(t[2])})"

def render_law(law):
    return f"{render(law.lhs)} = {render(law.rhs)}"

class ParseError(Exception):
    pass

def parse_term(s: str, strict: bool = False):
    """Parse 'x * (y * z)' style terms.
    DEFAULT (lenient, the v1 instrument): unparenthesized chains like 'x * y * z'
    are accepted and read LEFT-ASSOCIATIVELY — an interpretive convention applied
    to model outputs (audit 2026-07-07; the old docstring wrongly claimed parens
    were required). strict=True rejects ambiguous chains (>1 op at a level without
    parens) — the v2 option for confirmatory runs; choosing it is an instrument
    change (prereg it)."""
    toks = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c in '()*':
            toks.append(c); i += 1
        elif c.isalpha():
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                j += 1
            toks.append(s[i:j]); i = j
        else:
            raise ParseError(f"bad char {c!r} in {s!r}")
    pos = 0
    def atom():
        nonlocal pos
        if pos >= len(toks):
            raise ParseError("unexpected end")
        t = toks[pos]
        if t == '(':
            pos += 1
            e = expr()
            if pos >= len(toks) or toks[pos] != ')':
                raise ParseError("missing )")
            pos += 1
            return e
        if t in ('*', ')'):
            raise ParseError(f"unexpected {t}")
        pos += 1
        return ('var', t)
    def expr():
        nonlocal pos
        left = atom()
        n_ops_level = 0
        while pos < len(toks) and toks[pos] == '*':
            pos += 1
            right = atom()
            left = ('op', left, right)
            n_ops_level += 1
        if strict and n_ops_level > 1:
            raise ParseError(f"ambiguous unparenthesized chain in {s!r} (strict mode)")
        return left
    e = expr()
    if pos != len(toks):
        raise ParseError(f"trailing tokens in {s!r}")
    return e

def parse_law_str(s: str):
    if s.count('=') != 1:
        raise ParseError(f"law needs exactly one '=': {s!r}")
    l, r = s.split('=')
    return parse_term(l), parse_term(r)

def count_ops(t):
    return 0 if t[0] == 'var' else 1 + count_ops(t[1]) + count_ops(t[2])

# ---------- law library ----------

@dataclass(frozen=True)
class Law:
    lid: str          # local id
    name: str
    lhs: tuple
    rhs: tuple
    etp_node: Optional[str] = None  # verified by verify_etp.py; None iff n_ops > 4
    tclass: str = 'other'           # theory class stratum for reporting

    @property
    def n_ops(self):
        return count_ops(self.lhs) + count_ops(self.rhs)

    @property
    def vars(self):
        return sorted(variables_of(self.lhs) | variables_of(self.rhs))

def L(lid, name, s, etp=None, tclass='other'):
    lhs, rhs = parse_law_str(s)
    return Law(lid, name, lhs, rhs, etp, tclass)

# etp ids verified mechanically by verify_etp.py against the ETP equation list
# (2026-07-04); test_pilot.py re-checks them offline against verify_etp.ETP_STRINGS.
# Laws with more than 4 operation applications are outside the ETP enumeration and
# correctly carry etp=None. tclass groups laws into theory classes for stratified
# reporting (spec: 5-8 classes).
LAWS = {law.lid: law for law in [
    # -- degenerate endpoints of the strength order (0-2 ops)
    L('refl',      'reflexivity',            'x = x',                       etp='Eq1',    tclass='degenerate'),
    L('triv',      'triviality',             'x = y',                       etp='Eq2',    tclass='degenerate'),
    L('const',     'constant product',       'x * y = z * w',               etp='Eq46',   tclass='degenerate'),
    # -- idempotence-like (1-3 ops)
    L('idem',      'idempotence',            'x * x = x',                   etp='Eq3',    tclass='idempotence'),
    L('unipot',    'unipotence',             'x * x = y * y',               etp='Eq40',   tclass='idempotence'),
    L('ldupl',     'left duplication',       'x * y = x * (x * y)',         etp='Eq323',   tclass='idempotence'),
    # -- projections (1 op)
    L('lproj',     'left projection',        'x * y = x',                   etp='Eq4',    tclass='projection'),
    L('rproj',     'right projection',       'x * y = y',                   etp='Eq5',    tclass='projection'),
    # -- absorption (2 ops)
    L('labsorb',   'left absorption',        'x * (x * y) = y',             etp='Eq16',   tclass='absorption'),
    L('labsorb_sw','left absorption var-swap', 'x * (y * x) = y',           etp='Eq14',   tclass='absorption'),
    L('rabsorb',   'right absorption',       '(x * y) * y = x',             etp='Eq26',   tclass='absorption'),
    # -- commutation / permutation (2-4 ops)
    L('comm',      'commutativity',          'x * y = y * x',               etp='Eq43',   tclass='permutation'),
    L('lperm',     'left permutation',       'x * (y * z) = y * (x * z)',   etp='Eq4362', tclass='permutation'),
    L('rot',       'rotation',               'x * (y * z) = z * (x * y)',   etp='Eq4364', tclass='permutation'),
    # -- regrouping (4 ops)
    L('assoc',     'associativity',          'x * (y * z) = (x * y) * z',   etp='Eq4512', tclass='regrouping'),
    L('assoc_sw',  'assoc var-swap',         'x * (y * z) = (x * z) * y',   etp='Eq4515', tclass='regrouping'),
    L('flex',      'flexibility',            'x * (y * x) = (x * y) * x',   etp='Eq4435', tclass='regrouping'),
    L('lalt',      'left alternativity',     'x * (x * y) = (x * x) * y',   etp='Eq4396', tclass='regrouping'),
    L('ralt',      'right alternativity',    'x * (y * y) = (x * y) * y',   etp='Eq4473', tclass='regrouping'),
    # -- self-distributivity (5 ops; outside ETP enumeration)
    L('lselfdist', 'left self-distributivity',  'x * (y * z) = (x * y) * (x * z)', tclass='distributivity'),
    L('rselfdist', 'right self-distributivity', '(x * y) * z = (x * z) * (y * z)', tclass='distributivity'),
    # -- mediality (6 ops; outside ETP enumeration)
    L('medial',    'mediality',              '(x * y) * (z * w) = (x * z) * (y * w)', tclass='mediality'),
    L('medial_sw', 'mediality var-swap',     '(x * y) * (z * w) = (x * w) * (y * z)', tclass='mediality'),
    # -- central groupoid (3 ops)
    L('central',   'central groupoid',       '(x * y) * (y * z) = y',       etp='Eq168',  tclass='central'),
]}

# Construction-known implications (premise -> conclusion), used as certified gold
# without proof search. Everything implies refl; triv implies everything; E implies E.
def known_implication(p: Law, c: Law) -> Optional[bool]:
    if c.lid == 'refl':
        return True
    if p.lid == 'triv':
        return True
    if p.lid == c.lid:
        return True
    return None  # unknown here; must be certified by countermodel search or excluded

# ---------- substitution-instance implications (certified route #2) ----------
# Equational logic: a universally quantified equation entails every substitution
# instance of itself. If conclusion == sigma(premise) for some substitution sigma of
# terms for the premise's variables (orientation of '=' free on both sides), then
# premise -> conclusion, and sigma itself is the checkable certificate.

def _subst_match(pattern, target, env):
    """Match pattern onto target, binding pattern's vars to arbitrary subterms."""
    if pattern[0] == 'var':
        if pattern[1] in env:
            return env if env[pattern[1]] == target else None
        e = dict(env)
        e[pattern[1]] = target
        return e
    if target[0] != 'op':
        return None
    e = _subst_match(pattern[1], target[1], env)
    if e is None:
        return None
    return _subst_match(pattern[2], target[2], e)

def substitution_instance(premise: Law, conclusion: Law):
    """Return {var: rendered term} if conclusion is a substitution instance of
    premise (up to orienting either equation), else None."""
    for pl, pr in ((premise.lhs, premise.rhs), (premise.rhs, premise.lhs)):
        for cl, cr in ((conclusion.lhs, conclusion.rhs), (conclusion.rhs, conclusion.lhs)):
            env = _subst_match(pl, cl, {})
            if env is not None:
                env = _subst_match(pr, cr, env)
                if env is not None:
                    return {v: render(t) for v, t in env.items()}
    return None

def apply_substitution(t, sigma):
    """Apply {var: term} to a term (for re-verifying substitution certificates)."""
    if t[0] == 'var':
        return sigma.get(t[1], t)
    return ('op', apply_substitution(t[1], sigma), apply_substitution(t[2], sigma))

# ---------- NL surface templates (Family B registers) ----------
# register classes: canonical / paraphrase / instance / distractor-adjacent
NL_TEMPLATES = {
    'comm': {
        'canonical':  "the commutative law",
        'paraphrase': "combining two things in either order gives the same result",
        'instance':   "for any a and b, a combined with b equals b combined with a",
        'distractor': "the law where the grouping of a combination does not matter for two elements",
    },
    'assoc': {
        'canonical':  "the associative law",
        'paraphrase': "when combining three things, it does not matter which pair you combine first",
        'instance':   "for any a, b and c, a combined with (b combined with c) equals (a combined with b) combined with c",
        'distractor': "the law where the order of the two operands does not matter",
    },
    'idem': {
        'canonical':  "the idempotent law",
        'paraphrase': "combining a thing with itself gives the thing back",
        'instance':   "for any a, a combined with a equals a",
        'distractor': "the law where an element absorbs everything it is combined with",
    },
    'lproj': {
        'canonical':  "the left projection law",
        'paraphrase': "a combination always returns its first argument",
        'instance':   "for any a and b, a combined with b equals a",
        'distractor': "the law where a combination always returns its second argument",
    },
    'labsorb': {
        'canonical':  "the left absorption law",
        'paraphrase': "combining a thing with the combination of that thing and another recovers the other",
        'instance':   "for any a and b, a combined with (a combined with b) equals b",
        'distractor': "the law where combining a thing with itself gives the thing back",
    },
    'rproj': {
        'canonical':  "the right projection law",
        'paraphrase': "a combination always returns its second argument",
        'instance':   "for any a and b, a combined with b equals b",
        'distractor': "the law where a combination always returns its first argument",
    },
    'unipot': {
        'canonical':  "the unipotence law",
        'paraphrase': "combining any element with itself always gives one and the same result, whatever the element",
        'instance':   "for any a and b, a combined with a equals b combined with b",
        'distractor': "the law where combining a thing with itself gives the thing back",
    },
    'rabsorb': {
        'canonical':  "the right absorption law",
        'paraphrase': "combining something with a second thing and then with that same second thing again recovers the original",
        'instance':   "for any a and b, (a combined with b) combined with b equals a",
        'distractor': "the law where combining a thing with the combination of that thing and another recovers the other",
    },
    'lselfdist': {
        'canonical':  "the left self-distributive law",
        'paraphrase': "combining a thing with a combination gives the same result as combining it with each part separately and then combining the two results",
        'instance':   "for any a, b and c, a combined with (b combined with c) equals (a combined with b) combined with (a combined with c)",
        'distractor': "the law where it does not matter which pair of three elements you combine first",
    },
    'medial': {
        'canonical':  "the medial law",
        'paraphrase': "when combining two combinations, exchanging the two middle elements does not change the result",
        'instance':   "for any a, b, c and d, (a combined with b) combined with (c combined with d) equals (a combined with c) combined with (b combined with d)",
        'distractor': "the law where combining a thing with a combination distributes over both of its parts",
    },
}

# ---------- drift moves (Family A generators) ----------
# each returns (drifted_law, move_name) at comparable register (same rendering pipeline)

# nearest confusable law in the library
NEIGHBOR = {
    'comm': 'lperm', 'assoc': 'lperm',
    'idem': 'unipot', 'unipot': 'idem',
    'lproj': 'rproj', 'rproj': 'lproj',
    'labsorb': 'idem', 'rabsorb': 'labsorb',
    'lselfdist': 'rselfdist', 'medial': 'lselfdist',
    'flex': 'lalt',
}
# swapping which variable plays which role on one side of the intended law.
# assoc/labsorb/medial have targets genuinely distinct from their neighbor; for
# symmetric laws (comm) and the projections the swap coincides with the neighbor
# and is deduped (known limitation, documented).
VARSWAP = {
    'assoc': 'assoc_sw', 'labsorb': 'labsorb_sw', 'medial': 'medial_sw',
    'comm': 'lperm', 'lproj': 'rproj', 'rproj': 'lproj',
}

def drift_moves(intended: Law):
    moves = []
    if intended.lid != 'refl':
        moves.append((LAWS['refl'], 'weakening'))       # implied by everything: constraint dropped
    if intended.lid != 'triv':
        moves.append((LAWS['triv'], 'strengthening'))   # implies everything: over-constrained
    if intended.lid in NEIGHBOR:
        moves.append((LAWS[NEIGHBOR[intended.lid]], 'neighbor_confusion'))
    if intended.lid in VARSWAP:
        moves.append((LAWS[VARSWAP[intended.lid]], 'variable_role_swap'))
    # dedupe by target law, keep first move label
    seen, out = set(), []
    for law, mv in moves:
        if law.lid not in seen:
            seen.add(law.lid)
            out.append((law, mv))
    return out
