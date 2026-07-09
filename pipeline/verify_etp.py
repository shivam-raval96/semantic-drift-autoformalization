"""[VERIFY-ETP] Match every law in laws.py against the Equational Theories Project list.

ETP numbers equations up to variable relabeling and symmetry of '='. This script parses
the ETP Lean equation files (comment lines '-- equation N := lhs = rhs', operation ◇)
with the project's own term parser and reports, for each law, every ETP equation it
matches under a bijective variable renaming, with or without swapping the two sides.

Usage:
    python verify_etp.py [dir-with-Eqns*.lean]   # no arg: download from GitHub to ./etp_cache

The verified (id, exact ETP string) pairs live in ETP_STRINGS below; test_pilot.py
re-checks laws.py against them offline, so the only thing a human ever needs to
spot-check is that ETP_STRINGS faithfully quotes the explorer.
"""
import os, re, sys
from laws import LAWS, parse_law_str

ETP_FILES = ['Eqns1_999.lean', 'Eqns1000_1999.lean', 'Eqns2000_2999.lean',
             'Eqns3000_3999.lean', 'Eqns4000_4694.lean']
ETP_BASE = ('https://raw.githubusercontent.com/teorth/equational_theories/'
            'main/equational_theories/Equations/')

# Verified against the ETP equation list (repo main, fetched 2026-07-04) by running
# this script: each entry is the unique ETP match under renaming/side-swap.
# Key: our lid -> (ETP id, ETP equation string exactly as listed).
# 'medial' has 6 occurrences of the operation; ETP enumerates only <=4, so it has
# no ETP node — laws with n_ops > 4 legitimately carry etp_node=None.
ETP_STRINGS = {
    'refl':       (1,    'x = x'),
    'triv':       (2,    'x = y'),
    'idem':       (3,    'x = x ◇ x'),
    'lproj':      (4,    'x = x ◇ y'),
    'rproj':      (5,    'x = y ◇ x'),
    'labsorb_sw': (14,   'x = y ◇ (x ◇ y)'),
    'labsorb':    (16,   'x = y ◇ (y ◇ x)'),
    'rabsorb':    (26,   'x = (x ◇ y) ◇ y'),
    'unipot':     (40,   'x ◇ x = y ◇ y'),
    'comm':       (43,   'x ◇ y = y ◇ x'),
    'const':      (46,   'x ◇ y = z ◇ w'),
    'central':    (168,  'x = (y ◇ x) ◇ (x ◇ z)'),
    'ldupl':      (323,  'x ◇ y = x ◇ (x ◇ y)'),
    'lperm':      (4362, 'x ◇ (y ◇ z) = y ◇ (x ◇ z)'),
    'rot':        (4364, 'x ◇ (y ◇ z) = y ◇ (z ◇ x)'),
    'lalt':       (4396, 'x ◇ (x ◇ y) = (x ◇ x) ◇ y'),
    'flex':       (4435, 'x ◇ (y ◇ x) = (x ◇ y) ◇ x'),
    'ralt':       (4473, 'x ◇ (y ◇ y) = (x ◇ y) ◇ y'),
    'assoc':      (4512, 'x ◇ (y ◇ z) = (x ◇ y) ◇ z'),
    'assoc_sw':   (4515, 'x ◇ (y ◇ z) = (x ◇ z) ◇ y'),
}


def parse_etp(s):
    return parse_law_str(s.replace('◇', '*'))


def _match_terms(a, b, env):
    """Structural match of term a onto term b extending bijective var map env."""
    if a[0] == 'var' and b[0] == 'var':
        if a[1] in env:
            return env if env[a[1]] == b[1] else None
        if b[1] in env.values():
            return None
        e = dict(env); e[a[1]] = b[1]
        return e
    if a[0] == 'op' and b[0] == 'op':
        e = _match_terms(a[1], b[1], env)
        if e is None:
            return None
        return _match_terms(a[2], b[2], e)
    return None


def same_law(law_a, law_b):
    """(lhs_a, rhs_a) == (lhs_b, rhs_b) up to bijective renaming, allowing side swap."""
    la, ra = law_a
    for lb, rb in (law_b, (law_b[1], law_b[0])):
        env = _match_terms(la, lb, {})
        if env is not None and _match_terms(ra, rb, env) is not None:
            return True
    return False


def load_etp_equations(directory):
    eqs = {}
    pat = re.compile(r'^(?:--\s*)?equation\s+(\d+)\s*:=\s*(.+?)\s*$')
    for fname in ETP_FILES:
        with open(os.path.join(directory, fname), encoding='utf-8') as f:
            for line in f:
                m = pat.match(line)
                if m:
                    eqs[int(m.group(1))] = m.group(2)
    return eqs


def fetch(directory):
    import urllib.request
    os.makedirs(directory, exist_ok=True)
    for fname in ETP_FILES:
        path = os.path.join(directory, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(ETP_BASE + fname, path)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')  # ◇ vs Windows cp1252 console
    directory = sys.argv[1] if len(sys.argv) > 1 else 'etp_cache'
    if not all(os.path.exists(os.path.join(directory, f)) for f in ETP_FILES):
        fetch(directory)
    eqs = load_etp_equations(directory)
    print(f"loaded {len(eqs)} ETP equations")
    ok = True
    for lid, law in LAWS.items():
        ours = (law.lhs, law.rhs)
        hits = [n for n, s in eqs.items() if same_law(ours, parse_etp(s))]
        recorded = law.etp_node
        status = 'MISSING' if not hits else f"Eq{hits[0]}" + (f" (+{len(hits)-1} dupes)" if len(hits) > 1 else '')
        if hits:
            good = recorded == f'Eq{hits[0]}' and len(hits) == 1
        else:  # >4 op applications are outside the ETP enumeration
            good = recorded is None and law.n_ops > 4
            status += ' (n_ops>4, outside ETP)' if good else ''
        mark = 'ok ' if good else '!! '
        if not good:
            ok = False
        print(f"{mark}{lid:12s} laws.py={recorded or '-':8s} etp={status:12s} "
              f"{'; '.join(eqs[n] for n in hits[:2])}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
