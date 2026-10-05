#!/usr/bin/env python3
"""The Equational Theories Project's answers: which law implies which.

The ETP settled, for every ordered pair of its 4,694 laws, whether a magma
obeying the first must obey the second. This module turns that table into two
lookups an experiment can grade with:

- `identify(lhs, rhs)`: which ETP law, by number, an equation a model wrote is.
  ETP lists each law once up to renaming its variables and swapping its two
  sides, so both are folded away here. Mirroring the argument order of every
  operation (the dual law) is *not* folded away: a law's dual is a different
  law with its own number, and the implication table already answers for it.
- `implies(a, b)`: True, False, or None when ETP leaves the pair open.

Everything is pinned to one commit of `teorth/equational_theories`, and each
downloaded file is checked against its recorded hash, so two runs on different
machines cannot silently grade against different tables. The table ships as a
500 MB JSON file inside a 2 MB zip; it is parsed once and cached as one byte per
pair (22 MB, gzipped to a few MB) under `mech-interp-experiments/.etp-data/`.

Standard library only, so grading and `--analyze-only` run anywhere.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .vendor import EXPERIMENTS_DIR, ensure_on_path

ensure_on_path()

from storyform import ParseError, Term, canonical, parse_equation  # noqa: E402

ETP_REPO = "teorth/equational_theories"
ETP_COMMIT = "1aec8a7acf223b7c56e4830977b6e90d4ef1924b"
_RAW = "https://raw.githubusercontent.com/{}/{}/data/".format(ETP_REPO, ETP_COMMIT)

# The law list is byte-identical to the one benchmark.py caches from the main
# branch, so pairs sampled through the existing samplers carry ETP numbers that
# index this table correctly. build_dataset checks that rather than assuming it.
EQUATIONS_FILE = "equations.txt"
EQUATIONS_SHA256 = "e30e1a6735011ff36fdb228d37b232eb7cda9d4a836cbbefc69c07e6871c8010"

# The most recent published outcome table. In it, 190 of the 22,033,636 cells
# are still marked conjectured or unknown; those read as None, never as an
# answer.
OUTCOMES_FILE = "2024-11-10-outcomes.json.zip"
OUTCOMES_SHA256 = "eb8707f4de2d1487973a76200ced237ddb2650c1c483e12b826e0e1f49185cd0"

N_LAWS = 4694

DATA_DIR = EXPERIMENTS_DIR / ".etp-data"

# One byte per ordered pair in the cache.
FALSE, TRUE, UNKNOWN = 0, 1, 2
_CODES = {
    "explicit_proof_true": TRUE,
    "implicit_proof_true": TRUE,
    "explicit_proof_false": FALSE,
    "implicit_proof_false": FALSE,
}


def provenance() -> dict:
    """What a run directory should record about the table it graded against."""
    return {
        "etp_repo": ETP_REPO,
        "etp_commit": ETP_COMMIT,
        "etp_equations_sha256": EQUATIONS_SHA256,
        "etp_outcomes_file": OUTCOMES_FILE,
        "etp_outcomes_sha256": OUTCOMES_SHA256,
    }


# ------------------------------------------------------------------ download


def _fetch(name: str, expected_sha256: str, data_dir: Path) -> bytes:
    """A pinned ETP data file, downloaded once and verified every time."""
    path = Path(data_dir) / name
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(_RAW + name, timeout=120) as response:
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise SystemExit(
            "{} has sha256 {}, expected {}; refusing to grade against a "
            "different table".format(name, digest, expected_sha256)
        )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def load_equations(data_dir: Path = DATA_DIR) -> List[str]:
    """ETP's law list, law N at index N - 1."""
    text = _fetch(EQUATIONS_FILE, EQUATIONS_SHA256, data_dir).decode("utf-8")
    equations = text.splitlines()
    if len(equations) != N_LAWS:
        raise SystemExit("expected {} laws, found {}".format(N_LAWS, len(equations)))
    return equations


# ------------------------------------------------------------- implications


class Implications:
    """ETP's verdict on every ordered pair of laws."""

    def __init__(self, table: bytes, n: int = N_LAWS):
        if len(table) != n * n:
            raise ValueError("table has {} cells, expected {}".format(len(table), n * n))
        self._table = table
        self.n = n

    def implies(self, a: int, b: int) -> Optional[bool]:
        """Does every magma obeying law `a` obey law `b`? None if ETP is open."""
        if not (1 <= a <= self.n and 1 <= b <= self.n):
            raise ValueError("law numbers run 1..{}, got {} and {}".format(self.n, a, b))
        code = self._table[(a - 1) * self.n + (b - 1)]
        if code == UNKNOWN:
            return None
        return code == TRUE

    def equivalent(self, a: int, b: int) -> Optional[bool]:
        """True when each law implies the other; None if either direction is open."""
        forward, backward = self.implies(a, b), self.implies(b, a)
        if forward is False or backward is False:
            return False
        if forward is None or backward is None:
            return None
        return True


def _table_from_outcomes(zipped: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(zipped)) as archive:
        names = [n for n in archive.namelist() if n.endswith(".json")]
        if len(names) != 1:
            raise SystemExit("expected one JSON file in {}, found {}".format(OUTCOMES_FILE, names))
        with archive.open(names[0]) as handle:
            payload = json.load(handle)

    labels = payload["equations"]
    if labels != ["Equation{}".format(i) for i in range(1, N_LAWS + 1)]:
        raise SystemExit("the outcome table's law order is not Equation1..Equation{}".format(N_LAWS))
    rows = payload["outcomes"]
    table = bytearray(N_LAWS * N_LAWS)
    for i, row in enumerate(rows):
        if len(row) != N_LAWS:
            raise SystemExit("row {} of the outcome table has {} cells".format(i + 1, len(row)))
        table[i * N_LAWS:(i + 1) * N_LAWS] = bytes(_CODES.get(cell, UNKNOWN) for cell in row)
    return bytes(table)


def load_implications(data_dir: Path = DATA_DIR) -> Implications:
    """The implication table, built from the pinned download on first use."""
    cache = Path(data_dir) / "implications-{}.bin.gz".format(OUTCOMES_SHA256[:12])
    if cache.exists():
        return Implications(gzip.decompress(cache.read_bytes()))
    table = _table_from_outcomes(_fetch(OUTCOMES_FILE, OUTCOMES_SHA256, data_dir))
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(".partial")
    partial.write_bytes(gzip.compress(table))
    partial.replace(cache)
    return Implications(table)


# ---------------------------------------------------------- identifying laws


def law_key(lhs: Term, rhs: Term) -> str:
    """The same string for an equation however its variables are named and
    whichever side is written first."""
    return min(canonical(lhs, rhs), canonical(rhs, lhs))


class LawIndex:
    """Maps an equation to its ETP law number."""

    def __init__(self, equations: List[str]):
        self._by_key: Dict[str, int] = {}
        for number, text in enumerate(equations, start=1):
            key = law_key(*parse_equation(text))
            if key in self._by_key:
                raise ValueError(
                    "laws {} and {} are the same law; the list is meant to hold "
                    "each law once".format(self._by_key[key], number)
                )
            self._by_key[key] = number

    def identify(self, lhs: Term, rhs: Term) -> Optional[int]:
        """The law's ETP number, or None if it is not one of the 4,694.

        An equation with identical sides, such as `op(x, y) = op(x, y)`, holds
        in every magma, so it is law 1 (`x = x`) in all but spelling. ETP lists
        only the bare form, so the rest are folded onto it here.
        """
        if lhs == rhs:
            return 1
        return self._by_key.get(law_key(lhs, rhs))

    def identify_text(self, text: str) -> Optional[int]:
        try:
            return self.identify(*parse_equation(text))
        except ParseError:
            return None


class Etp:
    """The law index and the implication table together."""

    def __init__(self, index: LawIndex, implications: Implications):
        self.index = index
        self.implications = implications

    def implication(
        self, assume: Tuple[Term, Term], ask: Tuple[Term, Term]
    ) -> Tuple[Optional[int], Optional[int], Optional[bool]]:
        """(law number of ASSUME, law number of ASK, whether ASSUME implies ASK).

        The verdict is None when either law is outside the catalog or ETP has
        not settled the pair.
        """
        a = self.index.identify(*assume)
        b = self.index.identify(*ask)
        if a is None or b is None:
            return a, b, None
        return a, b, self.implications.implies(a, b)


_LOADED: Dict[str, Etp] = {}


def load(data_dir: Path = DATA_DIR) -> Etp:
    """The pinned law index and implication table, loaded once per process."""
    key = str(Path(data_dir).resolve())
    if key not in _LOADED:
        _LOADED[key] = Etp(LawIndex(load_equations(data_dir)), load_implications(data_dir))
    return _LOADED[key]
