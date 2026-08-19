#!/usr/bin/env python3
"""Truthdata: ETP implication outcomes and truth-balanced pair sampling.

The upstream Equational Theories Project resolved every implication
"law E implies law F" between the 4694 listed laws; its repository
publishes the full outcome matrix as a dated JSON snapshot. This module
downloads that snapshot once, compacts it into a 4694x4694 byte matrix
(one status code per ordered pair, ~21 MB, loads in milliseconds), and
answers implication_truth(e, f) queries against it. Only Lean-verified
statuses (explicit/implicit_proof_true/false) count as ground truth;
conjecture and unknown entries map to None and are never sampled.

sample_truth_balanced draws (E, F) pairs stratified by the pair's total
operation count with each bin split exactly 50/50 between true and false
implications. The draw never consults a rendering form, so every arm of
an experiment run with one seed covers the identical pair set. Vacuous
laws (E1 x = x, E2 x = y) are excluded structurally: bins start at 2 and
every split gives both laws at least one operation.

CLI:
    python3 truthdata.py build            # download + compact the matrix
    python3 truthdata.py stats --bins 2:8 # per-bin true/false availability
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from genform import parse_bins
from literalform import render_description
from storyform import ParseError, Term, Var, parse_equation, render_story

SNAPSHOT = "2024-11-10"
OUTCOMES_URL = (
    "https://raw.githubusercontent.com/teorth/equational_theories/main/data/"
    f"{SNAPSHOT}-outcomes.json.zip"
)
DATA_DIR = Path(__file__).resolve().parent / "data"
ZIP_PATH = DATA_DIR / f"{SNAPSHOT}-outcomes.json.zip"
MATRIX_PATH = DATA_DIR / f"outcomes-{SNAPSHOT}.bin"
MATRIX_META_PATH = DATA_DIR / f"outcomes-{SNAPSHOT}.meta.json"

# Status vocabulary of the upstream snapshot (extract_implications
# outcomes). Byte code = index into this tuple; the legend is duplicated
# into the meta file so the .bin is self-describing.
STATUSES = (
    "unknown",
    "explicit_proof_true",
    "implicit_proof_true",
    "explicit_conjecture_true",
    "implicit_conjecture_true",
    "explicit_conjecture_false",
    "implicit_conjecture_false",
    "explicit_proof_false",
    "implicit_proof_false",
)
_CODE_OF = {status: code for code, status in enumerate(STATUSES)}
TRUE_STATUSES = frozenset({"explicit_proof_true", "implicit_proof_true"})
FALSE_STATUSES = frozenset({"explicit_proof_false", "implicit_proof_false"})
_TRUE_CODES = frozenset(_CODE_OF[s] for s in TRUE_STATUSES)
_FALSE_CODES = frozenset(_CODE_OF[s] for s in FALSE_STATUSES)


# ----------------------------------------------------------------- Matrix


class TruthMatrix:
    """The full outcome matrix, one status-code byte per ordered pair.

    Row-major over 1-indexed ETP numbering: byte (e-1)*n + (f-1) holds
    the status of "equation e implies equation f".
    """

    def __init__(self, matrix: bytes, n: int, meta: dict):
        if len(matrix) != n * n:
            raise ValueError(f"matrix has {len(matrix)} bytes, expected {n}x{n}")
        self.matrix = matrix
        self.n = n
        self.meta = meta

    def _check(self, e_num: int, f_num: int) -> None:
        if not (1 <= e_num <= self.n and 1 <= f_num <= self.n):
            raise ValueError(f"equation numbers must be in 1..{self.n}")

    def status(self, e_num: int, f_num: int) -> str:
        """Upstream status string for "e_num implies f_num"."""
        self._check(e_num, f_num)
        return STATUSES[self.matrix[(e_num - 1) * self.n + (f_num - 1)]]

    def truth(self, e_num: int, f_num: int) -> Optional[bool]:
        """Lean-verified truth of the implication, or None if unproved."""
        self._check(e_num, f_num)
        code = self.matrix[(e_num - 1) * self.n + (f_num - 1)]
        if code in _TRUE_CODES:
            return True
        if code in _FALSE_CODES:
            return False
        return None

    def row(self, e_num: int) -> bytes:
        """All statuses of "e_num implies -", as raw codes."""
        self._check(e_num, e_num)
        return self.matrix[(e_num - 1) * self.n : e_num * self.n]

    def save(
        self, matrix_path: Path = MATRIX_PATH, meta_path: Path = MATRIX_META_PATH
    ) -> None:
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_path.write_bytes(self.matrix)
        meta_path.write_text(
            json.dumps(self.meta, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        matrix_path: Path = MATRIX_PATH,
        meta_path: Path = MATRIX_META_PATH,
    ) -> "TruthMatrix":
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        matrix = Path(matrix_path).read_bytes()
        if meta.get("legend") != list(STATUSES):
            raise SystemExit(
                f"{meta_path} legend does not match this module's STATUSES; "
                "delete the cache and rebuild"
            )
        return cls(matrix, meta["n"], meta)


def _encode_payload(payload: dict) -> Tuple[bytes, int, Dict[str, int]]:
    """Validate an outcomes payload and encode it as status-code bytes.

    Hard-fails on any surprise: unknown status strings, ragged rows, or
    equation names that do not line up with ETP numbering (the alignment
    is what ties matrix indices to data/equations.txt line numbers).
    """
    names = payload["equations"]
    outcomes = payload["outcomes"]
    n = len(names)
    if len(outcomes) != n:
        raise SystemExit(f"{n} equations but {len(outcomes)} outcome rows")
    for i, name in enumerate(names):
        if name != f"Equation{i + 1}":
            raise SystemExit(
                f"equation index {i} is named {name!r}, not 'Equation{i + 1}'; "
                "matrix indices would not match ETP numbering"
            )
    matrix = bytearray(n * n)
    counts = {status: 0 for status in STATUSES}
    for i, row in enumerate(outcomes):
        if len(row) != n:
            raise SystemExit(f"outcome row {i} has {len(row)} entries, expected {n}")
        base = i * n
        for j, status in enumerate(row):
            code = _CODE_OF.get(status)
            if code is None:
                raise SystemExit(f"unknown outcome status {status!r} at ({i}, {j})")
            matrix[base + j] = code
            counts[status] += 1
    return bytes(matrix), n, counts


def _check_orientation(matrix: TruthMatrix) -> None:
    """Fail hard if outcomes[i][j] is not "Equation i+1 implies Equation j+1".

    Three facts hold in the true data and all break under transposition
    or misalignment: every law implies itself (diagonal), every law
    implies E1 x = x (column 1), and E2 x = y implies every law (row 2).
    """
    n = matrix.n
    for k in range(1, n + 1):
        if matrix.truth(k, k) is not True:
            raise SystemExit(f"diagonal check failed at E{k}: {matrix.status(k, k)}")
        if matrix.truth(k, 1) is not True:
            raise SystemExit(
                f"column-E1 check failed at E{k}: {matrix.status(k, 1)} "
                "(matrix is misaligned or transposed)"
            )
        if matrix.truth(2, k) is not True:
            raise SystemExit(
                f"row-E2 check failed at E{k}: {matrix.status(2, k)} "
                "(matrix is misaligned or transposed)"
            )


def build_matrix(
    data_dir: Path = DATA_DIR, url: str = OUTCOMES_URL
) -> TruthMatrix:
    """Download the outcomes snapshot and compact it into the byte matrix.

    The JSON member holds 22 million short strings, so this one-time
    parse briefly needs a few GB of RAM; every later use reads the ~21 MB
    .bin instead. The zip stays cached next to it, and both digests go
    into the meta file so runs can pin the ground-truth version.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / Path(url).name
    if not zip_path.exists():
        print(f"downloading {url} ...")
        with urllib.request.urlopen(url, timeout=120) as response:
            zip_path.write_bytes(response.read())
    zip_bytes = zip_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        members = [m for m in archive.namelist() if m.endswith(".json")]
        if len(members) != 1:
            raise SystemExit(f"{zip_path} holds {members!r}, expected one .json")
        print(f"parsing {members[0]} (one-time, large) ...")
        payload = json.loads(archive.read(members[0]).decode("utf-8"))
    encoded, n, counts = _encode_payload(payload)
    meta = {
        "source_url": url,
        "snapshot": SNAPSHOT,
        "zip_sha256": hashlib.sha256(zip_bytes).hexdigest(),
        "matrix_sha256": hashlib.sha256(encoded).hexdigest(),
        "n": n,
        "legend": list(STATUSES),
        "status_counts": counts,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    matrix = TruthMatrix(encoded, n, meta)
    _check_orientation(matrix)
    matrix.save(
        data_dir / MATRIX_PATH.name, data_dir / MATRIX_META_PATH.name
    )
    return matrix


def ensure_matrix(data_dir: Path = DATA_DIR) -> TruthMatrix:
    """Load the compact matrix cache, building it on first use."""
    matrix_path = data_dir / MATRIX_PATH.name
    meta_path = data_dir / MATRIX_META_PATH.name
    if matrix_path.exists() and meta_path.exists():
        return TruthMatrix.load(matrix_path, meta_path)
    return build_matrix(data_dir)


_MATRIX: Optional[TruthMatrix] = None


def _singleton() -> TruthMatrix:
    global _MATRIX
    if _MATRIX is None:
        _MATRIX = ensure_matrix()
    return _MATRIX


def implication_status(e_num: int, f_num: int) -> str:
    """Upstream status of "E e_num implies E f_num" (ETP numbering)."""
    return _singleton().status(e_num, f_num)


def implication_truth(e_num: int, f_num: int) -> Optional[bool]:
    """Lean-verified truth of "E e_num implies E f_num", or None."""
    return _singleton().truth(e_num, f_num)


# --------------------------------------------------------------- Sampling


def _term_ops(term: Term) -> int:
    if isinstance(term, Var):
        return 0
    return 1 + _term_ops(term.left) + _term_ops(term.right)


def _term_depth(term: Term) -> int:
    if isinstance(term, Var):
        return 0
    return 1 + max(_term_depth(term.left), _term_depth(term.right))


def _group_by_ops(equations: List[str]) -> Dict[int, List[int]]:
    """Equation numbers grouped by per-equation operation count.

    Zero-op laws (E1, E2) are dropped here, which is what keeps every
    sampled pair free of vacuous laws.
    """
    by_ops: Dict[int, List[int]] = {}
    for number, text in enumerate(equations, start=1):
        try:
            lhs, rhs = parse_equation(text)
        except ParseError:
            continue
        ops = _term_ops(lhs) + _term_ops(rhs)
        if ops == 0:
            continue
        by_ops.setdefault(ops, []).append(number)
    return by_ops


def _bin_splits(by_ops: Dict[int, List[int]], target: int) -> List[Tuple[int, int]]:
    return [
        (a, target - a)
        for a in range(1, target)
        if by_ops.get(a) and by_ops.get(target - a)
    ]


def _renderable(equations: List[str], e_num: int, f_num: int) -> bool:
    """True when every arm's renderer accepts the pair.

    The check is arm-independent by construction — it tries all
    deterministic renderers — so the sampled pair set never depends on
    which form a run will use.
    """
    try:
        render_story(equations[e_num - 1], equations[f_num - 1])
        render_description(equations[e_num - 1], equations[f_num - 1])
    except (ParseError, ValueError):
        return False
    return True


def _pair_record(
    equations: List[str], e_num: int, f_num: int, truth: bool, status: str
) -> dict:
    e_lhs, e_rhs = parse_equation(equations[e_num - 1])
    f_lhs, f_rhs = parse_equation(equations[f_num - 1])
    ops_e = _term_ops(e_lhs) + _term_ops(e_rhs)
    ops_f = _term_ops(f_lhs) + _term_ops(f_rhs)
    return {
        "e_num": e_num,
        "f_num": f_num,
        "truth": truth,
        "status": status,
        "ops_e": ops_e,
        "ops_f": ops_f,
        "ops_total": ops_e + ops_f,
        "depth": max(map(_term_depth, (e_lhs, e_rhs, f_lhs, f_rhs))),
    }


def truth_availability(
    equations: List[str],
    matrix: TruthMatrix,
    bins: Tuple[int, ...] = tuple(range(2, 9)),
) -> Dict[int, Dict[str, int]]:
    """Per total-ops bin: how many ordered pairs are proof-true,
    proof-false, or excluded (conjecture/unknown status).

    Counts the same population sample_truth_balanced draws from — E1/E2
    and the diagonal already excluded — so it predicts exactly whether a
    per-bin quota can fill (modulo the rare unrenderable pair).
    """
    by_ops = _group_by_ops(equations)
    table: Dict[int, Dict[str, int]] = {}
    for target in bins:
        counts = {"true": 0, "false": 0, "excluded": 0}
        for a, b in _bin_splits(by_ops, target):
            for e_num in by_ops[a]:
                row = matrix.row(e_num)
                for f_num in by_ops[b]:
                    if e_num == f_num:
                        continue
                    code = row[f_num - 1]
                    if code in _TRUE_CODES:
                        counts["true"] += 1
                    elif code in _FALSE_CODES:
                        counts["false"] += 1
                    else:
                        counts["excluded"] += 1
        table[target] = counts
    return table


def _availability_message(
    equations: List[str], matrix: TruthMatrix, bins: Tuple[int, ...]
) -> str:
    lines = ["bin  proof-true  proof-false  excluded"]
    for target, counts in truth_availability(equations, matrix, bins).items():
        lines.append(
            f"{target:>3}  {counts['true']:>10}  {counts['false']:>11}  "
            f"{counts['excluded']:>8}"
        )
    return "\n".join(lines)


def sample_truth_balanced(
    equations: List[str],
    matrix: TruthMatrix,
    per_bin: int,
    seed: int,
    bins: Tuple[int, ...] = tuple(range(2, 9)),
) -> List[dict]:
    """Deterministically sample per_bin pairs per total-ops bin, each bin
    split exactly 50/50 between proof-true and proof-false implications.

    True implications are a small minority of ordered pairs, so each
    bin's true half is drawn from the fully enumerated true pool (a
    one-time matrix scan, seconds for the large bins); the abundant
    false half is rejection-sampled like sample_pairs_stratified. Pairs
    whose status is conjecture/unknown are never drawn. The result
    depends only on (equations, matrix, per_bin, seed, bins) and never
    on a rendering form.
    """
    if per_bin % 2:
        raise SystemExit(f"--per-bin must be even for a 50/50 truth split, got {per_bin}")
    half = per_bin // 2
    by_ops = _group_by_ops(equations)
    rng = random.Random(seed)
    pairs: List[dict] = []
    for target in bins:
        splits = _bin_splits(by_ops, target)
        if not splits:
            raise SystemExit(f"no equations available for ops bin {target}")

        true_pool: List[Tuple[int, int]] = []
        for a, b in splits:
            for e_num in by_ops[a]:
                row = matrix.row(e_num)
                true_pool.extend(
                    (e_num, f_num)
                    for f_num in by_ops[b]
                    if e_num != f_num and row[f_num - 1] in _TRUE_CODES
                )
        if len(true_pool) < half:
            raise SystemExit(
                f"ops bin {target} has only {len(true_pool)} proof-true pairs, "
                f"need {half}\n"
                + _availability_message(equations, matrix, bins)
            )
        true_pool.sort()
        rng.shuffle(true_pool)
        chosen = set()
        got = 0
        for e_num, f_num in true_pool:
            if got == half:
                break
            if _renderable(equations, e_num, f_num):
                chosen.add((e_num, f_num))
                pairs.append(
                    _pair_record(
                        equations, e_num, f_num, True, matrix.status(e_num, f_num)
                    )
                )
                got += 1
        if got < half:
            raise SystemExit(
                f"ops bin {target}: only {got} of {half} proof-true pairs are "
                "renderable"
            )

        got = 0
        attempts = 0
        while got < half:
            attempts += 1
            if attempts > 1000 * per_bin:
                raise SystemExit(
                    f"could not fill the proof-false half of ops bin {target}\n"
                    + _availability_message(equations, matrix, bins)
                )
            a, b = splits[rng.randrange(len(splits))]
            e_num = by_ops[a][rng.randrange(len(by_ops[a]))]
            f_num = by_ops[b][rng.randrange(len(by_ops[b]))]
            if e_num == f_num or (e_num, f_num) in chosen:
                continue
            if matrix.truth(e_num, f_num) is not False:
                continue
            if not _renderable(equations, e_num, f_num):
                continue
            chosen.add((e_num, f_num))
            pairs.append(
                _pair_record(
                    equations, e_num, f_num, False, matrix.status(e_num, f_num)
                )
            )
            got += 1
    return pairs


# --------------------------------------------------------------------- CLI


def main(argv: Optional[List[str]] = None) -> int:
    cli = argparse.ArgumentParser(
        description="Fetch the ETP implication-outcome matrix and report "
        "truth-balanced sampling availability."
    )
    commands = cli.add_subparsers(dest="command", required=True)
    build_cmd = commands.add_parser(
        "build", help="download the outcomes snapshot and build the byte matrix"
    )
    build_cmd.add_argument(
        "--force", action="store_true", help="rebuild even if the cache exists"
    )
    stats_cmd = commands.add_parser(
        "stats", help="print per-bin proof-true/false pair availability"
    )
    stats_cmd.add_argument(
        "--bins", default="2:8", metavar="MIN:MAX", help="total-ops bins (default 2:8)"
    )
    args = cli.parse_args(argv)

    if args.command == "build":
        if args.force:
            matrix = build_matrix()
        else:
            matrix = ensure_matrix()
        print(f"matrix: {matrix.n}x{matrix.n}, sha256 {matrix.meta['matrix_sha256']}")
        for status, count in matrix.meta["status_counts"].items():
            print(f"  {status:>26}: {count}")
        return 0

    from benchmark import load_equations

    equations, _ = load_equations()
    matrix = ensure_matrix()
    if len(equations) != matrix.n:
        raise SystemExit(
            f"{len(equations)} equations but a {matrix.n}x{matrix.n} matrix"
        )
    print(_availability_message(equations, matrix, tuple(parse_bins(args.bins))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
