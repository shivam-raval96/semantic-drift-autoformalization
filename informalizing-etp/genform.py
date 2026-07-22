#!/usr/bin/env python3
"""Genform: seeded synthesis of magma laws beyond the ETP complexity cap.

The ETP equation list stops at 4 operations per equation. Complexity
experiments past that cap need laws the list cannot supply, so this
module generates them: random equations with a chosen operation count,
emitted in the exact equations.txt format (fully parenthesized infix
with ◇) so every downstream component — parsing, rendering, grading,
complexity metrics — works on them unchanged. Ground truth never
consults ETP numbering (grading compares canonical forms computed from
the equation strings), so synthetic laws need no ETP entry.

Generation is a pure function of the seed: one RNG stream drives all
bins in ascending order, so a (seed, bins, per-bin) triple always
yields the byte-identical corpus. Per equation with m operations:

- The m operations are split between the sides uniformly, then
  normalized so the LHS is the smaller side (the ETP `x = x ◇ x`
  convention; grading is side-swap-invariant, so this is cosmetic).
- Each side's tree shape is drawn uniformly over all binary trees with
  its operation count, via Catalan-weighted subtree splits. A naive
  uniform split-size choice would over-weight deep path-like trees,
  making the depth mix at fixed ops a generator artifact.
- Variables follow the ETP's first-appearance style: walking the
  leaves left to right (LHS first), each leaf picks uniformly among
  the variables already used plus one fresh candidate, with fresh
  excluded once all 6 renderer-supported letters are in play. Fresh
  letters are assigned in x, y, z, w, u, v order, so emitted equations
  are their own canonical naming.
- Degenerate equations (identical sides) are redrawn, so no generated
  law is vacuous: every equation has at least one operation and two
  distinct sides.
- Laws are deduplicated up to variable renaming and side swap (duals
  are kept distinct — the ETP itself lists E4 and E5 separately). A
  bin that exhausts its law space below the requested count is
  accepted only if it still supports the sampler's per-bin pair quota.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

from literalform import VARIABLE_LETTERS
from storyform import Op, Term, Var, canonical, parse_equation

MAX_VARS = len(VARIABLE_LETTERS)

# C_0 .. C_12: enough for equations up to 12 operations on one side.
CATALAN = (1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012)


# ------------------------------------------------------------- Generation


def random_shape(rng: random.Random, k: int) -> Term:
    """A uniform draw over the CATALAN[k] binary tree shapes with k ops.

    Leaves are placeholder variables; _fill assigns real names later.
    The left subtree's op count i is drawn with weight C_i * C_{k-1-i},
    which makes the whole tree exactly uniform by induction.
    """
    if k == 0:
        return Var("?")
    draw = rng.randrange(CATALAN[k])
    for i in range(k):
        weight = CATALAN[i] * CATALAN[k - 1 - i]
        if draw < weight:
            return Op(random_shape(rng, i), random_shape(rng, k - 1 - i))
        draw -= weight
    raise AssertionError("Catalan weights must cover the draw")


def assign_variables(rng: random.Random, leaves: int) -> List[str]:
    """ETP-style fresh-or-reuse letters for an equation's leaves.

    Each leaf chooses uniformly among the variables used so far plus
    one fresh candidate (fresh excluded at the 6-letter renderer cap).
    Fresh letters appear in VARIABLE_LETTERS order, so the sequence is
    already named by first appearance.
    """
    names: List[str] = []
    used = 0
    for _ in range(leaves):
        limit = used + 1 if used < MAX_VARS else used
        choice = rng.randrange(limit)
        names.append(VARIABLE_LETTERS[choice])
        if choice == used:
            used += 1
    return names


def _fill(shape: Term, names: Iterator[str]) -> Term:
    if isinstance(shape, Var):
        return Var(next(names))
    return Op(_fill(shape.left, names), _fill(shape.right, names))


def random_equation(rng: random.Random, m: int) -> Tuple[Term, Term]:
    """One random law with exactly m operations across both sides."""
    a = rng.randrange(m + 1)
    lhs_ops = min(a, m - a)
    lhs_shape = random_shape(rng, lhs_ops)
    rhs_shape = random_shape(rng, m - lhs_ops)
    names = iter(assign_variables(rng, m + 2))
    return _fill(lhs_shape, names), _fill(rhs_shape, names)


def serialize_equation(lhs: Term, rhs: Term) -> str:
    """Render a law in the equations.txt convention: top-level ops bare,
    nested subterms parenthesized, ◇ as the operation symbol."""

    def serialize(term: Term, top: bool) -> str:
        if isinstance(term, Var):
            return term.name
        inner = f"{serialize(term.left, False)} ◇ {serialize(term.right, False)}"
        return inner if top else f"({inner})"

    return f"{serialize(lhs, True)} = {serialize(rhs, True)}"


def law_key(lhs: Term, rhs: Term) -> str:
    """Dedup key: same law up to variable renaming and side swap.

    Duals map to different keys on purpose — mirrored argument order is
    a different law, exactly as in the ETP list.
    """
    return min(canonical(lhs, rhs), canonical(rhs, lhs))


def generate_bin(
    rng: random.Random, m: int, count: int, min_pairs: int
) -> List[str]:
    """Up to count distinct m-op laws; fewer only if the space runs out
    while still supporting min_pairs ordered (E, F) pairs."""
    seen = set()
    laws: List[str] = []
    attempts = 0
    while len(laws) < count and attempts < 200 * count:
        attempts += 1
        lhs, rhs = random_equation(rng, m)
        if lhs == rhs:
            continue
        key = law_key(lhs, rhs)
        if key in seen:
            continue
        seen.add(key)
        laws.append(serialize_equation(lhs, rhs))
    if len(laws) < count and len(laws) * (len(laws) - 1) < min_pairs:
        raise SystemExit(
            f"ops bin {m}: only {len(laws)} distinct laws found — "
            f"not enough for {min_pairs} ordered pairs"
        )
    return laws


def generate_corpus(
    seed: int, bins: range, per_bin: int, min_pairs: int = 20
) -> List[str]:
    """All bins in ascending order from one seeded RNG stream."""
    rng = random.Random(seed)
    lines: List[str] = []
    for m in bins:
        lines.extend(generate_bin(rng, m, per_bin, min_pairs))
    return lines


# ------------------------------------------------------------- Self-check


def check_corpus(lines: List[str], bins: range) -> None:
    """Fail fast if any emitted line would misbehave downstream."""
    keys = set()
    ops_seen = []
    for line in lines:
        lhs, rhs = parse_equation(line)
        if serialize_equation(lhs, rhs) != line:
            raise SystemExit(f"round-trip mismatch: {line!r}")
        ops = _term_ops(lhs) + _term_ops(rhs)
        if ops not in bins:
            raise SystemExit(f"unexpected op count {ops}: {line!r}")
        if len(set(_leaf_names(lhs) + _leaf_names(rhs))) > MAX_VARS:
            raise SystemExit(f"too many variables: {line!r}")
        key = law_key(lhs, rhs)
        if key in keys:
            raise SystemExit(f"duplicate law: {line!r}")
        keys.add(key)
        ops_seen.append(ops)
    if ops_seen != sorted(ops_seen):
        raise SystemExit("bins are not in ascending order")


def _term_ops(term: Term) -> int:
    if isinstance(term, Var):
        return 0
    return 1 + _term_ops(term.left) + _term_ops(term.right)


def _leaf_names(term: Term) -> List[str]:
    if isinstance(term, Var):
        return [term.name]
    return _leaf_names(term.left) + _leaf_names(term.right)


# -------------------------------------------------------------------- CLI


def parse_bins(text: str) -> range:
    lo, sep, hi = text.partition(":")
    try:
        low = int(lo)
        high = int(hi) if sep else low
    except ValueError:
        raise SystemExit(f"--bins expects MIN:MAX, got {text!r}")
    if not 1 <= low <= high or high >= len(CATALAN):
        raise SystemExit(f"--bins must satisfy 1 <= MIN <= MAX < {len(CATALAN)}")
    return range(low, high + 1)


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(
        description="Generate synthetic magma laws in equations.txt format."
    )
    cli.add_argument("--seed", type=int, default=0, help="generator seed")
    cli.add_argument(
        "--bins",
        type=parse_bins,
        default=parse_bins("1:10"),
        metavar="MIN:MAX",
        help="per-equation operation counts to cover (default 1:10)",
    )
    cli.add_argument(
        "--per-bin",
        type=int,
        default=40,
        metavar="COUNT",
        help="distinct laws to generate per bin (default 40)",
    )
    cli.add_argument(
        "--min-pairs",
        type=int,
        default=20,
        metavar="N",
        help="accept a saturated bin only if it supports N ordered pairs",
    )
    cli.add_argument("--out", type=Path, required=True, help="output file")
    args = cli.parse_args(argv)

    lines = generate_corpus(args.seed, args.bins, args.per_bin, args.min_pairs)
    check_corpus(lines, args.bins)
    data = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(data, encoding="utf-8")

    counts: dict = {}
    for line in lines:
        lhs, rhs = parse_equation(line)
        ops = _term_ops(lhs) + _term_ops(rhs)
        counts[ops] = counts.get(ops, 0) + 1
    for ops in sorted(counts):
        print(f"bin {ops:>2}: {counts[ops]} laws")
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    print(f"{len(lines)} laws -> {args.out} (sha256 {digest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
