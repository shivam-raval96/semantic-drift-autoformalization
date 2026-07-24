#!/usr/bin/env python3
"""Shared helpers for the FT experiment: canonical class hashes, the
reference rigid-grammar serializer, and AST metrics.

Everything here defers to the repo's own parser/grader code paths
(storyform.canonical, checkform.answer_class_key) so the FT artifacts
collide and grade exactly as the benchmark does.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parent.parent / "informalizing-etp"
sys.path.insert(0, str(REPO))

from checkform import answer_class_key, dual, grade  # noqa: E402
from storyform import (  # noqa: E402
    Op,
    Term,
    Var,
    canonical,
    parse_equation,
    variables_in_order,
)

# The literal arm's letters, also the ETP first-appearance convention.
RG_LETTERS = ("x", "y", "z", "w", "u", "v")


# ---------------------------------------------------------------- Metrics


def term_ops(term: Term) -> int:
    if isinstance(term, Var):
        return 0
    return 1 + term_ops(term.left) + term_ops(term.right)


def term_depth(term: Term) -> int:
    if isinstance(term, Var):
        return 0
    return 1 + max(term_depth(term.left), term_depth(term.right))


def equation_ops(pair: Tuple[Term, Term]) -> int:
    return term_ops(pair[0]) + term_ops(pair[1])


def is_vacuous_pair(e: Tuple[Term, Term], f: Tuple[Term, Term]) -> bool:
    """True if either law never invokes the op (E1 x = x / E2 x = y)."""
    return equation_ops(e) == 0 or equation_ops(f) == 0


# ------------------------------------------------------------ Class keys

# Pair-level: the orbit of (E, F) under the grader's eight accepted
# transforms (swap either equation's sides, dualize both) — computed by
# checkform.answer_class_key itself so the audit can never drift from
# grading. Law-level: the orbit of one equation under side swap and
# dualization.


def serialize_infix(lhs: Term, rhs: Term) -> str:
    """Storyform-parseable infix form of an equation AST."""

    def go(t: Term) -> str:
        if isinstance(t, Var):
            return t.name
        return f"({go(t.left)} ∘ {go(t.right)})"

    return f"{go(lhs)} = {go(rhs)}"


def pair_class_key(e: Tuple[Term, Term], f: Tuple[Term, Term]) -> Tuple[str, str]:
    return answer_class_key(serialize_infix(*e), serialize_infix(*f))


def law_class_key(lhs: Term, rhs: Term) -> str:
    return min(
        canonical(lhs, rhs),
        canonical(rhs, lhs),
        canonical(dual(lhs), dual(rhs)),
        canonical(dual(rhs), dual(lhs)),
    )


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def pair_hash(e: Tuple[Term, Term], f: Tuple[Term, Term]) -> str:
    ck = pair_class_key(e, f)
    return short_hash(f"{ck[0]} => {ck[1]}")


def law_hash(lhs: Term, rhs: Term) -> str:
    return short_hash(law_class_key(lhs, rhs))


# ------------------------------------------------- Reference RG serializer


def rg_equation(lhs: Term, rhs: Term) -> str:
    """One equation as prefix RG text, variables renamed x, y, z, w, u, v
    by first appearance (each equation independently, as the ASSUME and
    ASK lines quantify independently)."""
    order = variables_in_order(lhs, rhs)
    if len(order) > len(RG_LETTERS):
        raise ValueError(f"equation uses {len(order)} variables; max {len(RG_LETTERS)}")
    renaming = {name: RG_LETTERS[i] for i, name in enumerate(order)}

    def go(t: Term) -> str:
        if isinstance(t, Var):
            return renaming[t.name]
        return f"op({go(t.left)}, {go(t.right)})"

    return f"{go(lhs)} = {go(rhs)}"


def rg_text(e: Tuple[Term, Term], f: Tuple[Term, Term]) -> str:
    """The two-line rigid-grammar rendering of an implication."""
    return f"ASSUME: {rg_equation(*e)}\nASK: {rg_equation(*f)}"


def verify_rg_round_trip(text: str, e: Tuple[Term, Term], f: Tuple[Term, Term]) -> None:
    """The RG text must grade exact against its own source pair."""
    verdict = grade(text, {"canonical_e": canonical(*e), "canonical_f": canonical(*f)})
    transform = verdict["transform"]
    if verdict["status"] != "correct" or transform != {
        "swap_e": False,
        "swap_f": False,
        "dual": False,
    }:
        raise AssertionError(f"RG round-trip failed: {verdict} for {text!r}")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
