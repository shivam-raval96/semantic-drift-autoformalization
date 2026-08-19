#!/usr/bin/env python3
"""Tests for the FT v2 grammar registry (grammars.py): round-trip fidelity
over hand-written and train_v1-sampled pairs, cross-grammar confusion,
corruption grading, extraction robustness, and the cleaner hazards."""

import json
import random
import sys
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parent
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import grammars  # noqa: E402
from grammars import GRAMMARS, BAnswerParseError, grade_b  # noqa: E402

checkform = grammars.checkform
canonical = grammars.storyform.canonical
parse_eq = grammars.storyform.parse_equation
Op = grammars.Op
Var = grammars.Var

TRAIN_PATH = V2.parents[0] / "train_v1" / "train.jsonl"
N_SAMPLED = 500

# Degenerate shapes: bare-variable sides, repeated variables, depth-4
# nesting, 6 variables in one equation, non-canonical variable names.
HAND_TEXTS = [
    ("x ∘ y = (y ∘ y) ∘ x", "x ∘ y = y ∘ x"),
    ("x = y ∘ x", "x = x ∘ y"),
    ("x ∘ x = x", "x = x ∘ (x ∘ x)"),
    ("x = (((y ∘ y) ∘ y) ∘ y) ∘ y", "x ∘ (x ∘ (x ∘ x)) = x"),
    ("((x ∘ y) ∘ z) ∘ w = u ∘ v", "x ∘ y = y ∘ x"),
    ("a ∘ b = b ∘ a", "p1 ∘ q2 = q2 ∘ p1"),
]

E387_PAIR = (parse_eq("x ∘ y = (y ∘ y) ∘ x"), parse_eq("x ∘ y = y ∘ x"))


def hand_pairs():
    return [(parse_eq(e), parse_eq(f)) for e, f in HAND_TEXTS]


@pytest.fixture(scope="module")
def corpus_pairs():
    lines = [
        line for line in TRAIN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows = random.Random(0).sample(lines, N_SAMPLED)
    return [GRAMMARS["a"].parse_answer(json.loads(row)["text"]) for row in rows]


# -------------------------------------------------------------- Round trip


@pytest.mark.parametrize("name", sorted(GRAMMARS))
def test_round_trip(name, corpus_pairs):
    g = GRAMMARS[name]
    for e, f in hand_pairs() + corpus_pairs:
        ce, cf = canonical(*e), canonical(*f)
        block = g.serialize_pair(e, f)
        e2, f2 = g.parse_answer(block)
        assert canonical(*e2) == ce, (name, block)
        assert canonical(*f2) == cf, (name, block)
        verdict = grade_b(block, name, ce, cf)
        assert verdict["status"] == "correct", (name, block, verdict)
        assert verdict["transform"] == {"swap_e": False, "swap_f": False, "dual": False}
        if name == "a":
            grammars.ftlib.verify_rg_round_trip(block, e, f)


def test_serialized_surface():
    e, f = E387_PAIR
    assert (
        GRAMMARS["a"].serialize_pair(e, f)
        == "ASSUME: op(x, y) = op(op(y, y), x)\nASK: op(x, y) = op(y, x)"
    )
    assert (
        GRAMMARS["b_near"].serialize_pair(e, f)
        == "GIVEN: f(x, y) = f(f(y, y), x)\nSHOW: f(x, y) = f(y, x)"
    )
    assert (
        GRAMMARS["b_far"].serialize_pair(e, f)
        == "LAW: (x ∘ y) = ((y ∘ y) ∘ x)\nDERIVE: (x ∘ y) = (y ∘ x)"
    )


def test_per_equation_renaming():
    # ASSUME and ASK quantify independently: each restarts at x, y, ...
    e = parse_eq("a ∘ b = b ∘ a")
    f = parse_eq("q ∘ p = p ∘ q")
    assert (
        GRAMMARS["b_near"].serialize_pair(e, f)
        == "GIVEN: f(x, y) = f(y, x)\nSHOW: f(x, y) = f(y, x)"
    )
    assert (
        GRAMMARS["b_far"].serialize_pair(e, f)
        == "LAW: (x ∘ y) = (y ∘ x)\nDERIVE: (x ∘ y) = (y ∘ x)"
    )


@pytest.mark.parametrize("name", sorted(GRAMMARS))
def test_determinism(name):
    g = GRAMMARS[name]
    for e, f in hand_pairs():
        assert g.serialize_pair(e, f) == g.serialize_pair(e, f)


# ------------------------------------------------- Cross-grammar confusion


def test_a_text_rejected_by_b_parsers():
    a_block = GRAMMARS["a"].serialize_pair(*E387_PAIR)
    for name in ("b_near", "b_far"):
        with pytest.raises(BAnswerParseError):
            GRAMMARS[name].parse_answer(a_block)


def test_b_text_unparseable_to_checkform():
    e, f = E387_PAIR
    b_block = GRAMMARS["b_near"].serialize_pair(e, f)
    verdict = checkform.grade(
        b_block, {"canonical_e": canonical(*e), "canonical_f": canonical(*f)}
    )
    assert verdict["status"] == "unparseable"


# -------------------------------------------------------------- Corruption


@pytest.mark.parametrize("name", ("b_near", "b_far"))
def test_arg_swap_grades_wrong(name):
    # Swapping ONE op node's arguments is not a uniform dualization, so it
    # must land outside the grader's symmetry orbit: "wrong", never
    # "unparseable" (the surface is still well-formed).
    e, f = E387_PAIR
    e_bad = (e[0], Op(e[1].right, e[1].left))
    block = GRAMMARS[name].serialize_pair(e_bad, f)
    verdict = grade_b(block, name, canonical(*e), canonical(*f))
    assert verdict["status"] == "wrong", verdict


# -------------------------------------------------- Extraction robustness


def test_extraction_messy_b_near():
    e, f = E387_PAIR
    text = (
        "Sure - here is the formalization.\n"
        "\n"
        "> **given**: `f(x, y) = f(f(y, y), x)`\n"
        "Some reasoning between the two lines.\n"
        "- SHOW:\n"
        "**f(x, y) = f(y, x)**\n"
    )
    verdict = grade_b(text, "b_near", canonical(*e), canonical(*f))
    assert verdict["status"] == "correct", verdict


def test_extraction_messy_b_far():
    e, f = E387_PAIR
    text = (
        "law: `(x ∘ y) = ((y ∘ y) ∘ x)`\n"
        "\n"
        "Derive:\n"
        "(x ∘ y) = (y ∘ x)\n"
    )
    verdict = grade_b(text, "b_far", canonical(*e), canonical(*f))
    assert verdict["status"] == "correct", verdict


def test_extraction_last_occurrence_wins():
    e, f = E387_PAIR
    text = (
        "GIVEN: f(x, x) = x\n"
        "SHOW: f(x, x) = x\n"
        "Wait, that is not right. Final answer:\n"
        "GIVEN: f(x, y) = f(f(y, y), x)\n"
        "SHOW: f(x, y) = f(y, x)\n"
    )
    verdict = grade_b(text, "b_near", canonical(*e), canonical(*f))
    assert verdict["status"] == "correct", verdict
    assert verdict["transform"] == {"swap_e": False, "swap_f": False, "dual": False}


def test_b_near_op_token_case_insensitive():
    e2, f2 = GRAMMARS["b_near"].parse_answer(
        "GIVEN: F(x, y) = F(y, x)\nSHOW: f(x, x) = x"
    )
    assert e2 == (Op(Var("x"), Var("y")), Op(Var("y"), Var("x")))
    assert f2 == (Op(Var("x"), Var("x")), Var("x"))


# ----------------------------------------------------------------- Hazards


def test_cleaner_leniency_matches_grammar_a():
    # The B cleaner is byte-identical to checkform._clean so the A-vs-B
    # unparseable comparison stays fair: a sentence-final period or a
    # backtick/bold wrapper is repaired in every grammar, not just A.
    (e_l, e_r), (f_l, f_r) = GRAMMARS["b_far"].parse_answer(
        "LAW: (x ∘ y) = x.\nDERIVE: `x = (y ∘ x)`"
    )
    assert e_r == Var("x")
    assert f_l == Var("x")
    (e2, _), _ = GRAMMARS["b_near"].parse_answer(
        "GIVEN: **f(x, y) = x**\nSHOW: f(x, y) = f(y, x)."
    )
    assert e2 == Op(Var("x"), Var("y"))


def test_b_near_variable_f_reserved():
    with pytest.raises(BAnswerParseError):
        GRAMMARS["b_near"].parse_answer("GIVEN: f = f(x, y)\nSHOW: x = f(x, y)")


def test_seven_variables_raise_in_serializers():
    lhs = Var("a")
    for name in "bcdefg":
        lhs = Op(lhs, Var(name))
    pair = ((lhs, Var("a")), E387_PAIR[1])
    for name in sorted(GRAMMARS):
        with pytest.raises(ValueError):
            GRAMMARS[name].serialize_pair(*pair)


def test_b_near_depth_cap():
    nested = "x"
    for _ in range(grammars._MAX_TERM_DEPTH + 5):
        nested = f"f({nested}, x)"
    with pytest.raises(BAnswerParseError):
        GRAMMARS["b_near"].parse_answer(f"GIVEN: {nested} = x\nSHOW: x = f(x, y)")
