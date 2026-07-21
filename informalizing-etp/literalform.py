#!/usr/bin/env python3
"""Literalform: direct natural-language rendering of ETP implications.

The contrasting arm to storyform: instead of a fuzzy themed story, the
implication "law E implies law F" becomes a plain-English description that
openly talks about an operation, its two ordered inputs, and named
variables. No setting, no agent, no palette — a literal translation.

Terms render as definition steps, mirroring storyform's named
intermediates: each application of the operation is introduced on its own
("apply the operation to x as its first input and y as its second input,
and call the result Value 1"), and later steps refer to earlier results
by name, so nesting never appears inline. Like storyform, the renderer
is a pure function of the (E, F) pair, and the description alone
determines both term trees.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from storyform import (
    Op,
    ParseError,
    Term,
    Var,
    canonical,
    parse_equation,
    variables_in_order,
    write_record,
)

# Fixed variable letters, assigned by order of first appearance (LHS
# first, then RHS). Six letters, matching the ETP maximum.
VARIABLE_LETTERS = ("x", "y", "z", "w", "u", "v")

STYLE = "literal"

# Intermediate-result label, the literal arm's counterpart to a theme's
# result_noun ("Batch 1" -> "Value 1"). Must not collide with any theme
# vocabulary (the no-fuzz test enforces this) or with a variable letter.
RESULT_NOUN = "Value"

OPENING = (
    "Consider a collection of objects together with an operation that "
    "combines two objects into one. The operation takes a first input and "
    "a second input, and the order of the inputs matters."
)

ASSUME_LEAD = "Suppose the following always holds."
QUESTION_LEAD = "Now consider the following question."


# -------------------------------------------------------------- Rendering


def _rename(lhs: Term, rhs: Term) -> Dict[str, str]:
    order = variables_in_order(lhs, rhs)
    if len(order) > len(VARIABLE_LETTERS):
        raise ValueError(
            f"law uses {len(order)} variables; at most "
            f"{len(VARIABLE_LETTERS)} are supported"
        )
    return {name: VARIABLE_LETTERS[i] for i, name in enumerate(order)}


Step = Tuple[str, str, int]  # (left input ref, right input ref, value number)


def _linearize(term: Term, name_of: Dict[str, str], counter: List[int]) -> Tuple[str, List[Step]]:
    """Post-order walk: each application becomes a numbered definition step.

    Returns the reference naming this term's result ("x" or "Value 3")
    and the steps that compute it. The counter is shared across both
    sides of a law so names never clash.
    """
    if isinstance(term, Var):
        return name_of[term.name], []
    left_ref, left_steps = _linearize(term.left, name_of, counter)
    right_ref, right_steps = _linearize(term.right, name_of, counter)
    counter[0] += 1
    number = counter[0]
    ref = f"{RESULT_NOUN} {number}"
    return ref, left_steps + right_steps + [(left_ref, right_ref, number)]


def _step_text(step: Step) -> str:
    left_ref, right_ref, number = step
    return (
        f"apply the operation to {left_ref} as its first input and "
        f"{right_ref} as its second input, and call the result "
        f"{RESULT_NOUN} {number}"
    )


def _quantifier(letters: List[str]) -> str:
    if len(letters) == 1:
        return f"for every choice of an object {letters[0]}"
    if len(letters) == 2:
        listing = f"{letters[0]} and {letters[1]}"
    else:
        listing = ", ".join(letters[:-1]) + f", and {letters[-1]}"
    return f"for every choice of objects {listing}"


def _capitalize(sentence: str) -> str:
    return sentence[0].upper() + sentence[1:]


def _law_parts(lhs: Term, rhs: Term) -> Tuple[List[str], List[Step], str, str]:
    """Quantifier letters, definition steps, and the two compared refs."""
    name_of = _rename(lhs, rhs)
    letters = [name_of[v] for v in variables_in_order(lhs, rhs)]
    counter = [0]
    lhs_ref, lhs_steps = _linearize(lhs, name_of, counter)
    rhs_ref, rhs_steps = _linearize(rhs, name_of, counter)
    return letters, lhs_steps + rhs_steps, lhs_ref, rhs_ref


def render_assumption(lhs: Term, rhs: Term) -> str:
    """The assumed law as one paragraph of definition steps."""
    letters, steps, lhs_ref, rhs_ref = _law_parts(lhs, rhs)
    quantifier = _capitalize(_quantifier(letters))
    equality = f"{lhs_ref} is always equal to {rhs_ref}"
    if steps:
        procedure = "; then ".join(_step_text(step) for step in steps)
        return f"{ASSUME_LEAD} {quantifier}, {procedure}. Then {equality}."
    return f"{ASSUME_LEAD} {quantifier}, {equality}."


def render_question(lhs: Term, rhs: Term) -> str:
    """The questioned law as one paragraph, closing with the question."""
    letters, steps, lhs_ref, rhs_ref = _law_parts(lhs, rhs)
    equality = f"{lhs_ref} is always equal to {rhs_ref}"
    if steps:
        quantifier = _capitalize(_quantifier(letters))
        procedure = "; then ".join(_step_text(step) for step in steps)
        return (
            f"{QUESTION_LEAD} {quantifier}, {procedure}. "
            f"Does it follow that {equality}?"
        )
    return f"{QUESTION_LEAD} Does it follow that, {_quantifier(letters)}, {equality}?"


def render_description(e_text: str, f_text: str) -> Tuple[str, dict]:
    """Render the implication "E implies F" as a literal question.

    Returns (description, metadata). Same record schema as storyform so
    checkform grades answers against it unchanged; the "theme" field holds
    the style key so record filenames stay deterministic and distinct.
    """
    e_lhs, e_rhs = parse_equation(e_text)
    f_lhs, f_rhs = parse_equation(f_text)

    paragraphs = [
        OPENING,
        render_assumption(e_lhs, e_rhs),
        render_question(f_lhs, f_rhs),
    ]
    description = "\n\n".join(paragraphs)

    e_renaming = _rename(e_lhs, e_rhs)
    f_renaming = _rename(f_lhs, f_rhs)
    metadata = {
        "theme": STYLE,
        "style": STYLE,
        "equation_e": e_text,
        "equation_f": f_text,
        "canonical_e": canonical(e_lhs, e_rhs),
        "canonical_f": canonical(f_lhs, f_rhs),
        "letters_e": e_renaming,
        "letters_f": f_renaming,
    }
    return description, metadata


# ------------------------------------------------------------ Back-parser


class LiteralBackparseError(ValueError):
    pass


_LETTER = f"(?:{'|'.join(VARIABLE_LETTERS)})"
_REF = rf"(?:{_LETTER}|{RESULT_NOUN} \d+)"
_STEP_RE = re.compile(
    rf"apply the operation to (?P<a>{_REF}) as its first input and "
    rf"(?P<b>{_REF}) as its second input, and call the result "
    rf"{RESULT_NOUN} (?P<n>\d+)"
)
_EQUALITY_RE = re.compile(rf"(?P<x>{_REF}) is always equal to (?P<y>{_REF})")
# Matches _quantifier's output ("an object x", "objects x and y", or the
# Oxford-comma listing "objects x, y, and z"), capitalized or not.
_QUANTIFIER_RE = re.compile(
    rf"[Ff]or every choice of (?:an object {_LETTER}"
    rf"|objects {_LETTER}(?:, {_LETTER})*,? and {_LETTER}), "
)


def _resolve(ref: str, values: Dict[int, Term]) -> Term:
    match = re.fullmatch(rf"{RESULT_NOUN} (\d+)", ref)
    if match:
        number = int(match.group(1))
        if number not in values:
            raise LiteralBackparseError(f"{ref!r} used before it was defined")
        return values[number]
    return Var(ref)


def _parse_procedure(text: str) -> Dict[int, Term]:
    """Build the value environment from a law's definition steps, in order."""
    values: Dict[int, Term] = {}
    for part in text.split("; then "):
        match = _STEP_RE.fullmatch(part)
        if not match:
            raise LiteralBackparseError(f"malformed step {part!r}")
        number = int(match.group("n"))
        if number != len(values) + 1:
            raise LiteralBackparseError(
                f"{RESULT_NOUN} {number} defined out of order"
            )
        values[number] = Op(
            _resolve(match.group("a"), values),
            _resolve(match.group("b"), values),
        )
    return values


def _parse_equality(text: str, values: Dict[int, Term]) -> Tuple[Term, Term]:
    match = _EQUALITY_RE.fullmatch(text)
    if not match:
        raise LiteralBackparseError(f"malformed equality {text!r}")
    return _resolve(match.group("x"), values), _resolve(match.group("y"), values)


def _parse_assumption(paragraph: str) -> Tuple[Term, Term]:
    prefix = ASSUME_LEAD + " "
    if not paragraph.startswith(prefix) or not paragraph.endswith("."):
        raise LiteralBackparseError("malformed assumption paragraph")
    body = paragraph[len(prefix) : -1]
    match = _QUANTIFIER_RE.match(body)
    if not match:
        raise LiteralBackparseError(f"no quantifier prefix in {body!r}")
    rest = body[match.end() :]
    if ". Then " in rest:
        procedure, equality_text = rest.split(". Then ", 1)
        values = _parse_procedure(procedure)
    else:
        equality_text, values = rest, {}
    return _parse_equality(equality_text, values)


def _parse_question(paragraph: str) -> Tuple[Term, Term]:
    prefix = QUESTION_LEAD + " "
    if not paragraph.startswith(prefix) or not paragraph.endswith("?"):
        raise LiteralBackparseError("malformed question paragraph")
    body = paragraph[len(prefix) : -1]
    ask = "Does it follow that "
    if body.startswith("Does it follow that, "):
        rest = body[len("Does it follow that, ") :]
        match = _QUANTIFIER_RE.match(rest)
        if not match:
            raise LiteralBackparseError(f"no quantifier prefix in {rest!r}")
        return _parse_equality(rest[match.end() :], {})
    match = _QUANTIFIER_RE.match(body)
    if not match:
        raise LiteralBackparseError(f"no quantifier prefix in {body!r}")
    rest = body[match.end() :]
    procedure, sep, equality_text = rest.partition(". " + ask)
    if not sep:
        raise LiteralBackparseError("question paragraph has no closing question")
    values = _parse_procedure(procedure)
    return _parse_equality(equality_text, values)


def backparse_literal(description: str) -> dict:
    """Recover both laws from a literal description's text alone."""
    paragraphs = description.split("\n\n")
    if len(paragraphs) != 3 or paragraphs[0] != OPENING:
        raise LiteralBackparseError("not a literal description")
    return {
        "style": STYLE,
        "habit_law": _parse_assumption(paragraphs[1]),
        "question_law": _parse_question(paragraphs[2]),
    }


# --------------------------------------------------------------------- CLI


def main(argv: Optional[List[str]] = None) -> int:
    cli = argparse.ArgumentParser(
        description="Render an ETP implication 'E implies F' as a literal "
        "natural-language question."
    )
    cli.add_argument("equation_e", help="the assumed law, e.g. 'x ∘ y = (y ∘ y) ∘ x'")
    cli.add_argument("equation_f", help="the questioned law, e.g. 'x ∘ y = y ∘ x'")
    cli.add_argument("--e-label", help="ETP label for E (metadata only), e.g. E387")
    cli.add_argument("--f-label", help="ETP label for F (metadata only), e.g. E43")
    cli.add_argument(
        "--json",
        action="store_true",
        help="print a JSON record pairing the description with its metadata",
    )
    cli.add_argument(
        "--out-dir",
        type=Path,
        metavar="DIR",
        help="write the JSON record to a file in DIR (created if missing)",
    )
    args = cli.parse_args(argv)

    try:
        description, metadata = render_description(args.equation_e, args.equation_f)
    except (ParseError, ValueError) as error:
        cli.error(str(error))
    if args.e_label:
        metadata["label_e"] = args.e_label
    if args.f_label:
        metadata["label_f"] = args.f_label

    if args.out_dir is not None:
        path = write_record(description, metadata, args.out_dir)
        print(path)
    elif args.json:
        import json

        print(json.dumps({"story": description, "metadata": metadata}, ensure_ascii=False))
    else:
        print(description)
    return 0


if __name__ == "__main__":
    sys.exit(main())
