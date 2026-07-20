#!/usr/bin/env python3
"""Literalform: direct natural-language rendering of ETP implications.

The contrasting arm to storyform: instead of a fuzzy themed story, the
implication "law E implies law F" becomes a plain-English description that
openly talks about an operation, its two ordered inputs, and named
variables. No setting, no agent, no palette — a literal translation.

Terms render in a words-only prefix grammar, so nesting is unambiguous
without parentheses: each "the result of applying the operation to ..."
opens a node that consumes its own "as its first input and ... as its
second input". Like storyform, the renderer is a pure function of the
(E, F) pair, and the description alone determines both term trees.
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

OPENING = (
    "Consider a collection of objects together with an operation that "
    "combines two objects into one. The operation takes a first input and "
    "a second input, and the order of the inputs matters."
)


# -------------------------------------------------------------- Rendering


def _rename(lhs: Term, rhs: Term) -> Dict[str, str]:
    order = variables_in_order(lhs, rhs)
    if len(order) > len(VARIABLE_LETTERS):
        raise ValueError(
            f"law uses {len(order)} variables; at most "
            f"{len(VARIABLE_LETTERS)} are supported"
        )
    return {name: VARIABLE_LETTERS[i] for i, name in enumerate(order)}


def render_term(term: Term, name_of: Dict[str, str]) -> str:
    """Words-only prefix rendering of one term (injective by construction)."""
    if isinstance(term, Var):
        return name_of[term.name]
    left = render_term(term.left, name_of)
    right = render_term(term.right, name_of)
    return (
        f"the result of applying the operation to {left} as its first "
        f"input and {right} as its second input"
    )


def _quantifier(letters: List[str]) -> str:
    if len(letters) == 1:
        return f"for every choice of an object {letters[0]}"
    if len(letters) == 2:
        listing = f"{letters[0]} and {letters[1]}"
    else:
        listing = ", ".join(letters[:-1]) + f", and {letters[-1]}"
    return f"for every choice of objects {listing}"


def render_law(lhs: Term, rhs: Term) -> str:
    """One law as a quantified equality clause (no trailing punctuation)."""
    name_of = _rename(lhs, rhs)
    letters = [name_of[v] for v in variables_in_order(lhs, rhs)]
    left = render_term(lhs, name_of)
    right = render_term(rhs, name_of)
    return f"{_quantifier(letters)}, {left} is always equal to {right}"


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
        f"Suppose the following always holds: {render_law(e_lhs, e_rhs)}.",
        f"Does it follow that {render_law(f_lhs, f_rhs)}?",
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


_OP_PHRASE = "the result of applying the operation to "
_FIRST_SEP = " as its first input and "
_SECOND_SEP = " as its second input"


class _PhraseReader:
    """Recursive-descent reader for the words-only prefix grammar."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def _expect(self, literal: str) -> None:
        if not self.text.startswith(literal, self.pos):
            context = self.text[self.pos : self.pos + 40]
            raise LiteralBackparseError(
                f"expected {literal!r} at {context!r}"
            )
        self.pos += len(literal)

    def read_term(self) -> Term:
        if self.text.startswith(_OP_PHRASE, self.pos):
            self.pos += len(_OP_PHRASE)
            left = self.read_term()
            self._expect(_FIRST_SEP)
            right = self.read_term()
            self._expect(_SECOND_SEP)
            return Op(left, right)
        for letter in VARIABLE_LETTERS:
            if self.text.startswith(letter, self.pos):
                end = self.pos + len(letter)
                if end == len(self.text) or not self.text[end].isalpha():
                    self.pos = end
                    return Var(letter)
        context = self.text[self.pos : self.pos + 40]
        raise LiteralBackparseError(f"expected a term at {context!r}")


_LETTER = f"(?:{'|'.join(VARIABLE_LETTERS)})"
# Matches _quantifier's output: "an object x", "objects x and y", or the
# Oxford-comma listing "objects x, y, and z".
_QUANTIFIER_RE = re.compile(
    rf"^for every choice of (?:an object {_LETTER}"
    rf"|objects {_LETTER}(?:, {_LETTER})*,? and {_LETTER}), "
)


def _parse_law_clause(clause: str) -> Tuple[Term, Term]:
    match = _QUANTIFIER_RE.match(clause)
    if not match:
        raise LiteralBackparseError(f"no quantifier prefix in {clause!r}")
    reader = _PhraseReader(clause[match.end() :])
    lhs = reader.read_term()
    reader._expect(" is always equal to ")
    rhs = reader.read_term()
    if reader.pos != len(reader.text):
        raise LiteralBackparseError(
            f"trailing text {reader.text[reader.pos:]!r}"
        )
    return lhs, rhs


def backparse_literal(description: str) -> dict:
    """Recover both laws from a literal description's text alone."""
    paragraphs = description.split("\n\n")
    if len(paragraphs) != 3 or paragraphs[0] != OPENING:
        raise LiteralBackparseError("not a literal description")
    suppose = paragraphs[1]
    question = paragraphs[2]
    prefix_e = "Suppose the following always holds: "
    prefix_f = "Does it follow that "
    if not suppose.startswith(prefix_e) or not suppose.endswith("."):
        raise LiteralBackparseError("malformed 'Suppose' paragraph")
    if not question.startswith(prefix_f) or not question.endswith("?"):
        raise LiteralBackparseError("malformed question paragraph")
    return {
        "style": STYLE,
        "habit_law": _parse_law_clause(suppose[len(prefix_e) : -1]),
        "question_law": _parse_law_clause(question[len(prefix_f) : -1]),
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
