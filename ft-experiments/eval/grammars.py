#!/usr/bin/env python3
"""Grammar registry for FT v2: three surface syntaxes over the repo's
equation ASTs, plus the transcode grading seam.

- "a"      trained RG:       ASSUME:/ASK:,  prefix op(a, b)
- "b_near" vocabulary swap:  GIVEN:/SHOW:,  prefix f(a, b)
- "b_far"  restructure:      LAW:/DERIVE:,  fully parenthesized infix (l ∘ r)

All three serialize the SAME ASTs, variables renamed x, y, z, w, u, v by
first appearance per equation independently (ftlib.rg_equation's
convention). Grading transcodes B answers back to grammar-A text and calls
checkform.grade unchanged, so every surface grades through the one
battle-tested canonicalization path (DESIGN.md "Implementation notes").
"""

from __future__ import annotations

import functools
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

STAGE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ftlib inserts the repo into sys.path, making storyform/checkform
# importable below; loading it by path keeps this module standalone.
ftlib = _load("v2_ftlib", STAGE.parents[0] / "data-gen" / "ftlib.py")

import checkform  # noqa: E402
import storyform  # noqa: E402
from storyform import Op, Term, Var  # noqa: E402

Equation = Tuple[Term, Term]

RG_LETTERS = ftlib.RG_LETTERS

# Mirrors checkform._MAX_TERM_DEPTH: anything near this bound is a runaway
# generation, and rejecting it keeps the recursive parsers (and the
# dual/canonical passes downstream) clear of Python's recursion limit.
_MAX_TERM_DEPTH = 50


class BAnswerParseError(ValueError):
    """A response holds no well-formed answer in the requested grammar."""


# ------------------------------------------------------------- Extraction

# Mirrors checkform._LINE_RE with the grammar's labels: optional
# quote/list/bold decoration, the label, a colon, then the equation (the
# \s* after the colon may cross the newline, so a value on the following
# line is still captured). The last occurrence of each label wins.


@functools.lru_cache(maxsize=None)
def _line_pattern(labels: Tuple[str, str]) -> re.Pattern:
    alt = "|".join(re.escape(label) for label in labels)
    return re.compile(
        rf"^[ \t>*+-]*(?:\*\*)?\s*(?P<label>{alt})\b\s*(?:\*\*)?\s*:\s*(?P<value>.+)$",
        re.IGNORECASE | re.MULTILINE,
    )


def _clean(value: str) -> str:
    """Byte-identical to checkform._clean, on purpose: the A-vs-B unparseable
    comparison is only fair if both graders repair the same cosmetic damage
    (trailing sentence periods, backtick/bold wrappers). Safe here because no
    valid B equation begins or ends with any stripped character: b_near ends
    in a letter/digit or ')', b_far in a letter/digit or ')', and neither may
    start with '`', '*', or '.'."""
    value = value.strip().strip("`*")
    return value.rstrip(".").strip()


def _extract(response_text: str, labels: Tuple[str, str]) -> Tuple[str, str]:
    """Pull the last labeled equation texts out of a raw response."""
    found = {}
    for match in _line_pattern(labels).finditer(response_text):
        found[match.group("label").upper()] = _clean(match.group("value"))
    for label in labels:
        if label not in found:
            raise BAnswerParseError(f"no '{label}:' line found in the response")
    return found[labels[0]], found[labels[1]]


# ------------------------------------------- Prefix parser (b_near, op 'f')


def _tokenize_prefix(text: str) -> List[str]:
    """checkform._tokenize's rules: '(),=' punctuation; a name is a letter
    then letters/digits/underscores."""
    tokens: List[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
        elif c in "(),=":
            tokens.append(c)
            i += 1
        elif c.isalpha():
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            raise BAnswerParseError(f"unexpected character {c!r}")
    return tokens


class _PrefixParser:
    """term := variable | OP '(' term ',' term ')' — checkform._PrefixParser
    with the op token a parameter (case-insensitive, reserved)."""

    def __init__(self, tokens: List[str], op_token: str):
        self.tokens = tokens
        self.op_token = op_token.lower()
        self.pos = 0

    def peek(self) -> Optional[str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def take(self) -> Optional[str]:
        token = self.peek()
        self.pos += 1
        return token

    def expect(self, token: str) -> None:
        got = self.take()
        if got != token:
            raise BAnswerParseError(f"expected {token!r}, got {got!r}")

    def parse_term(self, depth: int = 0) -> Term:
        if depth > _MAX_TERM_DEPTH:
            raise BAnswerParseError(f"term nested deeper than {_MAX_TERM_DEPTH}")
        token = self.take()
        if token is None or token in "(),=":
            raise BAnswerParseError(
                f"expected a variable or {self.op_token!r} + '(', got {token!r}"
            )
        if token.lower() == self.op_token:
            self.expect("(")
            left = self.parse_term(depth + 1)
            self.expect(",")
            right = self.parse_term(depth + 1)
            self.expect(")")
            return Op(left, right)
        return Var(token)


def parse_prefix_equation(text: str, op_token: str = "f") -> Equation:
    parser = _PrefixParser(_tokenize_prefix(text), op_token)
    lhs = parser.parse_term()
    parser.expect("=")
    rhs = parser.parse_term()
    if parser.peek() is not None:
        raise BAnswerParseError(f"trailing tokens in {text!r}")
    return lhs, rhs


# ------------------------------------------------- Infix parser (b_far)


def _parse_infix_equation(text: str) -> Equation:
    """storyform.parse_equation, with failures (including runaway nesting,
    which its uncapped recursive descent turns into RecursionError) mapped
    to BAnswerParseError; depth capped post-parse like the prefix side."""
    try:
        lhs, rhs = storyform.parse_equation(text)
    except (storyform.ParseError, RecursionError) as error:
        raise BAnswerParseError(str(error) or "runaway nesting") from error
    if max(ftlib.term_depth(lhs), ftlib.term_depth(rhs)) > _MAX_TERM_DEPTH:
        raise BAnswerParseError(f"term nested deeper than {_MAX_TERM_DEPTH}")
    return lhs, rhs


# ------------------------------------------------------------ Serializers


def _renaming(lhs: Term, rhs: Term) -> dict:
    order = storyform.variables_in_order(lhs, rhs)
    if len(order) > len(RG_LETTERS):
        raise ValueError(f"equation uses {len(order)} variables; max {len(RG_LETTERS)}")
    return {name: RG_LETTERS[i] for i, name in enumerate(order)}


def prefix_equation(lhs: Term, rhs: Term, op_token: str = "f") -> str:
    """One equation as prefix text with the given op token, variables
    renamed x, y, z, w, u, v by first appearance (per equation, matching
    ftlib.rg_equation)."""
    renaming = _renaming(lhs, rhs)

    def go(t: Term) -> str:
        if isinstance(t, Var):
            return renaming[t.name]
        return f"{op_token}({go(t.left)}, {go(t.right)})"

    return f"{go(lhs)} = {go(rhs)}"


def infix_equation(lhs: Term, rhs: Term) -> str:
    """One equation as fully parenthesized infix ∘ text — the shape
    ftlib.serialize_infix emits and storyform.parse_equation accepts —
    variables renamed as in prefix_equation."""
    renaming = _renaming(lhs, rhs)

    def go(t: Term) -> str:
        if isinstance(t, Var):
            return renaming[t.name]
        return f"({go(t.left)} ∘ {go(t.right)})"

    return f"{go(lhs)} = {go(rhs)}"


# --------------------------------------------------------------- Registry


@dataclass(frozen=True)
class Grammar:
    """One surface syntax: labels for the two answer lines, a reference
    serializer AST-pair -> labeled block, and a parser raw response ->
    AST pairs (raises BAnswerParseError on any failure)."""

    name: str
    labels: Tuple[str, str]
    serialize_pair: Callable[[Equation, Equation], str]
    parse_answer: Callable[[str], Tuple[Equation, Equation]]


def _serialize_a(e: Equation, f: Equation) -> str:
    return ftlib.rg_text(e, f)


def _parse_answer_a(text: str) -> Tuple[Equation, Equation]:
    try:
        assume_text, ask_text = checkform.extract_answer(text)
        return (
            checkform.parse_prefix_equation(assume_text),
            checkform.parse_prefix_equation(ask_text),
        )
    except checkform.AnswerParseError as error:
        raise BAnswerParseError(str(error)) from error


def _serialize_b_near(e: Equation, f: Equation) -> str:
    return f"GIVEN: {prefix_equation(*e)}\nSHOW: {prefix_equation(*f)}"


def _parse_answer_b_near(text: str) -> Tuple[Equation, Equation]:
    given_text, show_text = _extract(text, ("GIVEN", "SHOW"))
    return parse_prefix_equation(given_text), parse_prefix_equation(show_text)


def _serialize_b_far(e: Equation, f: Equation) -> str:
    return f"LAW: {infix_equation(*e)}\nDERIVE: {infix_equation(*f)}"


def _parse_answer_b_far(text: str) -> Tuple[Equation, Equation]:
    law_text, derive_text = _extract(text, ("LAW", "DERIVE"))
    return _parse_infix_equation(law_text), _parse_infix_equation(derive_text)


GRAMMARS = {
    "a": Grammar("a", ("ASSUME", "ASK"), _serialize_a, _parse_answer_a),
    "b_near": Grammar("b_near", ("GIVEN", "SHOW"), _serialize_b_near, _parse_answer_b_near),
    "b_far": Grammar("b_far", ("LAW", "DERIVE"), _serialize_b_far, _parse_answer_b_far),
}


# ------------------------------------------------- Opaque label layer (stage 1)

# Stage 1 teaches a model to associate an opaque label with each grammar's
# surface form. The label is deliberately meaningless so the association is
# learned rather than read off a descriptive name; the structural key and
# family stay code-side (never shown to the model). This is a thin view over
# GRAMMARS — the frozen v2 keys are untouched.
RG_LABELS = {"RG-1": "a", "RG-2": "b_near", "RG-3": "b_far"}
GRAMMAR_TO_LABEL = {key: label for label, key in RG_LABELS.items()}
GRAMMAR_FAMILY = {"a": "prefix", "b_near": "prefix", "b_far": "infix"}


# ----------------------------------------------------------------- Grading


def grade_b(response_text: str, grammar: str, canonical_e: str, canonical_f: str) -> dict:
    """Grade a response written in any registered grammar.

    The transcode seam: parse with the grammar's own parser, re-serialize
    the ASTs as grammar-A text, and hand that to checkform.grade
    unchanged — same verdict schema, same accepted symmetries. Parse
    failures (and answers rg_text cannot re-serialize, i.e. >6 variables
    in one equation) are this grammar's "unparseable".
    """
    g = GRAMMARS[grammar]
    try:
        e_pair, f_pair = g.parse_answer(response_text)
        rg = ftlib.rg_text(e_pair, f_pair)
    except ValueError as error:  # BAnswerParseError or rg_text's variable cap
        return {
            "status": "unparseable",
            "transform": None,
            "canonical_answer_e": None,
            "canonical_answer_f": None,
            "error": str(error),
        }
    return checkform.grade(rg, {"canonical_e": canonical_e, "canonical_f": canonical_f})
