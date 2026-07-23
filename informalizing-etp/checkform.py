#!/usr/bin/env python3
"""Checkform: grade a model's formalization of a question-story.

formalize_prompt.md teaches a model a tiny self-contained prefix syntax
(no Lean, no ETP conventions assumed) and asks for two lines:

    ASSUME: op(x, y) = op(op(y, y), x)
    ASK: op(x, y) = op(y, x)

Grading is purely syntactic: parse the two equations, canonicalize, and
compare against the record's ground truth (metadata's canonical_e /
canonical_f). Accepted symmetries, all faithful readings of the story:
variable renaming (always), swapping the sides of either equation (the
story's "always come out the same" is symmetric), and dualization
(mirroring every op's argument order uniformly across BOTH equations —
the story never says which participant is the first argument). The
direction ASSUME -> ASK is never lenient.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from storyform import Op, Term, Var, canonical, parse_equation

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "formalize_prompt.md"


class AnswerParseError(ValueError):
    pass


# ------------------------------------------------------------- Extraction

# An answer line: optional quote/list/bold decoration, the label, a colon,
# then the equation. The last occurrence of each label wins, so a model
# that restates or first echoes the worked example is still graded on its
# final answer.
_LINE_RE = re.compile(
    r"^[ \t>*+-]*(?:\*\*)?\s*(?P<label>ASSUME|ASK)\b\s*(?:\*\*)?\s*:\s*(?P<value>.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _clean(value: str) -> str:
    value = value.strip().strip("`*")
    return value.rstrip(".").strip()


def extract_answer(response_text: str) -> Tuple[str, str]:
    """Pull the last ASSUME/ASK equation texts out of a raw response."""
    found: Dict[str, str] = {}
    for match in _LINE_RE.finditer(response_text):
        found[match.group("label").upper()] = _clean(match.group("value"))
    for label in ("ASSUME", "ASK"):
        if label not in found:
            raise AnswerParseError(f"no '{label}:' line found in the response")
    return found["ASSUME"], found["ASK"]


# ----------------------------------------------------------- Prefix parser


def _tokenize(text: str) -> List[str]:
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
            raise AnswerParseError(f"unexpected character {c!r}")
    return tokens


# ETP laws nest at most 4 operations per side; anything remotely close to
# this bound is a runaway generation, and rejecting it keeps the recursive
# parser (and dual/canonical below) clear of Python's recursion limit.
_MAX_TERM_DEPTH = 50


class _PrefixParser:
    """term := variable | 'op' '(' term ',' term ')'  (op is case-insensitive)."""

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
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
            raise AnswerParseError(f"expected {token!r}, got {got!r}")

    def parse_term(self, depth: int = 0) -> Term:
        if depth > _MAX_TERM_DEPTH:
            raise AnswerParseError(f"term nested deeper than {_MAX_TERM_DEPTH}")
        token = self.take()
        if token is None or token in "(),=":
            raise AnswerParseError(f"expected a variable or 'op(', got {token!r}")
        if token.lower() == "op":
            self.expect("(")
            left = self.parse_term(depth + 1)
            self.expect(",")
            right = self.parse_term(depth + 1)
            self.expect(")")
            return Op(left, right)
        return Var(token)


def parse_prefix_equation(text: str) -> Tuple[Term, Term]:
    parser = _PrefixParser(_tokenize(text))
    lhs = parser.parse_term()
    parser.expect("=")
    rhs = parser.parse_term()
    if parser.peek() is not None:
        raise AnswerParseError(f"trailing tokens in {text!r}")
    return lhs, rhs


# ---------------------------------------------------------------- Grading


def dual(term: Term) -> Term:
    """Mirror every op's argument order (the opposite-magma reading)."""
    if isinstance(term, Var):
        return term
    return Op(dual(term.right), dual(term.left))


# Transform combinations in preference order: fewest flags first, so a
# symmetric equation that matches several ways reports the minimal one.
_TRANSFORMS = sorted(
    itertools.product((False, True), repeat=3), key=lambda flags: (sum(flags), flags)
)


def _transformed(
    e: Tuple[Term, Term], f: Tuple[Term, Term],
    swap_e: bool, swap_f: bool, dualize: bool,
) -> Tuple[Tuple[Term, Term], Tuple[Term, Term]]:
    if dualize:
        e = (dual(e[0]), dual(e[1]))
        f = (dual(f[0]), dual(f[1]))
    if swap_e:
        e = (e[1], e[0])
    if swap_f:
        f = (f[1], f[0])
    return e, f


def answer_class_key(equation_e: str, equation_f: str) -> Tuple[str, str]:
    """Grading-equivalence class of an (ASSUME, ASK) pair.

    Returns the lexicographically smallest (canonical_e, canonical_f)
    over the eight accepted transforms. Two answers grade identically
    against every record iff their keys are equal, and an answer is
    correct for a record iff its key equals the key of the record's
    (canonical_e, canonical_f).
    """
    e = parse_equation(equation_e)
    f = parse_equation(equation_f)
    return min(
        (canonical(*te), canonical(*tf))
        for swap_e, swap_f, dualize in _TRANSFORMS
        for te, tf in [_transformed(e, f, swap_e, swap_f, dualize)]
    )


def grade(response_text: str, metadata: dict) -> dict:
    """Grade a raw model response against a corpus record's metadata.

    Returns a verdict dict with a stable schema: status is "correct",
    "wrong", or "unparseable"; transform is the minimal accepted
    transformation of the answer that matched, or None.
    """
    verdict = {
        "status": "unparseable",
        "transform": None,
        "canonical_answer_e": None,
        "canonical_answer_f": None,
        "error": None,
    }
    try:
        assume_text, ask_text = extract_answer(response_text)
        e_lhs, e_rhs = parse_prefix_equation(assume_text)
        f_lhs, f_rhs = parse_prefix_equation(ask_text)
    except AnswerParseError as error:
        verdict["error"] = str(error)
        return verdict

    verdict["canonical_answer_e"] = canonical(e_lhs, e_rhs)
    verdict["canonical_answer_f"] = canonical(f_lhs, f_rhs)
    truth = (metadata["canonical_e"], metadata["canonical_f"])

    verdict["status"] = "wrong"
    for swap_e, swap_f, dualize in _TRANSFORMS:
        e, f = _transformed((e_lhs, e_rhs), (f_lhs, f_rhs), swap_e, swap_f, dualize)
        if (canonical(*e), canonical(*f)) == truth:
            verdict["status"] = "correct"
            verdict["transform"] = {
                "swap_e": swap_e,
                "swap_f": swap_f,
                "dual": dualize,
            }
            break
    return verdict


# ------------------------------------------------------------------ Prompt


def build_prompt(record: dict, template_path: Path = PROMPT_PATH) -> str:
    """Fill formalize_prompt.md with a corpus record's story."""
    template = Path(template_path).read_text(encoding="utf-8")
    return template.replace("{story}", record["story"])


# --------------------------------------------------------------------- CLI


def _load_record(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    cli = argparse.ArgumentParser(
        description="Build formalization prompts and grade model answers "
        "against corpus records."
    )
    commands = cli.add_subparsers(dest="command", required=True)

    prompt_cmd = commands.add_parser(
        "prompt", help="print the filled prompt for a corpus record"
    )
    prompt_cmd.add_argument("record", type=Path, help="corpus record JSON file")

    grade_cmd = commands.add_parser(
        "grade",
        help="grade a raw model response against a corpus record; "
        "exit 0 correct, 1 wrong, 2 unparseable",
    )
    grade_cmd.add_argument("record", type=Path, help="corpus record JSON file")
    grade_cmd.add_argument(
        "response", type=Path, help="file holding the raw model response"
    )

    args = cli.parse_args(argv)
    record = _load_record(args.record)

    if args.command == "prompt":
        print(build_prompt(record))
        return 0

    response_text = args.response.read_text(encoding="utf-8")
    verdict = grade(response_text, record["metadata"])
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return {"correct": 0, "wrong": 1, "unparseable": 2}[verdict["status"]]


if __name__ == "__main__":
    sys.exit(main())
