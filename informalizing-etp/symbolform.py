#!/usr/bin/env python3
"""Symbolform: rigid-grammar rendering of ETP implications.

The third presentation arm, alongside storyform's themed narrative and
literalform's plain-English description: the implication "law E implies
law F" rendered directly in the same tiny prefix grammar that
checkform.py parses — two lines, nothing else:

    ASSUME: op(x, y) = op(op(y, y), x)
    ASK: op(x, y) = op(y, x)

Variables are renamed to fixed letters by order of first appearance
(LHS first, then RHS, per law — literalform's convention), so neither
the ETP's own symbols nor its variable spellings survive into the text.
What the grammar means (universal quantification, ordered inputs) is
explained by the arm's prompt template, not here; the renderer stays a
pure function of the (E, F) pair, and each line round-trips through
checkform.parse_prefix_equation by construction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from literalform import VARIABLE_LETTERS
from storyform import (
    ParseError,
    Term,
    Var,
    canonical,
    parse_equation,
    variables_in_order,
    write_record,
)

STYLE = "symbolic"

ASSUME_LABEL = "ASSUME"
ASK_LABEL = "ASK"


def _rename(lhs: Term, rhs: Term) -> Dict[str, str]:
    order = variables_in_order(lhs, rhs)
    if len(order) > len(VARIABLE_LETTERS):
        raise ValueError(
            f"law uses {len(order)} variables; at most "
            f"{len(VARIABLE_LETTERS)} are supported"
        )
    return {name: VARIABLE_LETTERS[i] for i, name in enumerate(order)}


def to_prefix(term: Term, renaming: Dict[str, str]) -> str:
    """Serialize a term in checkform's prefix grammar."""
    if isinstance(term, Var):
        return renaming[term.name]
    return f"op({to_prefix(term.left, renaming)}, {to_prefix(term.right, renaming)})"


def _render_law(lhs: Term, rhs: Term, renaming: Dict[str, str]) -> str:
    return f"{to_prefix(lhs, renaming)} = {to_prefix(rhs, renaming)}"


def render_symbolic(e_text: str, f_text: str) -> Tuple[str, dict]:
    """Render the implication "E implies F" as the two prefix lines.

    Returns (text, metadata) with the same record schema as the other
    arms, so checkform-style tooling and the benchmark row format work
    unchanged; the "theme" field holds the style key.
    """
    e_lhs, e_rhs = parse_equation(e_text)
    f_lhs, f_rhs = parse_equation(f_text)
    e_renaming = _rename(e_lhs, e_rhs)
    f_renaming = _rename(f_lhs, f_rhs)
    text = "\n".join(
        (
            f"{ASSUME_LABEL}: {_render_law(e_lhs, e_rhs, e_renaming)}",
            f"{ASK_LABEL}: {_render_law(f_lhs, f_rhs, f_renaming)}",
        )
    )
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
    return text, metadata


# --------------------------------------------------------------------- CLI


def main(argv: Optional[List[str]] = None) -> int:
    cli = argparse.ArgumentParser(
        description="Render an ETP implication 'E implies F' in the rigid "
        "two-line prefix grammar."
    )
    cli.add_argument("equation_e", help="the assumed law, e.g. 'x ∘ y = (y ∘ y) ∘ x'")
    cli.add_argument("equation_f", help="the questioned law, e.g. 'x ∘ y = y ∘ x'")
    cli.add_argument("--e-label", help="ETP label for E (metadata only), e.g. E387")
    cli.add_argument("--f-label", help="ETP label for F (metadata only), e.g. E43")
    cli.add_argument(
        "--json",
        action="store_true",
        help="print a JSON record pairing the text with its metadata",
    )
    cli.add_argument(
        "--out-dir",
        type=Path,
        metavar="DIR",
        help="write the JSON record to a file in DIR (created if missing)",
    )
    args = cli.parse_args(argv)

    try:
        text, metadata = render_symbolic(args.equation_e, args.equation_f)
    except (ParseError, ValueError) as error:
        cli.error(str(error))
    if args.e_label:
        metadata["label_e"] = args.e_label
    if args.f_label:
        metadata["label_f"] = args.f_label

    if args.out_dir is not None:
        path = write_record(text, metadata, args.out_dir)
        print(path)
    elif args.json:
        import json

        print(json.dumps({"story": text, "metadata": metadata}, ensure_ascii=False))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
