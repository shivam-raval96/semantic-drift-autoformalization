#!/usr/bin/env python3
"""Storyform: deterministic informalization of ETP implications.

Renders the *statement* "law E implies law F" as a themed question-story:
a world whose custom always upholds the first habit (E), ending by asking
whether a second regularity (F) must also hold. The renderer is a pure
function of the (E, F) pair — same input, byte-identical story.

Pipeline: equation string -> AST -> SSA-style steps -> themed question-story.
See CLAUDE.md for the design invariants this module must uphold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

OP_SYMBOLS = ("∘", "◇", "*")


# --------------------------------------------------------------------- AST


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Op:
    left: "Term"
    right: "Term"


Term = Union[Var, Op]


class ParseError(ValueError):
    pass


def tokenize(text: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
        elif c in "()":
            tokens.append(("paren", c))
            i += 1
        elif c == "=":
            tokens.append(("eq", c))
            i += 1
        elif c in OP_SYMBOLS:
            tokens.append(("op", c))
            i += 1
        elif c.isalpha():
            j = i
            while j < len(text) and text[j].isalpha():
                j += 1
            tokens.append(("var", text[i:j]))
            i = j
        else:
            raise ParseError(f"unexpected character {c!r}")
    return tokens


class _Parser:
    """Recursive-descent parser for fully parenthesized ETP terms."""

    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Tuple[str, str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ("end", "")

    def take(self) -> Tuple[str, str]:
        token = self.peek()
        self.pos += 1
        return token

    def parse_operand(self) -> Term:
        kind, value = self.take()
        if kind == "var":
            return Var(value)
        if (kind, value) == ("paren", "("):
            term = self.parse_term()
            kind, value = self.take()
            if (kind, value) != ("paren", ")"):
                raise ParseError("expected ')'")
            return term
        raise ParseError(f"expected a variable or '(', got {value!r}")

    def parse_term(self) -> Term:
        left = self.parse_operand()
        if self.peek()[0] != "op":
            return left
        self.take()
        right = self.parse_operand()
        if self.peek()[0] == "op":
            raise ParseError(
                "ambiguous chain of operations; ETP terms are fully parenthesized"
            )
        return Op(left, right)


def parse_equation(text: str) -> Tuple[Term, Term]:
    parser = _Parser(tokenize(text))
    lhs = parser.parse_term()
    if parser.take()[0] != "eq":
        raise ParseError(f"expected '=' in {text!r}")
    rhs = parser.parse_term()
    if parser.peek()[0] != "end":
        raise ParseError(f"trailing tokens in {text!r}")
    return lhs, rhs


def variables_in_order(*terms: Term) -> List[str]:
    """Variable names by first appearance, walking terms left to right."""
    seen: List[str] = []

    def walk(t: Term) -> None:
        if isinstance(t, Var):
            if t.name not in seen:
                seen.append(t.name)
        else:
            walk(t.left)
            walk(t.right)

    for term in terms:
        walk(term)
    return seen


def canonical(lhs: Term, rhs: Term) -> str:
    """Serialize an equation with variables renamed by first appearance.

    Two equations are the same ETP law (up to variable naming and op
    symbol) iff their canonical forms are equal.
    """
    renaming = {
        name: f"v{i + 1}" for i, name in enumerate(variables_in_order(lhs, rhs))
    }

    def serialize(t: Term) -> str:
        if isinstance(t, Var):
            return renaming[t.name]
        return f"({serialize(t.left)} ∘ {serialize(t.right)})"

    return f"{serialize(lhs)} = {serialize(rhs)}"


# ------------------------------------------------------------------ Themes


@dataclass(frozen=True)
class Theme:
    key: str
    intro: str  # opening sentence naming setting, agent, and the habit
    subject: str  # "she" / "he"
    possessive: str  # "her" / "his"
    element_singular: str
    element_plural: str
    palette: Tuple[str, ...]  # at least 6 names, mapped by first appearance
    op_agent: str  # third-person op clause, e.g. "pours {a} into {b}"
    op_imperative: str  # imperative op clause, e.g. "pour {a} into {b}"
    result_noun: str  # "Batch", "Brew", ...
    same_habit: str  # "come out the exact same color"
    same_question: str  # "come out the same color"
    closing: str  # habit closer, restating the exceptionless custom
    question_intro: str  # sentence introducing the wonderer
    question_lead: str  # "In this workshop"


THEMES: Dict[str, Theme] = {
    theme.key: theme
    for theme in (
        Theme(
            key="paint",
            intro="In a certain paint workshop, the colorist follows one unbreakable habit.",
            subject="she",
            possessive="her",
            element_singular="pigment",
            element_plural="pigments",
            palette=("crimson", "ochre", "teal", "indigo", "saffron", "viridian"),
            op_agent="pours {a} into {b}",
            op_imperative="pour {a} into {b}",
            result_noun="Batch",
            same_habit="come out the exact same color",
            same_question="come out the same color",
            closing="That is simply how this workshop works, without exception.",
            question_intro="One morning her apprentice wonders about something.",
            question_lead="In this workshop",
        ),
        Theme(
            key="tea",
            intro="In a certain mountain teahouse, the tea master follows one unbreakable habit.",
            subject="he",
            possessive="his",
            element_singular="tea",
            element_plural="teas",
            palette=("jasmine", "oolong", "rooibos", "sencha", "chamomile", "darjeeling"),
            op_agent="pours {a} over the leaves of {b}",
            op_imperative="pour {a} over the leaves of {b}",
            result_noun="Brew",
            same_habit="taste exactly the same",
            same_question="taste the same",
            closing="That is simply how this teahouse works, without exception.",
            question_intro="One evening his newest server wonders about something.",
            question_lead="In this teahouse",
        ),
        Theme(
            key="graft",
            intro="In a certain hillside orchard, the gardener follows one unbreakable habit.",
            subject="she",
            possessive="her",
            element_singular="cutting",
            element_plural="cuttings",
            palette=("quince", "medlar", "damson", "mulberry", "persimmon", "loquat"),
            op_agent="grafts {a} onto {b}",
            op_imperative="graft {a} onto {b}",
            result_noun="Graft",
            same_habit="grow into exactly the same plant",
            same_question="grow into the same plant",
            closing="That is simply how this orchard works, without exception.",
            question_intro="One spring her neighbor wonders about something.",
            question_lead="In this orchard",
        ),
        Theme(
            key="signal",
            intro="At a certain mountaintop relay station, the operator follows one unbreakable habit.",
            subject="he",
            possessive="his",
            element_singular="signal",
            element_plural="signals",
            palette=("whistle", "hum", "chirp", "buzz", "drone", "trill"),
            op_agent="feeds {a} through {b}",
            op_imperative="feed {a} through {b}",
            result_noun="Relay",
            same_habit="sound exactly alike",
            same_question="sound exactly alike",
            closing="That is simply how this station works, without exception.",
            question_intro="One night his trainee wonders about something.",
            question_lead="At this station",
        ),
    )
}

THEME_ORDER: Tuple[str, ...] = ("paint", "tea", "graft", "signal")


def select_theme(e_text: str, f_text: str) -> str:
    """Deterministic theme choice: hash of the canonicalized pair."""
    e_canonical = canonical(*parse_equation(e_text))
    f_canonical = canonical(*parse_equation(f_text))
    digest = hashlib.sha256(f"{e_canonical} => {f_canonical}".encode("utf-8")).digest()
    return THEME_ORDER[int.from_bytes(digest[:8], "big") % len(THEME_ORDER)]


# ---------------------------------------------------------------- Renderer

# Naming phrase cycles with the intermediate's number so prose varies
# deterministically; the batch name always follows, keeping the grammar
# injective.
NAMING_AGENT = ("sets the result aside as", "calls the result", "calls that")
NAMING_IMPERATIVE = ("set the result aside as", "call the result", "call that")

NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth")

Step = Tuple[str, str, int]  # (left operand ref, right operand ref, batch number)


def _linearize(term: Term, name_of: Dict[str, str], counter: List[int], theme: Theme) -> Tuple[str, List[Step]]:
    """Post-order walk: each Op node becomes a numbered step.

    Returns the reference naming this term's result ("crimson" or
    "Batch 3") and the steps that compute it. The counter is shared
    across both sides of a law so intermediate names never clash.
    """
    if isinstance(term, Var):
        return name_of[term.name], []
    left_ref, left_steps = _linearize(term.left, name_of, counter, theme)
    right_ref, right_steps = _linearize(term.right, name_of, counter, theme)
    counter[0] += 1
    number = counter[0]
    ref = f"{theme.result_noun} {number}"
    return ref, left_steps + right_steps + [(left_ref, right_ref, number)]


def _law_setup(theme: Theme, lhs: Term, rhs: Term):
    order = variables_in_order(lhs, rhs)
    if len(order) > len(theme.palette):
        raise ValueError(
            f"law uses {len(order)} variables; theme {theme.key!r} has only "
            f"{len(theme.palette)} palette names"
        )
    name_of = {name: theme.palette[i] for i, name in enumerate(order)}
    counter = [0]
    lhs_ref, lhs_steps = _linearize(lhs, name_of, counter, theme)
    rhs_ref, rhs_steps = _linearize(rhs, name_of, counter, theme)
    return order, name_of, lhs_ref, lhs_steps, rhs_ref, rhs_steps


def _take_any(theme: Theme, order: List[str], name_of: Dict[str, str], habit: bool) -> str:
    names = [name_of[v] for v in order]
    at_all = " at all" if habit else ""
    again = "" if habit else "again "
    if len(names) == 1:
        return f"Take any {theme.element_singular}{at_all} — {again}call it {names[0]}."
    count = NUM_WORDS[len(names)]
    parts = [f"the {ORDINALS[i]} {name}" for i, name in enumerate(names)]
    if len(parts) == 2:
        listing = f"{parts[0]} and {parts[1]}"
    else:
        listing = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"Take any {count} {theme.element_plural}{at_all} — {again}call {listing}."


def _however(theme: Theme, variable_count: int) -> str:
    if variable_count == 1:
        chosen = f"starting {theme.element_singular}"
    else:
        chosen = f"{NUM_WORDS[variable_count]} starting {theme.element_plural}"
    return f"However {theme.subject} chooses {theme.possessive} {chosen}"


def _step_text(theme: Theme, step: Step, imperative: bool) -> str:
    left_ref, right_ref, number = step
    op = theme.op_imperative if imperative else theme.op_agent
    naming = NAMING_IMPERATIVE if imperative else NAMING_AGENT
    clause = op.format(a=left_ref, b=right_ref)
    return f"{clause} and {naming[(number - 1) % 3]} {theme.result_noun} {number}"


def _agent_procedure(theme: Theme, steps: List[Step]) -> str:
    texts = [_step_text(theme, step, imperative=False) for step in steps]
    return f"; then {theme.subject} ".join(texts)


def _imperative_procedure(theme: Theme, steps: List[Step]) -> str:
    texts = [_step_text(theme, step, imperative=True) for step in steps]
    return "; then ".join(texts)


def _capitalize(sentence: str) -> str:
    return sentence[0].upper() + sentence[1:]


def _compare_ref(term: Term, ref: str) -> str:
    # A bare-variable side has no steps; its "result" is the element itself.
    if isinstance(term, Var):
        return f"{ref} itself"
    return ref


def render_habit(theme: Theme, lhs: Term, rhs: Term) -> List[str]:
    order, name_of, lhs_ref, lhs_steps, rhs_ref, rhs_steps = _law_setup(theme, lhs, rhs)
    opening = f"{theme.intro} {_take_any(theme, order, name_of, habit=True)}"
    paragraphs: List[str] = []
    if lhs_steps and rhs_steps:
        opening += f" {_capitalize(theme.subject)} runs two procedures side by side."
        paragraphs.append(opening)
        paragraphs.append(
            f"In the first, {theme.subject} {_agent_procedure(theme, lhs_steps)}."
        )
        paragraphs.append(
            f"In the second, {theme.subject} {_agent_procedure(theme, rhs_steps)}."
        )
    elif lhs_steps or rhs_steps:
        paragraphs.append(opening)
        steps = lhs_steps or rhs_steps
        paragraphs.append(
            f"{_capitalize(theme.subject)} {_agent_procedure(theme, steps)}."
        )
    else:
        paragraphs.append(opening)
    left = _compare_ref(lhs, lhs_ref)
    right = _compare_ref(rhs, rhs_ref)
    paragraphs.append(
        f"{_however(theme, len(order))}, {left} and {right} always "
        f"{theme.same_habit}. {theme.closing}"
    )
    return paragraphs


def render_question(theme: Theme, lhs: Term, rhs: Term) -> List[str]:
    order, name_of, lhs_ref, lhs_steps, rhs_ref, rhs_steps = _law_setup(theme, lhs, rhs)
    sentences = [theme.question_intro, _take_any(theme, order, name_of, habit=False)]
    if lhs_steps:
        sentences.append(_capitalize(_imperative_procedure(theme, lhs_steps)) + ".")
    if rhs_steps:
        procedure = _imperative_procedure(theme, rhs_steps)
        if lhs_steps:
            sentences.append(f"Separately, {procedure}.")
        else:
            sentences.append(_capitalize(procedure) + ".")
    left = _compare_ref(lhs, lhs_ref)
    right = _compare_ref(rhs, rhs_ref)
    question = (
        f"{theme.question_lead}, must {left} and {right} always "
        f"{theme.same_question}?"
    )
    return [" ".join(sentences), question]


def render_story(e_text: str, f_text: str, theme_key: Optional[str] = None) -> Tuple[str, dict]:
    """Render the implication "E implies F" as a question-story.

    Returns (story, metadata). The formal side of the pair lives only in
    the metadata, never in the story text.
    """
    e_lhs, e_rhs = parse_equation(e_text)
    f_lhs, f_rhs = parse_equation(f_text)
    if theme_key is None:
        theme_key = select_theme(e_text, f_text)
    if theme_key not in THEMES:
        raise ValueError(f"unknown theme {theme_key!r}; choose from {sorted(THEMES)}")
    theme = THEMES[theme_key]

    paragraphs = render_habit(theme, e_lhs, e_rhs) + render_question(theme, f_lhs, f_rhs)
    story = "\n\n".join(paragraphs)

    e_order = variables_in_order(e_lhs, e_rhs)
    f_order = variables_in_order(f_lhs, f_rhs)
    metadata = {
        "theme": theme_key,
        "equation_e": e_text,
        "equation_f": f_text,
        "canonical_e": canonical(e_lhs, e_rhs),
        "canonical_f": canonical(f_lhs, f_rhs),
        "palette_e": {v: theme.palette[i] for i, v in enumerate(e_order)},
        "palette_f": {v: theme.palette[i] for i, v in enumerate(f_order)},
    }
    return story, metadata


# ------------------------------------------------------------------ Export


def _sanitize(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", label)


def record_filename(metadata: dict) -> str:
    """Deterministic filename for a corpus record.

    Uses ETP labels when both are present, otherwise a hash of the
    canonicalized pair; the theme key is included so the same pair
    rendered under different themes never collides.
    """
    label_e = metadata.get("label_e")
    label_f = metadata.get("label_f")
    if label_e and label_f:
        stem = f"{_sanitize(label_e)}-{_sanitize(label_f)}"
    else:
        digest = hashlib.sha256(
            f"{metadata['canonical_e']} => {metadata['canonical_f']}".encode("utf-8")
        ).hexdigest()[:12]
        stem = f"pair-{digest}"
    return f"{stem}-{metadata['theme']}.json"


def write_record(story: str, metadata: dict, out_dir: Path) -> Path:
    """Write one (story, metadata) record as a JSON file; return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / record_filename(metadata)
    record = {"story": story, "metadata": metadata}
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------- CLI


def main(argv: Optional[List[str]] = None) -> int:
    cli = argparse.ArgumentParser(
        description="Render an ETP implication 'E implies F' as a themed question-story."
    )
    cli.add_argument("equation_e", help="the assumed law, e.g. 'x ∘ y = (y ∘ y) ∘ x'")
    cli.add_argument("equation_f", help="the questioned law, e.g. 'x ∘ y = y ∘ x'")
    cli.add_argument(
        "--theme",
        choices=sorted(THEMES),
        help="theme override; default is a deterministic hash of the pair",
    )
    cli.add_argument("--e-label", help="ETP label for E (metadata only), e.g. E387")
    cli.add_argument("--f-label", help="ETP label for F (metadata only), e.g. E43")
    cli.add_argument(
        "--json",
        action="store_true",
        help="print a JSON record pairing the story with its formal metadata",
    )
    cli.add_argument(
        "--out-dir",
        type=Path,
        metavar="DIR",
        help="write the JSON record to a file in DIR (created if missing); "
        "the filename is deterministic, from the ETP labels or a hash of the pair",
    )
    args = cli.parse_args(argv)

    try:
        story, metadata = render_story(args.equation_e, args.equation_f, args.theme)
    except (ParseError, ValueError) as error:
        cli.error(str(error))
    if args.e_label:
        metadata["label_e"] = args.e_label
    if args.f_label:
        metadata["label_f"] = args.f_label

    if args.out_dir is not None:
        path = write_record(story, metadata, args.out_dir)
        print(path)
    elif args.json:
        print(json.dumps({"story": story, "metadata": metadata}, ensure_ascii=False))
    else:
        print(story)
    return 0


if __name__ == "__main__":
    sys.exit(main())
