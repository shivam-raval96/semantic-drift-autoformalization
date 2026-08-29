#!/usr/bin/env python3
"""Shared pieces for stage 1: opaque-label recognition over the three RGs.

Stage 1 fine-tunes a model to RECOGNIZE grammars by an opaque label
(RG-1/2/3), never to produce one. Two task types:

  identify  statement -> its label      (completion RG-1 / RG-2 / RG-3)
  validate  statement + label -> yes/no (completion Yes / No)

The grammar surfaces and parsers come straight from eval/grammars.py (no
new parser to trust); this module only adds the label view, the negative
constructors for validation, the frozen prompt wording, and the two answer
extractors the eval grades with. build_stage1, verify_stage1 and the eval
all import from here so the prompt/label conventions can never drift.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Tuple

STAGE = Path(__file__).resolve().parent            # data-gen/
ROOT = STAGE.parent                                # ft-experiments/


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: grammars.py defines a dataclass, whose field
    # resolution looks the module up in sys.modules (Python 3.8+ pitfall).
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# eval/grammars.py is standalone (it loads ftlib + the repo grader by path),
# so a path import works from any stage without touching sys.path.
grammars = _load("stage1_grammars", ROOT / "eval" / "grammars.py")

GRAMMARS = grammars.GRAMMARS
RG_LABELS = grammars.RG_LABELS                     # RG-N -> grammar key
GRAMMAR_TO_LABEL = grammars.GRAMMAR_TO_LABEL       # grammar key -> RG-N
GRAMMAR_FAMILY = grammars.GRAMMAR_FAMILY
BAnswerParseError = grammars.BAnswerParseError
checkform = grammars.checkform

GRAMMAR_KEYS = ("a", "b_near", "b_far")
LABELS_BY_KEY = {key: g.labels for key, g in GRAMMARS.items()}   # key -> (l0, l1)

Equation = Tuple[grammars.Term, grammars.Term]


# ------------------------------------------------------- equation source


def pair_from_completion(completion: str) -> Tuple[Equation, Equation]:
    """Reconstruct the (E, F) AST pair from a train_v2 grammar-A completion
    ('ASSUME: ...\\nASK: ...'), via checkform's own prefix parser."""
    e_line, f_line = completion.split("\n")
    if not e_line.startswith("ASSUME: ") or not f_line.startswith("ASK: "):
        raise ValueError(f"not a grammar-A completion: {completion!r}")
    e = checkform.parse_prefix_equation(e_line[len("ASSUME: "):])
    f = checkform.parse_prefix_equation(f_line[len("ASK: "):])
    return e, f


# ------------------------------------------------------------ surfaces


def serialize(grammar_key: str, e: Equation, f: Equation) -> str:
    """The two-line statement for a pair in one grammar."""
    return GRAMMARS[grammar_key].serialize_pair(e, f)


def parses_under(grammar_key: str, statement: str) -> bool:
    try:
        GRAMMARS[grammar_key].parse_answer(statement)
        return True
    except BAnswerParseError:
        return False


# ------------------------------------------------- validation negatives
#
# Each returns a statement that must FAIL the asked grammar's parser (the
# builder asserts this), so the gold answer is a hard "No".


def neg_wrong_grammar(other_key: str, e: Equation, f: Equation) -> str:
    """A well-formed statement in a DIFFERENT grammar (distinct line labels
    make it unparseable under the asked one)."""
    return serialize(other_key, e, f)


def neg_malformed(grammar_key: str, e: Equation, f: Equation) -> str:
    """The asked grammar's statement with its last ')' removed — unbalanced,
    so the recursive parser rejects it."""
    statement = serialize(grammar_key, e, f)
    idx = statement.rfind(")")
    if idx == -1:
        raise ValueError(f"no ')' to drop in {statement!r}")
    return statement[:idx] + statement[idx + 1:]


def neg_mismatched_labels(grammar_key: str, other_key: str,
                          e: Equation, f: Equation) -> str:
    """The asked grammar's bodies under ANOTHER grammar's line labels — the
    asked parser never finds its own labels, so it rejects."""
    statement = serialize(grammar_key, e, f)
    (l0, l1) = LABELS_BY_KEY[grammar_key]
    (m0, m1) = LABELS_BY_KEY[other_key]
    line0, line1 = statement.split("\n")
    body0 = line0[len(l0) + 2:]   # strip "<L0>: "
    body1 = line1[len(l1) + 2:]
    return f"{m0}: {body0}\n{m1}: {body1}"


# --------------------------------------------------------------- prompts
#
# Frozen wording. The label is used as an opaque token; nothing here defines
# what it means. Rows carry the fully assembled prompt so training and eval
# read byte-identical text.

IDENTIFY_PROMPT = (
    "Which rigid grammar is the following statement written in? "
    "Answer with the grammar's label only.\n\n{statement}"
)
VALIDATE_PROMPT = (
    "Is the following a valid statement in {label}? Answer Yes or No.\n\n{statement}"
)


def identify_prompt(statement: str) -> str:
    return IDENTIFY_PROMPT.format(statement=statement)


def validate_prompt(label: str, statement: str) -> str:
    return VALIDATE_PROMPT.format(label=label, statement=statement)


# ------------------------------------------------------ answer extraction
#
# Lenient in the same spirit as checkform's extraction: find the intended
# token anywhere in the response, last occurrence wins (models sometimes
# think aloud before answering).

_LABEL_RE = re.compile(r"\bRG[-\s]?([123])\b", re.IGNORECASE)
_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def extract_identify(response: str) -> str | None:
    matches = _LABEL_RE.findall(response)
    return f"RG-{matches[-1]}" if matches else None


def extract_validate(response: str) -> str | None:
    matches = _YESNO_RE.findall(response)
    if not matches:
        return None
    return matches[-1].capitalize()   # "Yes" / "No"


# --------------------------------------------------------------- grading
#
# One grader, shared by every eval path (Modal and Lambda), so extraction
# and scoring can never diverge from what the corpus was built with.


def grade_row(row: dict, response: str):
    """Return (predicted, correct, answered) for one graded response."""
    if row["task"] == "identify":
        predicted = extract_identify(response)
    else:
        predicted = extract_validate(response)
    answered = predicted is not None
    correct = bool(answered and predicted == row["completion"])
    return predicted, correct, answered


def _acc(subset: list) -> dict:
    n = len(subset)
    correct = sum(1 for r in subset if r["correct"])
    answered = sum(1 for r in subset if r["answered"])
    return {
        "n": n,
        "correct": correct,
        "accuracy_pct": round(100 * correct / n, 1) if n else 0.0,
        "answered_pct": round(100 * answered / n, 1) if n else 0.0,
    }


def summarize(graded: list) -> dict:
    """Accuracy overall and sliced by task / label / polarity / corruption.

    Each graded row needs task, label, polarity, corruption, correct,
    answered (the shape both eval entrypoints build)."""
    ident = [r for r in graded if r["task"] == "identify"]
    validate = [r for r in graded if r["task"] == "validate"]
    return {
        "overall": _acc(graded),
        "identify": _acc(ident),
        "validate": _acc(validate),
        "identify_by_label": {
            lab: _acc([r for r in ident if r["label"] == lab])
            for lab in sorted({r["label"] for r in ident})
        },
        "validate_by_polarity": {
            pol: _acc([r for r in validate if r["polarity"] == pol])
            for pol in ("yes", "no")
        },
        "validate_neg_by_corruption": {
            c: _acc([r for r in validate if r["corruption"] == c])
            for c in sorted({r["corruption"] for r in validate if r["corruption"]})
        },
    }
