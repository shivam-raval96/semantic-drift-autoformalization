#!/usr/bin/env python3
"""Proveform: grade a model's truth judgment of an implication.

The truth-judgment task (experiment 13) shows a model an implication
"law E implies law F" rendered in one of the presentation arms (story,
literal, symbolic) under the matching prove_*_prompt.md template and
asks for a final line

    ANSWER: True        (the questioned law holds in every structure
                         satisfying the assumption)
    ANSWER: False       (some structure satisfies the assumption but
                         violates the questioned law)

Grading is a string comparison against the Lean-verified truth from
truthdata.py — no LLM judging. Extraction mirrors checkform: the last
ANSWER line wins (a model that restates the format early is graded on
its final answer), decoration and case are tolerated, and the line is
anchored to its end so hedges like "ANSWER: True or False" stay
unparseable rather than silently reading as True.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_PATHS = {
    "story": _PROMPTS_DIR / "prove_story_prompt.md",
    "literal": _PROMPTS_DIR / "prove_literal_prompt.md",
    "symbolic": _PROMPTS_DIR / "prove_symbolic_prompt.md",
}

BUCKETS = ("correct", "wrong", "unparseable", "api-error")


class AnswerParseError(ValueError):
    pass


# An answer line: optional quote/list/bold/backtick decoration, the
# label, a colon, the verdict word, then only decoration or terminal
# punctuation to the end of the line. The full-line anchor is what keeps
# "ANSWER: True or False" and "ANSWER: probably True, since ..."
# unparseable — a graded answer must commit.
_ANSWER_RE = re.compile(
    r"^[ \t>*+`_-]*ANSWER\b[ \t]*(?:\*\*)?[ \t]*:[ \t`*_]*"
    r"(?P<value>true|false)"
    r"[ \t`*_]*[.!]?[ \t`*_]*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_answer(response_text: str) -> bool:
    """Pull the final True/False verdict out of a raw response."""
    value: Optional[str] = None
    for match in _ANSWER_RE.finditer(response_text):
        value = match.group("value")
    if value is None:
        raise AnswerParseError(
            "no conclusive 'ANSWER: True' or 'ANSWER: False' line found"
        )
    return value.lower() == "true"


def grade(response_text: str, truth: bool) -> dict:
    """Grade a raw model response against the implication's truth.

    Returns a verdict dict with a stable schema: status is "correct",
    "wrong", or "unparseable"; answer is the parsed bool or None.
    """
    try:
        answer = extract_answer(response_text)
    except AnswerParseError as error:
        return {"status": "unparseable", "answer": None, "error": str(error)}
    return {
        "status": "correct" if answer is truth else "wrong",
        "answer": answer,
        "error": None,
    }
