#!/usr/bin/env python3
"""Back-parser: recover both term trees from a story's text alone.

This enforces design invariant 6 (invertibility from narrative alone):
the rendering grammar is injective, so the story text by itself must
determine both laws — the term trees, the palette-name/variable
correspondence, and every argument order — with no annotations.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from storyform import THEMES, Op, Term, Theme, Var


class BackparseError(ValueError):
    pass


def _operand_pattern(theme: Theme) -> str:
    names = "|".join(re.escape(name) for name in theme.palette)
    return rf"{re.escape(theme.result_noun)} \d+|(?:{names})"


def _op_regex(template: str, operand: str) -> str:
    pattern = re.escape(template)
    pattern = pattern.replace(re.escape("{a}"), rf"(?P<a>{operand})")
    pattern = pattern.replace(re.escape("{b}"), rf"(?P<b>{operand})")
    return pattern


def _detect_theme(story: str) -> Theme:
    for theme in THEMES.values():
        if story.startswith(theme.intro):
            return theme
    raise BackparseError("story does not open with any known theme intro")


def _parse_take_any(theme: Theme, text: str) -> List[str]:
    """Recover the ordered palette names from a 'Take any ...' sentence."""
    match = re.search(r"Take any [^.]*\.", text)
    if not match:
        raise BackparseError("no 'Take any ...' sentence found")
    sentence = match.group(0)
    single = re.search(r"call it (\w+)", sentence)
    if single:
        return [single.group(1)]
    names = re.findall(
        r"the (?:first|second|third|fourth|fifth|sixth) (\w+)", sentence
    )
    if not names:
        raise BackparseError(f"could not read palette names from {sentence!r}")
    return names


def _resolve(ref: str, batches: Dict[int, Term], theme: Theme) -> Term:
    ref = ref.strip()
    match = re.fullmatch(rf"{re.escape(theme.result_noun)} (\d+)", ref, re.IGNORECASE)
    if match:
        number = int(match.group(1))
        if number not in batches:
            raise BackparseError(f"{ref!r} used before it was defined")
        return batches[number]
    name = re.sub(r" itself$", "", ref, flags=re.IGNORECASE).lower()
    if name not in theme.palette:
        raise BackparseError(f"unknown element name {name!r}")
    return Var(name)


def _parse_steps(theme: Theme, text: str, imperative: bool) -> Dict[int, Term]:
    """Build the batch environment from a law's step sentences, in order."""
    operand = _operand_pattern(theme)
    if imperative:
        op_clause = _op_regex(theme.op_imperative, operand)
        naming = r"(?:set the result aside as|call the result|call that)"
        pattern = rf"{op_clause} and {naming} {re.escape(theme.result_noun)} (?P<n>\d+)"
        flags = re.IGNORECASE
    else:
        op_clause = _op_regex(theme.op_agent, operand)
        naming = r"(?:sets the result aside as|calls the result|calls that)"
        # The subject is capitalized when a step sentence opens a paragraph
        # (single-procedure habits), lowercase inside one.
        pattern = (
            rf"\b(?:[Ss]he|[Hh]e) {op_clause} and {naming} "
            rf"{re.escape(theme.result_noun)} (?P<n>\d+)"
        )
        flags = 0
    batches: Dict[int, Term] = {}
    for match in re.finditer(pattern, text, flags):
        number = int(match.group("n"))
        if number in batches:
            raise BackparseError(f"{theme.result_noun} {number} defined twice")
        batches[number] = Op(
            _resolve(match.group("a"), batches, theme),
            _resolve(match.group("b"), batches, theme),
        )
    return batches


def _comparison_pattern(theme: Theme) -> str:
    names = "|".join(re.escape(name) for name in theme.palette)
    return rf"{re.escape(theme.result_noun)} \d+|(?:{names})(?: itself)?"


def _parse_habit_law(theme: Theme, text: str) -> Tuple[Term, Term]:
    batches = _parse_steps(theme, text, imperative=False)
    ref = _comparison_pattern(theme)
    match = re.search(
        rf"However (?:she|he) chooses [^,]*, (?P<x>{ref}) and (?P<y>{ref}) "
        rf"always {re.escape(theme.same_habit)}\.",
        text,
    )
    if not match:
        raise BackparseError("could not find the habit's comparison sentence")
    return (
        _resolve(match.group("x"), batches, theme),
        _resolve(match.group("y"), batches, theme),
    )


def _parse_question_law(theme: Theme, text: str) -> Tuple[Term, Term]:
    batches = _parse_steps(theme, text, imperative=True)
    ref = _comparison_pattern(theme)
    match = re.search(
        rf"must (?P<x>{ref}) and (?P<y>{ref}) "
        rf"always {re.escape(theme.same_question)}\?",
        text,
    )
    if not match:
        raise BackparseError("could not find the closing question")
    return (
        _resolve(match.group("x"), batches, theme),
        _resolve(match.group("y"), batches, theme),
    )


def backparse(story: str) -> dict:
    """Recover both laws from story text alone.

    Returns a dict with the detected theme key, each law as a pair of
    Terms over palette-name variables, and each law's ordered palette
    (the order-of-first-appearance correspondence).
    """
    theme = _detect_theme(story)
    split_at = story.find(theme.question_intro)
    if split_at < 0:
        raise BackparseError("story has no question section")
    habit_text = story[:split_at]
    question_text = story[split_at:]

    return {
        "theme": theme.key,
        "habit_palette": _parse_take_any(theme, habit_text),
        "habit_law": _parse_habit_law(theme, habit_text),
        "question_palette": _parse_take_any(theme, question_text),
        "question_law": _parse_question_law(theme, question_text),
    }
