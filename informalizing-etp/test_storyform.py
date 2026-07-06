#!/usr/bin/env python3
"""Tests for the storyform renderer, in CLAUDE.md's priority order:

1. Round-trip — the back-parser recovers both term trees from story text.
2. No-leakage — no notation, banned words, or variable letters in stories.
3. Determinism — same pair, byte-identical story.
4. Coverage — degenerate shapes render without error.
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

from backparse import backparse
from storyform import (
    THEME_ORDER,
    THEMES,
    ParseError,
    canonical,
    parse_equation,
    record_filename,
    render_story,
    select_theme,
    write_record,
)

# Implication pairs exercising the shapes that matter: the CLAUDE.md
# worked example, bare-variable sides, repeated variables, deep
# 4-operation nesting, and the 6-distinct-variable maximum. Only the
# first pair carries documented ETP numbering (E387 => E43); the rest
# are shape probes, not claims about ETP numbering.
PAIRS = [
    ("x ∘ y = (y ∘ y) ∘ x", "x ∘ y = y ∘ x"),
    ("x = x ◇ x", "x ◇ y = y ◇ x"),
    ("x * (y * (z * w)) = ((x * y) * z) * w", "x * x = x"),
    ("x = y ∘ x", "x = y"),
    ("x = x", "x ∘ x = x"),
    ("(x ∘ x) ∘ x = x", "x = x ∘ (x ∘ x)"),
    ("x = y ∘ (z ∘ (w ∘ (u ∘ v)))", "x ∘ y = y ∘ x"),
]

BANNED_WORDS = [
    "equation", "equations", "law", "laws", "magma", "magmas",
    "implies", "implication", "implications", "theorem", "theorems",
    "lemma", "axiom", "axioms", "formula", "formulas", "variable",
    "variables", "quantifier", "commutative", "associative", "lean",
]

BANNED_CHARS = "∘◇*=+^⇒→≠()[]{}<>"


def all_renderings():
    for e_text, f_text in PAIRS:
        for theme_key in THEME_ORDER:
            story, metadata = render_story(e_text, f_text, theme_key)
            yield e_text, f_text, theme_key, story, metadata


class RoundTripTest(unittest.TestCase):
    """Invariant 6: story text alone determines both term trees."""

    def test_roundtrip_all_pairs_all_themes(self):
        for e_text, f_text, theme_key, story, metadata in all_renderings():
            with self.subTest(e=e_text, f=f_text, theme=theme_key):
                recovered = backparse(story)
                self.assertEqual(recovered["theme"], theme_key)
                self.assertEqual(
                    canonical(*recovered["habit_law"]), metadata["canonical_e"]
                )
                self.assertEqual(
                    canonical(*recovered["question_law"]), metadata["canonical_f"]
                )

    def test_roundtrip_recovers_palette_order(self):
        for e_text, f_text, theme_key, story, metadata in all_renderings():
            with self.subTest(e=e_text, f=f_text, theme=theme_key):
                recovered = backparse(story)
                self.assertEqual(
                    recovered["habit_palette"], list(metadata["palette_e"].values())
                )
                self.assertEqual(
                    recovered["question_palette"], list(metadata["palette_f"].values())
                )


class NoLeakageTest(unittest.TestCase):
    """Invariant 2: no notation, formal words, or variable letters."""

    def test_no_banned_words(self):
        for _, _, theme_key, story, _ in all_renderings():
            lowered = story.lower()
            for word in BANNED_WORDS:
                with self.subTest(theme=theme_key, word=word):
                    self.assertIsNone(re.search(rf"\b{word}\b", lowered), story)

    def test_no_operator_symbols_or_math_chars(self):
        for _, _, theme_key, story, _ in all_renderings():
            for char in BANNED_CHARS:
                with self.subTest(theme=theme_key, char=char):
                    self.assertNotIn(char, story)

    def test_no_single_letter_variables(self):
        # Standalone letters b-z (bar 'i') would leak variable names;
        # 'a' the article is fine.
        for _, _, theme_key, story, _ in all_renderings():
            with self.subTest(theme=theme_key):
                self.assertIsNone(
                    re.search(r"\b[b-hj-z]\b", story, re.IGNORECASE), story
                )

    def test_no_digits_attached_to_letters(self):
        # "Batch 1" is fine; "E387" would not be.
        for _, _, theme_key, story, _ in all_renderings():
            with self.subTest(theme=theme_key):
                self.assertIsNone(re.search(r"[A-Za-z]\d|\d[A-Za-z]", story), story)


class DeterminismTest(unittest.TestCase):
    """Invariant 3: same (E, F) pair, byte-identical story."""

    def test_render_twice_identical(self):
        for e_text, f_text in PAIRS:
            with self.subTest(e=e_text, f=f_text):
                first, _ = render_story(e_text, f_text)
                second, _ = render_story(e_text, f_text)
                self.assertEqual(first, second)

    def test_theme_selection_deterministic_and_format_insensitive(self):
        for e_text, f_text in PAIRS:
            with self.subTest(e=e_text, f=f_text):
                choice = select_theme(e_text, f_text)
                self.assertIn(choice, THEME_ORDER)
                self.assertEqual(choice, select_theme(e_text, f_text))
        # Whitespace and op-symbol spelling do not change the choice.
        self.assertEqual(
            select_theme("x ∘ y = (y ∘ y) ∘ x", "x ∘ y = y ∘ x"),
            select_theme("x◇y = (y◇y)◇x", "x◇y=y◇x"),
        )


class CoverageTest(unittest.TestCase):
    """Degenerate shapes render without error; parser rejects ambiguity."""

    def test_degenerate_shapes_render_everywhere(self):
        degenerate = [
            ("x = y", "x = y ∘ x"),  # both habit sides bare
            ("x = x", "x = x"),  # single repeated variable
            ("((x ∘ x) ∘ x) ∘ x = x", "x ∘ (x ∘ (x ∘ x)) = x"),  # deep, one var
        ]
        for e_text, f_text in degenerate:
            for theme_key in THEME_ORDER:
                with self.subTest(e=e_text, f=f_text, theme=theme_key):
                    story, metadata = render_story(e_text, f_text, theme_key)
                    recovered = backparse(story)
                    self.assertEqual(
                        canonical(*recovered["habit_law"]), metadata["canonical_e"]
                    )
                    self.assertEqual(
                        canonical(*recovered["question_law"]), metadata["canonical_f"]
                    )

    def test_parser_rejects_unparenthesized_chain(self):
        with self.assertRaises(ParseError):
            parse_equation("x ∘ y ∘ z = x")

    def test_parser_accepts_all_op_symbols(self):
        for symbol in ("∘", "◇", "*"):
            lhs, rhs = parse_equation(f"x {symbol} y = y {symbol} x")
            self.assertEqual(canonical(lhs, rhs), "(v1 ∘ v2) = (v2 ∘ v1)")

    def test_too_many_variables_rejected(self):
        with self.assertRaises(ValueError):
            render_story("x = y ∘ (z ∘ (w ∘ (u ∘ (v ∘ q))))", "x = y", "paint")


class ExportTest(unittest.TestCase):
    """JSON records land in files with deterministic names, and the
    story inside a written file still round-trips through the back-parser."""

    def test_write_record_with_labels(self):
        story, metadata = render_story(
            "x ∘ y = (y ∘ y) ∘ x", "x ∘ y = y ∘ x", "paint"
        )
        metadata["label_e"] = "E387"
        metadata["label_f"] = "E43"
        with tempfile.TemporaryDirectory() as tmp:
            path = write_record(story, metadata, Path(tmp) / "corpus")
            self.assertEqual(path.name, "E387-E43-paint.json")
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["story"], story)
            self.assertEqual(record["metadata"], metadata)
            recovered = backparse(record["story"])
            self.assertEqual(
                canonical(*recovered["habit_law"]), metadata["canonical_e"]
            )
            self.assertEqual(
                canonical(*recovered["question_law"]), metadata["canonical_f"]
            )

    def test_filename_without_labels_is_deterministic(self):
        _, metadata = render_story("x = x ◇ x", "x ◇ y = y ◇ x", "tea")
        name = record_filename(metadata)
        self.assertRegex(name, r"^pair-[0-9a-f]{12}-tea\.json$")
        self.assertEqual(name, record_filename(metadata))

    def test_same_pair_different_themes_do_not_collide(self):
        _, paint_meta = render_story("x = y ∘ x", "x = y", "paint")
        _, tea_meta = render_story("x = y ∘ x", "x = y", "tea")
        self.assertNotEqual(record_filename(paint_meta), record_filename(tea_meta))


class WorkedExampleTest(unittest.TestCase):
    """The habit section of E387 => E43 under the paint theme matches
    CLAUDE.md's worked example (compared with normalized whitespace)."""

    def test_habit_matches_claude_md(self):
        story, _ = render_story("x ∘ y = (y ∘ y) ∘ x", "x ∘ y = y ∘ x", "paint")
        expected_habit = (
            "In a certain paint workshop, the colorist follows one unbreakable "
            "habit. Take any two pigments at all — call the first crimson and the "
            "second ochre. She runs two procedures side by side.\n\n"
            "In the first, she pours crimson into ochre and sets the result aside "
            "as Batch 1.\n\n"
            "In the second, she pours ochre into ochre and calls the result "
            "Batch 2; then she pours Batch 2 into crimson and calls that Batch 3.\n\n"
            "However she chooses her two starting pigments, Batch 1 and Batch 3 "
            "always come out the exact same color. That is simply how this "
            "workshop works, without exception."
        )
        self.assertTrue(story.startswith(expected_habit), story)
        self.assertTrue(story.rstrip().endswith("?"), story)


if __name__ == "__main__":
    unittest.main()
