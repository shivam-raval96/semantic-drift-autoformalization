#!/usr/bin/env python3
"""Tests for the literalform renderer, mirroring test_storyform's priorities:

1. Round-trip — the literal back-parser recovers both term trees.
2. No-fuzz — no theme vocabulary or op symbols in descriptions.
3. Determinism — same pair, byte-identical description.
4. Coverage — degenerate shapes render without error.
5. Grading — checkform grades a perfect answer correct against the record.
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

from checkform import build_prompt, grade
from literalform import (
    LiteralBackparseError,
    backparse_literal,
    render_description,
)
from storyform import THEMES, canonical, write_record

LITERAL_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "literal_prompt.md"

# Same shape probes as test_storyform: the worked pair, bare-variable
# sides, repeated variables, deep 4-operation nesting, and the
# 6-distinct-variable maximum.
PAIRS = [
    ("x ∘ y = (y ∘ y) ∘ x", "x ∘ y = y ∘ x"),
    ("x = x ◇ x", "x ◇ y = y ◇ x"),
    ("x * (y * (z * w)) = ((x * y) * z) * w", "x * x = x"),
    ("x = y ∘ x", "x = y"),
    ("x = x", "x ∘ x = x"),
    ("(x ∘ x) ∘ x = x", "x = x ∘ (x ∘ x)"),
    ("x = y ∘ (z ∘ (w ∘ (u ∘ v)))", "x ∘ y = y ∘ x"),
]


def all_renderings():
    for e_text, f_text in PAIRS:
        yield e_text, f_text, *render_description(e_text, f_text)


class RoundTripTest(unittest.TestCase):
    """The description text alone determines both term trees."""

    def test_roundtrip_all_pairs(self):
        for e_text, f_text, description, metadata in all_renderings():
            with self.subTest(e=e_text, f=f_text):
                recovered = backparse_literal(description)
                self.assertEqual(
                    canonical(*recovered["habit_law"]), metadata["canonical_e"]
                )
                self.assertEqual(
                    canonical(*recovered["question_law"]), metadata["canonical_f"]
                )

    def test_backparse_rejects_story_text(self):
        from storyform import render_story

        story, _ = render_story("x ∘ y = y ∘ x", "x ∘ x = x", "paint")
        with self.assertRaises(LiteralBackparseError):
            backparse_literal(story)


class NoFuzzTest(unittest.TestCase):
    """Descriptions are literal: no theme vocabulary, no op symbols,
    no ETP numbering."""

    def test_no_theme_vocabulary(self):
        fuzz_words = set()
        for theme in THEMES.values():
            fuzz_words.update(theme.palette)
            fuzz_words.add(theme.result_noun.lower())
            fuzz_words.add(theme.element_singular.lower())
        for _, _, description, _ in all_renderings():
            lowered = description.lower()
            for word in fuzz_words:
                with self.subTest(word=word):
                    self.assertIsNone(re.search(rf"\b{word}\b", lowered))

    def test_no_op_symbols_or_letter_digit_runs(self):
        for _, _, description, _ in all_renderings():
            for char in "∘◇*=()[]{}":
                self.assertNotIn(char, description)
            self.assertIsNone(re.search(r"[A-Za-z]\d|\d[A-Za-z]", description))


class DeterminismTest(unittest.TestCase):
    def test_render_twice_identical(self):
        for e_text, f_text in PAIRS:
            with self.subTest(e=e_text, f=f_text):
                first, _ = render_description(e_text, f_text)
                second, _ = render_description(e_text, f_text)
                self.assertEqual(first, second)

    def test_op_symbol_spelling_does_not_change_output(self):
        a, _ = render_description("x ∘ y = y ∘ x", "x ∘ x = x")
        b, _ = render_description("x ◇ y = y ◇ x", "x ◇ x = x")
        self.assertEqual(a, b)


class CoverageTest(unittest.TestCase):
    def test_degenerate_shapes_roundtrip(self):
        degenerate = [
            ("x = y", "x = y ∘ x"),
            ("x = x", "x = x"),
            ("((x ∘ x) ∘ x) ∘ x = x", "x ∘ (x ∘ (x ∘ x)) = x"),
        ]
        for e_text, f_text in degenerate:
            with self.subTest(e=e_text, f=f_text):
                description, metadata = render_description(e_text, f_text)
                recovered = backparse_literal(description)
                self.assertEqual(
                    canonical(*recovered["habit_law"]), metadata["canonical_e"]
                )
                self.assertEqual(
                    canonical(*recovered["question_law"]), metadata["canonical_f"]
                )

    def test_too_many_variables_rejected(self):
        with self.assertRaises(ValueError):
            render_description("x = y ∘ (z ∘ (w ∘ (u ∘ (v ∘ q))))", "x = y")


class GradingTest(unittest.TestCase):
    """checkform grades literalform records with no changes."""

    @staticmethod
    def _prefix(canonical_text: str) -> str:
        # canonical() emits infix like "(v1 ∘ v2) = v1"; rewrite to the
        # answer grammar's prefix op(...) form.
        def term_to_prefix(text: str) -> str:
            text = text.strip()
            if not text.startswith("("):
                return text
            depth = 0
            for i, c in enumerate(text):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                elif c == "∘" and depth == 1:
                    left = term_to_prefix(text[1:i])
                    right = term_to_prefix(text[i + 1 : -1])
                    return f"op({left}, {right})"
            raise AssertionError(f"malformed canonical term {text!r}")

        lhs, rhs = canonical_text.split(" = ")
        return f"{term_to_prefix(lhs)} = {term_to_prefix(rhs)}"

    def test_perfect_answer_grades_correct(self):
        for e_text, f_text, _, metadata in all_renderings():
            with self.subTest(e=e_text, f=f_text):
                response = (
                    f"ASSUME: {self._prefix(metadata['canonical_e'])}\n"
                    f"ASK: {self._prefix(metadata['canonical_f'])}\n"
                )
                verdict = grade(response, metadata)
                self.assertEqual(verdict["status"], "correct", verdict)

    def test_swapped_direction_grades_wrong(self):
        _, _, _, metadata = next(iter(all_renderings()))
        response = (
            f"ASSUME: {self._prefix(metadata['canonical_f'])}\n"
            f"ASK: {self._prefix(metadata['canonical_e'])}\n"
        )
        self.assertEqual(grade(response, metadata)["status"], "wrong")

    def test_prompt_template_fills_description(self):
        description, metadata = render_description("x ∘ y = y ∘ x", "x ∘ x = x")
        prompt = build_prompt(
            {"story": description, "metadata": metadata},
            template_path=LITERAL_PROMPT_PATH,
        )
        self.assertIn(description, prompt)
        self.assertNotIn("{story}", prompt)
        self.assertIn("ASSUME", prompt)


class ExportTest(unittest.TestCase):
    def test_write_record_filename_uses_literal_style(self):
        description, metadata = render_description("x ∘ y = y ∘ x", "x ∘ x = x")
        metadata["label_e"] = "E43"
        metadata["label_f"] = "E3"
        with tempfile.TemporaryDirectory() as tmp:
            path = write_record(description, metadata, Path(tmp))
            self.assertEqual(path.name, "E43-E3-literal.json")
            record = json.loads(path.read_text(encoding="utf-8"))
            recovered = backparse_literal(record["story"])
            self.assertEqual(
                canonical(*recovered["habit_law"]), metadata["canonical_e"]
            )


if __name__ == "__main__":
    unittest.main()
