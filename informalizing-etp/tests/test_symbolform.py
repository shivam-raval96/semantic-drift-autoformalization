#!/usr/bin/env python3
"""Tests for symbolform, the rigid-grammar rendering arm:

1. Round-trip — each rendered line parses under checkform's own prefix
   grammar back to the law it came from.
2. Determinism — same pair, byte-identical text.
3. No leakage — no ETP op symbols, labels, or original variable spellings.
4. Coverage — degenerate shapes (bare-variable sides, repeated variables,
   maximal 4-operation nesting) render without error.
"""

import re
import unittest

from checkform import parse_prefix_equation
from storyform import canonical
from symbolform import render_symbolic

# A spread of real ETP shapes: the CLAUDE.md worked example, a
# bare-variable LHS, repeated variables, deep 4-op nesting, and laws
# whose ETP spelling starts at y/z (renaming must normalize them).
PAIRS = (
    ("x ∘ y = (y ∘ y) ∘ x", "x ∘ y = y ∘ x"),
    ("x = x ∘ (y ∘ x)", "x ∘ x = x"),
    ("x = (x ∘ ((x ∘ x) ∘ x)) ∘ x", "x = x ∘ x"),
    ("y ∘ z = z ∘ y", "x ∘ (y ∘ z) = (x ∘ y) ∘ z"),
)


class RoundTripTest(unittest.TestCase):
    def test_lines_parse_back_to_the_pair(self):
        for e_text, f_text in PAIRS:
            text, metadata = render_symbolic(e_text, f_text)
            assume_line, ask_line = text.splitlines()
            self.assertTrue(assume_line.startswith("ASSUME: "))
            self.assertTrue(ask_line.startswith("ASK: "))
            e_terms = parse_prefix_equation(assume_line[len("ASSUME: ") :])
            f_terms = parse_prefix_equation(ask_line[len("ASK: ") :])
            self.assertEqual(canonical(*e_terms), metadata["canonical_e"])
            self.assertEqual(canonical(*f_terms), metadata["canonical_f"])

    def test_letters_assigned_by_first_appearance(self):
        _, metadata = render_symbolic("y ∘ z = z ∘ y", "x ∘ y = y ∘ x")
        self.assertEqual(metadata["letters_e"], {"y": "x", "z": "y"})
        self.assertEqual(metadata["letters_f"], {"x": "x", "y": "y"})


class DeterminismTest(unittest.TestCase):
    def test_repeated_renders_identical(self):
        for pair in PAIRS:
            self.assertEqual(render_symbolic(*pair), render_symbolic(*pair))


class LeakageTest(unittest.TestCase):
    def test_no_op_symbols_or_labels(self):
        for pair in PAIRS:
            text, _ = render_symbolic(*pair)
            for symbol in "∘◇*":
                self.assertNotIn(symbol, text)
            self.assertIsNone(re.search(r"\bE\d+\b", text))

    def test_only_the_two_lines(self):
        text, _ = render_symbolic(*PAIRS[0])
        self.assertEqual(len(text.splitlines()), 2)


class MetadataTest(unittest.TestCase):
    def test_record_schema_matches_the_other_arms(self):
        _, metadata = render_symbolic(*PAIRS[0])
        for key in (
            "theme",
            "style",
            "equation_e",
            "equation_f",
            "canonical_e",
            "canonical_f",
            "letters_e",
            "letters_f",
        ):
            self.assertIn(key, metadata)
        self.assertEqual(metadata["theme"], "symbolic")
        self.assertEqual(metadata["style"], "symbolic")


if __name__ == "__main__":
    unittest.main()
