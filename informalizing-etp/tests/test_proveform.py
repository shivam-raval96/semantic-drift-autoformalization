#!/usr/bin/env python3
"""Tests for proveform, the truth-judgment grader:

1. Extraction — the final ANSWER line is found in raw model responses,
   tolerant of decoration and case, last occurrence winning.
2. Rejection — hedged, malformed, or missing answers stay unparseable.
3. Grading — one test per verdict class, plus determinism.
4. Templates — each prove_*_prompt.md has exactly one story slot, states
   the ANSWER-line contract, and leaks nothing about the ETP.
"""

import re
import unittest

from proveform import (
    BUCKETS,
    AnswerParseError,
    PROMPT_PATHS,
    extract_answer,
    grade,
)


class ExtractionTest(unittest.TestCase):
    def test_plain_lines(self):
        self.assertIs(extract_answer("ANSWER: True"), True)
        self.assertIs(extract_answer("ANSWER: False"), False)

    def test_answer_buried_in_prose(self):
        response = (
            "Let me test the assumption on a two-element table.\n"
            "The custom holds but the question fails.\n\n"
            "ANSWER: False\n"
        )
        self.assertIs(extract_answer(response), False)

    def test_last_answer_wins(self):
        response = (
            "The format requires a final line like ANSWER: True.\n"
            "After deriving the identity, the law is forced.\n"
            "ANSWER: True"
        )
        self.assertIs(extract_answer(response), True)

    def test_decoration_and_case(self):
        for text, expected in (
            ("**ANSWER:** True", True),
            ("> ANSWER: False.", False),
            ("- answer: TRUE", True),
            ("`ANSWER: false`", False),
            ("ANSWER: **True**", True),
            ("Answer: true.", True),
        ):
            self.assertIs(extract_answer(text), expected, text)

    def test_missing_answer_raises(self):
        with self.assertRaises(AnswerParseError):
            extract_answer("The questioned law follows from the assumption.")

    def test_unlabeled_verdict_raises(self):
        with self.assertRaises(AnswerParseError):
            extract_answer("The answer is True.")

    def test_hedged_answer_raises(self):
        for text in (
            "ANSWER: True or False",
            "ANSWER: probably True",
            "ANSWER: True, since the derivation above applies",
            "ANSWER: maybe",
        ):
            with self.assertRaises(AnswerParseError, msg=text):
                extract_answer(text)


class GradeTest(unittest.TestCase):
    def test_correct(self):
        verdict = grade("Derivation.\nANSWER: True", truth=True)
        self.assertEqual(verdict["status"], "correct")
        self.assertIs(verdict["answer"], True)
        self.assertIsNone(verdict["error"])

    def test_wrong(self):
        verdict = grade("Derivation.\nANSWER: True", truth=False)
        self.assertEqual(verdict["status"], "wrong")
        self.assertIs(verdict["answer"], True)

    def test_unparseable(self):
        verdict = grade("I could not decide.", truth=True)
        self.assertEqual(verdict["status"], "unparseable")
        self.assertIsNone(verdict["answer"])
        self.assertIn("ANSWER", verdict["error"])

    def test_statuses_cover_buckets(self):
        for status in ("correct", "wrong", "unparseable"):
            self.assertIn(status, BUCKETS)

    def test_determinism(self):
        response = "Some reasoning.\nANSWER: False"
        first = grade(response, truth=False)
        second = grade(response, truth=False)
        self.assertEqual(first, second)


class TemplateTest(unittest.TestCase):
    BANNED = ("magma", "Magma", "Lean", "ETP", "Equational")

    def test_templates_exist_with_one_story_slot(self):
        for form, path in PROMPT_PATHS.items():
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("{story}"), 1, form)

    def test_templates_state_the_answer_contract(self):
        for form, path in PROMPT_PATHS.items():
            text = path.read_text(encoding="utf-8")
            self.assertIn("ANSWER: True", text, form)
            self.assertIn("ANSWER: False", text, form)

    def test_templates_leak_nothing(self):
        for form, path in PROMPT_PATHS.items():
            text = path.read_text(encoding="utf-8")
            for word in self.BANNED:
                self.assertNotIn(word, text, f"{form}: {word}")
            self.assertIsNone(re.search(r"\bE\d+\b", text), form)


if __name__ == "__main__":
    unittest.main()
