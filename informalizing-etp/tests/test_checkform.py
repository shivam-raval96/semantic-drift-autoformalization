#!/usr/bin/env python3
"""Tests for checkform, the formalization-answer grader:

1. Extraction — the ASSUME/ASK lines are found in raw model responses.
2. Parsing — the prefix syntax parses exactly and rejects malformed input.
3. Grading — one test per verdict class, including the symmetries that
   must be accepted (side swap, uniform dualization) and the ones that
   must not (reversed implication, dual applied to one equation only).
4. Determinism and corpus round-trip.
"""

import json
import tempfile
import unittest
from pathlib import Path

from checkform import (
    AnswerParseError,
    answer_class_key,
    build_prompt,
    dual,
    extract_answer,
    grade,
    main,
    parse_prefix_equation,
)
from storyform import Op, Var, canonical, parse_equation, render_story

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"

# Ground truth: the CLAUDE.md worked example E387 => E43 (F symmetric),
# and a second pair whose questioned law is NOT symmetric under
# dualization, needed to detect mixed-convention answers.
_, META_SYMMETRIC_F = render_story("x ∘ y = (y ∘ y) ∘ x", "x ∘ y = y ∘ x", "paint")
_, META_ASYMMETRIC_F = render_story(
    "x ∘ y = (y ∘ y) ∘ x", "x ∘ y = (x ∘ x) ∘ y", "paint"
)

PERFECT = "ASSUME: op(x, y) = op(op(y, y), x)\nASK: op(x, y) = op(y, x)"


def to_prefix(term) -> str:
    if isinstance(term, Var):
        return term.name
    return f"op({to_prefix(term.left)}, {to_prefix(term.right)})"


def answer_from_equations(e_text: str, f_text: str) -> str:
    e_lhs, e_rhs = parse_equation(e_text)
    f_lhs, f_rhs = parse_equation(f_text)
    return (
        f"ASSUME: {to_prefix(e_lhs)} = {to_prefix(e_rhs)}\n"
        f"ASK: {to_prefix(f_lhs)} = {to_prefix(f_rhs)}"
    )


class ExtractionTest(unittest.TestCase):
    def test_answer_buried_in_prose(self):
        response = (
            "Let me think about the workshop.\n"
            "The custom combines pigments in order.\n\n"
            "ASSUME: op(crimson, ochre) = ochre\n"
            "ASK: op(ochre, crimson) = crimson\n"
        )
        self.assertEqual(
            extract_answer(response),
            ("op(crimson, ochre) = ochre", "op(ochre, crimson) = crimson"),
        )

    def test_fenced_and_decorated_lines(self):
        response = (
            "Here is my answer:\n"
            "```\n"
            "**ASSUME**: `op(a, b) = op(b, a)`\n"
            "- ASK: op(a, a) = a.\n"
            "```\n"
        )
        self.assertEqual(
            extract_answer(response), ("op(a, b) = op(b, a)", "op(a, a) = a")
        )

    def test_last_occurrence_wins(self):
        response = (
            "From the worked example:\n"
            "ASSUME: op(atlas, op(atlas, ledger)) = ledger\n"
            "ASK: op(atlas, ledger) = op(ledger, atlas)\n\n"
            "For this story instead:\n"
            "ASSUME: op(x, y) = y\n"
            "ASK: op(y, x) = x\n"
        )
        self.assertEqual(extract_answer(response), ("op(x, y) = y", "op(y, x) = x"))

    def test_missing_line_raises(self):
        with self.assertRaises(AnswerParseError):
            extract_answer("ASSUME: op(x, y) = y\nno question line here")
        with self.assertRaises(AnswerParseError):
            extract_answer("ASK: op(x, y) = y")

    def test_case_insensitive_labels(self):
        response = "assume: op(x, y) = y\nAsk: x = y"
        self.assertEqual(extract_answer(response), ("op(x, y) = y", "x = y"))


class PrefixParserTest(unittest.TestCase):
    def test_deep_nesting(self):
        lhs, rhs = parse_prefix_equation(
            "op(x, op(y, op(z, op(w, u)))) = op(op(x, y), z)"
        )
        self.assertEqual(
            canonical(lhs, rhs),
            "(v1 ∘ (v2 ∘ (v3 ∘ (v4 ∘ v5)))) = ((v1 ∘ v2) ∘ v3)",
        )

    def test_bare_variable_sides(self):
        lhs, rhs = parse_prefix_equation("x = y")
        self.assertEqual(lhs, Var("x"))
        self.assertEqual(rhs, Var("y"))

    def test_multiword_and_numbered_variables(self):
        lhs, rhs = parse_prefix_equation("op(crimson, x1) = deep_blue")
        self.assertEqual(canonical(lhs, rhs), "(v1 ∘ v2) = v3")

    def test_op_keyword_case_insensitive(self):
        lhs, rhs = parse_prefix_equation("OP(x, y) = Op(y, x)")
        self.assertEqual(canonical(lhs, rhs), "(v1 ∘ v2) = (v2 ∘ v1)")

    def test_malformed_inputs_raise(self):
        bad = [
            "op(x y) = y",  # missing comma
            "op(x, y = y",  # unbalanced parens
            "op(x, y, z) = x",  # wrong arity
            "x ∘ y = y",  # infix leakage
            "op(x, y)",  # no equals sign
            "op(x, y) = y = x",  # trailing tokens
            "op(, y) = x",  # empty operand
            "",  # empty
            "op(x, " * 5000 + "x",  # runaway nesting, never closed
            "op(x, " * 200 + "x" + ")" * 200 + " = x",  # nested past the depth cap
        ]
        for text in bad:
            with self.subTest(text=text):
                with self.assertRaises(AnswerParseError):
                    parse_prefix_equation(text)


class DualTest(unittest.TestCase):
    def test_dual_mirrors_recursively(self):
        term = Op(Op(Var("x"), Var("y")), Var("z"))
        self.assertEqual(dual(term), Op(Var("z"), Op(Var("y"), Var("x"))))
        self.assertEqual(dual(dual(term)), term)


class GradingTest(unittest.TestCase):
    def assertVerdict(self, response, metadata, status, transform=None):
        verdict = grade(response, metadata)
        self.assertEqual(verdict["status"], status, verdict)
        self.assertEqual(verdict["transform"], transform, verdict)
        return verdict

    def test_exact_convention(self):
        self.assertVerdict(
            PERFECT,
            META_SYMMETRIC_F,
            "correct",
            {"swap_e": False, "swap_f": False, "dual": False},
        )

    def test_renamed_variables_only(self):
        response = (
            "ASSUME: op(crimson, ochre) = op(op(ochre, ochre), crimson)\n"
            "ASK: op(crimson, ochre) = op(ochre, crimson)"
        )
        self.assertVerdict(
            response,
            META_SYMMETRIC_F,
            "correct",
            {"swap_e": False, "swap_f": False, "dual": False},
        )

    def test_swapped_assume_sides(self):
        response = (
            "ASSUME: op(op(y, y), x) = op(x, y)\nASK: op(x, y) = op(y, x)"
        )
        self.assertVerdict(
            response,
            META_SYMMETRIC_F,
            "correct",
            {"swap_e": True, "swap_f": False, "dual": False},
        )

    def test_swapped_ask_sides(self):
        response = (
            "ASSUME: op(x, y) = op(op(y, y), x)\n"
            "ASK: op(op(x, x), y) = op(x, y)"
        )
        self.assertVerdict(
            response,
            META_ASYMMETRIC_F,
            "correct",
            {"swap_e": False, "swap_f": True, "dual": False},
        )

    def test_dualized_consistently(self):
        # The opposite argument-order convention, applied to both lines.
        response = (
            "ASSUME: op(y, x) = op(x, op(y, y))\nASK: op(y, x) = op(x, y)"
        )
        self.assertVerdict(
            response,
            META_SYMMETRIC_F,
            "correct",
            {"swap_e": False, "swap_f": False, "dual": True},
        )

    def test_dualized_and_swapped(self):
        response = (
            "ASSUME: op(x, op(y, y)) = op(y, x)\nASK: op(y, x) = op(x, y)"
        )
        self.assertVerdict(
            response,
            META_SYMMETRIC_F,
            "correct",
            {"swap_e": True, "swap_f": False, "dual": True},
        )

    def test_reversed_implication_is_wrong(self):
        response = (
            "ASSUME: op(x, y) = op(y, x)\nASK: op(x, y) = op(op(y, y), x)"
        )
        self.assertVerdict(response, META_SYMMETRIC_F, "wrong")

    def test_dual_on_one_equation_only_is_wrong(self):
        # ASSUME uses the opposite convention, ASK the straight one; no
        # uniform transformation reconciles them.
        response = (
            "ASSUME: op(y, x) = op(x, op(y, y))\n"
            "ASK: op(x, y) = op(op(x, x), y)"
        )
        self.assertVerdict(response, META_ASYMMETRIC_F, "wrong")

    def test_wrong_tree_shape_is_wrong(self):
        response = (
            "ASSUME: op(x, y) = op(y, op(y, x))\nASK: op(x, y) = op(y, x)"
        )
        self.assertVerdict(response, META_SYMMETRIC_F, "wrong")

    def test_unparseable_reports_error(self):
        verdict = grade("no answer lines at all", META_SYMMETRIC_F)
        self.assertEqual(verdict["status"], "unparseable")
        self.assertIsNotNone(verdict["error"])
        verdict = grade("ASSUME: x + y = y\nASK: x = y", META_SYMMETRIC_F)
        self.assertEqual(verdict["status"], "unparseable")

    def test_runaway_nesting_is_unparseable(self):
        # Regression: a degenerate generation that repeats "op(" until the
        # token budget must grade unparseable, not blow the recursion limit.
        response = "ASSUME: op(x, y) = op(y, x)\nASK: " + "op(x, " * 5000 + "x"
        verdict = grade(response, META_SYMMETRIC_F)
        self.assertEqual(verdict["status"], "unparseable")
        self.assertIsNotNone(verdict["error"])

    def test_grading_is_deterministic(self):
        for response in (PERFECT, "ASSUME: op(x, y) = x\nASK: x = y"):
            with self.subTest(response=response):
                self.assertEqual(
                    grade(response, META_SYMMETRIC_F),
                    grade(response, META_SYMMETRIC_F),
                )


class CorpusRoundTripTest(unittest.TestCase):
    """A perfect answer built from each corpus record's own equations
    grades correct with the identity transform."""

    def test_corpus_records(self):
        records = sorted(CORPUS_DIR.glob("*.json"))
        self.assertTrue(records, "no corpus records found")
        for path in records:
            record = json.loads(path.read_text(encoding="utf-8"))
            metadata = record["metadata"]
            response = answer_from_equations(
                metadata["equation_e"], metadata["equation_f"]
            )
            with self.subTest(record=path.name):
                verdict = grade(response, metadata)
                self.assertEqual(verdict["status"], "correct", verdict)
                self.assertEqual(
                    verdict["transform"],
                    {"swap_e": False, "swap_f": False, "dual": False},
                )


class PromptAndCliTest(unittest.TestCase):
    def _write_record(self, directory: Path) -> Path:
        story, metadata = render_story(
            "x ∘ y = (y ∘ y) ∘ x", "x ∘ y = y ∘ x", "paint"
        )
        path = directory / "record.json"
        path.write_text(
            json.dumps({"story": story, "metadata": metadata}), encoding="utf-8"
        )
        return path

    def test_build_prompt_embeds_story(self):
        story, metadata = render_story(
            "x ∘ y = (y ∘ y) ∘ x", "x ∘ y = y ∘ x", "paint"
        )
        prompt = build_prompt({"story": story, "metadata": metadata})
        self.assertIn(story, prompt)
        self.assertNotIn("{story}", prompt)
        # The prompt must not leak the record's formal side.
        for leak in ("canonical", "∘", "E387", "Lean", "magma"):
            self.assertNotIn(leak, prompt)

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            record = self._write_record(tmp_path)
            cases = [
                (PERFECT, 0),
                ("ASSUME: op(x, y) = x\nASK: x = y", 1),
                ("nothing formal here", 2),
            ]
            for i, (response, expected_code) in enumerate(cases):
                response_path = tmp_path / f"response{i}.txt"
                response_path.write_text(response, encoding="utf-8")
                with self.subTest(expected_code=expected_code):
                    code = main(["grade", str(record), str(response_path)])
                    self.assertEqual(code, expected_code)


class AnswerClassKeyTest(unittest.TestCase):
    """Experiment 11: the grading-equivalence-class key used for voting."""

    E = "x ∘ y = (y ∘ y) ∘ x"
    F = "x ∘ y = (x ∘ x) ∘ y"  # not symmetric under dualization

    def orbit(self, e_text, f_text):
        """All 8 accepted transforms of the pair, as canonical strings."""
        e = parse_equation(e_text)
        f = parse_equation(f_text)
        members = []
        for dualize in (False, True):
            te = (dual(e[0]), dual(e[1])) if dualize else e
            tf = (dual(f[0]), dual(f[1])) if dualize else f
            for swap_e in (False, True):
                for swap_f in (False, True):
                    se = (te[1], te[0]) if swap_e else te
                    sf = (tf[1], tf[0]) if swap_f else tf
                    members.append((canonical(*se), canonical(*sf)))
        return members

    def test_invariant_across_the_full_orbit(self):
        keys = {
            answer_class_key(e_text, f_text)
            for e_text, f_text in self.orbit(self.E, self.F)
        }
        self.assertEqual(len(keys), 1)

    def test_idempotent(self):
        key = answer_class_key(self.E, self.F)
        self.assertEqual(answer_class_key(*key), key)

    def test_variable_names_and_op_symbol_are_immaterial(self):
        self.assertEqual(
            answer_class_key("x ∘ y = y ∘ x", "x ∘ y = (y ∘ y) ∘ x"),
            answer_class_key("a ◇ b = b ◇ a", "a ◇ b = (b ◇ b) ◇ a"),
        )

    def test_matches_grade_on_correct_answers(self):
        truth_key = answer_class_key(
            META_ASYMMETRIC_F["canonical_e"], META_ASYMMETRIC_F["canonical_f"]
        )
        correct_answers = [
            (self.E, self.F),  # exact
            ("(y ∘ y) ∘ x = x ∘ y", self.F),  # swapped assume sides
            ("y ∘ x = x ∘ (y ∘ y)", "y ∘ x = y ∘ (x ∘ x)"),  # dualized both
        ]
        for e_text, f_text in correct_answers:
            with self.subTest(e=e_text):
                verdict = grade(
                    answer_from_equations(e_text, f_text), META_ASYMMETRIC_F
                )
                self.assertEqual(verdict["status"], "correct")
                self.assertEqual(
                    answer_class_key(
                        verdict["canonical_answer_e"], verdict["canonical_answer_f"]
                    ),
                    truth_key,
                )

    def test_differs_from_grade_rejected_answers(self):
        truth_key = answer_class_key(
            META_ASYMMETRIC_F["canonical_e"], META_ASYMMETRIC_F["canonical_f"]
        )
        rejected = [
            (self.F, self.E),  # reversed implication
            (self.E, "y ∘ x = y ∘ (x ∘ x)"),  # dual applied to ASK only
        ]
        for e_text, f_text in rejected:
            with self.subTest(e=e_text):
                verdict = grade(
                    answer_from_equations(e_text, f_text), META_ASYMMETRIC_F
                )
                self.assertEqual(verdict["status"], "wrong")
                self.assertNotEqual(
                    answer_class_key(
                        verdict["canonical_answer_e"], verdict["canonical_answer_f"]
                    ),
                    truth_key,
                )


if __name__ == "__main__":
    unittest.main()
