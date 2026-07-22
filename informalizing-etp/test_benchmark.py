#!/usr/bin/env python3
"""Tests for benchmark's two-stage arm and regime wrappers:

1. Two-stage pipeline — the dry-run path (stage 1 answers with the
   deterministic literalform rendering, stage 2 with a synthesized
   correct answer) grades exact and records both stages.
2. Error paths — a failed stage yields an api-error row; stage 2 is
   never called after a stage-1 failure, and a stage-2 failure keeps
   the stage-1 response.
3. abstract_prompt.md — one {story} slot, and both worked-example
   blockquotes stay byte-consistent with their sources (the librarian
   story in formalize_prompt.md, the literalform rendering of its laws).
4. Regime wrappers — the story/literal wordings are byte-identical to
   the pre-two-stage constants; the abstract wording applies to stage 1.
5. Experiment 08 additions — the --prompt-template override reaches the
   sample's prompt, drop_vacuous removes exactly the zero-op-law pairs,
   and the two hint templates stay consistent with their sources
   (formalize_prompt.md plus the shared hint paragraph; the literalform
   abstraction in the example arm).
6. Experiment 09 additions — sample_pairs_balanced draws both laws of a
   pair from the same per-equation-ops bin (including bins past the ETP
   cap, over a genform corpus), deterministically per seed; the label
   prefix threads through labels and pair ids and defaults to "E".
"""

import re
import tempfile
import unittest
from pathlib import Path

from benchmark import (
    ABSTRACT_PROMPT_PATH,
    STORY_PROMPT_PATH,
    build_reasoning_payload,
    drop_vacuous,
    make_sample,
    run_one,
    sample_pairs_balanced,
    synthesize_response,
    wrap_prompt,
)
from genform import generate_corpus
from literalform import render_description

EQUATIONS = ["x ∘ y = (y ∘ y) ∘ x", "x ∘ y = y ∘ x"]

# The worked example's laws: librarian story in formalize_prompt.md.
LIBRARIAN_E = "x ∘ (x ∘ y) = y"
LIBRARIAN_F = "x ∘ y = y ∘ x"

# Experiment 08's hint templates and the paragraph they share.
HINT_PROMPT_PATH = Path(__file__).resolve().parent / "formalize_hint_prompt.md"
HINT_EXAMPLE_PROMPT_PATH = (
    Path(__file__).resolve().parent / "formalize_hint_example_prompt.md"
)
HINT_PARAGRAPH = (
    "The approach you should take to finding a formalization is first\n"
    "abstracting away unnecessary details about the story, and only then\n"
    "translating into the output format.\n"
)

# Byte-exact copies of the wrapper wordings used by experiments 01-04;
# the two-stage refactor must not change what earlier arms send.
OLD_OFF_SUFFIX = (
    "\n\nRespond with only the two required lines, and no other text before them."
)
OLD_ON_PREFIX_STORY = (
    "Work through the story step by step first — write out what "
    "expression each numbered intermediate stands for, one at a time — "
    "and only then finish with the two required lines.\n\n"
)


def dry_caller(model, prompt, sample, stage=2):
    """Mirror of benchmark.main's dry-run caller closure."""
    metadata = sample["metadata"]
    if stage == 1:
        content = render_description(metadata["equation_e"], metadata["equation_f"])[0]
    else:
        content = synthesize_response(metadata)
    return {"content": content, "error": None, "usage": None, "latency_s": 0.0}


def normalized(text: str) -> str:
    """Strip blockquote markers and collapse all whitespace runs."""
    return re.sub(r"\s+", " ", re.sub(r"(?m)^> ?", "", text)).strip()


class TwoStageDryRunTest(unittest.TestCase):
    def test_pipeline_grades_exact(self):
        sample = make_sample(EQUATIONS, 1, 2, "two-stage")
        row = run_one(sample, "test/model", dry_caller, "off")
        self.assertEqual(row["form"], "two-stage")
        self.assertEqual(row["bucket"], "exact")
        self.assertIsNone(row["api_error"])
        self.assertIn("Value 1", row["stage1_response"])
        self.assertIn("ASSUME:", row["response"])
        self.assertIsNotNone(row["sent_prompt_hash"])
        self.assertIsNotNone(row["stage1_sent_prompt_hash"])
        self.assertNotEqual(row["sent_prompt_hash"], row["stage1_sent_prompt_hash"])

    def test_stage1_prompt_uses_abstract_template(self):
        sample = make_sample(EQUATIONS, 1, 2, "two-stage")
        self.assertIn("rewrite what it describes", sample["prompt"])
        self.assertIn(sample["story"], sample["prompt"])

    def test_stage2_sees_stage1_output(self):
        sent = []

        def spy_caller(model, prompt, sample, stage=2):
            sent.append((stage, prompt))
            return dry_caller(model, prompt, sample, stage)

        sample = make_sample(EQUATIONS, 1, 2, "two-stage")
        run_one(sample, "test/model", spy_caller, None)
        self.assertEqual([stage for stage, _ in sent], [1, 2])
        stage1_output = render_description(EQUATIONS[0], EQUATIONS[1])[0]
        self.assertIn(stage1_output, sent[1][1])


class TwoStageErrorTest(unittest.TestCase):
    def test_stage1_error_is_api_error_and_skips_stage2(self):
        calls = []

        def failing_caller(model, prompt, sample, stage=2):
            calls.append(stage)
            return {"content": None, "error": "boom", "usage": None, "latency_s": 0.1}

        sample = make_sample(EQUATIONS, 1, 2, "two-stage")
        row = run_one(sample, "test/model", failing_caller, "off")
        self.assertEqual(row["bucket"], "api-error")
        self.assertIsNone(row["response"])
        self.assertTrue(row["api_error"].startswith("stage 1"))
        self.assertEqual(calls, [1])
        self.assertIsNone(row["sent_prompt_hash"])

    def test_stage2_error_keeps_stage1_response(self):
        def failing_stage2(model, prompt, sample, stage=2):
            if stage == 1:
                return dry_caller(model, prompt, sample, stage)
            return {"content": None, "error": "boom", "usage": None, "latency_s": 0.1}

        sample = make_sample(EQUATIONS, 1, 2, "two-stage")
        row = run_one(sample, "test/model", failing_stage2, "off")
        self.assertEqual(row["bucket"], "api-error")
        self.assertTrue(row["api_error"].startswith("stage 2"))
        self.assertIn("Value 1", row["stage1_response"])


class AbstractPromptTest(unittest.TestCase):
    def test_single_story_slot(self):
        text = ABSTRACT_PROMPT_PATH.read_text(encoding="utf-8")
        self.assertEqual(text.count("{story}"), 1)

    def test_worked_example_description_matches_literalform(self):
        template = normalized(ABSTRACT_PROMPT_PATH.read_text(encoding="utf-8"))
        description = normalized(render_description(LIBRARIAN_E, LIBRARIAN_F)[0])
        self.assertIn(description, template)

    def test_worked_example_story_matches_formalize_prompt(self):
        template = normalized(ABSTRACT_PROMPT_PATH.read_text(encoding="utf-8"))
        source = STORY_PROMPT_PATH.read_text(encoding="utf-8")
        quoted = [line for line in source.splitlines() if line.startswith(">")]
        story = normalized("\n".join(quoted))
        self.assertIn(story, template)


class RegimeWrapperTest(unittest.TestCase):
    def test_story_and_literal_wrappers_unchanged(self):
        for form in ("story", "literal"):
            self.assertTrue(wrap_prompt("P", "off", form=form).endswith(OLD_OFF_SUFFIX))
        self.assertTrue(wrap_prompt("P", "on", form="story").startswith(OLD_ON_PREFIX_STORY))

    def test_abstract_wrapper_applies(self):
        off = wrap_prompt("P", "off", form="abstract")
        self.assertTrue(off.endswith("rewritten description, and no other text before it."))
        on = wrap_prompt("P", "on", form="abstract")
        self.assertTrue(on.startswith("Work through the story"))
        self.assertIn("complete rewritten description", on)

    def test_qwen_soft_switch_survives(self):
        wrapped = wrap_prompt("P", "off", model="qwen/qwen3-32b", form="abstract")
        self.assertTrue(wrapped.endswith("\n/no_think"))


class NativeReasoningTest(unittest.TestCase):
    def test_off_uses_none_when_supported(self):
        info = {
            "supported_parameters": ["reasoning"],
            "reasoning": {
                "mandatory": False,
                "supported_efforts": ["high", "medium", "low", "none"],
            },
        }
        self.assertEqual(
            build_reasoning_payload("off", info),
            {"effort": "none", "exclude": True},
        )

    def test_off_disables_optional_reasoning_without_effort_none(self):
        info = {
            "supported_parameters": ["reasoning"],
            "reasoning": {"mandatory": False},
        }
        self.assertEqual(
            build_reasoning_payload("off", info),
            {"enabled": False, "exclude": True},
        )

    def test_off_uses_lowest_effort_for_mandatory_reasoning(self):
        info = {
            "supported_parameters": ["reasoning"],
            "reasoning": {
                "mandatory": True,
                "supported_efforts": ["high", "medium", "low"],
            },
        }
        self.assertEqual(
            build_reasoning_payload("off", info),
            {"effort": "low", "exclude": True},
        )

    def test_on_enables_supported_reasoning(self):
        info = {"supported_parameters": ["reasoning"]}
        self.assertEqual(build_reasoning_payload("on", info), {"enabled": True})

    def test_unsupported_reasoning_has_no_native_payload(self):
        self.assertIsNone(build_reasoning_payload("off", {}))


class PromptTemplateOverrideTest(unittest.TestCase):
    def test_override_reaches_the_sample_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "override.md"
            template.write_text("MARKER TOP\n\n{story}\n", encoding="utf-8")
            sample = make_sample(EQUATIONS, 1, 2, "story", template_path=template)
        self.assertTrue(sample["prompt"].startswith("MARKER TOP"))
        self.assertIn(sample["story"], sample["prompt"])

    def test_default_template_is_unchanged_without_override(self):
        sample = make_sample(EQUATIONS, 1, 2, "story")
        self.assertIn("translate what it describes", sample["prompt"])


class DropVacuousTest(unittest.TestCase):
    def test_drops_exactly_the_zero_op_pairs_in_order(self):
        equations = ["x = x", "x = y", "x ∘ y = (y ∘ y) ∘ x", "x ∘ y = y ∘ x"]
        samples = [
            make_sample(equations, e, f, "story")
            for e in range(1, 5)
            for f in range(1, 5)
            if e != f
        ]
        self.assertNotIn(None, samples)
        kept = drop_vacuous(samples)
        self.assertEqual([s["pair_id"] for s in kept], ["E3-E4", "E4-E3"])


class BalancedSamplerTest(unittest.TestCase):
    # A synthetic list spanning the ETP cap: bins 1-3 plus a 5-op bin
    # the ETP list could never supply.
    EQUATIONS = generate_corpus(seed=1, bins=range(1, 4), per_bin=8) + (
        generate_corpus(seed=1, bins=range(5, 6), per_bin=8)
    )

    def test_both_laws_carry_the_bin_op_count(self):
        samples = sample_pairs_balanced(
            self.EQUATIONS, 3, seed=0, bins=(1, 2, 3, 5), label_prefix="S"
        )
        self.assertEqual(len(samples), 12)
        for index, sample in enumerate(samples):
            target = (1, 2, 3, 5)[index // 3]
            self.assertEqual(sample["ops_e"], target)
            self.assertEqual(sample["ops_f"], target)
            self.assertEqual(sample["ops_total"], 2 * target)

    def test_pairs_are_distinct_and_labeled(self):
        samples = sample_pairs_balanced(
            self.EQUATIONS, 3, seed=0, bins=(1, 2, 3, 5), label_prefix="S"
        )
        pair_ids = [sample["pair_id"] for sample in samples]
        self.assertEqual(len(pair_ids), len(set(pair_ids)))
        for sample in samples:
            self.assertRegex(sample["pair_id"], r"^S\d+-S\d+$")
            e_label, f_label = sample["pair_id"].split("-")
            self.assertNotEqual(e_label, f_label)
            self.assertEqual(sample["metadata"]["label_e"], e_label)
            self.assertEqual(sample["metadata"]["label_f"], f_label)

    def test_same_seed_same_pairs_across_forms(self):
        story = sample_pairs_balanced(self.EQUATIONS, 3, seed=0, bins=(2, 3))
        literal = sample_pairs_balanced(
            self.EQUATIONS, 3, seed=0, bins=(2, 3), form="literal"
        )
        self.assertEqual(
            [s["pair_id"] for s in story], [s["pair_id"] for s in literal]
        )

    def test_label_prefix_defaults_to_etp_numbering(self):
        samples = sample_pairs_balanced(self.EQUATIONS, 2, seed=0, bins=(2,))
        for sample in samples:
            self.assertRegex(sample["pair_id"], r"^E\d+-E\d+$")

    def test_missing_bin_is_refused(self):
        with self.assertRaises(SystemExit):
            sample_pairs_balanced(self.EQUATIONS, 2, seed=0, bins=(4,))


class HintTemplateTest(unittest.TestCase):
    def test_hint_arm_is_formalize_prompt_plus_the_paragraph(self):
        hint = HINT_PROMPT_PATH.read_text(encoding="utf-8")
        base = STORY_PROMPT_PATH.read_text(encoding="utf-8")
        self.assertEqual(hint.replace(HINT_PARAGRAPH + "\n", "", 1), base)

    def test_both_arms_share_the_paragraph_and_one_story_slot(self):
        for path in (HINT_PROMPT_PATH, HINT_EXAMPLE_PROMPT_PATH):
            text = path.read_text(encoding="utf-8")
            self.assertIn(HINT_PARAGRAPH, text)
            self.assertEqual(text.count("{story}"), 1)

    def test_example_arm_abstraction_matches_literalform(self):
        template = normalized(HINT_EXAMPLE_PROMPT_PATH.read_text(encoding="utf-8"))
        description = normalized(render_description(LIBRARIAN_E, LIBRARIAN_F)[0])
        self.assertIn(description, template)

    def test_example_arm_story_matches_formalize_prompt(self):
        template = normalized(HINT_EXAMPLE_PROMPT_PATH.read_text(encoding="utf-8"))
        source = STORY_PROMPT_PATH.read_text(encoding="utf-8")
        quoted = [line for line in source.splitlines() if line.startswith(">")]
        story = normalized("\n".join(quoted))
        self.assertIn(story, template)

    def test_example_arm_translates_from_the_lettered_abstraction(self):
        text = HINT_EXAMPLE_PROMPT_PATH.read_text(encoding="utf-8")
        self.assertIn("ASSUME: op(x, op(x, y)) = y", text)
        self.assertIn("ASK: op(x, y) = op(y, x)", text)


if __name__ == "__main__":
    unittest.main()
