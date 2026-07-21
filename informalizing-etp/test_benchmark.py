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
"""

import re
import unittest
from pathlib import Path

from benchmark import (
    ABSTRACT_PROMPT_PATH,
    STORY_PROMPT_PATH,
    make_sample,
    run_one,
    synthesize_response,
    wrap_prompt,
)
from literalform import render_description

EQUATIONS = ["x ∘ y = (y ∘ y) ∘ x", "x ∘ y = y ∘ x"]

# The worked example's laws: librarian story in formalize_prompt.md.
LIBRARIAN_E = "x ∘ (x ∘ y) = y"
LIBRARIAN_F = "x ∘ y = y ∘ x"

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


if __name__ == "__main__":
    unittest.main()
