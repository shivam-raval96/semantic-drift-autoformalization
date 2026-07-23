#!/usr/bin/env python3
"""Tests for voteform, the majority-vote reducer over repeated passes:

1. Voting — majority beats pass order; ties fall to the earliest pass;
   swap/dual variants of one law pool into a single class; unparseable
   and api-error passes abstain; an all-abstain (pair, model) falls back
   to the earliest graded row over an api-error row.
2. Prefixes — vote@1 reproduces pass 1's rows modulo the vote field.
3. Output — the synthetic run dirs load in charts.py and aggregate to
   one row per (pair, model); the baseline subset keeps only the
   passes' models.
"""

import json
import tempfile
import unittest
from pathlib import Path

import voteform
from benchmark import make_sample, run_one
from charts import load_run

EQUATIONS = ["x ∘ y = (y ∘ y) ∘ x", "x ∘ y = y ∘ x"]
MODEL = "test/model"

EXACT = "ASSUME: op(x, y) = op(op(y, y), x)\nASK: op(x, y) = op(y, x)"
SWAPPED = "ASSUME: op(op(y, y), x) = op(x, y)\nASK: op(x, y) = op(y, x)"
WRONG_A = "ASSUME: op(x, y) = op(y, x)\nASK: op(x, y) = op(y, x)"
WRONG_B = "ASSUME: x = y\nASK: x = y"
UNPARSEABLE = "I cannot answer."
API_ERROR = None  # caller returns an error instead of content


def canned_caller(content):
    def caller(model, prompt, sample, stage=2):
        if content is None:
            return {"content": None, "error": "boom", "usage": None, "latency_s": None}
        return {"content": content, "error": None, "usage": None, "latency_s": 0.1}

    return caller


def pass_meta(models):
    return {
        "timestamp": "2026-07-22T00:00:00+00:00",
        "seed": 0,
        "n": 1,
        "models": models,
        "form": "story",
        "reasoning_regime": "off",
        "label_prefix": "E",
        "equations_sha256": "eqsha",
        "prompt_template": "formalize_prompt.md",
        "prompt_template_sha256": "tmplsha",
        "max_tokens": 4096,
        "temperature": 0.7,
    }


class VoteFixture(unittest.TestCase):
    """Base: write pass dirs holding one pair graded by real run_one rows."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.sample = make_sample(EQUATIONS, 1, 2, "story")

    def write_pass(self, name, responses, models=(MODEL,), meta_overrides=None):
        """One pass dir; responses maps model -> canned response text."""
        run_dir = self.root / name
        run_dir.mkdir(parents=True)
        meta = pass_meta(list(models))
        meta.update(meta_overrides or {})
        (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (run_dir / "samples.jsonl").write_text(
            json.dumps(self.sample, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with (run_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
            for model in models:
                row = run_one(
                    self.sample, model, canned_caller(responses[model]), "off"
                )
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return run_dir

    def run_vote(self, responses, ks="3", extra_args=()):
        dirs = [
            self.write_pass(f"pass{i}", {MODEL: response})
            for i, response in enumerate(responses, start=1)
        ]
        out_root = self.root / "out"
        voteform.main(
            [str(d) for d in dirs]
            + ["--ks", ks, "--out-root", str(out_root), "--name", "run"]
            + list(extra_args)
        )
        return out_root

    def voted_row(self, out_root, k):
        lines = (out_root / f"run-vote{k}" / "results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(len(lines), 1)
        return json.loads(lines[0])


class VotingRuleTest(VoteFixture):
    def test_majority_beats_pass_order(self):
        out = self.run_vote([WRONG_A, EXACT, EXACT])
        row = self.voted_row(out, 3)
        self.assertEqual(row["bucket"], "exact")
        self.assertEqual(row["vote"]["votes_for_winner"], 2)
        self.assertEqual(row["vote"]["source_pass"], 2)
        self.assertFalse(row["vote"]["tie"])

    def test_tie_goes_to_earliest_pass(self):
        out = self.run_vote([WRONG_A, EXACT, UNPARSEABLE])
        row = self.voted_row(out, 3)
        self.assertEqual(row["bucket"], "wrong")
        self.assertEqual(row["vote"]["source_pass"], 1)
        self.assertTrue(row["vote"]["tie"])

    def test_swap_variants_pool_into_one_class(self):
        out = self.run_vote([WRONG_A, SWAPPED, EXACT])
        row = self.voted_row(out, 3)
        self.assertEqual(row["vote"]["votes_for_winner"], 2)
        self.assertEqual(row["vote"]["distinct_classes"], 2)
        # earliest member of the winning class is the swapped pass
        self.assertEqual(row["vote"]["source_pass"], 2)
        self.assertEqual(row["bucket"], "correct-swapped")

    def test_unparseable_and_api_error_abstain(self):
        out = self.run_vote([UNPARSEABLE, API_ERROR, WRONG_B])
        row = self.voted_row(out, 3)
        self.assertEqual(row["bucket"], "wrong")
        self.assertEqual(row["vote"]["valid_ballots"], 1)
        self.assertEqual(row["vote"]["abstentions"], 2)
        self.assertEqual(row["vote"]["source_pass"], 3)

    def test_all_abstain_prefers_graded_over_api_error(self):
        out = self.run_vote([API_ERROR, UNPARSEABLE, API_ERROR])
        row = self.voted_row(out, 3)
        self.assertEqual(row["bucket"], "unparseable")
        self.assertEqual(row["vote"]["valid_ballots"], 0)
        self.assertEqual(row["vote"]["source_pass"], 2)
        self.assertIsNone(row["vote"]["winner_class"])

    def test_vote1_is_pass1_relabeled(self):
        out = self.run_vote([WRONG_A, EXACT, EXACT], ks="1")
        row = self.voted_row(out, 1)
        pass1_row = json.loads(
            (self.root / "pass1" / "results.jsonl").read_text(encoding="utf-8")
        )
        vote = row.pop("vote")
        self.assertEqual(row, pass1_row)
        self.assertEqual(vote["k"], 1)


class OutputSchemaTest(VoteFixture):
    def test_vote_dir_loads_in_charts_and_aggregates(self):
        out = self.run_vote([EXACT, EXACT, WRONG_A])
        run = load_run(out / "run-vote3")
        self.assertEqual(run["label"], "vote@3")
        self.assertEqual(len(run["rows"]), 1)
        summary = json.loads(
            (out / "run-vote3" / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary[MODEL]["graded"], 1)
        self.assertEqual(summary[MODEL]["correct_rate"], 1.0)
        meta = run["meta"]
        self.assertEqual(meta["vote"]["k"], 3)
        self.assertEqual(len(meta["vote"]["pass_dirs"]), 3)

    def test_mismatched_passes_refused(self):
        dirs = [
            self.write_pass("pass1", {MODEL: EXACT}),
            self.write_pass(
                "pass2", {MODEL: EXACT}, meta_overrides={"temperature": 0.9}
            ),
        ]
        with self.assertRaises(SystemExit):
            voteform.main(
                [str(d) for d in dirs]
                + ["--ks", "1", "--out-root", str(self.root / "out")]
            )

    def test_baseline_subset_restricts_models(self):
        baseline = self.write_pass(
            "baseline",
            {MODEL: EXACT, "other/model": EXACT},
            models=(MODEL, "other/model"),
            meta_overrides={"temperature": 0.0},
        )
        out = self.run_vote(
            [EXACT, WRONG_A, EXACT],
            extra_args=["--baseline", str(baseline)],
        )
        run = load_run(out / "run-t0-baseline")
        self.assertEqual(run["models"], [MODEL])
        self.assertEqual(len(run["rows"]), 1)
        self.assertEqual(run["label"], "temp0 single-pass")


if __name__ == "__main__":
    unittest.main()
