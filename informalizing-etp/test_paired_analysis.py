import json
import tempfile
import unittest
from pathlib import Path

from paired_analysis import (
    add_model_correlations,
    average_ranks,
    build_observations,
    classify_story_error,
    error_taxonomy,
    grouped_summaries,
    load_result_rows,
    markdown_report,
    pearson,
    summarize,
)


class PairedAnalysisTest(unittest.TestCase):
    def test_summary_counts_and_conditionals(self):
        rows = [
            {"story_correct": True, "literal_correct": True},
            {"story_correct": True, "literal_correct": True},
            {"story_correct": False, "literal_correct": True},
            {"story_correct": True, "literal_correct": False},
            {"story_correct": False, "literal_correct": False},
            {"story_correct": False, "literal_correct": False},
        ]
        result = summarize(rows)
        self.assertEqual(result["both_correct"], 2)
        self.assertEqual(result["literal_only"], 1)
        self.assertEqual(result["story_only"], 1)
        self.assertEqual(result["neither"], 2)
        self.assertAlmostEqual(result["p_story_given_literal_correct"], 2 / 3)
        self.assertAlmostEqual(result["p_story_given_literal_wrong"], 1 / 3)
        self.assertAlmostEqual(result["phi"], 1 / 3)

    def test_successful_retry_is_not_replaced_by_later_api_error(self):
        rows = [
            {"pair_id": "E1-E2", "model": "m", "bucket": "api-error"},
            {"pair_id": "E1-E2", "model": "m", "bucket": "exact"},
            {"pair_id": "E1-E2", "model": "m", "bucket": "api-error"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            loaded = load_result_rows(path)
        self.assertEqual(loaded[("E1-E2", "m")]["bucket"], "exact")

    def test_correlations_and_tied_ranks(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(pearson([1, 2, 3], [6, 4, 2]), -1.0)
        self.assertEqual(average_ranks([10, 20, 20, 40]), [1.0, 2.5, 2.5, 4.0])

    def test_story_error_taxonomy(self):
        sample = {
            "metadata": {
                "canonical_e": "v1 = (v2 ∘ v2)",
                "canonical_f": "v1 = v1",
            }
        }
        self.assertEqual(
            classify_story_error(
                {
                    "bucket": "wrong",
                    "response": "ASSUME: jasmine = op(oolong, op(oolong, oolong))\n"
                    "ASK: jasmine = jasmine",
                },
                sample,
            ),
            "assume_wrong_only",
        )
        self.assertEqual(
            classify_story_error(
                {
                    "bucket": "wrong",
                    "response": "ASSUME: jasmine = op(oolong, oolong)\n"
                    "ASK: op(jasmine, jasmine) = jasmine",
                },
                sample,
            ),
            "ask_wrong_only",
        )
        self.assertEqual(
            classify_story_error({"bucket": "unparseable"}, sample),
            "unparseable",
        )

        inconsistent_sample = {
            "metadata": {
                "canonical_e": "(v1 ∘ v2) = v1",
                "canonical_f": "(v1 ∘ v2) = v2",
            }
        }
        self.assertEqual(
            classify_story_error(
                {
                    "bucket": "wrong",
                    "response": "ASSUME: op(a, b) = a\nASK: op(a, b) = a",
                },
                inconsistent_sample,
            ),
            "inconsistent_operation_direction",
        )

        taxonomy = error_taxonomy(
            [
                {
                    "sampling": "stratified",
                    "regime": "off",
                    "pair_id": "E1-E2",
                    "story_error_type": "assume_wrong_only",
                },
                {
                    "sampling": "stratified",
                    "regime": "off",
                    "pair_id": "E3-E4",
                    "story_error_type": "unparseable",
                },
            ]
        )
        self.assertEqual([row["count"] for row in taxonomy], [1, 1])
        self.assertEqual([row["share"] for row in taxonomy], [0.5, 0.5])

    def test_direct_run_directories_and_dynamic_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            story = root / "story"
            literal = root / "literal"
            models = ["model/a", "model/b"]
            samples = [
                {"pair_id": "E1-E2", "ops_total": 1},
                {"pair_id": "E3-E4", "ops_total": 2},
            ]
            for run_dir, form in ((story, "story"), (literal, "literal")):
                run_dir.mkdir()
                (run_dir / "run_meta.json").write_text(
                    json.dumps(
                        {
                            "models": models,
                            "form": form,
                            "stratify_ops": 20,
                            "reasoning_regime": "off",
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "samples.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in samples),
                    encoding="utf-8",
                )

            story_rows = [
                {"pair_id": "E1-E2", "model": "model/a", "bucket": "exact"},
                {
                    "pair_id": "E1-E2",
                    "model": "model/b",
                    "bucket": "unparseable",
                    "response": "",
                },
                {"pair_id": "E3-E4", "model": "model/a", "bucket": "exact"},
                {
                    "pair_id": "E3-E4",
                    "model": "model/b",
                    "bucket": "unparseable",
                    "response": "",
                },
            ]
            literal_rows = [
                {"pair_id": pair, "model": model, "bucket": "exact"}
                for pair in ("E1-E2", "E3-E4")
                for model in models
            ]
            (story / "results.jsonl").write_text(
                "\n".join(json.dumps(row) for row in story_rows), encoding="utf-8"
            )
            (literal / "results.jsonl").write_text(
                "\n".join(json.dumps(row) for row in literal_rows), encoding="utf-8"
            )

            observations, warnings = build_observations(
                story, literal, direct_runs=True
            )

        self.assertEqual(len(observations), 4)
        self.assertEqual(warnings, [])
        summaries = grouped_summaries(observations, ("sampling", "regime"))
        model_rows = grouped_summaries(
            observations, ("sampling", "regime", "model")
        )
        add_model_correlations(summaries, model_rows)
        report = markdown_report(summaries, warnings)
        self.assertIn("Stratified 2", report)
        self.assertIn("use 2 models", report)


if __name__ == "__main__":
    unittest.main()
