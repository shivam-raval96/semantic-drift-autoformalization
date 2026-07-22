import json
import tempfile
import unittest
from pathlib import Path

from paired_analysis import average_ranks, load_result_rows, pearson, summarize


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


if __name__ == "__main__":
    unittest.main()
