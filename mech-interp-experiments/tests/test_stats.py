#!/usr/bin/env python3
"""Tests for shared.stats.

    cd mech-interp-experiments && python3 -m unittest discover -s tests -t .
"""

import unittest

from shared import stats


class WilsonTest(unittest.TestCase):
    def test_interval_brackets_the_rate(self):
        r = stats.rate(43, 52)
        self.assertLess(r.low, r.value)
        self.assertLess(r.value, r.high)

    def test_interval_stays_inside_zero_to_one(self):
        for successes, total in ((0, 20), (20, 20), (1, 3)):
            low, high = stats.wilson_interval(successes, total)
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)

    def test_more_data_narrows_the_interval(self):
        narrow = stats.rate(80, 100)
        wide = stats.rate(8, 10)
        self.assertLess(narrow.high - narrow.low, wide.high - wide.low)

    def test_empty_sample_reports_no_rate(self):
        r = stats.rate(0, 0)
        self.assertNotEqual(r.value, r.value)  # nan
        self.assertIn("0 examples", str(r))


class McNemarTest(unittest.TestCase):
    def test_agreement_everywhere_is_not_significant(self):
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)

    def test_symmetric_discordance_is_not_significant(self):
        self.assertEqual(stats.mcnemar_exact(5, 5), 1.0)

    def test_one_sided_discordance_is_significant(self):
        self.assertLess(stats.mcnemar_exact(10, 0), 0.01)

    def test_p_value_never_exceeds_one(self):
        for gained in range(6):
            for lost in range(6):
                self.assertLessEqual(stats.mcnemar_exact(gained, lost), 1.0)


class PairingTest(unittest.TestCase):
    BASELINE = [
        {"pair_id": "a", "status": "correct"},
        {"pair_id": "b", "status": "wrong"},
        {"pair_id": "c", "status": "wrong"},
        {"pair_id": "d", "status": "unparseable"},
    ]

    def test_counts_split_by_who_got_what_right(self):
        variant = [
            {"pair_id": "a", "status": "wrong"},     # lost
            {"pair_id": "b", "status": "correct"},   # gained
            {"pair_id": "c", "status": "wrong"},     # neither
            {"pair_id": "d", "status": "correct"},   # gained
        ]
        result = stats.pair_records(self.BASELINE, variant)
        self.assertEqual((result.gained, result.lost), (2, 1))
        self.assertEqual((result.both, result.neither), (0, 1))
        self.assertAlmostEqual(result.delta, 0.25)

    def test_records_are_matched_by_id_not_position(self):
        shuffled = list(reversed(self.BASELINE))
        result = stats.pair_records(self.BASELINE, shuffled)
        self.assertEqual((result.gained, result.lost), (0, 0))

    def test_only_shared_problems_are_compared(self):
        result = stats.pair_records(self.BASELINE, self.BASELINE[:2])
        self.assertEqual(result.n, 2)

    def test_duplicate_ids_are_refused(self):
        with self.assertRaises(ValueError):
            stats.pair_records(self.BASELINE, self.BASELINE + self.BASELINE[:1])


class CorrelationTest(unittest.TestCase):
    def test_monotone_but_curved_data_ranks_perfectly(self):
        xs = [1, 2, 3, 4]
        ys = [1, 4, 9, 16]
        self.assertAlmostEqual(stats.spearman(xs, ys), 1.0)
        self.assertLess(stats.pearson(xs, ys), 1.0)

    def test_constant_input_has_no_correlation(self):
        self.assertIsNone(stats.pearson([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(stats.spearman([1, 1, 1], [1, 2, 3]))

    def test_ties_share_their_rank(self):
        self.assertAlmostEqual(stats.spearman([1, 2, 2, 3], [1, 2, 2, 3]), 1.0)


if __name__ == "__main__":
    unittest.main()
