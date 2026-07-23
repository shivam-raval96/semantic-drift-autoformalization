#!/usr/bin/env python3
"""Tests for genform, the synthetic-law generator:

1. Format — every emitted line round-trips through parse_equation and
   re-serializes byte-identically, so downstream components see exactly
   the equations.txt conventions.
2. Bin contract — each bin's laws carry exactly its operation count,
   stay within the renderers' 6-variable cap, keep depth <= ops, and
   never have identical sides (no vacuous or degenerate laws).
3. Dedup — no two laws in a corpus share a key (renaming + side swap
   collapsed, duals kept distinct).
4. Determinism — one seed, one byte-identical corpus; a different seed
   differs.
5. Anchors — the 1-op bin saturates at exactly the five ETP laws
   E3-E7, and a bin too small for the sampler's pair quota is refused.
"""

import random
import unittest

from genform import (
    check_corpus,
    generate_bin,
    generate_corpus,
    law_key,
    parse_bins,
    random_shape,
    serialize_equation,
)
from storyform import Op, Var, parse_equation

ETP_ONE_OP_LAWS = (
    "x = x ◇ x",
    "x = x ◇ y",
    "x = y ◇ x",
    "x = y ◇ y",
    "x = y ◇ z",
)


def term_ops(term):
    if isinstance(term, Var):
        return 0
    return 1 + term_ops(term.left) + term_ops(term.right)


def term_depth(term):
    if isinstance(term, Var):
        return 0
    return 1 + max(term_depth(term.left), term_depth(term.right))


def leaf_names(term):
    if isinstance(term, Var):
        return [term.name]
    return leaf_names(term.left) + leaf_names(term.right)


class CorpusFormatTest(unittest.TestCase):
    def setUp(self):
        self.bins = range(1, 11)
        self.lines = generate_corpus(seed=9, bins=self.bins, per_bin=40)

    def test_round_trip_is_byte_identical(self):
        for line in self.lines:
            lhs, rhs = parse_equation(line)
            self.assertEqual(serialize_equation(lhs, rhs), line)

    def test_bin_contract(self):
        for line in self.lines:
            lhs, rhs = parse_equation(line)
            ops = term_ops(lhs) + term_ops(rhs)
            self.assertIn(ops, self.bins)
            self.assertLessEqual(len(set(leaf_names(lhs) + leaf_names(rhs))), 6)
            self.assertLessEqual(max(term_depth(lhs), term_depth(rhs)), ops)
            self.assertLessEqual(term_ops(lhs), term_ops(rhs))
            self.assertNotEqual(lhs, rhs)

    def test_no_duplicate_laws(self):
        keys = [law_key(*parse_equation(line)) for line in self.lines]
        self.assertEqual(len(keys), len(set(keys)))

    def test_self_check_accepts_the_corpus(self):
        check_corpus(self.lines, self.bins)  # raises SystemExit on failure

    def test_bins_ascend_and_high_bins_fill(self):
        ops = [sum(map(term_ops, parse_equation(line))) for line in self.lines]
        self.assertEqual(ops, sorted(ops))
        for m in range(3, 11):
            self.assertEqual(ops.count(m), 40)


class DeterminismTest(unittest.TestCase):
    def test_same_seed_same_corpus(self):
        first = generate_corpus(seed=9, bins=range(1, 6), per_bin=10)
        second = generate_corpus(seed=9, bins=range(1, 6), per_bin=10)
        self.assertEqual(first, second)

    def test_different_seed_differs(self):
        first = generate_corpus(seed=9, bins=range(1, 6), per_bin=10)
        second = generate_corpus(seed=10, bins=range(1, 6), per_bin=10)
        self.assertNotEqual(first, second)


class SaturationTest(unittest.TestCase):
    def test_one_op_bin_is_exactly_etp_e3_to_e7(self):
        laws = generate_bin(random.Random(0), m=1, count=40, min_pairs=20)
        generated = {law_key(*parse_equation(line)) for line in laws}
        expected = {law_key(*parse_equation(line)) for line in ETP_ONE_OP_LAWS}
        self.assertEqual(generated, expected)

    def test_bin_too_small_for_pair_quota_is_refused(self):
        with self.assertRaises(SystemExit):
            generate_bin(random.Random(0), m=1, count=40, min_pairs=21)


class ShapeTest(unittest.TestCase):
    def test_shapes_have_requested_op_count(self):
        rng = random.Random(0)
        for k in range(0, 11):
            shape = random_shape(rng, k)
            self.assertEqual(term_ops(shape), k)

    def test_all_two_op_shapes_appear(self):
        rng = random.Random(0)
        seen = set()
        for _ in range(100):
            shape = random_shape(rng, 2)
            seen.add(isinstance(shape.left, Op))
        self.assertEqual(seen, {True, False})


class BinsArgumentTest(unittest.TestCase):
    def test_parse_bins(self):
        self.assertEqual(parse_bins("1:10"), range(1, 11))
        self.assertEqual(parse_bins("5"), range(5, 6))
        with self.assertRaises(SystemExit):
            parse_bins("0:4")
        with self.assertRaises(SystemExit):
            parse_bins("4:2")


if __name__ == "__main__":
    unittest.main()
