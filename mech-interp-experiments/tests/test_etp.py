#!/usr/bin/env python3
"""Tests for shared.etp, on a hand-made law list and implication table.

The real table is a pinned download; these tests need no network and check the
two things grading depends on: an equation is recognized as its law however it
is written, and the table is read in the right orientation.

    cd mech-interp-experiments && python3 -m unittest discover -s tests -t .
"""

import unittest

from shared import etp
from storyform import parse_equation

LAWS = [
    "x = x",                          # 1
    "x = y",                          # 2
    "x = x ◇ y",                      # 3
    "x ◇ y = y ◇ x",                  # 4
    "x ◇ (y ◇ z) = (x ◇ y) ◇ z",      # 5
]


def table(rows):
    return bytes(cell for row in rows for cell in row)


T, F, U = etp.TRUE, etp.FALSE, etp.UNKNOWN


class LawIndexTest(unittest.TestCase):
    def setUp(self):
        self.index = etp.LawIndex(LAWS)

    def identify(self, text):
        return self.index.identify(*parse_equation(text))

    def test_every_law_is_itself(self):
        for number, text in enumerate(LAWS, start=1):
            self.assertEqual(self.identify(text), number)

    def test_renaming_variables_and_swapping_sides_are_the_same_law(self):
        self.assertEqual(self.identify("b ◇ a = a ◇ b"), 4)
        self.assertEqual(self.identify("(p ◇ q) ◇ r = p ◇ (q ◇ r)"), 5)
        self.assertEqual(self.identify("a ◇ b = a"), 3)

    def test_identical_sides_are_law_one(self):
        self.assertEqual(self.identify("x ◇ y = x ◇ y"), 1)

    def test_the_dual_is_a_different_law(self):
        # Mirroring x ◇ y gives y ◇ x: law 3's dual, which this list lacks.
        self.assertIsNone(self.identify("x = y ◇ x"))

    def test_a_law_listed_twice_is_refused(self):
        with self.assertRaises(ValueError):
            etp.LawIndex(["x = x ◇ y", "y ◇ x = y"])


class ImplicationsTest(unittest.TestCase):
    def setUp(self):
        # Row a, column b: does law a imply law b?
        self.table = etp.Implications(table([
            [T, F, U],
            [T, T, T],
            [T, F, T],
        ]), n=3)

    def test_orientation(self):
        self.assertTrue(self.table.implies(2, 1))
        self.assertFalse(self.table.implies(1, 2))

    def test_open_pairs_are_none(self):
        self.assertIsNone(self.table.implies(1, 3))

    def test_equivalence_needs_both_directions(self):
        self.assertTrue(self.table.equivalent(1, 1))
        self.assertFalse(self.table.equivalent(1, 2))
        self.assertIsNone(self.table.equivalent(1, 3))
        self.assertFalse(self.table.equivalent(3, 2))

    def test_out_of_range_laws_are_refused(self):
        with self.assertRaises(ValueError):
            self.table.implies(0, 1)
        with self.assertRaises(ValueError):
            self.table.implies(1, 4)

    def test_a_table_of_the_wrong_size_is_refused(self):
        with self.assertRaises(ValueError):
            etp.Implications(bytes(8), n=3)


class EtpTest(unittest.TestCase):
    def test_implication_of_an_answer(self):
        rows = [[T] * 5 for _ in range(5)]
        rows[3][4] = F  # commutativity does not imply associativity
        oracle = etp.Etp(etp.LawIndex(LAWS), etp.Implications(table(rows), n=5))
        commutative = parse_equation("x ◇ y = y ◇ x")
        associative = parse_equation("x ◇ (y ◇ z) = (x ◇ y) ◇ z")
        outside = parse_equation("x = y ◇ (y ◇ (y ◇ (y ◇ (y ◇ x))))")

        self.assertEqual(oracle.implication(commutative, associative), (4, 5, False))
        self.assertEqual(oracle.implication(commutative, commutative), (4, 4, True))
        self.assertEqual(oracle.implication(commutative, outside), (4, None, None))


if __name__ == "__main__":
    unittest.main()
