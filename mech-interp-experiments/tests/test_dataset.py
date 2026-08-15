#!/usr/bin/env python3
"""Tests for shared.dataset.

The contract worth protecting: inside a family of cells the rendered text is
the same length at every depth. That is the whole reason a cell is keyed on the
side split and the variable count rather than on depth alone, so the length
assertion is the test that would actually catch a regression.

    cd mech-interp-experiments && python3 -m unittest discover -s tests -t .
"""

import unittest

from shared.dataset import (
    build_pool,
    equation_shape,
    index_by_shape,
    orderable,
    parse_cells,
    render_all_forms,
    sample_depth_balanced,
)
from literalform import render_description
from storyform import render_story


class ShapeTest(unittest.TestCase):
    def test_split_is_canonical_and_depth_is_the_deeper_side(self):
        # The same cell but for depth: three operations all on the right, a
        # bare left side, two variables. The first is balanced, the second a
        # spine where each step consumes the previous result.
        self.assertEqual(equation_shape("x = (x ◇ y) ◇ (y ◇ x)"), (0, 3, 2, 2))
        self.assertEqual(equation_shape("x = ((x ◇ y) ◇ y) ◇ y"), (0, 3, 2, 3))

    def test_unparseable_equations_are_skipped(self):
        self.assertIsNone(equation_shape("not an equation"))


class DepthSamplingTest(unittest.TestCase):
    PER_CELL = 3
    FAMILY = (0, 7, 4)  # bare minor side, seven-operation major side, four variables

    @classmethod
    def setUpClass(cls):
        cls.pool = build_pool(per_bin=800)
        index = index_by_shape(cls.pool)
        cls.cells = tuple(
            sorted(
                shape
                for shape, laws in index.items()
                if shape[:3] == cls.FAMILY and orderable(laws) >= cls.PER_CELL
            )
        )

    def sample(self, seed=0, cells=None):
        return sample_depth_balanced(
            self.pool, self.PER_CELL, seed, cells=cells or self.cells
        )

    def test_family_spans_several_depths(self):
        self.assertGreater(
            len({cell[3] for cell in self.cells}), 1, "no depth contrast to test"
        )

    def test_every_pair_matches_its_cell(self):
        for sample in self.sample():
            minor, major, variables, depth = sample["shape"]
            self.assertEqual((minor, major, variables), self.FAMILY)
            self.assertEqual(sample["ops_total"], 2 * (minor + major))
            self.assertEqual(sample["depth"], depth)

    def test_length_is_constant_across_depth(self):
        # Per surface form: the themes differ in wording length, and the
        # default theme is a hash of the pair, so the theme is pinned here.
        for render in (
            lambda meta: render_story(
                meta["equation_e"], meta["equation_f"], theme_key="graft"
            )[0],
            lambda meta: render_description(
                meta["equation_e"], meta["equation_f"]
            )[0],
        ):
            lengths = {
                len(render(sample["metadata"]).split()) for sample in self.sample()
            }
            self.assertEqual(len(lengths), 1, "lengths differ: {}".format(sorted(lengths)))

    def test_length_is_constant_in_all_six_surface_forms(self):
        lengths = {}
        for sample in self.sample():
            for form, text in render_all_forms(sample["metadata"]).items():
                lengths.setdefault(form, set()).add(len(text.split()))
        for form, counts in sorted(lengths.items()):
            self.assertEqual(
                len(counts), 1, "{} varies in length: {}".format(form, sorted(counts))
            )

    def test_labels_do_not_look_like_etp_numbering(self):
        for sample in self.sample():
            self.assertTrue(sample["pair_id"].startswith("G"))

    def test_same_seed_same_pairs(self):
        ids = [sample["pair_id"] for sample in self.sample()]
        self.assertEqual(ids, [sample["pair_id"] for sample in self.sample()])
        self.assertNotEqual(
            ids, [sample["pair_id"] for sample in self.sample(seed=1)]
        )

    def test_unfillable_cell_is_refused(self):
        with self.assertRaises(SystemExit):
            self.sample(cells=((0, 7, 4, 99),))


class ParseCellsTest(unittest.TestCase):
    def test_range_expands_within_a_family(self):
        self.assertEqual(
            parse_cells("0:7:4:4-6"), ((0, 7, 4, 4), (0, 7, 4, 5), (0, 7, 4, 6))
        )

    def test_duplicates_collapse(self):
        self.assertEqual(parse_cells("0:7:4:5,0:7:4:5"), ((0, 7, 4, 5),))

    def test_malformed_specs_are_refused(self):
        for text in ("0:7:4", "0:7:4:", "a:7:4:5", "0:7:4:6-4", "7:0:4:5", ""):
            with self.assertRaises(SystemExit):
                parse_cells(text)


if __name__ == "__main__":
    unittest.main()
