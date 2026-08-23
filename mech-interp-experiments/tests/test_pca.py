#!/usr/bin/env python3
"""Tests for the frozen-frame behaviour that the whole design depends on."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np

    from shared import pca
except ImportError:  # numpy is only needed by the analysis half of the project
    np = None


@unittest.skipIf(np is None, "numpy is not installed")
class BasisTest(unittest.TestCase):
    def cloud(self, n=60, d=8, seed=0, shift=0.0):
        rng = np.random.default_rng(seed)
        base = rng.normal(size=(n, 3)) @ rng.normal(size=(3, d))
        return base + shift

    def test_axes_are_orthonormal_and_ordered(self):
        basis = pca.fit(self.cloud(), k=3)
        self.assertEqual(basis.axes.shape[0], 3)
        np.testing.assert_allclose(
            basis.axes @ basis.axes.T, np.eye(3), atol=1e-9
        )
        shares = basis.explained
        self.assertTrue(all(a >= b for a, b in zip(shares, shares[1:])))

    def test_projecting_the_fit_set_is_centred_on_the_origin(self):
        vectors = self.cloud()
        basis = pca.fit(vectors, k=2)
        coords = basis.project(vectors)
        np.testing.assert_allclose(coords.mean(axis=0), np.zeros(2), atol=1e-9)

    def test_project_uses_the_basis_centre_not_the_new_group_mean(self):
        """The point of a frozen frame: a displaced group must look displaced.

        Re-centring on the incoming group's own mean would move every group to
        the origin and destroy exactly the comparison this is for.
        """
        vectors = self.cloud()
        basis = pca.fit(vectors, k=2)
        moved = vectors + 5.0 * basis.axes[0]
        coords = basis.project(moved)
        self.assertAlmostEqual(float(coords[:, 0].mean()), 5.0, places=6)

    def test_a_projected_group_does_not_move_the_axes(self):
        """Fitting without an outlying group is what keeps it off the axes."""
        core = self.cloud(seed=1)
        outlier = self.cloud(n=60, seed=2) * 40.0 + 500.0

        without = pca.fit(core, k=2)
        with_it = pca.fit(np.vstack([core, outlier]), k=2)

        # The outlier dominates a fit it is included in, and not one it is not.
        alignment = abs(float(without.axes[0] @ with_it.axes[0]))
        self.assertLess(alignment, 0.99)
        # It still gets coordinates in the frame it had no vote in.
        self.assertEqual(without.project(outlier).shape, (60, 2))

    def test_residual_share_is_zero_when_the_axes_span_the_data(self):
        rng = np.random.default_rng(3)
        flat = rng.normal(size=(40, 2)) @ rng.normal(size=(2, 9))
        basis = pca.fit(flat, k=2)
        self.assertAlmostEqual(basis.residual_share(flat), 0.0, places=9)

    def test_residual_share_grows_when_a_group_leaves_the_frame(self):
        rng = np.random.default_rng(4)
        flat = rng.normal(size=(40, 2)) @ rng.normal(size=(2, 9))
        basis = pca.fit(flat, k=2)
        escaped = flat + rng.normal(size=(40, 9)) * 3.0
        self.assertGreater(basis.residual_share(escaped), 0.3)

    def test_center_within_groups_removes_each_group_mean(self):
        rng = np.random.default_rng(5)
        vectors = np.vstack([rng.normal(size=(10, 4)) + 8.0,
                             rng.normal(size=(10, 4)) - 8.0])
        groups = ["a"] * 10 + ["b"] * 10
        centred = pca.center_within_groups(vectors, groups)
        np.testing.assert_allclose(centred[:10].mean(axis=0), np.zeros(4), atol=1e-9)
        np.testing.assert_allclose(centred[10:].mean(axis=0), np.zeros(4), atol=1e-9)

    def test_center_within_groups_rejects_a_length_mismatch(self):
        with self.assertRaises(ValueError):
            pca.center_within_groups(np.zeros((4, 2)), ["a", "b"])

    def test_spread_ratio_is_small_for_tight_groups(self):
        rng = np.random.default_rng(6)
        centres = rng.normal(size=(12, 5)) * 20.0
        vectors = np.vstack([c + rng.normal(size=(3, 5)) * 0.1 for c in centres])
        groups = [i // 3 for i in range(36)]
        self.assertLess(pca.spread_ratio(vectors, groups), 0.05)

    def test_spread_ratio_is_near_one_when_the_grouping_means_nothing(self):
        rng = np.random.default_rng(7)
        vectors = rng.normal(size=(60, 5))
        groups = [i % 20 for i in range(60)]
        self.assertGreater(pca.spread_ratio(vectors, groups), 0.6)

    def test_centroids_average_within_each_group(self):
        coords = np.array([[0.0, 0.0], [2.0, 4.0], [10.0, 10.0]])
        middles = pca.centroids(coords, ["a", "a", "b"])
        np.testing.assert_allclose(middles["a"], [1.0, 2.0])
        np.testing.assert_allclose(middles["b"], [10.0, 10.0])

    def test_a_basis_survives_a_round_trip_through_json(self):
        basis = pca.fit(self.cloud(), k=2, labels=["graft", "literal"])
        restored = pca.Basis.from_json(basis.to_json())
        vectors = self.cloud(seed=9)
        np.testing.assert_allclose(
            basis.project(vectors), restored.project(vectors), atol=1e-12
        )
        self.assertEqual(restored.labels_fit, ["graft", "literal"])

    def test_fit_refuses_a_single_vector(self):
        with self.assertRaises(ValueError):
            pca.fit(np.zeros((1, 4)), k=2)


if __name__ == "__main__":
    unittest.main()
