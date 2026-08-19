#!/usr/bin/env python3
"""Tests for truthdata, the outcomes matrix and truth-balanced sampler.

All synthetic and offline: a fabricated payload exercises encoding and
validation, a small equation list with a hand-built matrix exercises the
sampler's invariants (50/50 truth split per bin, no vacuous laws, no
diagonal, proof-only statuses, determinism, loud failure on unfillable
quotas). The network download path is exercised only by the CLI.
"""

import tempfile
import unittest
from pathlib import Path

from truthdata import (
    STATUSES,
    TruthMatrix,
    _encode_payload,
    sample_truth_balanced,
    truth_availability,
)

# Six one-op laws (numbers 3-8) behind the two vacuous laws, mirroring
# the ETP list's opening. Bin 2 pairs draw both laws from this group.
EQUATIONS = [
    "x = x",
    "x = y",
    "x = x ◇ x",
    "x = x ◇ y",
    "x = y ◇ x",
    "x = y ◇ y",
    "x ◇ y = x",
    "x ◇ y = y",
]

TRUE_PAIRS = {(3, 4), (4, 3), (5, 6), (6, 5), (7, 8), (3, 6)}


def build_matrix() -> TruthMatrix:
    """A consistent synthetic matrix over EQUATIONS.

    Diagonal, the E1 column, and the E2 row are true (as in the real
    data); TRUE_PAIRS are proof-true; one pair is left unproved to check
    it is never sampled; everything else is proof-false.
    """
    n = len(EQUATIONS)
    names = [f"Equation{i + 1}" for i in range(n)]
    outcomes = []
    for e in range(1, n + 1):
        row = []
        for f in range(1, n + 1):
            if e == f or f == 1 or e == 2:
                row.append("implicit_proof_true")
            elif (e, f) in TRUE_PAIRS:
                row.append("explicit_proof_true")
            elif (e, f) == (4, 5):
                row.append("explicit_conjecture_false")
            else:
                row.append("implicit_proof_false")
        outcomes.append(row)
    matrix, size, counts = _encode_payload({"equations": names, "outcomes": outcomes})
    return TruthMatrix(matrix, size, {"n": size, "legend": list(STATUSES), "status_counts": counts})


class EncodeTest(unittest.TestCase):
    def test_status_and_truth_lookup(self):
        matrix = build_matrix()
        self.assertEqual(matrix.status(3, 4), "explicit_proof_true")
        self.assertIs(matrix.truth(3, 4), True)
        self.assertIs(matrix.truth(4, 6), False)
        self.assertIsNone(matrix.truth(4, 5))  # conjecture, not proof

    def test_misnamed_equation_rejected(self):
        with self.assertRaises(SystemExit):
            _encode_payload({"equations": ["EquationX"], "outcomes": [["unknown"]]})

    def test_ragged_row_rejected(self):
        with self.assertRaises(SystemExit):
            _encode_payload(
                {"equations": ["Equation1"], "outcomes": [["unknown", "unknown"]]}
            )

    def test_unknown_status_rejected(self):
        with self.assertRaises(SystemExit):
            _encode_payload({"equations": ["Equation1"], "outcomes": [["proved"]]})


class RoundTripTest(unittest.TestCase):
    def test_save_and_load_identical(self):
        matrix = build_matrix()
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "outcomes.bin"
            meta_path = Path(tmp) / "outcomes.meta.json"
            matrix.save(matrix_path, meta_path)
            loaded = TruthMatrix.load(matrix_path, meta_path)
        self.assertEqual(loaded.matrix, matrix.matrix)
        self.assertEqual(loaded.n, matrix.n)
        for e in range(1, matrix.n + 1):
            for f in range(1, matrix.n + 1):
                self.assertEqual(loaded.status(e, f), matrix.status(e, f))


class AvailabilityTest(unittest.TestCase):
    def test_counts_the_sampled_population(self):
        table = truth_availability(EQUATIONS, build_matrix(), bins=(2,))
        # 6 one-op laws -> 30 ordered pairs; E1/E2 and the diagonal are
        # structurally outside the population.
        counts = table[2]
        self.assertEqual(counts["true"], len(TRUE_PAIRS))
        self.assertEqual(counts["excluded"], 1)  # the (4, 5) conjecture
        self.assertEqual(sum(counts.values()), 30)


class SamplerTest(unittest.TestCase):
    def sample(self, per_bin=4, seed=0):
        return sample_truth_balanced(
            EQUATIONS, build_matrix(), per_bin=per_bin, seed=seed, bins=(2,)
        )

    def test_balanced_and_clean(self):
        pairs = self.sample()
        self.assertEqual(len(pairs), 4)
        self.assertEqual(sum(p["truth"] for p in pairs), 2)
        for pair in pairs:
            self.assertNotIn(pair["e_num"], (1, 2))
            self.assertNotIn(pair["f_num"], (1, 2))
            self.assertNotEqual(pair["e_num"], pair["f_num"])
            self.assertIn("proof", pair["status"])
            self.assertEqual(pair["ops_total"], 2)
            self.assertEqual((pair["ops_e"], pair["ops_f"], pair["depth"]), (1, 1, 1))
        self.assertEqual(len({(p["e_num"], p["f_num"]) for p in pairs}), 4)

    def test_truth_labels_match_the_matrix(self):
        matrix = build_matrix()
        for pair in self.sample():
            self.assertIs(pair["truth"], matrix.truth(pair["e_num"], pair["f_num"]))

    def test_deterministic_in_seed(self):
        self.assertEqual(self.sample(seed=7), self.sample(seed=7))
        self.assertNotEqual(self.sample(seed=0), self.sample(seed=1))

    def test_odd_per_bin_rejected(self):
        with self.assertRaises(SystemExit):
            self.sample(per_bin=5)

    def test_unfillable_true_quota_fails_loudly(self):
        with self.assertRaises(SystemExit) as caught:
            self.sample(per_bin=2 * len(TRUE_PAIRS) + 2)
        self.assertIn("proof-true", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
