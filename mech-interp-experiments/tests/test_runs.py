#!/usr/bin/env python3
"""Tests for shared.runs.

The property that matters: a condition interrupted part-way through must not
leave anything a later run would mistake for finished work.

    cd mech-interp-experiments && python3 -m unittest discover -s tests -t .
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from shared import runs


class ConditionTest(unittest.TestCase):
    def test_settings_are_readable_and_serializable(self):
        condition = runs.Condition("depth-4-B256", depth=4, budget=256)
        self.assertEqual(condition["depth"], 4)
        self.assertEqual(condition.get("missing", "default"), "default")
        self.assertEqual(
            condition.as_dict(), {"name": "depth-4-B256", "depth": 4, "budget": 256}
        )

    def test_names_that_are_not_filename_safe_are_refused(self):
        for name in ("", "with space", "with/slash"):
            with self.assertRaises(ValueError):
                runs.Condition(name)


class RunDirectoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.meta = {"model": "test-model", "seed": 0, "budgets": [0, 256]}
        self.run = runs.RunDirectory.open(self.tmp, "demo", self.meta)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_records_are_published_on_success(self):
        with self.run.writing("cell-a") as writer:
            writer.write({"pair_id": "p1", "status": "correct"})
            writer.write({"pair_id": "p2", "status": "wrong"})
        self.assertTrue(self.run.has("cell-a"))
        self.assertEqual(len(self.run.read("cell-a")), 2)

    def test_an_interrupted_condition_is_not_published(self):
        with self.assertRaises(RuntimeError):
            with self.run.writing("cell-a") as writer:
                writer.write({"pair_id": "p1", "status": "correct"})
                raise RuntimeError("connection dropped")
        self.assertFalse(self.run.has("cell-a"))
        with self.assertRaises(FileNotFoundError):
            self.run.read("cell-a")

    def test_a_rerun_after_an_interruption_starts_that_condition_over(self):
        conditions = [runs.Condition("cell-a"), runs.Condition("cell-b")]
        with self.run.writing("cell-a") as writer:
            writer.write({"pair_id": "p1", "status": "correct"})
        self.assertEqual(
            [c.name for c in runs.pending(self.run, conditions)], ["cell-b"]
        )
        self.assertEqual(
            [c.name for c in runs.pending(self.run, conditions, force=True)],
            ["cell-a", "cell-b"],
        )

    def test_provenance_records_both_code_versions(self):
        meta = self.run.read_meta()
        self.assertEqual(meta["model"], "test-model")
        self.assertEqual(meta["seed"], 0)
        for field in ("repo_commit", "informalizing_etp_tree", "python", "created"):
            self.assertIsNotNone(meta.get(field), field)

    def test_resuming_with_changed_settings_is_refused(self):
        with self.assertRaises(SystemExit):
            runs.RunDirectory.open(self.tmp, "demo", dict(self.meta, seed=1))

    def test_forcing_overwrites_the_recorded_settings(self):
        again = runs.RunDirectory.open(
            self.tmp, "demo", dict(self.meta, seed=1), force=True
        )
        self.assertEqual(again.read_meta()["seed"], 1)

    def test_resuming_from_a_different_machine_is_allowed(self):
        # Provenance fields describing the box, not the experiment, must not
        # block a resume: runs move between a laptop and a rented GPU.
        meta_path = self.run.path / "run_meta.json"
        stored = json.loads(meta_path.read_text())
        stored["hostname"] = "some-other-box"
        stored["platform"] = "Linux-6.0"
        meta_path.write_text(json.dumps(stored))
        runs.RunDirectory.open(self.tmp, "demo", self.meta)  # must not raise

    def test_summary_round_trips(self):
        self.run.write_summary({"accuracy": 0.5})
        self.assertEqual(self.run.read_summary(), {"accuracy": 0.5})

    def test_reported_rates_survive_serialization(self):
        from shared.stats import rate

        self.run.write_summary({"accuracy": rate(4, 8)})
        stored = self.run.read_summary()["accuracy"]
        self.assertEqual(stored["successes"], 4)
        self.assertEqual(stored["total"], 8)
        self.assertEqual(len(stored["ci95"]), 2)

    def test_figures_live_beside_the_run_not_inside_it(self):
        path = self.run.figure_path("plot.png")
        self.assertEqual(path.parent.name, "figures")
        self.assertNotIn("runs", path.parent.parts)


if __name__ == "__main__":
    unittest.main()
