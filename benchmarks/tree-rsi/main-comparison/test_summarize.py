import unittest

import numpy as np

import summarize


def observation(error, *, case="case-A", seed=41, block=3):
    return {
        "algorithm": "rsi",
        "status": "completed",
        "case": case,
        "seed": seed,
        "block": block,
        "relative_l2": error,
        "output_ranks": [[0, 1, 1]],
        "diagnostics": {
            "edges": [
                {
                    "child": 0,
                    "parent": 1,
                    "rank": 1,
                    "relative_pivot": 0.004,
                    "exact_columns": False,
                    "rows": 2,
                    "columns": 2,
                }
            ]
        },
    }


def two_node_manifest():
    return {
        "n": 2,
        "edges": [[0, 1, 2]],
        "operand_edges": [[[0, 1, 1]], [[0, 1, 1]]],
    }


class SummaryTests(unittest.TestCase):
    def test_empty_and_partial_groups_report_attempts_without_fake_medians(self):
        _, empty = summarize.summarize_measurements([], seeds=[7], blocks=3)
        self.assertEqual((empty["attempts"], empty["completed"], empty["failed"]), (0, 0, 0))
        self.assertIsNone(empty["seconds_median"])
        self.assertEqual(empty["actual_max_bond_ranks"], [])
        self.assertFalse(empty["timing_stable"])

        calls = [
            {"status": "completed", "seed": 7, "seconds": 0.25, "relative_l2": 1e-10, "output_ranks": [[0, 1, 2]]},
            {"status": "timeout", "seed": 7},
            {"status": "error", "seed": 7},
        ]
        complete, partial = summarize.summarize_measurements(calls, seeds=[7], blocks=3)
        self.assertEqual(len(complete), 1)
        self.assertEqual((partial["attempts"], partial["completed"], partial["failed"]), (3, 1, 2))
        self.assertEqual(partial["seconds_median"], 0.25)
        self.assertIsNone(partial["repeat_cv_by_seed"][0])
        self.assertFalse(partial["timing_stable"])

    def test_failure_focus_and_cut_lower_bound_are_derived_from_the_observation(self):
        result = summarize.diagnose_rsi_failure(
            [observation(0.8, seed=41, block=3)],
            {"case-A": np.array([1.0, 0.0, 0.0, 1.0])},
            {"case-A": two_node_manifest()},
            {"case-A": 2},
        )
        self.assertEqual(result["status"], "observed_cap_sufficient_rsi_accuracy_failure")
        self.assertEqual((result["case"], result["seed"], result["block"]), ("case-A", 41, 3))
        self.assertEqual(result["focus_edge"], [0, 1])
        self.assertEqual(result["observed_edge_rank"], 1)
        self.assertEqual(result["reference_cut_rank_1e12"], 2)
        self.assertAlmostEqual(result["observed_rank_cut_relative_l2_lower_bound"], 1 / np.sqrt(2))
        self.assertIn("does not establish causality", result["local_edge_note"])

    def test_no_failure_has_an_explicit_empty_status(self):
        result = summarize.diagnose_rsi_failure(
            [observation(0.0)],
            {"case-A": np.array([1.0, 0.0, 0.0, 1.0])},
            {"case-A": two_node_manifest()},
            {"case-A": 2},
        )
        self.assertEqual(result["status"], "no_observed_cap_sufficient_rsi_accuracy_failure")

    def test_disconnected_fixture_is_rejected(self):
        manifest = two_node_manifest()
        manifest["n"] = 4
        manifest["edges"] = [[0, 1, 2], [1, 2, 2], [2, 0, 2]]
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            summarize.structural_rank_bounds(manifest)

    def test_saved_error_cannot_be_smaller_than_the_cut_rank_lower_bound(self):
        with self.assertRaisesRegex(RuntimeError, "contradicts"):
            summarize.diagnose_rsi_failure(
                [observation(0.1)],
                {"case-A": np.array([1.0, 0.0, 0.0, 1.0])},
                {"case-A": two_node_manifest()},
                {"case-A": 2},
            )


if __name__ == "__main__":
    unittest.main()
