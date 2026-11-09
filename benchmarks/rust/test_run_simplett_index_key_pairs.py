"""Regression tests for the noise and thermal decision in the paired driver."""

import unittest

import run_simplett_index_key_pairs as driver

CASES = [
    f"n{sites}_chi{rank}_batch{batch}"
    for sites, rank, batch in [
        (32, 1, 64),
        (100, 1, 64),
        (100, 1, 1024),
        (100, 8, 1024),
        (300, 1, 64),
        (300, 8, 64),
    ]
]
MEANS_NS = {"original": 100.0, "shared": 110.0, "final": 80.0}


def thermal(speed_limit):
    if speed_limit is None:
        return "unavailable on this platform"
    return f"Note: No thermal warning level has been recorded\nCPU_Speed_Limit = {speed_limit}\n"


def runs(speed_limit):
    """Nine run records: 30 cells x three arms x three rounds, all noise checks passing."""
    records = []
    for round_no in (1, 2, 3):
        for arm, mean_ns in MEANS_NS.items():
            observation = {"loadavg": [0.0, 0.0, 0.0], "thermal": thermal(speed_limit)}
            cells = {}
            for route in driver.ROUTES:
                for case in CASES:
                    cells[f"cache_keys/{route}/{case}"] = {
                        "mean_ns": mean_ns,
                        "ci95_low_ns": mean_ns * 0.99,
                        "ci95_high_ns": mean_ns * 1.01,
                    }
            records.append({"round": round_no, "arm": arm, "before": observation, "after": observation, "cells": cells})
    return records


class ThermalGateTests(unittest.TestCase):
    def test_speed_limit_below_100_cannot_pass(self):
        summary = driver.summarize(runs(50))
        self.assertEqual(summary["decision"], "THROTTLED")
        # The host was otherwise quiet and fast enough: only throttling blocks the run.
        self.assertEqual(summary["noise"]["minimum_cpu_speed_limit"], 50)
        self.assertFalse(summary["noise_pass"]["thermal"])
        self.assertEqual({k: v for k, v in summary["noise_pass"].items() if k != "thermal"},
                         {"load": True, "ci_halfwidth": True, "cv": True})
        self.assertTrue(all(summary["timing_pass"].values()))

    def test_speed_limit_at_100_still_passes(self):
        summary = driver.summarize(runs(100))
        self.assertEqual(summary["decision"], "PASS")
        self.assertEqual(summary["noise"]["minimum_cpu_speed_limit"], 100)
        self.assertTrue(summary["noise_pass"]["thermal"])

    def test_unavailable_thermal_observation_keeps_the_previous_decision(self):
        summary = driver.summarize(runs(None))
        self.assertEqual(summary["decision"], "PASS")
        self.assertIsNone(summary["noise"]["minimum_cpu_speed_limit"])
        self.assertTrue(summary["noise_pass"]["thermal"])


if __name__ == "__main__":
    unittest.main()
