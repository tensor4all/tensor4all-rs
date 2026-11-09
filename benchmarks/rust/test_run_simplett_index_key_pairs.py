"""Regression tests for the noise, thermal and evidence decisions in the paired driver."""

import gzip
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

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


class EvidenceArchiveTests(unittest.TestCase):
    def test_incomplete_run_set_is_archived_before_the_cell_set_check(self):
        records = runs(100)
        missing = sorted(records[-1]["cells"])[0]
        records[-1]["cells"].pop(missing)
        pending = list(records)

        def fake_build_arm(repo, arm, rev, harness_bytes, work, target_dir):
            for path in (work / "src", work / "target", work / "bin"):
                path.mkdir(parents=True, exist_ok=True)
            binary = work / "bin" / arm
            binary.write_text(arm)
            return {
                "commit": rev, "harness_sha256": driver.HARNESS_SHA256, "harness_overlaid": False,
                "binary_sha256": f"binary-{arm}", "rustc": "rustc", "cargo": "cargo",
            }, binary

        def fake_idle_check(seconds, max_load):
            observation = driver.observe()
            return {"seconds": seconds, "maximum_load_1m": max_load,
                    "before": observation, "after": observation, "passed": True}

        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "run"
            with mock.patch.object(driver, "build_arm", fake_build_arm), \
                    mock.patch.object(driver, "run_arm", lambda *a: pending.pop(0)), \
                    mock.patch.object(driver, "idle_check", fake_idle_check), \
                    mock.patch.object(sys, "argv", ["driver", "--output", str(out)]):
                with self.assertRaises(SystemExit) as caught:
                    driver.main()
            self.assertTrue(caught.exception.code)
            self.assertIn("experiment.json.gz", str(caught.exception.code))
            self.assertFalse((out / "summary.json").exists(), "a rejected experiment must not summarize")
            archived = json.loads(gzip.open(out / "experiment.json.gz", "rt").read())

        self.assertEqual(len(archived["runs"]), len(records))
        self.assertNotIn(missing, archived["runs"][-1]["cells"])
        self.assertEqual(archived["runs"][0]["before"]["loadavg"], [0.0, 0.0, 0.0])
        self.assertIn("CPU_Speed_Limit = 100", archived["runs"][0]["after"]["thermal"])


if __name__ == "__main__":
    unittest.main()
