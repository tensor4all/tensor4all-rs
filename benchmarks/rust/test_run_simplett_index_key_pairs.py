"""Regression tests for the noise, thermal and evidence decisions in the paired driver."""

import contextlib
import gzip
import hashlib
import json
import pathlib
import subprocess
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


# The tests stub the git and host seams, so they need neither this checkout's history
# nor this machine's load, and the harness blob can differ from the recorded one.
HARNESS_BYTES = b"harness source"
OBSERVATION = {"time": 0.0, "loadavg": [0.0, 0.0, 0.0], "thermal": "CPU_Speed_Limit = 100"}
HOST = {"platform": "test", "machine": "test", "logical_cpus": 1, "python": "3"}
REAL_RUN = subprocess.run


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


def fake_subprocess_run(cmd, **kwargs):
    """Answer the driver's `git cat-file`/`git show` probes from constants."""
    if cmd[0] == "git":
        blob = HARNESS_BYTES if cmd[1] == "show" else b""
        return subprocess.CompletedProcess(cmd, 0, stdout=blob, stderr=b"")
    return REAL_RUN(cmd, **kwargs)


def fake_build_arm(repo, arm, rev, harness_bytes, work, target_dir, failing=None):
    for path in (work / "src", work / "target", work / "bin"):
        path.mkdir(parents=True, exist_ok=True)
    binary = work / "bin" / arm
    exit_code = 3 if arm == failing else 0
    binary.write_text(f"#!/bin/sh\nexit {exit_code}\n")
    binary.chmod(0o755)
    return {
        "commit": rev, "harness_sha256": driver.HARNESS_SHA256, "harness_overlaid": False,
        "binary_sha256": f"binary-{arm}", "rustc": "rustc", "cargo": "cargo",
    }, binary


def fake_idle_check(seconds, max_load):
    return {"seconds": seconds, "maximum_load_1m": max_load,
            "before": OBSERVATION, "after": OBSERVATION, "passed": True}


def fake_read_cells(_criterion_home, baseline):
    """One cell per arm, whose values differ per arm; the fake executable writes no Criterion files."""
    arm = baseline.rsplit("-r", 1)[0]
    mean_ns = MEANS_NS[arm]
    return {f"cache_keys/{arm}": {"mean_ns": mean_ns, "ci95_low_ns": mean_ns * 0.99,
                                   "ci95_high_ns": mean_ns * 1.01, "samples": {"arm": arm}}}


def run_main(tmp, build_arm=fake_build_arm, **overrides):
    """Drive `main` against a temporary output directory with the seams stubbed."""
    out = pathlib.Path(tmp) / "run"
    # `--repo` points at a directory without history: the git seam is stubbed, so
    # neither a shallow checkout nor a different host can fail this test.
    argv = ["driver", "--output", str(out), "--repo", str(pathlib.Path(tmp) / "no-checkout")]
    patches = {
        "HARNESS_SHA256": hashlib.sha256(HARNESS_BYTES).hexdigest(),
        "build_arm": build_arm,
        "idle_check": fake_idle_check,
        "observe": lambda: OBSERVATION,
        "host_description": lambda: HOST,
    }
    patches.update(overrides)
    with mock.patch.object(subprocess, "run", fake_subprocess_run), \
            mock.patch.object(sys, "argv", argv):
        with contextlib.ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(driver, name, value))
            try:
                driver.main()
            except SystemExit as err:
                return out, err
    raise AssertionError("main returned without refusing")


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

        def replay(_binary, _arm, _round, _out, _cpu, _filter, sink=None):
            """Stand in for `run_arm`, which appends its own record before it can fail."""
            record = pending.pop(0)
            sink.append(record)
            return record

        with tempfile.TemporaryDirectory() as tmp:
            out, err = run_main(tmp, run_arm=replay)
            self.assertTrue(err.code)
            self.assertIn("experiment.json.gz", str(err.code))
            self.assertFalse((out / "summary.json").exists(), "a rejected experiment must not summarize")
            archived = json.loads(gzip.open(out / "experiment.json.gz", "rt").read())

        self.assertEqual(len(archived["runs"]), len(records))
        self.assertNotIn(missing, archived["runs"][-1]["cells"])
        self.assertEqual(archived["runs"][0]["before"]["loadavg"], [0.0, 0.0, 0.0])
        self.assertIn("CPU_Speed_Limit = 100", archived["runs"][0]["after"]["thermal"])

    def test_failed_arm_is_archived_and_still_refused(self):
        # Two arms succeed, then the third one's own subprocess fails, so `run_arm`
        # never attaches cells for it; all three records must reach the archive.
        with tempfile.TemporaryDirectory() as tmp:
            out, err = run_main(tmp, build_arm=lambda *a: fake_build_arm(*a, failing="final"),
                                read_cells=fake_read_cells)
            self.assertEqual(str(err.code), "final-r1 exited with 3; see logs/final-r1.txt")
            self.assertTrue((out / "experiment.json.gz").exists(),
                            "a failed arm must keep its observations")
            self.assertFalse((out / "summary.json").exists(), "a failed arm must not summarize")
            archived = json.loads(gzip.open(out / "experiment.json.gz", "rt").read())

        self.assertEqual([(r["arm"], r["round"], r["exit_code"]) for r in archived["runs"]],
                         [("original", 1, 0), ("shared", 1, 0), ("final", 1, 3)])
        for index, arm in ((0, "original"), (1, "shared")):
            self.assertEqual(archived["runs"][index]["cells"], fake_read_cells(None, f"{arm}-r1"),
                             "a completed arm keeps its measured cells")
        self.assertNotIn("cells", archived["runs"][-1], "a failed arm has no cells to summarize")
        self.assertEqual(archived["runs"][-1]["after"]["loadavg"], [0.0, 0.0, 0.0])
        self.assertIn("CPU_Speed_Limit = 100", archived["runs"][-1]["before"]["thermal"])
        self.assertEqual(archived["protocol"]["host"], HOST)

    def test_arm_whose_cells_cannot_be_read_is_archived_with_its_observations(self):
        # The executable exits 0, so only the Criterion files can fail here: unreadable
        # cells must still cost the arm neither its load nor its thermal observations.
        def unreadable_cells(*_args):
            raise json.JSONDecodeError("Expecting value", "<criterion estimates>", 0)

        with tempfile.TemporaryDirectory() as tmp:
            out, err = run_main(tmp, read_cells=unreadable_cells)
            self.assertFalse((out / "summary.json").exists(), "unreadable cells must not summarize")
            archived = json.loads(gzip.open(out / "experiment.json.gz", "rt").read())

        self.assertIn("could not be read", str(err.code))
        self.assertEqual([(r["arm"], r["round"], r["exit_code"]) for r in archived["runs"]],
                         [("original", 1, 0)])
        failed = archived["runs"][0]
        self.assertNotIn("cells", failed, "unreadable cells must not be archived as measurements")
        self.assertIn("Expecting value", failed["cells_error"], "the archive says why the cells are missing")
        self.assertEqual(failed["after"]["loadavg"], [0.0, 0.0, 0.0])
        self.assertIn("CPU_Speed_Limit = 100", failed["before"]["thermal"])


if __name__ == "__main__":
    unittest.main()
