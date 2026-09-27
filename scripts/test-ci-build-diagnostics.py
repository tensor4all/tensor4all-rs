#!/usr/bin/env python3
"""Check diagnostic timing boundaries and propagation of build failures."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class BuildDiagnosticsTests(unittest.TestCase):
    def test_failure_and_build_boundary_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(Path(__file__).with_name("ci-build-diagnostics.py")),
                       "probe", "--", sys.executable, "-c",
                       "print('Fresh dependency v1'); print('Compiling workspace v1'); "
                       "print('Finished release target(s) in 1s'); "
                       "print('--crate-name book_tests --extern core=/exact/core.rlib'); "
                       "raise SystemExit(7)"]
            result = subprocess.run(command, env={**os.environ, "CI_BUILD_DIAGNOSTICS": directory},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 7)
            summary = json.loads((Path(directory) / "probe.json").read_text())
            self.assertEqual(summary["returncode"], 7)
            self.assertEqual(summary["compiled"], ["Compiling workspace v1"])
            self.assertEqual(summary["fresh"], ["Fresh dependency v1"])
            self.assertAlmostEqual(summary["elapsed_seconds"],
                                   summary["build_finished_seconds"] + summary["post_build_seconds"])
            self.assertIn("--extern core=/exact/core.rlib", (Path(directory) / "probe.log").read_text())

    def test_missing_build_marker_and_bounded_fingerprint_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(Path(__file__).with_name("ci-build-diagnostics.py")),
                       "probe", "--", sys.executable, "-c",
                       "[print(f'fingerprint diagnostic {i}') for i in range(80)]"]
            result = subprocess.run(command, env={**os.environ, "CI_BUILD_DIAGNOSTICS": directory},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            summary = json.loads((Path(directory) / "probe.json").read_text())
            self.assertIsNone(summary["build_finished_seconds"])
            self.assertIsNone(summary["post_build_seconds"])
            self.assertEqual(len(summary["fingerprints_first_60"]), 60)
            self.assertIn("fingerprint diagnostic 79", (Path(directory) / "probe.log").read_text())


if __name__ == "__main__":
    unittest.main()
