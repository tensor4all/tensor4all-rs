#!/usr/bin/env python3
"""Unit tests for scripts/check-test-targets.py."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "check_test_targets", Path(__file__).with_name("check-test-targets.py")
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class UndeclaredTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tests = Path(self._tmp.name) / "tests"
        self.tests.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, rel: str, text: str = "") -> Path:
        path = self.tests / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_declared_modules_and_separate_targets_pass(self) -> None:
        main = self.write("main.rs", "mod alpha;\nmod suite;\n")
        self.write("alpha.rs")
        self.write("suite/mod.rs")
        separate = self.write("global_state.rs")
        self.write("support/helper.rs")
        self.assertEqual(MODULE.undeclared(self.tests, {main, separate}), [])

    def test_new_file_without_declaration_fails(self) -> None:
        main = self.write("main.rs", "mod alpha;\n")
        self.write("alpha.rs")
        forgotten = self.write("beta.rs")
        self.write("other_suite/main.rs")
        self.assertEqual(
            MODULE.undeclared(self.tests, {main}),
            [forgotten, self.tests / "other_suite"],
        )

    def test_autodiscovered_targets_pass(self) -> None:
        a = self.write("a.rs")
        b = self.write("suite/main.rs")
        self.assertEqual(MODULE.undeclared(self.tests, {a, b}), [])


class HeavyTargetTests(unittest.TestCase):
    def test_heavy_file_must_be_a_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tests = Path(tmp) / "tests"
            tests.mkdir()
            heavy = tests / "heavy_slow.rs"
            heavy.write_text("")
            main = tests / "main.rs"
            main.write_text("")
            self.assertEqual(
                MODULE.heavy_targets(tests, {main, heavy}), [heavy.resolve()]
            )
            self.assertEqual(MODULE.heavy_targets(tests, {main}), [])


if __name__ == "__main__":
    sys.exit(unittest.main())
