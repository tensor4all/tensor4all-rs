#!/usr/bin/env python3
"""Exercise resolved-package and feature leak detection, including opt-in graphs."""

from pathlib import Path
import runpy
import subprocess
import unittest
from unittest.mock import patch


CHECK = runpy.run_path(str(Path(__file__).with_name("check-m6-dependencies.py")))
VERIFY = CHECK["verify_tree"]
TREE = 'tensor4all-treetn v0.3.0 (/checkout/crates/treetn) features=[default,tenferro-cpu-platform]\n'
SIMPLETT = 'tensor4all-simplett v0.3.0 (/checkout/crates/simplett) features=[]\n'
BRIDGE = TREE.replace('default,', 'default,simplett-bridge,') + SIMPLETT


class GraphChecks(unittest.TestCase):
    def test_native_graph(self):
        VERIFY(TREE, ["tensor4all-treetn"])

    def test_missing_root_and_unrecognized_format_fail_closed(self):
        for tree in ("", "tensor4all-treetn v0.3.0\n"):
            with self.subTest(tree=tree), self.assertRaisesRegex(ValueError, "missing expected"):
                VERIFY(tree, ["tensor4all-treetn"])

    def test_unrelated_packages_are_rejected(self):
        for package in ("simplett", "tensorci", "aci", "quanticstransform",
                        "itensorlike", "interpolativeqtt"):
            tree = TREE + f'tensor4all-{package} v0.3.0 features=[]\n'
            with self.subTest(package=package), self.assertRaisesRegex(ValueError, "unexpected dependencies"):
                VERIFY(tree, ["tensor4all-treetn"])

    def test_bridge_feature_alone_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "simplett-bridge enabled=True"):
            VERIFY(BRIDGE.replace(SIMPLETT, ""), ["tensor4all-treetn"])

    def test_explicit_bridge_requires_feature_and_package(self):
        VERIFY(BRIDGE, ["tensor4all-treetn"], bridge=True)
        with self.assertRaisesRegex(ValueError, "expected=True"):
            VERIFY(TREE, ["tensor4all-treetn"], bridge=True)
        with self.assertRaisesRegex(ValueError, "did not resolve SimpleTT"):
            VERIFY(BRIDGE.replace(SIMPLETT, ""), ["tensor4all-treetn"], bridge=True)

    def test_legacy_comparisons_require_tci(self):
        with self.assertRaisesRegex(ValueError, "did not resolve TCI"):
            VERIFY(BRIDGE, ["tensor4all-treetn"], bridge=True, comparisons=True)
        VERIFY(BRIDGE + 'tensor4all-tensorci v0.3.0 features=[]\n',
               ["tensor4all-treetn"], bridge=True, comparisons=True)

    def test_fourier_legacy_feature_is_rejected_even_with_comparisons(self):
        tree = BRIDGE + 'tensor4all-tensorci v0.3.0 features=[]\n'
        tree += 'tensor4all-quanticstransform v0.3.0 features=[simplett-transforms]\n'
        with self.assertRaisesRegex(ValueError, "legacy simplett-transforms"):
            VERIFY(tree, ["tensor4all-treetn"], bridge=True, fourier=True, comparisons=True)

    def test_native_fourier_and_itensorlike_opt_in(self):
        tree = TREE + 'tensor4all-quanticstransform v0.3.0 features=[tenferro-cpu-platform]\n'
        tree += 'tensor4all-itensorlike v0.3.0 features=[default]\n'
        VERIFY(tree, ["tensor4all-treetn"], fourier=True, itensorlike=True)

    def test_cargo_failure_is_not_an_empty_success(self):
        with patch.object(subprocess, "check_output", side_effect=subprocess.CalledProcessError(1, "cargo")):
            with self.assertRaises(subprocess.CalledProcessError):
                CHECK["check_graph"]("resolution failure", Path("Cargo.toml"), ["tensor4all-treetn"])


if __name__ == "__main__":
    unittest.main()
