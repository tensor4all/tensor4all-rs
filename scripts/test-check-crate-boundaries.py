#!/usr/bin/env python3
"""Tests for the crate-boundary checker (tensor4all/tensor4all-rs#566 Task 9).

Fixture manifests must cover: the tensorbackend normal tenferro route passes;
a new feature crate with a normal tenferro dependency fails; a renamed
``package = "tenferro-*"`` dependency fails; a dev-only tenferro dependency is
allowed; an acyclic dev graph passes; a dev cycle fails with the full path; a stale exception tuple
fails; a listed feature-only self dev-dependency passes, while an unlisted one or
one whose sources name the crate's own extern path fails. Rule 4 (#1380): core
may name tensorbackend bridge items, another crate fails unless the site is
pinned, nested groups and crate aliases are seen while comments and scalar
vocabulary are not, a stale pin fails, and ``--update-baseline`` shrinks the pin
but refuses growth without ``--allow-growth``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("check-crate-boundaries.py")
PYTHON = sys.executable

TENSORBACKEND_MANIFEST = """\
[package]
name = "tensor4all-tensorbackend"
version = "0.1.0"
edition = "2021"

[dependencies]
tenferro-einsum.workspace = true
tenferro-tensor.workspace = true
"""

PLAIN_MANIFEST = """\
[package]
name = "tensor4all-demo"
version = "0.1.0"
edition = "2021"

[dependencies]
tensor4all-tensorbackend = { path = "../tensor4all-tensorbackend" }
"""

WORKSPACE_MANIFEST = """\
[workspace]
members = ["crates/*"]
resolver = "2"
"""


def write_fixture(files: dict[str, str]) -> Path:
    directory = tempfile.mkdtemp(prefix="crate-boundaries-")
    root = Path(directory)
    (root / "Cargo.toml").write_text(WORKSPACE_MANIFEST, encoding="utf-8")
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def run_check(
    root: Path, exceptions: str | None = "", extra_args: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    """Run the checker with an isolated exception map (test fixture).

    ``exceptions=None`` keeps the built-in map; ``""`` (default) uses no
    exceptions so fixture crates never trip the removed-crate stale check.
    """
    import os

    env = dict(os.environ)
    if exceptions is None:
        env.pop("T4A_TEST_TENFERRO_EXCEPTIONS", None)
    else:
        env["T4A_TEST_TENFERRO_EXCEPTIONS"] = exceptions
    return subprocess.run(
        [PYTHON, str(SCRIPT), "--root-dir", str(root), *extra_args],
        capture_output=True,
        text=True,
        env=env,
    )


class CrateBoundaryTests(unittest.TestCase):
    def test_tensorbackend_normal_tenferro_route_passes(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST,
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("crate-boundary-ok", result.stdout)

    def test_new_feature_crate_normal_tenferro_dependency_fails(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST
                + "tenferro-tensor.workspace = true\n",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tensor4all-demo", result.stderr)
        self.assertIn("tenferro-tensor", result.stderr)
        self.assertIn("not allowed", result.stderr)

    def test_renamed_package_tenferro_dependency_fails(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST
                + 'sneaky = { workspace = true, package = "tenferro-tensor" }\n',
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tenferro-tensor", result.stderr)

    def test_dev_only_tenferro_dependency_is_allowed(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST
                + "\n[dev-dependencies]\ntenferro-tensor.workspace = true\n",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_acyclic_dev_graph_passes(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST,
                "crates/tensor4all-demo2/Cargo.toml": """\
[package]
name = "tensor4all-demo2"
version = "0.1.0"
edition = "2021"

[dependencies]
tensor4all-demo = { path = "../tensor4all-demo" }

[dev-dependencies]
tensor4all-tensorbackend = { path = "../tensor4all-tensorbackend" }
""",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dev_cycle_fails_with_full_path(self) -> None:
        # The exact cycle the checker exists to reject: alpha dev-depends on
        # beta while beta normal-depends on alpha (generic fixture, not tied
        # to any specific crate).
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-alpha/Cargo.toml": """\
[package]
name = "tensor4all-alpha"
version = "0.1.0"
edition = "2021"

[dependencies]
tensor4all-tensorbackend = { path = "../tensor4all-tensorbackend" }

[dev-dependencies]
tensor4all-beta = { path = "../tensor4all-beta" }
""",
                "crates/tensor4all-beta/Cargo.toml": """\
[package]
name = "tensor4all-beta"
version = "0.1.0"
edition = "2021"

[dependencies]
tensor4all-alpha = { path = "../tensor4all-alpha" }
""",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tensor4all-alpha -> tensor4all-beta(dev)", result.stderr)
        self.assertIn("tensor4all-beta -> tensor4all-alpha(normal)", result.stderr)

    def _self_dev_fixture(self, name: str, source: str) -> Path:
        return write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                f"crates/{name}/Cargo.toml": f"""\
[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[dependencies]
tensor4all-tensorbackend = {{ path = "../tensor4all-tensorbackend" }}

[dev-dependencies]
{name} = {{ path = ".", features = ["global-defaults"] }}
""",
                f"crates/{name}/src/lib.rs": source,
            }
        )

    def test_listed_feature_only_self_dev_dependency_passes(self) -> None:
        root = self._self_dev_fixture(
            "tensor4all-core",
            "/// ```\n/// use tensor4all_core::Thing;\n/// ```\npub struct Thing;\n",
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unlisted_self_dev_dependency_fails(self) -> None:
        root = self._self_dev_fixture("tensor4all-demo", "pub struct Thing;\n")
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tensor4all-demo dev-depends on itself", result.stderr)

    def test_self_dev_dependency_with_own_extern_path_fails(self) -> None:
        # A unit test naming the crate by its extern path would meet a second
        # build of the crate: the hazard a dev cycle carries.
        root = self._self_dev_fixture(
            "tensor4all-core",
            "pub struct Thing;\n#[cfg(test)]\nmod tests {\n    use tensor4all_core::Thing;\n}\n",
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("src/lib.rs:4", result.stderr)
        self.assertIn("must not name `tensor4all_core::`", result.stderr)

    def test_dev_cycle_survives_coexisting_pure_normal_cycle(self) -> None:
        # A pure-normal cycle must not suppress a dev-containing cycle with
        # the same node set.
        root = write_fixture(
            {
                "crates/tensor4all-a/Cargo.toml": """\
[package]
name = "tensor4all-a"
version = "0.1.0"
edition = "2021"

[dependencies]
tensor4all-b = { path = "../tensor4all-b" }

[dev-dependencies]
tensor4all-b = { path = "../tensor4all-b" }
""",
                "crates/tensor4all-b/Cargo.toml": """\
[package]
name = "tensor4all-b"
version = "0.1.0"
edition = "2021"

[dependencies]
tensor4all-a = { path = "../tensor4all-a" }
""",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tensor4all-a -> tensor4all-b(dev)", result.stderr)

    def test_subtable_renamed_tenferro_dependency_fails(self) -> None:
        # ``[dependencies.name]`` sub-tables must not bypass the tenferro rule.
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": """\
[package]
name = "tensor4all-demo"
version = "0.1.0"
edition = "2021"

[dependencies.sneaky]
package = "tenferro-tensor"
workspace = true
""",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tenferro-tensor", result.stderr)
        self.assertIn("not allowed", result.stderr)

    def test_target_specific_tenferro_dependency_fails(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": """\
[package]
name = "tensor4all-demo"
version = "0.1.0"
edition = "2021"

[target.'cfg(unix)'.dependencies]
tenferro-tensor.workspace = true
""",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tenferro-tensor", result.stderr)
        self.assertIn("not allowed", result.stderr)

    def test_quoted_subtable_tenferro_dependency_fails(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": """\
[package]
name = "tensor4all-demo"
version = "0.1.0"
edition = "2021"

[dependencies."tenferro-tensor"]
workspace = true
""",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tenferro-tensor", result.stderr)
        self.assertIn("not allowed", result.stderr)

    def test_single_quoted_package_rename_fails(self) -> None:
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST
                + "sneaky = { workspace = true, package = 'tenferro-tensor' }\n",
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tenferro-tensor", result.stderr)
        self.assertIn("not allowed", result.stderr)

    def test_removed_exception_crate_is_stale(self) -> None:
        # An exception tuple whose crate no longer exists must be flagged.
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST,
            }
        )
        result = run_check(root, exceptions="tensor4all-ghost:tenferro-tensor")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("no crate of that name exists", result.stderr)
        self.assertIn("tensor4all-ghost", result.stderr)

    def test_stale_exception_tuple_fails(self) -> None:
        # A crate whose sanctioned tuple lists a dependency it no longer has.
        root = write_fixture(
            {
                "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
                "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST
                + "tenferro-tensor.workspace = true\n",
            }
        )
        result = run_check(
            root, exceptions="tensor4all-demo:tenferro-tensor,tenferro-einsum"
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("stale tenferro exception", result.stderr)
        self.assertIn("tenferro-einsum", result.stderr)


    # ---- rule 4: tensorbackend bridge items (#1380) ----

    def bridge_fixture(self, files: dict[str, str]) -> Path:
        base = {
            "crates/tensor4all-tensorbackend/Cargo.toml": TENSORBACKEND_MANIFEST,
            "crates/tensor4all-demo/Cargo.toml": PLAIN_MANIFEST,
        }
        base.update(files)
        return write_fixture(base)

    def write_baseline(self, root: Path, entries: str) -> None:
        (root / "scripts").mkdir(exist_ok=True)
        (root / "scripts/tensorbackend-bridge-sites-baseline.json").write_text(
            entries, encoding="utf-8"
        )

    def test_core_may_name_bridge_items(self) -> None:
        root = self.bridge_fixture(
            {
                "crates/tensor4all-core/src/lib.rs": (
                    "use tensor4all_tensorbackend::{Storage, NativeTensorReadInput};\n"
                ),
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unpinned_bridge_item_in_another_crate_fails(self) -> None:
        root = self.bridge_fixture(
            {
                "crates/tensor4all-demo/src/lib.rs": (
                    "use tensor4all_tensorbackend::{explicit::{Session}, Storage};\n"
                ),
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("crates/tensor4all-demo/src/lib.rs", result.stderr)
        self.assertIn("`Storage`", result.stderr)
        self.assertNotIn("`Session`", result.stderr)

    def test_bridge_item_in_an_isolated_test_workspace_fails(self) -> None:
        root = self.bridge_fixture(
            {
                "tests/feature-configurations/demo/tests/demo.rs": (
                    "let s = tensor4all_tensorbackend::Storage::default();\n"
                ),
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("tests/feature-configurations/demo/tests/demo.rs", result.stderr)

    def test_crate_alias_is_followed(self) -> None:
        root = self.bridge_fixture(
            {
                "crates/tensor4all-demo/src/lib.rs": (
                    "use tensor4all_tensorbackend as backend;\n"
                    "fn f() { backend::conj_native_tensor(&t); }\n"
                ),
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("`conj_native_tensor`", result.stderr)

    def test_comments_and_vocabulary_are_not_sites(self) -> None:
        root = self.bridge_fixture(
            {
                "crates/tensor4all-demo/src/lib.rs": (
                    "// tensor4all_tensorbackend::Storage is core's business\n"
                    "use tensor4all_tensorbackend::{StorageKind, StorageScalar}; "
                    "// tensor4all_tensorbackend::Storage\n"
                ),
            }
        )
        result = run_check(root)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pinned_site_passes_and_stale_pin_fails(self) -> None:
        source = "use tensor4all_tensorbackend::Storage;\n"
        root = self.bridge_fixture({"crates/tensor4all-demo/src/lib.rs": source})
        self.write_baseline(
            root,
            '[{"path": "crates/tensor4all-demo/src/lib.rs", "item": "Storage", "count": 1}]\n',
        )
        self.assertEqual(run_check(root).returncode, 0)

        (root / "crates/tensor4all-demo/src/lib.rs").write_text("", encoding="utf-8")
        result = run_check(root)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("stale bridge baseline entry `Storage`", result.stderr)

    def test_update_baseline_shrinks_and_refuses_growth(self) -> None:
        root = self.bridge_fixture(
            {"crates/tensor4all-demo/src/lib.rs": "use tensor4all_tensorbackend::Storage;\n"}
        )
        env_ci = os.environ.pop("CI", None)
        try:
            refused = run_check(root, extra_args=("--update-baseline",))
            self.assertEqual(refused.returncode, 1, refused.stdout)
            self.assertIn("refusing to record new bridge site", refused.stderr)

            grown = run_check(root, extra_args=("--update-baseline", "--allow-growth"))
            self.assertEqual(grown.returncode, 0, grown.stderr)
            self.assertEqual(run_check(root).returncode, 0)

            (root / "crates/tensor4all-demo/src/lib.rs").write_text("", encoding="utf-8")
            shrunk = run_check(root, extra_args=("--update-baseline",))
            self.assertEqual(shrunk.returncode, 0, shrunk.stderr)
            baseline = (root / "scripts/tensorbackend-bridge-sites-baseline.json").read_text(
                encoding="utf-8"
            )
            self.assertEqual(baseline, "[]\n")
            self.assertEqual(run_check(root).returncode, 0)
        finally:
            if env_ci is not None:
                os.environ["CI"] = env_ci


if __name__ == "__main__":
    unittest.main()
