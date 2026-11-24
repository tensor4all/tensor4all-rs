#!/usr/bin/env python3
"""Tests for the implicit-engine-site ratchet (tensor4all/tensor4all-rs#1392)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("check-implicit-engine-sites.py")
SPEC = importlib.util.spec_from_file_location("check_implicit_engine_sites", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CORE = "crates/tensor4all-core/src/ops.rs"
BACKEND = "crates/tensor4all-tensorbackend/src/context.rs"


def patterns(source: str, path: str = CORE) -> list[str]:
    return [site.pattern for site in MODULE.scan_source(path, source)]


class PatternTests(unittest.TestCase):
    def test_literal_none_context_argument(self) -> None:
        for name in MODULE.NONE_ROUTE_FUNCTIONS:
            with self.subTest(name=name):
                source = f"fn f() {{ {name}(&runtime, None, |s| s.run()); }}\n"
                self.assertEqual(patterns(source), ["none-context-arg"])

    def test_wrapped_literal_none_is_found_on_its_own_line(self) -> None:
        source = (
            "fn conj(&self) -> Result<Self> {\n"
            "    with_eager_session(\n"
            "        &self.runtime(),\n"
            "        None,\n"
            "        |session| session.conj(&self.inner, (1, 2)),\n"
            "    )\n"
            "}\n"
        )
        [site] = MODULE.scan_source(CORE, source)
        self.assertEqual(
            (site.pattern, site.line, site.scope, site.text),
            ("none-context-arg", 4, "conj", "with_eager_session(arg1=None)"),
        )

    def test_explicit_context_argument_is_not_a_site(self) -> None:
        source = "fn f() { with_eager_session(&rt, Some(context), |s| g(None)); }\n"
        self.assertEqual(patterns(source), [])

    def test_global_accessors_but_not_their_definitions(self) -> None:
        for name in (
            "default_context",
            "default_eager_ctx",
            "default_cpu_execution_context",
            "with_default_session",
        ):
            with self.subTest(name=name):
                call = f"fn f() {{ let c = tensor4all_tensorbackend::{name}(); }}\n"
                self.assertEqual(patterns(call), ["global-accessor"])
                definition = f"pub fn {name}() -> Ctx {{ todo() }}\n"
                self.assertEqual(patterns(definition), [])
                import_only = f"use tensor4all_tensorbackend::{name};\n"
                self.assertEqual(patterns(import_only), [])

    def test_option_context_parameter_spellings(self) -> None:
        for ty in (
            "Option<&ExecutionContext>",
            "Option<&CpuExecutionContext>",
            "Option<&CudaExecutionContext>",
            "Option<&tensor4all_tensorbackend::CpuExecutionContext>",
            "Option< &'a ExecutionContext >",
        ):
            with self.subTest(ty=ty):
                source = f"fn f(context: {ty}) {{}}\n"
                self.assertEqual(patterns(source), ["option-context-param"])

    def test_context_reference_parameter_spellings(self) -> None:
        for ty in (
            "&ExecutionContext",
            "&CpuExecutionContext",
            "&CudaExecutionContext",
            "&tensor4all_tensorbackend::ExecutionContext",
            "&'a CpuExecutionContext",
            "&Arc<CpuExecutionContext>",
        ):
            with self.subTest(ty=ty):
                source = f"fn f(\n    a: usize,\n    context: {ty},\n) {{}}\n"
                self.assertEqual(patterns(source), ["context-ref-param"])

    def test_engine_parameter_is_not_a_site(self) -> None:
        source = "fn f(exec: &mut dyn ExecutionEngine, ctx: ExecutionContextError) {}\n"
        self.assertEqual(patterns(source), [])

    def test_session_calls_outside_tensorbackend_only(self) -> None:
        source = (
            "fn f() {\n"
            "    context.with_session(|e| run(e));\n"
            "    context\n        .with_session(|e| run(e));\n"
            "    with_concrete_session(&ctx, |e| run(e));\n"
            "}\n"
        )
        self.assertEqual(patterns(source), ["session-call"] * 3)
        self.assertEqual(patterns(source, BACKEND), [])
        definition = "impl X { pub fn with_session<R>(&self) -> R { todo() } }\n"
        self.assertEqual(patterns(definition), [])

    def test_comments_strings_and_doctests_are_ignored(self) -> None:
        source = (
            "/// ```\n"
            "/// let c = default_cpu_execution_context();\n"
            "/// ```\n"
            "// with_eager_session(&rt, None, f)\n"
            "/* outer /* default_context() */ still comment */\n"
            'fn f() -> &str { "default_context() and .with_session(" }\n'
            "fn g() -> char { '\"' }\n"
            "fn h<'a>(x: &'a str) -> &'a str { r#\"default_eager_ctx()\"# }\n"
            "fn k() { default_eager_ctx(); }\n"
        )
        [site] = MODULE.scan_source(CORE, source)
        self.assertEqual((site.line, site.text, site.scope), (9, "default_eager_ctx()", "k"))

    def test_enclosing_scope_names_the_function(self) -> None:
        source = (
            "impl T {\n"
            "    fn first(&self) { let a = 1; }\n"
            "    fn second(&self, context: Option<&ExecutionContext>) {\n"
            "        if x { default_context(); }\n"
            "    }\n"
            "}\n"
            "struct Holder<'a> { context: &'a ExecutionContext }\n"
        )
        sites = MODULE.scan_source(CORE, source)
        self.assertEqual(
            [(s.pattern, s.scope) for s in sites],
            [
                ("option-context-param", "second"),
                ("global-accessor", "second"),
                ("context-ref-param", "Holder"),
            ],
        )

    def test_scope_excludes_test_and_bench_sources(self) -> None:
        self.assertTrue(MODULE.is_in_scope("crates/tensor4all-core/src/lib.rs"))
        self.assertTrue(MODULE.is_in_scope("crates/tensor4all-core/src/a/b.rs"))
        for path in (
            "crates/tensor4all-core/src/defaults/tests.rs",
            "crates/tensor4all-core/src/defaults/tests/mod.rs",
            "crates/tensor4all-core/src/benches/x.rs",
            "crates/tensor4all-core/tests/integration.rs",
            "crates/tensor4all-core/benches/b.rs",
            "crates/tensor4all-core/examples/e.rs",
            "crates/tensor4all-core/src/notes.md",
        ):
            with self.subTest(path=path):
                self.assertFalse(MODULE.is_in_scope(path))


SITE_FILE = "crates/tensor4all-demo/src/lib.rs"
ORIGINAL = (
    "pub fn demo(t: &T) -> R {\n"
    "    with_eager_session(&t.rt, None, |s| s.run())\n"
    "}\n"
)


class RatchetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.write(SITE_FILE, ORIGINAL)
        self.write("crates/tensor4all-demo/tests/it.rs",
                   "fn t() { default_context(); }\n")
        (self.root / "scripts").mkdir()
        result = self.run_check("--update-baseline", "--allow-growth")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def write(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def run_check(self, *extra: str, env: dict | None = None):
        environment = {k: v for k, v in os.environ.items() if k != "CI"}
        environment.update(env or {})
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.root), *extra],
            capture_output=True, text=True, env=environment,
        )

    def test_unchanged_tree_passes_and_test_dirs_are_not_pinned(self) -> None:
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("OK, 1 pinned sites", result.stdout)

    def test_new_site_fails_naming_file_line_and_pattern(self) -> None:
        self.write(SITE_FILE, ORIGINAL + "pub fn more() { let c = default_cpu_execution_context(); }\n")
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertIn(f"{SITE_FILE}:4: global-accessor: default_cpu_execution_context()",
                      result.stdout)
        self.assertIn("exec: &mut dyn ExecutionEngine", result.stdout)

    def test_second_site_in_same_function_fails(self) -> None:
        self.write(SITE_FILE, ORIGINAL.replace(
            "s.run())\n", "s.run())?;\n    with_eager_session(\n        &t.rt,\n        None,\n        f,\n    )\n"))
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("2 found, baseline pins 1", result.stdout)
        self.assertIn(f"{SITE_FILE}:5: none-context-arg", result.stdout)

    def test_line_drift_and_reflow_do_not_fail(self) -> None:
        self.write(SITE_FILE, "\n\n// moved down\nuse x::y;\n" + ORIGINAL.replace(
            "with_eager_session(&t.rt, None, |s| s.run())",
            "with_eager_session(\n        &t.rt,\n        None,\n        |s| s.run(),\n    )"))
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_removed_site_reports_stale_baseline_and_update_shrinks_it(self) -> None:
        self.write(SITE_FILE, "pub fn demo(t: &T, exec: &mut dyn ExecutionEngine) {}\n")
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("stale baseline entry", result.stdout)
        self.assertEqual(self.run_check("--update-baseline").returncode, 0)
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("OK, 0 pinned sites", result.stdout)

    def test_update_refuses_growth_without_flag(self) -> None:
        self.write(SITE_FILE, ORIGINAL + "fn g() { default_eager_ctx(); }\n")
        result = self.run_check("--update-baseline")
        self.assertEqual(result.returncode, 1)
        self.assertIn("refusing to grow global-accessor: 0 -> 1", result.stderr)
        self.assertEqual(self.run_check().returncode, 1)

    def test_moving_a_site_is_accepted_by_update(self) -> None:
        self.write(SITE_FILE, ORIGINAL.replace("demo", "renamed"))
        self.assertEqual(self.run_check().returncode, 1)
        self.assertEqual(self.run_check("--update-baseline").returncode, 0)
        self.assertEqual(self.run_check().returncode, 0)

    def test_update_refuses_in_ci(self) -> None:
        result = self.run_check("--update-baseline", env={"CI": "true"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("refuses to run in CI", result.stderr)


class RepositoryBaselineTests(unittest.TestCase):
    def test_repository_baseline_is_sorted_one_entry_per_line(self) -> None:
        path = SCRIPT.parents[1] / MODULE.DEFAULT_BASELINE
        text = path.read_text(encoding="utf-8")
        self.assertEqual(text, MODULE.render_baseline(MODULE.load_baseline(path)))


if __name__ == "__main__":
    unittest.main()
