#!/usr/bin/env python3
"""Check that every integration test file is compiled by some test target.

Packages that merge their integration tests into one `integration` target set
``autotests = false``, so Cargo no longer discovers ``tests/*.rs`` on its own.
A new file that is neither a ``[[test]]`` target nor declared as ``mod <name>;``
in a target's root file would then be skipped silently. This script fails on
such files.

Checked entries are the top-level ``tests/*.rs`` files and the
``tests/<dir>/`` directories holding a ``main.rs`` or ``mod.rs``.

A ``tests/heavy_*.rs`` file must be a module, not a target of its own: the
heavy-tests workflow selects paths with ``test(/(^|::)heavy_/)``, which matches
when a heavy module name is part of the test path.

Private ``heavy_*`` unit-test modules under ``src/`` are checked too: each
``heavy_*.rs`` file, or ``heavy_*/`` directory holding a ``mod.rs``, must be
declared as ``mod heavy_<name>;`` in its parent module file. An undeclared one
is never compiled, so its tests would silently run in no tier.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

MOD_RE = re.compile(
    r"^\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;", re.M
)


def undeclared(tests_dir: Path, roots: set[Path]) -> list[Path]:
    """Return test entries under ``tests_dir`` that no target compiles."""
    roots = {r.resolve() for r in roots}
    declared: set[str] = set()
    for root in roots:
        if root.parent == tests_dir.resolve() and root.is_file():
            declared.update(MOD_RE.findall(root.read_text(encoding="utf-8")))

    missing = []
    for entry in sorted(tests_dir.iterdir()):
        if entry.is_file() and entry.suffix == ".rs":
            if entry.resolve() not in roots and entry.stem not in declared:
                missing.append(entry)
        elif entry.is_dir():
            files = [entry / "main.rs", entry / "mod.rs"]
            if not any(f.is_file() for f in files):
                continue
            if any(f.resolve() in roots for f in files):
                continue
            if entry.name not in declared:
                missing.append(entry)
    return missing


def heavy_targets(tests_dir: Path, roots: set[Path]) -> list[Path]:
    """Return ``heavy_*.rs`` files that are test targets of their own."""
    tests_dir = tests_dir.resolve()
    return sorted(
        r
        for r in (root.resolve() for root in roots)
        if r.parent == tests_dir and r.name.startswith("heavy_")
    )


def undeclared_heavy_modules(src_dir: Path) -> list[Path]:
    """Return ``heavy_*`` modules under ``src_dir`` that no parent declares."""
    src_dir = src_dir.resolve()
    missing = []
    for path in sorted(src_dir.rglob("heavy_*")):
        if path.is_file() and path.suffix == ".rs":
            name = path.stem
        elif path.is_dir() and (path / "mod.rs").is_file():
            name = path.name
        else:
            continue
        directory = path.parent
        if directory == src_dir:
            parents = [src_dir / "lib.rs", src_dir / "main.rs"]
        else:
            parents = [directory / "mod.rs", directory.with_suffix(".rs")]
        declared: set[str] = set()
        for parent in parents:
            if parent.is_file():
                declared.update(MOD_RE.findall(parent.read_text(encoding="utf-8")))
        if name not in declared:
            missing.append(path)
    return missing


def main() -> int:
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    failures = []
    for package in meta["packages"]:
        src_dir = Path(package["manifest_path"]).parent / "src"
        if src_dir.is_dir():
            for entry in undeclared_heavy_modules(src_dir):
                failures.append(
                    f"{entry} ({package['name']}) is not declared as `mod` in "
                    "its parent module, so its heavy tests are never compiled"
                )
        tests_dir = Path(package["manifest_path"]).parent / "tests"
        if not tests_dir.is_dir():
            continue
        roots = {
            Path(t["src_path"]) for t in package["targets"] if "test" in t["kind"]
        }
        for entry in undeclared(tests_dir, roots):
            failures.append(
                f"{entry} ({package['name']}) is not compiled by any test "
                "target; declare it as `mod` in tests/main.rs or add a "
                "`[[test]]` target"
            )
        for entry in heavy_targets(tests_dir, roots):
            failures.append(
                f"{entry} ({package['name']}) is a test target of its own; "
                "declare it as `mod` in the package's tests/main.rs so "
                "`test(/(^|::)heavy_/)` selects it"
            )

    for failure in failures:
        print(f"error: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
