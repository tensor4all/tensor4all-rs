#!/usr/bin/env python3
"""Report panic-style calls in non-test library source files.

This is an audit helper for issue #485. It intentionally reports only source
lines that are likely to be runtime library code: rustdoc comments, line
comments, files under src/**/tests, and cfg(test) modules are skipped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


PATTERN = re.compile(r"\bpanic!\s*\(|\bunreachable!\s*\(|\.unwrap\(\)|\.expect\s*\(")


def brace_delta(line: str) -> int:
    """Best-effort brace balance for Rust blocks."""
    return line.count("{") - line.count("}")


def audit_file(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    pending_cfg_test = False
    test_depth: int | None = None

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()

        if test_depth is not None:
            test_depth += brace_delta(line)
            if test_depth <= 0:
                test_depth = None
            continue

        if stripped.startswith("//"):
            continue

        if stripped.startswith("#[cfg(test)]"):
            pending_cfg_test = True
            continue

        if pending_cfg_test:
            if not stripped:
                continue
            if re.match(r"(pub\s+)?mod\s+\w+", stripped):
                test_depth = max(brace_delta(line), 1)
                pending_cfg_test = False
                continue
            pending_cfg_test = False

        if PATTERN.search(line):
            hits.append((lineno, stripped))

    return hits


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    source_root = root / "crates"
    hits: list[tuple[Path, int, str]] = []

    for path in sorted(source_root.glob("*/src/**/*.rs")):
        if "tests" in path.parts or path.name in {"tests.rs", "test_utils.rs"}:
            continue
        for lineno, line in audit_file(path):
            hits.append((path.relative_to(root), lineno, line))

    for path, lineno, line in hits:
        print(f"{path}:{lineno}: {line}")

    if hits:
        print(f"\nFound {len(hits)} runtime panic-style hit(s).", file=sys.stderr)
        return 1

    print("No runtime panic-style hits found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
