#!/usr/bin/env python3
"""Ratchet for implicit execution-engine sites (#1392, stage A1 of #1376).

#1376 makes the execution engine an explicit argument of every numerical
operation. The migration takes several phases while other PRs keep landing,
so this check stops *new* implicit-engine sites without touching the existing
ones (``#[deprecated]`` is ruled out because Clippy runs with ``-D warnings``).

The sites are enumerated from source on every run: every ``.rs`` file under
``crates/*/src`` is lexed (comments and string/char literals blanked out, so
prose, doctests and messages never match), and five patterns are matched on
the remaining code:

``none-context-arg``
    a literal ``None`` argument to ``with_eager_session``,
    ``with_eager_session_typed``, ``with_engine`` or ``with_execution_engine``.
    Arguments are split with a balanced-delimiter scan, so a call that
    ``rustfmt`` wrapped over several lines is still found. Since #1378 core's
    eager entries take an ``EngineRoute`` instead of an ``Option``, and the
    former ``None`` is spelled ``EngineRoute::Legacy``: it means the same
    implicit route, so every ``EngineRoute::Legacy`` expression (a ``match``
    arm pattern ``EngineRoute::Legacy =>`` is not one) and every
    ``EngineRoute::context_or_legacy(None)`` is a site of this pattern too.
``global-accessor``
    a call of ``default_context()``, ``default_eager_ctx()``,
    ``default_cpu_execution_context()``, ``with_default_session(``,
    ``with_default_backend(``, ``with_default_graph_compiler(``, the
    owner-resolving implicit entry ``with_runtime_engine(`` and its
    process-global fallback bridge ``with_runtime_engine_or_global(`` (#1378;
    their ``fn`` definitions are not sites), and every naming of the global
    ``implicit_route::eager_runtime`` / ``legacy::eager_runtime`` path (the
    ``context.eager_runtime()`` method is explicit and not a site). A
    renaming import (``use ..::default_context as ctx;``) is a site too.
``option-context-param``
    an ``Option<&ExecutionContext>`` / ``Option<&CpuExecutionContext>`` /
    ``Option<&CudaExecutionContext>`` type, with or without a path prefix or
    lifetime.
``context-ref-param``
    a ``name: &ExecutionContext`` / ``&CpuExecutionContext`` /
    ``&CudaExecutionContext`` binding (parameter, field or ``let``), including
    ``&Arc<..ExecutionContext>``. Design revision 3 replaces these with
    ``exec: &mut dyn ExecutionEngine``.
``session-call``
    a ``.with_session(`` or ``with_concrete_session(`` call outside
    ``crates/tensor4all-tensorbackend``: the session shape revision 3 removes
    (design doc section 4.6).

Test-only sources are out of scope: any file named ``tests.rs`` and anything
under a ``tests`` or ``benches`` directory.

Every site found is compared with ``scripts/implicit-engine-sites-baseline.json``.
A baseline entry is identified by (path, pattern, enclosing item, normalised
site text) and holds a count, never a line number, so an unrelated edit above a
site does not break anyone. The check fails when

* a group has more sites than the baseline records: a **new** site. The
  report names ``path:line``, the pattern and the site text;
* a group has fewer sites than the baseline records: the baseline is stale,
  and the PR that removed the site shrinks it with ``--update-baseline``.

``--update-baseline`` rewrites the baseline from the tree. It refuses to run
in CI, and it refuses to record growth in any pattern's total unless
``--allow-growth`` is also given: adding debt is a deliberate act that the PR
must justify. Moving a site between files or functions keeps the totals and is
accepted.

The remedy for a new site is design doc section 6.2: take
``exec: &mut dyn ExecutionEngine`` and pass it on, or, for a caller with no
engine yet, call an existing ``legacy`` bridge (``tensor4all_core::legacy``
today). Bridge operations such as ``legacy::mat_mul`` are deliberately not
sites: the #1376 rules use them to stop the signature cascade during phase B.

This check is deleted in stage D4 of #1376, once nothing is left to guard.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_BASELINE = Path("scripts") / "implicit-engine-sites-baseline.json"
TENSORBACKEND_PREFIX = "crates/tensor4all-tensorbackend/"
REMEDY = (
    "take `exec: &mut dyn ExecutionEngine` and pass it on "
    "(docs/design/explicit-engine-migration.md section 6.2); a caller with no "
    "engine yet calls an existing `legacy` bridge instead (#1376)."
)
PATTERNS = (
    "none-context-arg",
    "global-accessor",
    "option-context-param",
    "context-ref-param",
    "session-call",
)

NONE_ROUTE_FUNCTIONS = (
    "with_eager_session_typed",
    "with_eager_session",
    "with_engine",
    "with_execution_engine",
)
NONE_ROUTE_RE = re.compile(
    r"\b(" + "|".join(NONE_ROUTE_FUNCTIONS) + r")\s*(?:::\s*<[^;{}]*?>\s*)?\("
)
# #1378: the `EngineRoute` spelling of the former literal `None` route.
LEGACY_ROUTE_RE = re.compile(r"\bEngineRoute\s*::\s*Legacy\b")
MATCH_ARM_AFTER_RE = re.compile(r"\s*(?:=>|\|)")
LEGACY_FROM_NONE_RE = re.compile(
    r"\bEngineRoute\s*::\s*context_or_legacy\s*\(\s*None\s*\)"
)
GLOBAL_ACCESSOR_RE = re.compile(
    r"\b(default_context|default_eager_ctx|default_cpu_execution_context"
    r"|with_default_session|with_default_backend|with_default_graph_compiler"
    r"|with_runtime_engine_or_global|with_runtime_engine)"
    r"\s*(?:::\s*<[^;{}]*?>\s*)?\("
)
CONTEXT_TYPE = r"(?:[A-Za-z_]\w*\s*::\s*)*(?:Cpu|Cuda)?ExecutionContext\b"
OPTION_CONTEXT_RE = re.compile(
    r"\bOption\s*<\s*&\s*(?:'\w+\s+)?(?:mut\s+)?" + CONTEXT_TYPE
)
CONTEXT_REF_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*:\s*&\s*(?:'\w+\s+)?(?:mut\s+)?"
    r"(?:(?:[A-Za-z_]\w*\s*::\s*)*Arc\s*<\s*)?" + CONTEXT_TYPE
)
SESSION_CALL_RE = re.compile(r"(\.\s*with_session|\bwith_concrete_session)\s*\(")
FN_DEF_BEFORE_RE = re.compile(r"\bfn\s+$")
GLOBAL_EAGER_RUNTIME_RE = re.compile(
    r"\b(?P<module>implicit_route|legacy)\s*::\s*(?:\{[^}]*?)?\b(?P<name>eager_runtime)\b"
)
# `use ..::default_context as ctx;` would hide later `ctx()` calls, so the
# renaming import is itself a site.
ACCESSOR_ALIAS_RE = re.compile(
    r"\b(default_context|default_eager_ctx|default_cpu_execution_context"
    r"|with_default_session|with_default_backend|with_default_graph_compiler"
    r"|with_runtime_engine_or_global|with_runtime_engine)\s+as\s+([A-Za-z_]\w*)"
)
BRACE_RE = re.compile(r"[{};]")
ITEM_RE = re.compile(r"\b(fn|struct|enum|trait)\s+([A-Za-z_]\w*)")


@dataclass(frozen=True)
class Site:
    path: str
    line: int
    pattern: str
    scope: str
    text: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.path, self.pattern, self.scope, self.text)


LEXER_RE = re.compile(r"//|/\*|(?<![\w])(?:br|r)#*\"|(?<![\w])b\"|\"|'")


def strip_code(source: str) -> str:
    """Blank out comments and string/char literals, keeping offsets and newlines."""
    out = list(source)
    n = len(source)

    def blank(start: int, end: int) -> None:
        for k in range(start, min(end, n)):
            if out[k] != "\n":
                out[k] = " "

    i = 0
    while True:
        m = LEXER_RE.search(source, i)
        if m is None:
            break
        i, token = m.start(), m.group(0)
        if token == "//":
            end = source.find("\n", i)
            end = n if end < 0 else end
            blank(i, end)
            i = end
        elif token == "/*":
            depth, j = 1, i + 2
            while j < n and depth:
                if source.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif source.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif token.startswith(("r", "br")):
            closing = '"' + token[token.index("r") + 1:-1]
            j = source.find(closing, m.end())
            j = n if j < 0 else j + len(closing)
            blank(i, j)
            i = j
        elif token in ('"', 'b"'):
            j = m.end()
            while j < n and source[j] != '"':
                j += 2 if source[j] == "\\" else 1
            blank(i, j + 1)
            i = j + 1
        else:
            # A char literal ('a', '\n', '\u{..}') or a lifetime ('a).
            if source.startswith("'\\", i):
                j = source.find("'", i + 2)
                j = n if j < 0 else j
                blank(i, j + 1)
                i = j + 1
            elif i + 2 < n and source[i + 2] == "'":
                blank(i, i + 3)
                i += 3
            else:
                i += 1
    return "".join(out)


def split_call_args(code: str, open_paren: int) -> list[tuple[str, int]] | None:
    """Split the arguments of the call whose `(` is at `open_paren`."""
    depth = 0
    args: list[tuple[str, int]] = []
    start = open_paren + 1
    for j in range(open_paren, len(code)):
        ch = code[j]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                args.append((code[start:j], start))
                return args
        elif ch == "," and depth == 1:
            args.append((code[start:j], start))
            start = j + 1
    return None


def enclosing_scopes(code: str, offsets: list[int]) -> dict[int, str]:
    """Name the innermost `fn`/`struct`/`enum`/`trait` around each offset."""
    wanted = sorted(set(offsets))
    result: dict[int, str] = {}
    stack: list[tuple[int, str]] = []  # (brace depth at open, name)
    pending: str | None = None
    depth = 0
    events: list[tuple[int, str, str]] = []
    for m in ITEM_RE.finditer(code):
        events.append((m.start(), "item", m.group(2)))
    for m in BRACE_RE.finditer(code):
        events.append((m.start(), m.group(0), ""))
    events.sort()
    w = 0
    for pos, kind, name in events:
        while w < len(wanted) and wanted[w] < pos:
            result[wanted[w]] = pending or (stack[-1][1] if stack else "<module>")
            w += 1
        if kind == "item":
            pending = name
        elif kind == "{":
            depth += 1
            if pending is not None:
                stack.append((depth, pending))
                pending = None
        elif kind == "}":
            if stack and stack[-1][0] == depth:
                stack.pop()
            depth -= 1
        elif kind == ";" and pending is not None:
            # `fn f(..);` in a trait or `struct S(..);` ends without a body.
            pending = None
    while w < len(wanted):
        result[wanted[w]] = pending or (stack[-1][1] if stack else "<module>")
        w += 1
    return result


def normalise(text: str) -> str:
    """Collapse whitespace so a `rustfmt` reflow does not change a site's identity."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r" ?([^\w ]) ?", r"\1", text)


# A file without any of these words cannot hold a site; skip lexing it.
PREFILTER = ("with_", "default_", "ExecutionContext", "eager_runtime", "EngineRoute")


def scan_source(rel_path: str, source: str) -> list[Site]:
    if not any(word in source for word in PREFILTER):
        return []
    code = strip_code(source)
    raw: list[tuple[int, str, str]] = []  # (offset, pattern, text)

    for m in NONE_ROUTE_RE.finditer(code):
        args = split_call_args(code, m.end() - 1)
        if not args:
            continue
        for index, (arg, start) in enumerate(args):
            if arg.strip() == "None":
                raw.append((start + arg.index("None"), "none-context-arg",
                            f"{m.group(1)}(arg{index}=None)"))

    for m in LEGACY_ROUTE_RE.finditer(code):
        if MATCH_ARM_AFTER_RE.match(code, m.end()):
            continue
        raw.append((m.start(), "none-context-arg", "EngineRoute::Legacy"))
    for m in LEGACY_FROM_NONE_RE.finditer(code):
        raw.append((m.start(), "none-context-arg", "EngineRoute::context_or_legacy(None)"))

    for m in GLOBAL_ACCESSOR_RE.finditer(code):
        if FN_DEF_BEFORE_RE.search(code[max(0, m.start() - 8):m.start()]):
            continue
        raw.append((m.start(), "global-accessor", f"{m.group(1)}()"))

    # The global `implicit_route::eager_runtime()` (also `legacy::eager_runtime`)
    # is counted where its path is named, in a call or a `use`, because a bare
    # `eager_runtime()` is usually the explicit `context.eager_runtime()` method.
    for m in GLOBAL_EAGER_RUNTIME_RE.finditer(code):
        raw.append((m.start("name"), "global-accessor", f"{m.group('module')}::eager_runtime"))

    for m in ACCESSOR_ALIAS_RE.finditer(code):
        raw.append((m.start(), "global-accessor", f"{m.group(1)} as {m.group(2)}"))

    for m in OPTION_CONTEXT_RE.finditer(code):
        raw.append((m.start(), "option-context-param", normalise(m.group(0))))

    for m in CONTEXT_REF_RE.finditer(code):
        raw.append((m.start(), "context-ref-param", normalise(m.group(0))))

    if not rel_path.startswith(TENSORBACKEND_PREFIX):
        for m in SESSION_CALL_RE.finditer(code):
            if FN_DEF_BEFORE_RE.search(code[max(0, m.start() - 8):m.start()]):
                continue
            raw.append((m.start(), "session-call", normalise(m.group(1)) + "("))

    if not raw:
        return []
    scopes = enclosing_scopes(code, [offset for offset, _, _ in raw])
    sites = [
        Site(rel_path, code.count("\n", 0, offset) + 1, pattern, scopes[offset], text)
        for offset, pattern, text in raw
    ]
    sites.sort(key=lambda s: (s.line, s.pattern, s.text))
    return sites


def is_in_scope(rel_path: str) -> bool:
    parts = rel_path.split("/")
    if len(parts) < 4 or parts[0] != "crates" or parts[2] != "src":
        return False
    if not rel_path.endswith(".rs") or parts[-1] == "tests.rs":
        return False
    return not any(part in ("tests", "benches") for part in parts[3:-1])


def scan_tree(root: Path) -> list[Site]:
    sites: list[Site] = []
    for path in sorted((root / "crates").glob("*/src/**/*.rs")):
        rel = path.relative_to(root).as_posix()
        if is_in_scope(rel):
            sites.extend(scan_source(rel, path.read_text(encoding="utf-8")))
    return sites


def load_baseline(path: Path) -> Counter:
    counts: Counter = Counter()
    if not path.is_file():
        return counts
    for entry in json.loads(path.read_text(encoding="utf-8")):
        key = (entry["path"], entry["pattern"], entry["scope"], entry["site"])
        if entry["pattern"] not in PATTERNS:
            raise ValueError(f"unknown pattern in baseline: {entry['pattern']!r}")
        counts[key] += int(entry["count"])
    return counts


def render_baseline(counts: Counter) -> str:
    lines = []
    for (path, pattern, scope, site), count in sorted(counts.items()):
        entry = {"path": path, "pattern": pattern, "scope": scope, "site": site,
                 "count": count}
        lines.append("  " + json.dumps(entry, ensure_ascii=False))
    return "[\n" + ",\n".join(lines) + "\n]\n" if lines else "[]\n"


def totals(counts: Counter) -> dict[str, int]:
    result = {pattern: 0 for pattern in PATTERNS}
    for (_, pattern, _, _), count in counts.items():
        result[pattern] += count
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--update-baseline", action="store_true",
                        help="rewrite the baseline; refuses CI and refuses growth")
    parser.add_argument("--allow-growth", action="store_true",
                        help="with --update-baseline, record a larger total; the "
                        "growth must be justified in the PR")
    parser.add_argument("--list", action="store_true",
                        help="print every site found, one per line")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    baseline_path = (args.baseline or root / DEFAULT_BASELINE).resolve()
    sites = scan_tree(root)
    current: Counter = Counter(site.key for site in sites)
    try:
        baseline = load_baseline(baseline_path)
    except (ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"implicit-engine ratchet: unreadable baseline {baseline_path}: {error}",
              file=sys.stderr)
        return 2

    if args.list:
        for site in sites:
            print(f"{site.path}:{site.line}: {site.pattern}: {site.text} [in {site.scope}]")

    now, before = totals(current), totals(baseline)
    summary = ", ".join(f"{p} {now[p]}" for p in PATTERNS)

    if args.update_baseline:
        if os.environ.get("CI"):
            print("implicit-engine ratchet: --update-baseline refuses to run in CI",
                  file=sys.stderr)
            return 2
        grown = [p for p in PATTERNS if now[p] > before[p] and not args.allow_growth]
        if grown:
            for p in grown:
                print(f"implicit-engine ratchet: refusing to grow {p}: "
                      f"{before[p]} -> {now[p]}", file=sys.stderr)
            print("Remove the new site. Recording new debt needs --allow-growth and a "
                  "justification in the PR.", file=sys.stderr)
            return 1
        baseline_path.write_text(render_baseline(current), encoding="utf-8")
        print(f"implicit-engine ratchet: wrote {baseline_path} "
              f"({sum(now.values())} sites: {summary})")
        return 0

    by_key: dict[tuple, list[Site]] = defaultdict(list)
    for site in sites:
        by_key[site.key].append(site)

    failures = 0
    for key in sorted(set(current) | set(baseline)):
        have, pinned = current[key], baseline[key]
        if have > pinned:
            failures += 1
            path, pattern, scope, text = key
            print(f"error: new implicit-engine site ({pattern}) in `{scope}`: "
                  f"{have} found, baseline pins {pinned}")
            for site in by_key[key]:
                print(f"  {site.path}:{site.line}: {site.pattern}: {site.text}")
            print(f"  remedy: {REMEDY}")
        elif have < pinned:
            failures += 1
            path, pattern, scope, text = key
            print(f"error: stale baseline entry {path} {pattern} `{scope}` {text}: "
                  f"baseline pins {pinned}, {have} found. A site was removed or "
                  "moved; shrink the baseline with "
                  "`python3 scripts/check-implicit-engine-sites.py --update-baseline`.")
    if failures:
        print(f"implicit-engine ratchet: FAILED ({failures} group(s) differ from "
              f"{baseline_path.name}; tree has {sum(now.values())} sites: {summary})")
        return 1
    print(f"implicit-engine ratchet: OK, {sum(now.values())} pinned sites "
          f"({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
