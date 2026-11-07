#!/usr/bin/env python3
"""Review PR diffs against REPOSITORY_RULES.md using a delta-scoped LLM check.

Ported from ``tenferro-rs/scripts/repository-rules-review.py`` by way of
strided-rs, and adapted to this workspace: the rule sections and their path
routing differ, and the deterministic checks enforce the rules that are exactly
machine-checkable -- the two from
https://github.com/tensor4all/tensor4all-rs/issues/566 and the shared
"Unit Test Organization" rule -- rather than tenferro's AD boundary.

All rule checks are delta-scoped on purpose. #566 records existing violations of
both of its rules, so a repo-wide gate would fail on that backlog; these stop
new violations from landing while the backlog burns down separately.
tensor4all-rs#1105 moved the last inline `mod tests {` block out, so the inline
unit-test check has no backlog and is exact.

The #917 file-size check is selected with ``--file-growth`` and ``--dry-run``.
Hosted CI enables it; it produces advisory warnings only.

Local API key loading uses the ``python-dotenv`` library. Install dev helpers with:

    python3 -m pip install -r scripts/requirements-dev.txt

When ``.env`` exists at the repository root it is loaded automatically.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "REPOSITORY_RULES.md"
TIPS_PATH = ROOT / "PERFORMANCE_TIPS.md"
PROMPT_PATH = ROOT / "ai" / "prompts" / "repository-rules-review.md"
PROMPT_VERSION = "1"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
MAX_DIFF_CHARS = 60_000
MAX_FILE_DIFF_CHARS = 40_000
MAX_FINDINGS_PER_CHUNK = 8
HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)

# Only tensorbackend may depend on tenferro directly (Dense Layout And Linear
# Algebra). #566 Phase 3 records the existing violations in core, simplett,
# and treetci, so this check is delta-scoped: it rejects new dependency
# lines, not the backlog. dev-dependencies are out of scope --
# the rule is about the runtime route, and test code may reach for tenferro
# fixtures directly.
TENFERRO_ROUTE_CRATE = "tensor4all-tensorbackend"
# Wiring-only crates: no code, only dependency tables that select tenferro
# features. tensor4all-cpu-platform picks the CPU backend by target.
TENFERRO_WIRING_CRATES = frozenset({"tensor4all-cpu-platform"})
TENFERRO_DEP_TABLES = frozenset({"dependencies", "build-dependencies"})
CARGO_TABLE = re.compile(r"^\[([^\]]+)\]\s*$")
# `tenferro-linalg = ...`, `"tenferro-linalg" = ...`, `tenferro-linalg.workspace`
TENFERRO_DEP_KEY = re.compile(r"""^\s*["']?(tenferro-[\w-]+)["']?\s*(?:\.|=)""")
# Cargo renaming hides the real crate behind an arbitrary key:
#   linalg = { package = "tenferro-linalg", workspace = true }
TENFERRO_PACKAGE_VALUE = re.compile(r"""\bpackage\s*=\s*["'](tenferro-[\w-]+)["']""")

# `ignore` and `no_run` doctests are prohibited without explicit approval
# (Documentation Examples). #566 Phase 1 lists the two existing `no_run` sites.
PROHIBITED_DOCTEST_FENCE = re.compile(
    r"^\s*(?://[/!]|\*)?\s*(?:`{3,}|~{3,})\s*([\w,\s-]*)$"
)
PROHIBITED_DOCTEST_ATTRS = frozenset({"ignore", "no_run"})

# Inline unit-test blocks belong in their own file (shared rules
# `rules/rust/index.md`, "Unit Test Organization"): the module keeps only
# `#[cfg(test)] mod tests;` and the suite lives in `src/<module>/tests.rs`.
# The `pub(crate)` forms are covered too. tensor4all-rs#1105 moved the last
# inline block out, so this rejects every new one instead of a backlog; a
# genuinely tiny leaf module is handled by the maintainer waiver label.
INLINE_TEST_MODULE = re.compile(r"^\s*(?:pub(?:\(crate\))?\s+)?mod tests\s*\{")

SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)\bAuthorization:\s*Bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
SECRET_NAME = (
    r"[\w.-]*(?:api[_-]?key|token|secret|password|passwd|pwd|client[_-]?secret|"
    r"private[_-]?key)[\w.-]*"
)
# A typed declaration puts the type between the name and the value:
#   const API_KEY: &str = "....";     let api_key: String = "...".into();
# Matching only `name <sep> value` redacts the type and leaves the literal, so
# allow a short annotation before the separator that precedes the value.
SECRET_ANNOTATION = r"(?:[ \t]*:[^=\n]{0,40})?"
# The value alternatives are ordered quoted-first so a quoted secret is
# consumed whole. Putting the bare-token alternative first would stop at the
# first space inside a quoted passphrase and upload the remainder. (Spelling
# out an example assignment here would make this file trip its own guard.)
SECRET_VALUE = r"""(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s#]+)"""
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<name>" + SECRET_NAME + r")(?P<sep>" + SECRET_ANNOTATION
    + r"[ \t]*[:=][ \t]*)(?P<value>" + SECRET_VALUE + r")"
)
# Quoted credentials may legitimately contain spaces (a diceware passphrase is
# prose by construction), so the value shape cannot be the discriminator.
QUOTED_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<name>" + SECRET_NAME + r")" + SECRET_ANNOTATION + r"\s*[:=]\s*"
    r"""(?:"[^"\r\n]{8,}"|'[^'\r\n]{8,}')"""
)
# An assignment whose value has not started yet on this line: the credential
# lands on the following line, where no credential-shaped name is in sight.
OPEN_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<name>" + SECRET_NAME + r")" + SECRET_ANNOTATION + r"[ \t]*[:=][ \t]*$"
)
# A value standing alone on its own line, as the continuation of the above.
STANDALONE_VALUE = re.compile(r"""^[ \t]*(?:"[^"\r\n]{4,}"|'[^'\r\n]{4,}')[ \t]*[,;]?[ \t]*$""")
# A quote opened on this line and closed on a later one hides the value from
# any single-line pattern, so treat the opening alone as disqualifying.
UNTERMINATED_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<name>" + SECRET_NAME + r")" + SECRET_ANNOTATION
    + r"""\s*[:=]\s*["'][^"'\r\n]*$"""
)
# Since the value may be prose, the name has to carry the discrimination.
# `token_type`, `secret_name`, and `key_path` describe a credential rather than
# holding one, and their values are ordinary text.
SECRET_NAME_METADATA = re.compile(
    r"(?i)[_-]?(?:type|kind|name|names|id|ids|len|length|size|count|path|paths"
    r"|file|files|dir|env|var|vars|header|prefix|suffix|field|fields|url|uri"
    r"|list|set|map|schema|format|scheme|class|enum|error|regex|pattern|label"
    r"|source|store|provider|policy|status|state|mode|version)$"
)


def is_credential_name(name: str) -> bool:
    """Whether a matched identifier holds a credential rather than describes one."""
    return not SECRET_NAME_METADATA.search(name)


SEVERITY_ALIASES = {
    "block": "block",
    "blocker": "block",
    "critical": "block",
    "error": "block",
    "fail": "block",
    "failure": "block",
    "warn": "warn",
    "warning": "warn",
    "minor": "warn",
    "info": "warn",
    "informational": "warn",
}

ALWAYS_SECTIONS = frozenset(
    {
        "Public Boundary Safety Audits",
        "Work Logs And Design Records",
        "No Hidden Dense Materialization In Production Paths",
        "Tensor Network Test Comparisons",
    }
)

# Sections a human reviewer owns; never routed to the LLM. "Base Branch
# Synchronization" is a git-workflow protocol -- nothing inside a diff can
# satisfy or violate it, so showing it to a diff-scoped reviewer only invites
# invented findings.
# "Audit procedure" (PERFORMANCE_TIPS.md) instructs the auditor; a diff cannot
# violate it.
HUMAN_ONLY_SECTIONS = frozenset({"Base Branch Synchronization", "Audit procedure"})

PERFORMANCE_SECTIONS = (
    "Pointwise TT Readout In A Loop",
    "Uncached Target Function Across Sweeps",
    "Wrapper That Disables Batching",
    "Full Quantics Grid Sweeps",
    "Per-Call Reconstruction Inside Loops",
    "Owned Vector Cache Keys",
    "Std Hasher On Integer Keys Of A Per-Point Map",
    "Cache Key Rebuilt Per Request",
    "Owned Copy Of A Dense Payload On A Per-Call Path",
    "Input Validation Repeated Per Pass",
    "Batch Entry Point Driven One Point At A Time",
    "Silent Slow-Path Fallback",
    "Unpinned Timing Claims",
)

SECTION_TRIGGERS: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    (
        re.compile(r"^crates/[^/]+/src/.*\.rs$"),
        frozenset({"File Growth Dry Run"}),
    ),
    (
        re.compile(r"tensor4all-capi|/capi|\.h$|cbindgen|_capi"),
        frozenset(
            {
                "Language Bindings",
                "Index Identity Semantics",
                "Unsafe Code Boundary",
            }
        ),
    ),
    (
        re.compile(r"julia|python|Tensor4all\.jl|binding"),
        frozenset({"Language Bindings"}),
    ),
    (
        re.compile(
            r"tensor4all-tensorbackend|matrix\.rs|/storage|linalg|gemm|/svd"
            r"|/qr|matrixlu|factorize"
        ),
        frozenset(
            {
                "No Hidden Dense Materialization In Production Paths",
                "Unsafe Code Boundary",
            }
        ),
    ),
    (
        re.compile(
            r"tensor4all-(?:treetn|simplett|itensorlike)"
            r"|tensortrain|treetn|/tt\b|mps|mpo"
        ),
        frozenset(
            {
                "No Hidden Dense Materialization In Production Paths",
                "Tensor Network Test Comparisons",
                "Index Identity Semantics",
            }
        ),
    ),
    (
        re.compile(
            r"tensor4all-(?:tensorci|treetci|quanticstci|aci"
            r"|interpolativeqtt|quanticstransform)|crossinterpolate|pivot"
        ),
        frozenset(
            {
                "No Hidden Dense Materialization In Production Paths",
                "Tensor Network Test Comparisons",
            }
        ),
    ),
    # PERFORMANCE_TIPS.md sections: TT evaluation, TCI, caches, contraction.
    (
        re.compile(
            r"tensor4all-(?:tensorci|treetci|quanticstci|aci|treeaci|simplett"
            r"|treetn|partitionedtreetn)|evaluat|cache|contract"
            r"|benchmarks/"
        ),
        frozenset(PERFORMANCE_SECTIONS),
    ),
    (
        re.compile(
            r"graph|petgraph|topology|node_name_network|site_index_network"
            r"|/restructure|/transform"
        ),
        frozenset({"Tensor Network Test Comparisons"}),
    ),
    (
        re.compile(r"cache|evaluator|memo|/context"),
        frozenset({"No Hidden Dense Materialization In Production Paths"}),
    ),
    (
        re.compile(r"(?:^|/)ad(?:/|\.rs$)|autodiff|differentia|grad|vjp|jvp"),
        frozenset({"Differentiability And AD Preservation"}),
    ),
    (
        re.compile(r"index|/inds|index_ops|prime|tags"),
        frozenset({"Index Identity Semantics"}),
    ),
    (
        re.compile(r"unsafe|/ffi|raw|_ptr|get_unchecked"),
        frozenset({"Unsafe Code Boundary"}),
    ),
    (
        re.compile(r"Cargo\.toml$"),
        frozenset({"File Organization"}),
    ),
    (
        re.compile(r"(?:^|/)tests?/|_tests?\.rs$|(?:^|/)tests?\.rs$"),
        frozenset(
            {
                "Tensor Network Test Comparisons",
            }
        ),
    ),
    (
        re.compile(r"coverage-thresholds\.json$|check-coverage"),
        frozenset({"Tensor Network Test Comparisons"}),
    ),
    (
        re.compile(r"^docs/worklogs/|^docs/design/"),
        frozenset({"Work Logs And Design Records"}),
    ),
    (
        re.compile(r"^docs/book/|^docs/tutorial-code/|tutorial"),
        frozenset(
            {
                "Online Tutorial Synchronization",
                "Tensor Network Test Comparisons",
            }
        ),
    ),
    (
        re.compile(r"^docs/api/"),
        frozenset({"File Organization"}),
    ),
    (
        re.compile(r"\.md$|^README"),
        frozenset(
            {
                "Online Tutorial Synchronization",
                "No Hidden Dense Materialization In Production Paths",
            }
        ),
    ),
    (
        re.compile(r"\.rs$"),
        frozenset(
            {
                "File Organization",
                "No Hidden Dense Materialization In Production Paths",
            }
        ),
    ),
)


# Signals in the changed lines themselves. Path routing cannot see that a file
# under a generic path just gained an `unsafe` block or an `anyhow::Result`.
CONTENT_TRIGGERS: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    (
        re.compile(r"\bunsafe\b|get_unchecked|from_raw_parts|\bas_ptr\b|\bas_mut_ptr\b"),
        frozenset({"Unsafe Code Boundary"}),
    ),
    (
        re.compile(r"anyhow::Result|anyhow!|\bbail!\("),
        frozenset({"Tensor Network Test Comparisons"}),
    ),
    (
        re.compile(r"to_dense\(|contract_to_tensor\(|fulltensor\(|max_dense_elements"),
        frozenset({"No Hidden Dense Materialization In Production Paths"}),
    ),
    (
        re.compile(r"tenferro_\w+::|CpuBackend|with_default_backend"),
        frozenset({"No Hidden Dense Materialization In Production Paths"}),
    ),
    (
        re.compile(r"\.id\(\)|IndexLike::Id|prime_level"),
        frozenset({"Index Identity Semantics"}),
    ),
    (
        re.compile(r"petgraph|NodeNameNetwork|SiteIndexNetwork"),
        frozenset({"Tensor Network Test Comparisons"}),
    ),
    (
        re.compile(r"OnceLock|lazy_static|thread_local!|struct \w*Cache\b"),
        frozenset({"No Hidden Dense Materialization In Production Paths"}),
    ),
    (
        re.compile(r"#\[cfg\(test\)\]|\binclude!\("),
        frozenset({"Tensor Network Test Comparisons"}),
    ),
    (
        re.compile(r"t4a_[a-z_]+|catch_unwind|extern \"C\""),
        frozenset({"Language Bindings"}),
    ),
)


@dataclass(frozen=True)
class Finding:
    id: str
    severity: str
    rule_section: str
    file: str
    line: int | None
    summary: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity,
            "rule_section": self.rule_section,
            "file": self.file,
            "line": self.line,
            "summary": self.summary,
            "detail": self.detail,
        }


def run_git(args: list[str], cwd: Path = ROOT, *, check: bool = True) -> str:
    # Without this git C-quotes non-ASCII pathnames ("\346\227\245..."), and
    # the quoted form matches no real path: per_file_diffs yields nothing and
    # the extension-based deterministic checks never recognise the file.
    completed = subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def untracked_files() -> list[str]:
    """Paths git does not track yet, honouring .gitignore."""
    output = run_git(["ls-files", "--others", "--exclude-standard"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def untracked_file_diff(path: str) -> str:
    """Synthesize an all-added diff for a file with no committed counterpart.

    `git diff <base>` compares the working tree against a commit, so an
    untracked path has no object to compare and is omitted entirely. In
    worktree mode -- the documented local preview -- that means a brand new
    file is reviewed by nothing at all and the preview reports a false pass.
    `--no-index` against /dev/null gives a normal diff without touching the
    index; it exits 1 because the inputs differ, which is not an error here.
    """
    return run_git(
        ["diff", "--unified=3", "--no-index", "--", "/dev/null", path],
        check=False,
    )


def changed_files(base: str, head: str, *, worktree: bool = False) -> list[str]:
    if worktree:
        output = run_git(["diff", "--name-only", base])
        tracked = [line.strip() for line in output.splitlines() if line.strip()]
        return sorted({*tracked, *untracked_files()})
    output = run_git(["diff", "--name-only", f"{base}...{head}"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def unified_diff(base: str, head: str, *, worktree: bool = False) -> str:
    if worktree:
        pieces = [run_git(["diff", "--unified=3", base])]
        pieces.extend(untracked_file_diff(path) for path in untracked_files())
        return "\n".join(piece for piece in pieces if piece.strip())
    return run_git(["diff", "--unified=3", f"{base}...{head}"])


def per_file_diffs(
    base: str,
    head: str,
    files: list[str],
    *,
    worktree: bool = False,
) -> dict[str, str]:
    diffs: dict[str, str] = {}
    untracked = set(untracked_files()) if worktree else set()
    for path in files:
        if path in untracked:
            diff = untracked_file_diff(path)
        elif worktree:
            diff = run_git(["diff", "--unified=3", base, "--", path])
        else:
            diff = run_git(["diff", "--unified=3", f"{base}...{head}", "--", path])
        if diff.strip():
            diffs[path] = diff
    return diffs


def configure_dotenv(*, explicit: Path | None, skip: bool) -> None:
    """Load environment variables with python-dotenv."""
    if skip:
        return

    path = explicit if explicit is not None else ROOT / ".env"
    if not path.is_file():
        if explicit is not None:
            print(f"dotenv file not found: {path}", file=sys.stderr)
            raise SystemExit(1)
        return

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        print(
            "python-dotenv is required to load .env; install with: "
            "python3 -m pip install -r scripts/requirements-dev.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    load_dotenv(path, override=False)


def parse_repository_rules_sections(
    paths: tuple[Path, ...] = (RULES_PATH, TIPS_PATH),
) -> dict[str, str]:
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    sections: dict[str, str] = {}
    current_title: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_title is not None:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = line.removeprefix("## ").strip()
            current_lines = [line]
            continue
        if current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        sections[current_title] = "\n".join(current_lines).strip()
    return sections


def select_rule_sections(
    files: list[str],
    added: dict[str, list[tuple[int, str]]] | None = None,
) -> list[str]:
    """Pick the rule sections to show the reviewer.

    Path routing alone misses rule-relevant code added under a generic path:
    an `unsafe` block in `crates/tensor4all-core/src/lib.rs` matched no
    trigger, so `Unsafe Code Boundary` was never supplied -- and the prompt
    forbids inventing requirements that were not supplied, making the rule
    unenforceable there. Content triggers close that gap without making every
    safety section unconditional.
    """
    selected = set(ALWAYS_SECTIONS)
    for path in files:
        for pattern, section_names in SECTION_TRIGGERS:
            if pattern.search(path):
                selected.update(section_names)

    if added:
        for entries in added.values():
            for _line_no, text in entries:
                for pattern, section_names in CONTENT_TRIGGERS:
                    if pattern.search(text):
                        selected.update(section_names)
    return sorted(selected - HUMAN_ONLY_SECTIONS)


def build_rules_payload(section_names: list[str]) -> str:
    sections = parse_repository_rules_sections()
    chunks: list[str] = []
    for name in section_names:
        body = sections.get(name)
        if body:
            chunks.append(body)
    if not chunks:
        return sections.get("Tensor Network Test Comparisons", "")
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Hunk-aware diff parsing (tensor4all/tensor4all-meta#88).
#
# `git diff` is a state machine. `@@ -a,b +c,d @@` opens a hunk whose body is
# exactly `b` old-side and `d` new-side lines, and `+++ ` is a file header ONLY
# while no body is outstanding. Deciding "this line is a header" from its prefix
# alone let an ADDED line whose content begins with `++ ` -- rendered as
# `+++ b/whatever` -- become the current file, so the pull-request author chose
# the path every later addition was attributed to and every path-keyed rule read
# or skipped the wrong file.
#
# Every helper below shares this one parse so the entry points cannot drift
# apart again.
# ---------------------------------------------------------------------------

DEV_NULL = "/dev/null"

# Lines git emits that are never file content and never file headers.
DIFF_META_PREFIXES = (
    "index ",
    "old mode ",
    "new mode ",
    "deleted file mode ",
    "new file mode ",
    "similarity index ",
    "dissimilarity index ",
    "copy from ",
    "copy to ",
    "rename from ",
    "rename to ",
    "Binary files ",
)


@dataclass(frozen=True)
class HunkLine:
    """One body line of a hunk. ``kind`` is ``"+"``, ``"-"`` or ``" "``."""

    kind: str
    text: str
    old_lineno: int | None
    new_lineno: int | None


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[HunkLine, ...]

    def added(self) -> list[HunkLine]:
        return [line for line in self.lines if line.kind == "+"]


@dataclass(frozen=True)
class FileDiff:
    old_path: str | None
    new_path: str | None
    hunks: tuple[Hunk, ...]

    @property
    def path(self) -> str | None:
        """Path rules key on: the new side, or the old one for a delete."""

        return self.new_path or self.old_path


@dataclass
class _OpenHunk:
    """A hunk whose `@@` budget is still deciding what its body is."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    old_left: int
    new_left: int
    old_no: int
    new_no: int
    lines: list[HunkLine]

    @classmethod
    def from_header(cls, match: re.Match[str]) -> "_OpenHunk":
        # HUNK_HEADER groups: 1 old_start, 2 old_count, 3 new_start,
        # 4 new_count, 5 trailing section heading.
        old_count = int(match.group(2) or 1)
        new_count = int(match.group(4) or 1)
        old_start = int(match.group(1))
        new_start = int(match.group(3))
        return cls(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            old_left=old_count,
            new_left=new_count,
            old_no=old_start,
            new_no=new_start,
            lines=[],
        )

    @property
    def spent(self) -> bool:
        """True once the header's line budget has been fully accounted for."""

        return self.old_left <= 0 and self.new_left <= 0

    def consume(self, raw: str) -> bool:
        """Take one body line; return False when ``raw`` cannot be body content."""

        marker = raw[:1]
        if marker == "+":
            self.lines.append(HunkLine("+", raw[1:], None, self.new_no))
            self.new_no += 1
            self.new_left -= 1
        elif marker == "-":
            self.lines.append(HunkLine("-", raw[1:], self.old_no, None))
            self.old_no += 1
            self.old_left -= 1
        elif marker == " " or raw == "":
            self.lines.append(
                HunkLine(" ", raw[1:] if raw else "", self.old_no, self.new_no)
            )
            self.old_no += 1
            self.new_no += 1
            self.old_left -= 1
            self.new_left -= 1
        else:
            return False
        return True

    def close(self) -> Hunk:
        return Hunk(
            old_start=self.old_start,
            old_count=self.old_count,
            new_start=self.new_start,
            new_count=self.new_count,
            lines=tuple(self.lines),
        )


_GIT_C_ESCAPES = {
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "\\": "\\",
    '"': '"',
}


def _unquote_diff_path(path: str) -> str:
    """Undo git's C-style quoting of a path containing spaces or escapes.

    git quotes with `core.quotePath` on (the default) and escapes non-ASCII as
    three-digit OCTAL BYTES -- `"src/na\\303\\257ve file.rs"` -- which is not JSON
    and not `unicode_escape` either: the escapes are bytes of the UTF-8 encoding,
    so they have to be reassembled before decoding. Getting this wrong is not
    cosmetic: the path then matches no entry in `git diff --name-status -z`, and
    the cross-check in `paths_consistent` blocks every diff that touches a
    non-ASCII path.
    """

    if not (len(path) >= 2 and path.startswith('"') and path.endswith('"')):
        return path
    body = path[1:-1]
    raw = bytearray()
    index = 0
    while index < len(body):
        char = body[index]
        if char != "\\":
            raw.extend(char.encode("utf-8"))
            index += 1
            continue
        index += 1
        if index >= len(body):
            raw.extend(b"\\")
            break
        escape = body[index]
        octal = body[index : index + 3]
        if escape in _GIT_C_ESCAPES:
            raw.extend(_GIT_C_ESCAPES[escape].encode("utf-8"))
            index += 1
        elif len(octal) == 3 and all(digit in "01234567" for digit in octal):
            raw.append(int(octal, 8))
            index += 3
        else:
            raw.extend(escape.encode("utf-8"))
            index += 1
    return raw.decode("utf-8", "surrogateescape")


def _strip_diff_side(path: str, side: str) -> str | None:
    """``b/foo`` -> ``foo``; ``/dev/null`` -> None."""

    path = _unquote_diff_path(path.strip())
    if path == DEV_NULL:
        return None
    prefix = side + "/"
    if path.startswith(prefix):
        return path[len(prefix) :]
    return path


def _split_diff_git_header(header: str) -> tuple[str | None, str | None]:
    """Split the ``a/x b/y`` tail of a ``diff --git`` line.

    Plain paths are separated by " b/"; a path git had to quote (spaces,
    non-ASCII) arrives as two quoted strings.
    """

    header = header.strip()
    if header.startswith('"'):
        parts = re.findall(r'"(?:[^"\\]|\\.)*"', header)
        if len(parts) == 2:
            return _strip_diff_side(parts[0], "a"), _strip_diff_side(parts[1], "b")
    marker = header.rfind(" b/")
    if marker == -1:
        return None, None
    return (
        _strip_diff_side(header[:marker], "a"),
        _strip_diff_side(header[marker + 1 :], "b"),
    )


def parse_diff(diff_text: str) -> list[FileDiff]:
    """Parse ``git diff`` output into :class:`FileDiff` records.

    A ``+++ `` line is a header only while no hunk body is outstanding; inside a
    hunk the ``@@`` budget decides, so a ``+++``-looking line is content.
    """

    files: list[FileDiff] = []
    old_path: str | None = None
    new_path: str | None = None
    hunks: list[Hunk] = []
    open_hunk: _OpenHunk | None = None

    def close_hunk() -> None:
        nonlocal open_hunk
        if open_hunk is not None:
            hunks.append(open_hunk.close())
            open_hunk = None

    def flush_file() -> None:
        nonlocal old_path, new_path, hunks
        close_hunk()
        if old_path is None and new_path is None and not hunks:
            return
        files.append(FileDiff(old_path=old_path, new_path=new_path, hunks=tuple(hunks)))
        old_path = None
        new_path = None
        hunks = []

    for raw in diff_text.splitlines():
        if open_hunk is not None:
            if raw.startswith("\\"):
                # "\ No newline at end of file" is metadata, not a body line.
                continue
            if open_hunk.consume(raw):
                if open_hunk.spent:
                    close_hunk()
                continue
            # Truncated or malformed hunk: stop reading content and re-read this
            # line as a header candidate instead of swallowing it.
            close_hunk()
            if not raw.startswith(("diff --git ", "@@", "--- ")):
                continue

        if raw.startswith("diff --git "):
            flush_file()
            # The header names both sides; `--- `/`+++ ` refine it below and a
            # rename/copy pair overrides both. Keeping this default makes a
            # mode-only or binary file block report its paths at all.
            old_path, new_path = _split_diff_git_header(raw[len("diff --git ") :])
            continue
        if raw.startswith(("rename from ", "copy from ")):
            old_path = _unquote_diff_path(raw.split(" ", 2)[2].strip())
            continue
        if raw.startswith(("rename to ", "copy to ")):
            new_path = _unquote_diff_path(raw.split(" ", 2)[2].strip())
            continue
        if raw.startswith("--- "):
            old_path = _strip_diff_side(raw[4:], "a")
            continue
        if raw.startswith("+++ "):
            new_path = _strip_diff_side(raw[4:], "b")
            continue
        match = HUNK_HEADER.match(raw)
        if match:
            open_hunk = _OpenHunk.from_header(match)
            if open_hunk.spent:
                close_hunk()
            continue
        if raw.startswith(DIFF_META_PREFIXES):
            continue
        # Anything else outside a hunk is noise; it never names a file.

    flush_file()
    return files


def added_lines_with_text(diff_text: str) -> dict[str, list[tuple[int, str]]]:
    """Map each file to its added ``(new_line_number, text)`` pairs.

    Only the new side names a file: a deletion's ``+++ /dev/null`` yields no
    bucket, which is what the pre-#88 helper reported and what every call site
    already tolerates.
    """

    result: dict[str, list[tuple[int, str]]] = {}
    for file_diff in parse_diff(diff_text):
        if file_diff.new_path is None:
            continue
        bucket = result.setdefault(file_diff.new_path, [])
        for hunk in file_diff.hunks:
            for line in hunk.lines:
                if line.kind == "+" and line.new_lineno is not None:
                    bucket.append((line.new_lineno, line.text))
    return result


def added_line_numbers(
    added: dict[str, list[tuple[int, str]]],
) -> dict[str, set[int]]:
    """Reduce the parsed added lines to the line numbers used for anchoring."""
    return {path: {line for line, _ in entries} for path, entries in added.items()}


def changed_paths(diff_text: str) -> set[str]:
    """Every non-``/dev/null`` path the diff names, old side or new."""

    paths: set[str] = set()
    for file_diff in parse_diff(diff_text):
        if file_diff.old_path:
            paths.add(file_diff.old_path)
        if file_diff.new_path:
            paths.add(file_diff.new_path)
    return paths


def parse_name_status(text: str) -> dict[str, tuple[str, str | None]]:
    """Parse ``git diff --name-status -z`` output.

    Returns ``{path: (status, other_path)}`` where ``other_path`` is the old path
    of a rename or copy and ``None`` otherwise.
    """

    fields = [chunk for chunk in text.split("\0") if chunk != ""]
    result: dict[str, tuple[str, str | None]] = {}
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if status[:1] in ("R", "C"):
            if index + 1 >= len(fields):
                break
            result[fields[index + 1]] = (status, fields[index])
            index += 2
            continue
        if index >= len(fields):
            break
        result[fields[index]] = (status, None)
        index += 1
    return result


@dataclass(frozen=True)
class PathConsistency:
    """Difference between the paths a diff claims and the paths git lists."""

    unexpected: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.unexpected and not self.missing


def paths_consistent(diff_text: str, name_status_text: str) -> PathConsistency:
    """Cross-check the diff's claimed paths against `git diff --name-status -z`.

    Git reports the paths it used to build the diff, NUL-separated and therefore
    unambiguous. A diff naming a path git does not list has been tampered with;
    a gate must fail closed on a non-empty result rather than trust either side.
    This is the second layer behind `parse_diff`: the parser no longer invents
    paths, and if a later edit reintroduces one this still refuses.
    """

    parsed = parse_name_status(name_status_text)
    listed = set(parsed)
    for _status, other in parsed.values():
        if other:
            listed.add(other)
    claimed = changed_paths(diff_text)
    return PathConsistency(
        unexpected=tuple(sorted(claimed - listed)),
        missing=tuple(sorted(listed - claimed)),
    )


def path_consistency_finding(
    diff_text: str, name_status_text: str
) -> Finding | None:
    """Block when the diff and `git --name-status` disagree about the paths."""

    consistency = paths_consistent(diff_text, name_status_text)
    if consistency.ok:
        return None
    unexpected = ", ".join(consistency.unexpected) or "(none)"
    missing = ", ".join(consistency.missing) or "(none)"
    return Finding(
        id="diff-path-mismatch",
        severity="block",
        rule_section="Repository Rules Review",
        file=consistency.unexpected[0]
        if consistency.unexpected
        else (consistency.missing[0] if consistency.missing else "<unknown>"),
        line=None,
        summary="Diff names paths that git --name-status does not report",
        detail=(
            "The diff under review and `git diff --name-status -z` for the same "
            "base/head pair disagree, so at least one path was taken from the "
            f"diff text rather than from git: not listed by git: {unexpected}; "
            f"listed by git but absent from the diff: {missing}. The gate reads "
            "paths from diff headers, so it cannot be trusted until they match."
        ),
    )


def name_status(base: str, head: str, *, worktree: bool = False) -> str:
    """`git diff --name-status -z` for the same pair the reviewed diff came from.

    `unified_diff` synthesises all-added diffs for untracked files in worktree
    mode (they have no committed counterpart), so those paths are appended here
    as `A` records or the cross-check would report them as unexpected.
    """

    if worktree:
        listed = run_git(["diff", "--name-status", "-z", base])
        return listed + "".join(f"A\0{path}\0" for path in untracked_files())
    return run_git(["diff", "--name-status", "-z", f"{base}...{head}"])


def split_diff_chunks(file_diffs: dict[str, str]) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for path in sorted(file_diffs):
        piece = file_diffs[path]
        if len(piece) > MAX_FILE_DIFF_CHARS:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            chunks.extend(split_large_file_diff(piece))
            continue

        if current_len + len(piece) > MAX_DIFF_CHARS and current:
            chunks.append("\n".join(current))
            current = [piece]
            current_len = len(piece)
        else:
            current.append(piece)
            current_len += len(piece)

    if current:
        chunks.append("\n".join(current))
    return chunks


def joined_line_len(lines: list[str]) -> int:
    return len("\n".join(lines))


def line_deltas(line: str) -> tuple[int, int]:
    """How one diff body line advances the old and new file cursors."""
    if line.startswith("+"):
        return (0, 1)
    if line.startswith("-"):
        return (1, 0)
    if line.startswith("\\"):
        return (0, 0)
    return (1, 1)


def format_hunk_header(
    old_start: int, old_count: int, new_start: int, new_count: int, suffix: str
) -> str:
    return f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{suffix}"


def split_overlong_diff_line(prefix: list[str], line: str) -> list[str]:
    prefix_len = joined_line_len(prefix)
    line_budget = MAX_FILE_DIFF_CHARS - prefix_len - 1
    if line_budget <= 0:
        return ["\n".join([*prefix, line])]

    marker = line[:1] if line[:1] in {"+", "-", " "} else ""
    payload = line[1:] if marker else line
    payload_budget = max(1, line_budget - len(marker))
    chunks: list[str] = []
    for start in range(0, len(payload), payload_budget):
        piece = f"{marker}{payload[start : start + payload_budget]}"
        chunks.append("\n".join([*prefix, piece]))
    return chunks


def split_oversized_hunk(header: list[str], hunk: list[str]) -> list[str]:
    """Split one oversized hunk, rewriting the header for every emitted chunk.

    Repeating the original header would tell the model that each chunk starts
    at the hunk's first line, so findings in later chunks come back with line
    numbers thousands of lines too small. `filter_findings` then drops them, or
    worse keeps them against an unrelated added line that happens to collide.
    """
    if not hunk:
        return []

    parsed = HUNK_HEADER.match(hunk[0])
    if parsed is None:
        # Not a header we can renumber; fall back to repeating it verbatim
        # rather than inventing offsets.
        return split_oversized_hunk_verbatim(header, hunk)

    old_start = int(parsed.group(1))
    new_start = int(parsed.group(3))
    suffix = parsed.group(5)

    chunks: list[str] = []
    body: list[str] = []
    body_old = body_new = 0
    cursor_old, cursor_new = old_start, new_start

    def flush() -> None:
        nonlocal body, body_old, body_new, cursor_old, cursor_new
        if not body:
            return
        hunk_header = format_hunk_header(
            cursor_old, body_old, cursor_new, body_new, suffix
        )
        chunks.append("\n".join([*header, hunk_header, *body]))
        cursor_old += body_old
        cursor_new += body_new
        body = []
        body_old = body_new = 0

    for line in hunk[1:]:
        delta_old, delta_new = line_deltas(line)
        single_header = format_hunk_header(
            cursor_old + body_old, delta_old, cursor_new + body_new, delta_new, suffix
        )
        if joined_line_len([*header, single_header, line]) > MAX_FILE_DIFF_CHARS:
            flush()
            single_header = format_hunk_header(
                cursor_old, delta_old, cursor_new, delta_new, suffix
            )
            chunks.extend(
                split_overlong_diff_line([*header, single_header], line)
            )
            cursor_old += delta_old
            cursor_new += delta_new
            continue

        candidate_header = format_hunk_header(
            cursor_old, body_old + delta_old, cursor_new, body_new + delta_new, suffix
        )
        if (
            body
            and joined_line_len([*header, candidate_header, *body, line])
            > MAX_FILE_DIFF_CHARS
        ):
            flush()
        body.append(line)
        body_old += delta_old
        body_new += delta_new

    flush()
    if not chunks:
        chunks.append("\n".join([*header, hunk[0]]))
    return chunks


def split_oversized_hunk_verbatim(header: list[str], hunk: list[str]) -> list[str]:
    """Fallback for a hunk header this script cannot parse."""
    hunk_header = hunk[0]
    prefix = [*header, hunk_header]
    chunks: list[str] = []
    current = list(prefix)

    for line in hunk[1:]:
        if joined_line_len([*prefix, line]) > MAX_FILE_DIFF_CHARS:
            if current != prefix:
                chunks.append("\n".join(current))
                current = list(prefix)
            chunks.extend(split_overlong_diff_line(prefix, line))
            continue

        candidate = [*current, line]
        if current != prefix and joined_line_len(candidate) > MAX_FILE_DIFF_CHARS:
            chunks.append("\n".join(current))
            current = [*prefix, line]
        else:
            current = candidate

    if current != prefix:
        chunks.append("\n".join(current))
    else:
        chunks.append("\n".join(prefix))
    return chunks


def split_large_file_diff(diff_text: str) -> list[str]:
    """Split one file diff while preserving file headers in every chunk.

    The header/hunk split uses the same ``@@`` budget as `parse_diff`: a body
    line never opens a hunk, so a chunk boundary can never be chosen by diff
    content.
    """

    lines = diff_text.splitlines()
    header: list[str] = []
    hunks: list[list[str]] = []
    current_hunk: list[str] | None = None
    open_hunk: _OpenHunk | None = None

    for line in lines:
        if open_hunk is not None and current_hunk is not None:
            if line.startswith("\\") or open_hunk.consume(line):
                current_hunk.append(line)
                if open_hunk.spent:
                    open_hunk = None
                continue
            # Truncated or malformed budget: re-read this line as a header.
            open_hunk = None
        match = HUNK_HEADER.match(line)
        if line.startswith("@@"):
            if current_hunk is not None:
                hunks.append(current_hunk)
            current_hunk = [line]
            open_hunk = _OpenHunk.from_header(match) if match else None
            if open_hunk is not None and open_hunk.spent:
                open_hunk = None
        elif current_hunk is None:
            header.append(line)
        else:
            current_hunk.append(line)

    if current_hunk is not None:
        hunks.append(current_hunk)
    if not hunks:
        return [
            diff_text[start : start + MAX_FILE_DIFF_CHARS]
            for start in range(0, len(diff_text), MAX_FILE_DIFF_CHARS)
        ]

    chunks: list[str] = []
    current_lines = list(header)
    current_len = joined_line_len(current_lines)

    for hunk in hunks:
        hunk_len = joined_line_len(hunk)
        separator_len = 1 if current_lines else 0
        if joined_line_len([*header, *hunk]) > MAX_FILE_DIFF_CHARS:
            if current_lines != header:
                chunks.append("\n".join(current_lines))
                current_lines = list(header)
                current_len = joined_line_len(current_lines)
            chunks.extend(split_oversized_hunk(header, hunk))
            continue

        if (
            current_lines != header
            and current_len + separator_len + hunk_len > MAX_FILE_DIFF_CHARS
        ):
            chunks.append("\n".join(current_lines))
            current_lines = list(header)
            current_len = len("\n".join(current_lines))

        if current_lines:
            current_len += 1
        current_lines.extend(hunk)
        current_len += hunk_len

    if current_lines != header:
        chunks.append("\n".join(current_lines))
    return chunks


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    def mask(match: re.Match[str]) -> str:
        if not is_credential_name(match.group("name")):
            return match.group(0)
        return f"{match.group('name')}{match.group('sep')}[REDACTED_SECRET]"

    return SECRET_ASSIGNMENT.sub(mask, redacted)


def redact_file_diffs(file_diffs: dict[str, str]) -> dict[str, str]:
    return {path: redact_sensitive_text(diff) for path, diff in file_diffs.items()}


def contains_sensitive_text(text: str) -> bool:
    if any(pattern.search(text) for pattern in SECRET_VALUE_PATTERNS):
        return True
    for pattern in (QUOTED_SECRET_ASSIGNMENT, UNTERMINATED_SECRET_ASSIGNMENT):
        for match in pattern.finditer(text):
            if is_credential_name(match.group("name")):
                return True
    return False


def sensitive_diff_location(diff_text: str) -> tuple[str, int] | None:
    """Locate the first added line that must not be uploaded.

    Lines are examined in diff order rather than per file, because a credential
    can be split across two: the assignment stays unchanged on a context line
    and only the value line is replaced. Checking added lines in isolation sees
    a bare string literal with no credential-shaped name anywhere on it.

    The scan walks hunks from `parse_diff`, so a line whose content renders as
    ``+++ ...`` is an ordinary added line here instead of being skipped as a file
    header -- which is exactly how such a line used to avoid
    `contains_sensitive_text`.
    """
    for file_diff in parse_diff(diff_text):
        if file_diff.new_path is None:
            continue
        for hunk in file_diff.hunks:
            # Set when the previous surviving line opened an assignment whose
            # value has not appeared yet, so the next line's literal is that value.
            awaiting_value = False
            for line in hunk.lines:
                if line.kind == "-":
                    # A deleted line neither survives nor breaks the continuation.
                    continue
                body = line.text
                is_added = line.kind == "+"
                if is_added and line.new_lineno is not None:
                    if contains_sensitive_text(body):
                        return file_diff.new_path, line.new_lineno
                    if awaiting_value and STANDALONE_VALUE.match(body):
                        return file_diff.new_path, line.new_lineno
                opener = OPEN_SECRET_ASSIGNMENT.search(body)
                awaiting_value = bool(opener) and is_credential_name(
                    opener.group("name")
                )
    return None


def sensitive_diff_finding(diff_text: str) -> Finding | None:
    location = sensitive_diff_location(diff_text)
    if location is None:
        return None
    file, line = location
    return Finding(
        id="sensitive-diff",
        severity="block",
        rule_section="External LLM Review",
        file=file,
        line=line,
        summary="Sensitive-looking diff content detected before LLM upload",
        detail=(
            "External LLM review was skipped because the diff contains token, "
            "secret, password, authorization header, or private-key shaped text. "
            "Remove the sensitive value or use a maintainer-approved waiver if "
            "this is a verified false positive."
        ),
    )


def parse_findings(raw: Any) -> tuple[str, list[Finding]]:
    if not isinstance(raw, dict):
        raise ValueError("model response must be a JSON object")

    verdict = raw.get("verdict")
    if verdict not in {"pass", "fail"}:
        raise ValueError("verdict must be 'pass' or 'fail'")

    findings_raw = raw.get("findings", [])
    if not isinstance(findings_raw, list):
        raise ValueError("findings must be a list")

    findings: list[Finding] = []
    for index, item in enumerate(findings_raw[:MAX_FINDINGS_PER_CHUNK]):
        if not isinstance(item, dict):
            raise ValueError(f"findings[{index}] must be an object")
        severity_raw = str(item.get("severity", "warn")).strip().lower()
        severity = SEVERITY_ALIASES.get(severity_raw, "warn")
        line = item.get("line")
        if line is not None and not isinstance(line, int):
            raise ValueError(f"findings[{index}].line must be an integer or null")
        findings.append(
            Finding(
                id=str(item.get("id", f"finding-{index + 1}")),
                severity=severity,
                rule_section=str(item.get("rule_section", "unknown")),
                file=str(item.get("file", "")),
                line=line,
                summary=str(item.get("summary", "")),
                detail=str(item.get("detail", "")),
            )
        )

    return verdict, findings


def extract_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as err:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError(f"model response was not valid JSON: {err}") from err
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as embedded_err:
            raise ValueError(
                f"model response was not valid JSON: {embedded_err}"
            ) from embedded_err

    if not isinstance(parsed, dict):
        raise ValueError("parsed JSON must be an object")
    return parsed


def llm_response_error_finding(error: BaseException) -> Finding:
    return Finding(
        id="llm-review-unusable",
        severity="block",
        rule_section="External LLM Review",
        file="",
        line=None,
        summary="External LLM review did not complete",
        detail=(
            "The repository-rules review could not send the request or parse "
            f"the model response: {type(error).__name__}: {error}"
        ),
    )


def api_key_error_finding(reason: str) -> Finding:
    return Finding(
        id="llm-api-key-invalid",
        severity="block",
        rule_section="External LLM Review",
        file="",
        line=None,
        summary="DEEPSEEK_API_KEY is unusable",
        detail=(
            f"{reason} Re-set the repository secret; the value itself is never "
            "printed. Until then the LLM pass cannot run."
        ),
    )


def api_key_problem(api_key: str) -> str | None:
    """Describe why a key cannot be sent, without echoing the key.

    HTTP header values are encoded as latin-1, so a key carrying non-ASCII
    text -- a pasted ellipsis from a masked console display, or mojibake --
    raises UnicodeEncodeError before any request leaves the machine. That is a
    ValueError subclass, so it used to surface as "did not produce usable
    JSON", pointing the reader at the model instead of at the secret.
    """
    if not api_key:
        return "The secret is empty."
    if not api_key.isascii():
        offsets = [
            str(index)
            for index, char in enumerate(api_key)
            if not char.isascii()
        ]
        return (
            "The secret contains non-ASCII characters at offset(s) "
            f"{', '.join(offsets[:10])}, which cannot be sent in an HTTP "
            "Authorization header. A masked value copied from a console "
            "display is the usual cause."
        )
    if any(char.isspace() for char in api_key):
        return "The secret contains whitespace."
    return None


def filter_findings(
    findings: list[Finding],
    files: list[str],
    added_lines: dict[str, set[int]],
    *,
    allow_global: bool = True,
) -> list[Finding]:
    allowed_files = set(files)
    kept: list[Finding] = []
    for finding in findings:
        if not finding.file:
            if allow_global:
                kept.append(finding)
            continue
        if finding.file and finding.file not in allowed_files:
            continue
        if finding.line is None and finding.severity == "block":
            continue
        if finding.line is not None:
            if finding.line not in added_lines.get(finding.file, set()):
                continue
        kept.append(finding)
    return kept


def reconcile_verdict(findings: list[Finding]) -> str:
    return "fail" if any(item.severity == "block" for item in findings) else "pass"


# Every way the request can fail below the JSON layer. OSError covers
# socket.timeout, TimeoutError, ConnectionResetError, ssl.SSLError, and
# urllib.error.URLError/HTTPError. http.client.HTTPException is a sibling of
# OSError, not a subclass, so a truncated chunked response (IncompleteRead)
# needs naming separately. socket.timeout only aliases TimeoutError from
# Python 3.10 on, so listing TimeoutError alone is not enough on 3.9.
TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    http.client.HTTPException,
)
NETWORK_RETRIES = 2
RETRY_BACKOFF_SECONDS = 5.0
# Wall-clock ceiling for all LLM traffic in one run. The workflow allows 20
# minutes; a run that blows past it is killed mid-request, so neither the
# exception handler nor the report ever executes and the gate fails with no
# diagnostic at all -- the exact failure this retry logic exists to prevent.
# Finishing under our own budget guarantees a report is always posted.
DEFAULT_BUDGET_SECONDS = 900.0


def call_deepseek(
    *,
    api_key: str,
    model: str,
    api_url: str,
    system_prompt: str,
    user_content: str,
    timeout: float,
    deadline: float | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: BaseException | None = None
    for attempt in range(1, NETWORK_RETRIES + 1):
        attempt_timeout = timeout
        if deadline is not None:
            attempt_timeout = min(timeout, max(1.0, deadline - time.monotonic()))
        try:
            with urllib.request.urlopen(request, timeout=attempt_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except TRANSPORT_ERRORS as exc:
            # Timeouts and connection resets are common enough on large diffs
            # that one blocked PR per blip is not an acceptable failure mode.
            last_error = exc
            if attempt == NETWORK_RETRIES:
                raise
            if deadline is not None and time.monotonic() + RETRY_BACKOFF_SECONDS >= deadline:
                # Retrying would run past the budget and get the job killed,
                # which loses the report entirely. Fail now, with a diagnostic.
                raise
            print(
                f"LLM request attempt {attempt}/{NETWORK_RETRIES} failed "
                f"({type(exc).__name__}: {exc}); retrying in "
                f"{RETRY_BACKOFF_SECONDS:.0f}s",
                file=sys.stderr,
            )
            time.sleep(RETRY_BACKOFF_SECONDS)
    else:  # pragma: no cover - the loop either breaks or raises
        raise last_error if last_error else RuntimeError("no response")
    content = body["choices"][0]["message"]["content"]
    return extract_json_payload(content)


def review_chunk(
    *,
    api_key: str,
    model: str,
    api_url: str,
    system_prompt: str,
    rules_text: str,
    changed: list[str],
    diff_chunk: str,
    timeout: float,
    deadline: float | None = None,
) -> tuple[str, list[Finding]]:
    user_content = "\n\n".join(
        [
            f"Prompt version: {PROMPT_VERSION}",
            "Changed files:",
            "\n".join(f"- {path}" for path in changed),
            "Applicable REPOSITORY_RULES sections:",
            rules_text,
            "Output limits:",
            f"- Return at most {MAX_FINDINGS_PER_CHUNK} findings for this diff chunk.",
            "- Do not split one root cause into multiple findings.",
            "- Do not invent requirements that are not explicit in the supplied rules.",
            "Unified diff (review only added/changed lines):",
            diff_chunk,
        ]
    )
    parsed = call_deepseek(
        api_key=api_key,
        model=model,
        api_url=api_url,
        system_prompt=system_prompt,
        user_content=user_content,
        timeout=timeout,
        deadline=deadline,
    )
    return parse_findings(parsed)


def merge_findings(all_findings: list[Finding]) -> list[Finding]:
    merged: dict[tuple[str, str, str, int | None], Finding] = {}
    for finding in all_findings:
        key = (finding.id, finding.file, finding.summary, finding.line)
        existing = merged.get(key)
        if existing is None or (
            finding.severity == "block" and existing.severity != "block"
        ):
            merged[key] = finding
    return list(merged.values())


def budget_exhausted_finding(reviewed: int, total: int, budget: float) -> Finding:
    return Finding(
        id="llm-budget-exhausted",
        severity="warn",
        rule_section="External LLM Review",
        file="",
        line=None,
        summary="External LLM review stopped at its time budget",
        detail=(
            f"Reviewed {reviewed} of {total} diff chunk(s) before the "
            f"{budget:.0f}s budget ran out, so part of this "
            "diff was not reviewed. Deterministic checks still covered all of "
            "it. Split the PR or raise --budget-seconds to review the rest."
        ),
    )


def llm_skipped_finding(reason: str) -> Finding:
    return Finding(
        id="llm-skipped",
        severity="warn",
        rule_section="External LLM Review",
        file="",
        line=None,
        summary="External LLM review was skipped",
        detail=reason,
    )


def summarize_llm_review(
    *,
    chunk_sizes: list[int],
    elapsed_seconds: float,
    returned_count: int,
    kept_count: int,
) -> str:
    dropped = returned_count - kept_count
    return (
        f"LLM review: {len(chunk_sizes)} chunk(s) "
        f"({', '.join(f'{size} chars' for size in chunk_sizes)}) "
        f"in {elapsed_seconds:.1f}s; "
        f"{returned_count} finding(s) returned, {kept_count} kept, "
        f"{dropped} dropped by diff-anchor filtering."
    )


def format_report(
    *,
    base: str,
    head: str,
    verdict: str,
    findings: list[Finding],
    waived: bool,
    llm_summary: str | None = None,
) -> str:
    lines = [
        f"Repository rules review ({base}...{head})",
        f"Verdict: {verdict}",
    ]
    if llm_summary:
        lines.append(llm_summary)
    if waived:
        lines.append("Waived by maintainer label.")
    if not findings:
        lines.append("No findings.")
        return "\n".join(lines)

    lines.append("Findings:")
    for finding in findings:
        location = finding.file or "<unknown>"
        if finding.line is not None:
            location = f"{location}:{finding.line}"
        lines.append(
            f"- [{finding.severity}] {finding.id} ({finding.rule_section}) "
            f"{location}: {finding.summary}"
        )
        if finding.detail:
            lines.append(f"  {finding.detail}")
    return "\n".join(lines)


def file_text_at(path: str, *, ref: str | None, worktree: bool) -> str:
    """Read one file as it exists on the reviewed side of the diff."""
    if worktree:
        return (ROOT / path).read_text(encoding="utf-8")
    if ref is None:
        raise ValueError("ref is required when worktree is false")
    return run_git(["show", f"{ref}:{path}"])


def crate_of(path: str) -> str | None:
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] == "crates":
        return parts[1]
    return None


def strip_toml_comment(line: str) -> str:
    """Drop a TOML comment, respecting `#` inside quoted values."""
    quote: str | None = None
    for index, char in enumerate(line):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            continue
        if char == "#":
            return line[:index]
    return line


def package_name_of(text: str) -> str | None:
    """The `name` under `[package]`, so a manifest can be identified by package."""
    table: str | None = None
    for line in text.splitlines():
        stripped = strip_toml_comment(line).strip()
        header = CARGO_TABLE.match(stripped)
        if header:
            table = header.group(1)
            continue
        if table != "package":
            continue
        match = re.match(r"""^name\s*=\s*["']([\w-]+)["']""", stripped)
        if match:
            return match.group(1)
    return None


def dependency_table_of(table: str) -> tuple[str, str | None] | None:
    """Classify a Cargo table header as a dependency table.

    Returns ``(kind, subtable_name)``. `[dependencies]` yields
    ``("dependencies", None)`` and `[dependencies.tenferro-linalg]` yields
    ``("dependencies", "tenferro-linalg")``, including the
    `[target.'cfg(...)'.dependencies]` forms. Workspace dependency declarations
    are excluded because they do not add a direct crate dependency.
    """
    parts = [part.strip().strip("\"'") for part in table.split(".")]
    for index, part in enumerate(parts):
        if part in TENFERRO_DEP_TABLES:
            if index > 0 and parts[index - 1] == "workspace":
                return None
            remainder = parts[index + 1 :]
            return (part, remainder[0] if remainder else None)
    return None


def tenferro_dependency_lines(text: str) -> dict[int, str]:
    """Map line number to crate name for every direct tenferro dependency.

    Cargo.toml names `tenferro-*` in unrelated places -- feature names under
    `[features]`, and `dep:` activations inside a feature list -- so the table
    context has to be tracked rather than grepped. Three declaration forms all
    count, and a bare-key regex sees only the first:

        tenferro-linalg.workspace = true            # bare key
        linalg = { package = "tenferro-linalg" }    # Cargo renaming
        [dependencies.tenferro-linalg]              # subtable
    """
    found: dict[int, str] = {}
    table: tuple[str, str | None] | None = None

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = strip_toml_comment(raw_line)
        header = CARGO_TABLE.match(line.strip())
        if header:
            table = dependency_table_of(header.group(1))
            if table is not None and table[1] and table[1].startswith("tenferro-"):
                found[line_no] = table[1]
            continue
        if table is None or table[0] not in TENFERRO_DEP_TABLES:
            continue

        renamed = TENFERRO_PACKAGE_VALUE.search(line)
        if renamed:
            found[line_no] = renamed.group(1)
            continue
        # A subtable already declared its crate on the header line; its body
        # keys are settings such as `workspace = true`, not dependency names.
        if table[1] is not None:
            continue
        key = TENFERRO_DEP_KEY.match(line)
        if key:
            found[line_no] = key.group(1)
    return found


def tenferro_route_violations(
    added: dict[str, list[tuple[int, str]]],
    *,
    ref: str | None,
    worktree: bool,
) -> list[str]:
    """Report newly added direct tenferro dependencies outside tensorbackend."""
    violations: list[str] = []
    for path in sorted(added):
        if not path.endswith("Cargo.toml"):
            continue
        added_numbers = {line for line, _ in added[path]}
        if not added_numbers:
            continue
        text = file_text_at(path, ref=ref, worktree=worktree)
        # Exempt the one package allowed to depend on tenferro, identified by
        # its package name rather than its directory. Skipping every manifest
        # outside `crates/` would exempt the workspace members that live
        # elsewhere -- xtask, tools/api-dump, docs/tutorial-code, docs/book-tests.
        package = package_name_of(text) or crate_of(path)
        if package == TENFERRO_ROUTE_CRATE or package in TENFERRO_WIRING_CRATES:
            continue
        declared = tenferro_dependency_lines(text)
        for line_no in sorted(added_numbers & declared.keys()):
            violations.append(f"{path}:{line_no}: {declared[line_no]}")
    return violations


def prohibited_doctest_violations(
    added: dict[str, list[tuple[int, str]]],
) -> list[str]:
    """Report added `ignore` / `no_run` doctest fences."""
    violations: list[str] = []
    for path in sorted(added):
        if not (path.endswith(".rs") or path.endswith(".md")):
            continue
        for line_no, text in added[path]:
            fence = PROHIBITED_DOCTEST_FENCE.match(text)
            if not fence:
                continue
            attrs = {
                attr.strip()
                for attr in fence.group(1).replace(",", " ").split()
            }
            banned = sorted(attrs & PROHIBITED_DOCTEST_ATTRS)
            if banned:
                violations.append(f"{path}:{line_no}: ```{' '.join(banned)}")
    return violations


def inline_test_module_violations(
    added: dict[str, list[tuple[int, str]]],
) -> list[str]:
    """Report added inline `mod tests { ... }` blocks."""
    violations: list[str] = []
    for path in sorted(added):
        if not path.endswith(".rs"):
            continue
        for line_no, text in added[path]:
            if INLINE_TEST_MODULE.match(text):
                violations.append(f"{path}:{line_no}: {text.strip()}")
    return violations


def deterministic_checks(
    files: list[str],
    *,
    added: dict[str, list[tuple[int, str]]] | None = None,
    ref: str | None = None,
    worktree: bool = False,
) -> list[Finding]:
    findings: list[Finding] = []
    if added is None:
        return findings

    route = tenferro_route_violations(added, ref=ref, worktree=worktree)
    if route:
        findings.append(
            Finding(
                id="tenferro-route",
                severity="block",
                rule_section="No Hidden Dense Materialization In Production Paths",
                file=route[0].split(":")[0],
                line=None,
                summary=(
                    f"New direct tenferro dependency outside "
                    f"{TENFERRO_ROUTE_CRATE}"
                ),
                detail=(
                    f"Route dense tensor and linear-algebra work through "
                    f"{TENFERRO_ROUTE_CRATE}. Existing violations are tracked "
                    "in tensor4all/tensor4all-rs#566 Phase 3; this check only "
                    "rejects new ones.\n" + "\n".join(route)
                ),
            )
        )

    doctests = prohibited_doctest_violations(added)
    if doctests:
        findings.append(
            Finding(
                id="prohibited-doctest",
                severity="block",
                rule_section="Documentation Examples",
                file=doctests[0].split(":")[0],
                line=None,
                summary="Added `ignore` or `no_run` doctest",
                detail=(
                    "Rustdoc examples and mdBook snippets must run; `ignore` "
                    "and `no_run` are prohibited without explicit approval. "
                    "Use the maintainer waiver label when a fence is "
                    "genuinely approved.\n" + "\n".join(doctests)
                ),
            )
        )

    inline_tests = inline_test_module_violations(added)
    if inline_tests:
        findings.append(
            Finding(
                id="inline-unit-tests",
                severity="block",
                rule_section="Unit Test Organization",
                file=inline_tests[0].split(":")[0],
                line=None,
                summary="New inline `mod tests {` block in a source file",
                detail=(
                    "Keep unit tests out of production source files. Leave "
                    "`#[cfg(test)] mod tests;` in the module and put the suite "
                    "in `src/<module>/tests.rs` (shared rule "
                    "tensor4all/tensor4all-agent-rules `rules/rust/index.md`, "
                    "Unit Test Organization). A genuinely tiny leaf module can "
                    "be waived with the maintainer waiver label.\n"
                    + "\n".join(inline_tests)
                ),
            )
        )
    return findings


def numstat_entries(output: str) -> list[tuple[str, int, int]]:
    """Read NUL-delimited numstat, retaining rename destinations and binary skips."""
    fields = iter(output.split("\0"))
    entries = []
    for field in fields:
        if not field:
            continue
        added, deleted, path = field.split("\t", 2)
        if not path:
            next(fields)  # Rename source; counts describe the source-to-target delta.
            path = next(fields)
        if added != "-" and deleted != "-":
            entries.append((path, int(added), int(deleted)))
    return entries


def file_growth_path(path: str) -> bool:
    """Select crate source files, excluding conventional test/generated paths."""
    parts = path.split("/")
    if len(parts) < 4 or parts[0] != "crates" or parts[2] != "src":
        return False
    name = parts[-1]
    return (
        name.endswith(".rs")
        and not {"tests", "test", "generated"}.intersection(parts[3:-1])
        and name not in {"tests.rs", "test.rs", "generated.rs"}
        and not name.startswith(("test_", "generated_"))
        and not name.endswith(("_test.rs", "_tests.rs", "_generated.rs", ".generated.rs"))
    )


def file_line_count(text: str) -> int:
    """Count physical lines, including an unterminated final line, as Git does."""
    return text.count("\n") + int(bool(text) and not text.endswith("\n"))


def file_growth_findings(
    base: str,
    head: str,
    *,
    worktree: bool = False,
    soft_limit: int = 1000,
    large_limit: int = 2000,
    large_min_delta: int = 50,
) -> list[Finding]:
    """Preview advisory whole-file growth against the merge base, never a hard cap.

    Numstat supplies the delta; reading the destination supplies its total size.
    Renames preserve the original size, so moving a large file alone is silent.
    No reviewed code is executed, and no historical file-size baseline is needed.
    The net-growth floor applies only to already-large files, not first crossings.
    """
    comparison = [base] if worktree else [f"{base}...{head}"]
    entries = numstat_entries(run_git([
        "diff", "--numstat", "-z", "--find-renames", "--no-ext-diff",
        "--no-textconv", *comparison, "--", "crates/",
    ]))
    if worktree:
        for path in untracked_files():
            if file_growth_path(path):
                entries.extend(numstat_entries(run_git([
                    "diff", "--numstat", "-z", "--no-index", "--no-ext-diff",
                    "--no-textconv", "--", "/dev/null", path,
                ], check=False)))

    findings = []
    for path, added, deleted in sorted(entries):
        if added <= deleted or not file_growth_path(path):
            continue
        text = file_text_at(path, ref=head, worktree=worktree)
        header = "\n".join(text.split("\n", 20)[:20])
        if re.search(r"@generated\b|\bDO NOT EDIT\b", header, re.IGNORECASE):
            continue
        after = file_line_count(text)
        before = after - added + deleted
        if before <= soft_limit < after:
            trigger = f"crosses the {soft_limit}-line soft review trigger"
        elif before > large_limit and added - deleted >= large_min_delta:
            trigger = (
                f"grows a file already above {large_limit} lines "
                f"by at least {large_min_delta} net lines"
            )
        else:
            continue
        findings.append(Finding(
            id=f"file-growth:{path}",
            severity="warn",
            rule_section="File Growth Dry Run",
            file=path,
            line=None,
            summary=f"{before} -> {after} lines (+{added}/-{deleted}); {trigger}",
            detail=(
                "Experimental, advisory only; counts include comments, blank lines "
                "and inline tests. Explain the coherent responsibility in the PR "
                "body, or split at a meaningful responsibility boundary. Never "
                "split solely to lower the count. Thresholds remain provisional (#917)."
            ),
        ))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Merge base ref")
    parser.add_argument("--head", default="HEAD", help="Head ref (default: HEAD)")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write machine-readable report JSON to this path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM call; run diff/section selection and deterministic checks only",
    )
    parser.add_argument(
        "--file-growth", action="store_true",
        help="Opt into the experimental advisory file-size check (requires --dry-run)",
    )
    parser.add_argument(
        "--file-growth-soft-limit", type=int, default=1000,
        help="Provisional crossing threshold in whole-file lines (default: 1000)",
    )
    parser.add_argument(
        "--file-growth-large-limit", type=int, default=2000,
        help="Provisional already-large threshold, at least the soft limit (default: 2000)",
    )
    parser.add_argument(
        "--file-growth-large-min-delta", type=int, default=50,
        help="Minimum net added lines for already-large files; 0 allows any growth "
        "(default: 50). First crossings are unaffected.",
    )
    parser.add_argument(
        "--waived",
        action="store_true",
        help="Treat review as waived (maintainer label in CI)",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        metavar="PATH",
        help="Load environment from this dotenv file (default: .env at repo root when present)",
    )
    parser.add_argument(
        "--no-dotenv",
        action="store_true",
        help="Do not load .env even when the file exists",
    )
    parser.add_argument(
        "--worktree",
        action="store_true",
        help="Diff the working tree against --base (includes uncommitted changes)",
    )
    parser.add_argument(
        "--llm-skipped-reason",
        help="Record a maintainer-approved reason when --dry-run intentionally skips LLM review",
    )
    parser.add_argument(
        "--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("DEEPSEEK_API_URL", DEFAULT_API_URL),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=DEFAULT_BUDGET_SECONDS,
        help="Wall-clock ceiling for all LLM requests, so the job is never "
        "killed before the report is written",
    )
    args = parser.parse_args(argv)

    if args.file_growth and not args.dry_run:
        parser.error("--file-growth requires --dry-run during the #917 experiment")
    if not 0 < args.file_growth_soft_limit <= args.file_growth_large_limit:
        parser.error("file-growth limits must satisfy 0 < soft <= large")
    if args.file_growth_large_min_delta < 0:
        parser.error("--file-growth-large-min-delta must be nonnegative")

    if args.llm_skipped_reason and not args.dry_run:
        print("--llm-skipped-reason requires --dry-run", file=sys.stderr)
        return 1

    configure_dotenv(explicit=args.dotenv, skip=args.no_dotenv)

    for rules_path in (RULES_PATH, TIPS_PATH):
        if not rules_path.is_file():
            print(f"Missing rules file: {rules_path}", file=sys.stderr)
            return 1
    if not PROMPT_PATH.is_file():
        print(f"Missing prompt file: {PROMPT_PATH}", file=sys.stderr)
        return 1

    files = changed_files(args.base, args.head, worktree=args.worktree)
    if not files:
        report = {
            "verdict": "pass",
            "waived": args.waived,
            "findings": [],
            "summary": "No changed files; review skipped.",
        }
        print(json.dumps(report, indent=2))
        return 0

    diff_text = unified_diff(args.base, args.head, worktree=args.worktree)
    # Second layer behind the hunk-aware parser: git's own NUL-separated path
    # list for the same range. The diff text is PR-controlled; this is not.
    name_status_text = name_status(args.base, args.head, worktree=args.worktree)
    added_text = added_lines_with_text(diff_text)
    added_lines = added_line_numbers(added_text)
    section_names = select_rule_sections(files, added_text)
    rules_text = build_rules_payload(section_names)
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    findings = deterministic_checks(
        files, added=added_text, ref=args.head, worktree=args.worktree
    )
    if args.file_growth:
        findings.extend(file_growth_findings(
            args.base, args.head, worktree=args.worktree,
            soft_limit=args.file_growth_soft_limit,
            large_limit=args.file_growth_large_limit,
            large_min_delta=args.file_growth_large_min_delta,
        ))
    sensitive_finding = sensitive_diff_finding(diff_text)
    if sensitive_finding:
        findings.append(sensitive_finding)
    path_finding = path_consistency_finding(diff_text, name_status_text)
    if path_finding:
        findings.append(path_finding)
    if args.llm_skipped_reason:
        findings.append(llm_skipped_finding(args.llm_skipped_reason))

    if args.waived:
        report_body = format_report(
            base=args.base,
            head=args.head,
            verdict="pass",
            findings=findings,
            waived=True,
        )
        print(report_body)
        payload = {
            "verdict": "pass",
            "waived": True,
            "findings": [item.to_dict() for item in findings],
            "changed_files": files,
            "rule_sections": section_names,
        }
        if args.output_json:
            args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0

    llm_summary: str | None = None
    llm_stats: dict[str, Any] | None = None
    if not args.dry_run and not sensitive_finding:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            print("DEEPSEEK_API_KEY is not set", file=sys.stderr)
            return 1
        key_problem = api_key_problem(api_key)

        if key_problem:
            print(f"DEEPSEEK_API_KEY is unusable: {key_problem}", file=sys.stderr)
            findings.append(api_key_error_finding(key_problem))
            chunks = []
        else:
            file_diffs = per_file_diffs(
                args.base,
                args.head,
                files,
                worktree=args.worktree,
            )
            chunks = split_diff_chunks(redact_file_diffs(file_diffs))
        chunk_sizes = [len(chunk) for chunk in chunks]
        print(
            f"LLM review: model {args.model}, {len(chunks)} chunk(s), "
            f"sizes {chunk_sizes} chars",
            file=sys.stderr,
        )
        llm_findings: list[Finding] = []
        llm_started = time.monotonic()
        llm_deadline = llm_started + args.budget_seconds
        reviewed = 0
        for index, chunk in enumerate(chunks, start=1):
            chunk_started = time.monotonic()
            if chunk_started >= llm_deadline:
                print(
                    f"LLM review: budget exhausted after {reviewed}/{len(chunks)} "
                    "chunk(s)",
                    file=sys.stderr,
                )
                findings.append(
                    budget_exhausted_finding(
                        reviewed, len(chunks), args.budget_seconds
                    )
                )
                break
            try:
                _, chunk_findings = review_chunk(
                    api_key=api_key,
                    model=args.model,
                    api_url=args.api_url,
                    system_prompt=system_prompt,
                    rules_text=rules_text,
                    changed=files,
                    diff_chunk=chunk,
                    timeout=args.timeout,
                    deadline=llm_deadline,
                )
            except (KeyError, ValueError, *TRANSPORT_ERRORS) as exc:
                print(
                    f"LLM chunk {index}/{len(chunks)}: failed after "
                    f"{time.monotonic() - chunk_started:.1f}s: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                findings.append(llm_response_error_finding(exc))
                break
            print(
                f"LLM chunk {index}/{len(chunks)}: {len(chunk)} chars, "
                f"{len(chunk_findings)} finding(s), "
                f"{time.monotonic() - chunk_started:.1f}s",
                file=sys.stderr,
            )
            llm_findings.extend(chunk_findings)
            reviewed += 1
        merged_llm_findings = merge_findings(llm_findings)
        kept_llm_findings = filter_findings(
            merged_llm_findings,
            files,
            added_lines,
            allow_global=False,
        )
        llm_elapsed = time.monotonic() - llm_started
        llm_summary = summarize_llm_review(
            chunk_sizes=chunk_sizes,
            elapsed_seconds=llm_elapsed,
            returned_count=len(merged_llm_findings),
            kept_count=len(kept_llm_findings),
        )
        llm_stats = {
            "chunk_sizes": chunk_sizes,
            "elapsed_seconds": round(llm_elapsed, 3),
            "findings_returned": len(merged_llm_findings),
            "findings_kept": len(kept_llm_findings),
        }
        findings.extend(kept_llm_findings)

    block_findings = [item for item in findings if item.severity == "block"]
    verdict = reconcile_verdict(block_findings)

    report_body = format_report(
        base=args.base,
        head=args.head,
        verdict=verdict,
        findings=findings,
        waived=False,
        llm_summary=llm_summary,
    )
    print(report_body)

    payload = {
        "verdict": verdict,
        "waived": False,
        "findings": [item.to_dict() for item in findings],
        "block_findings": [item.to_dict() for item in block_findings],
        "changed_files": files,
        "rule_sections": section_names,
        "prompt_version": PROMPT_VERSION,
        "llm_review": llm_stats,
    }
    if args.output_json:
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 1 if block_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
