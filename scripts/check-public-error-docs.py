#!/usr/bin/env python3
"""Audit public Rust ``Result`` APIs for concrete ``# Errors`` docs.

The Rust compiler cannot make ``clippy::missing_errors_doc`` a repository
contract by itself: it checks only whether a heading exists, and it does not
cover all public trait methods consistently across toolchain versions. This
small source audit complements Clippy by checking the public API surface and
requiring the section to name a concrete error variant or failure condition.
It intentionally does not rewrite source; missing documentation is a review
failure that must be fixed at the API's source of truth.

Ported from tenferro-rs (tensor4all/tensor4all-rs#566 Phase 1); the
``--changed-from REV`` mode audits only Rust source lines added since ``REV``
so the pre-existing undocumented-API backlog burns down incrementally.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PUBLIC_FN_RE = re.compile(
    r"^\s*pub\s+(?!(?:\([^)]*\)))"
    r"(?:(?:async|const|unsafe)\s+)*fn\b"
)
TRAIT_FN_RE = re.compile(r"^\s*(?:(?:async|unsafe)\s+)*fn\b")
RESULT_RE = re.compile(r"\b(?:[A-Za-z_][\w:]*::)?Result\s*<")
# `type StorageResult<T> = std::result::Result<T, StorageError>;` declares an
# alias whose signatures `RESULT_RE` would miss. Struct names that merely end
# in `Result` (e.g. `FactorizeResult<T>`) are not aliases and stay excluded.
RESULT_ALIAS_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?type\s+([A-Za-z_]\w*)\s*(?:<[^=]*>)?\s*=\s*"
    r"(?:(?:std|core)::)?(?:result::)?Result\s*<"
)


def result_type_re(paths: list[Path]) -> re.Pattern[str]:
    """Match `Result<` plus every `Result` alias declared in ``paths``."""
    aliases: set[str] = set()
    for path in paths:
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                match = RESULT_ALIAS_RE.match(line)
                if match:
                    aliases.add(match.group(1))
    if not aliases:
        return RESULT_RE
    names = "|".join(sorted(re.escape(alias) for alias in aliases))
    return re.compile(r"\b(?:[A-Za-z_][\w:]*::)?(?:Result|" + names + r")\s*<")
TRAIT_RE = re.compile(r"^\s*pub\s+(?:unsafe\s+)?trait\b")
DOC_RE = re.compile(r"^\s*///")
DOC_ATTRIBUTE_RE = re.compile(r'^\s*#\[doc\s*=\s*("(?:\\.|[^"\\])*")\s*\]\s*$')
ATTRIBUTE_RE = re.compile(r"^\s*#\[")
ERROR_HEADING_RE = re.compile(r"^\s*#\s+Errors\s*$", re.IGNORECASE)
DEFERRED_HEADING_RE = re.compile(r"^\s*#\s+Deferred errors\s*$", re.IGNORECASE)
NEXT_HEADING_RE = re.compile(r"^\s*#\s+")
# A traced API may mention a deferred check in its prose. Such a statement is
# required to live under an explicit ``# Deferred errors`` heading so callers
# can distinguish graph-build failures from compile/execution failures.
DEFERRED_HINT_RE = re.compile(
    r"\b(?:deferred|compile(?:d)? or execution|compilation or execution|"
    r"after binding|at execution,)\b",
    re.IGNORECASE,
)
# A category label such as "backend error" or "validation failure" is not
# enough: the section must identify an observable variant or a condition that
# lets a caller decide how to recover. Keep this deliberately narrower than a
# prose spell-checker so boilerplate cannot satisfy the repository contract.
CONCRETE_ERROR_RE = re.compile(
    r"(?:"
    r"\bInfallible\b|"
    r"\b(?:Error|ErrorKind)::(?!Validation\b)[A-Za-z0-9_]+\b|"
    r"\bValidationError::[A-Za-z0-9_]+\b|"
    r"\b[A-Z][A-Za-z0-9_]*(?:Error|Failure|Mismatch|OutOfBounds|Overflow|"
    r"Underflow|Unavailable|NonConvergence|Singular|Unsupported|Invalid|"
    r"Poisoned|Conversion|State)\b|"
    r"\b(?:shape|rank|axis|dtype|configuration|config|input|output|"
    r"cotangent|tangent|operand|index|placement|buffer|metadata|program|"
    r"graph|cache|device|plugin|lock|file|stream|serialization|worker|"
    r"operation|extension|family|backend|runtime|divisor|exponent)\b"
    r"[^.\n]{0,80}\b(?:mismatch|out of bounds|overflow|underflow|invalid|"
    r"incompatible|missing|poisoned|unavailable|unsupported|singular|"
    r"non[- ]?convergen\w*|zero|failure|(?:do not )?match)\b|"
    r"\binvalid\s+(?:shape|shapes|rank|axis|axes|dtype|dtypes|configuration|"
    r"config|input|output|notation|subscripts|metadata|binding|indices|"
    r"positions|range|dimensions?)\b|"
    r"\b(?:no|missing|without)\b[^.\n]{0,50}\b(?:shape|rank|axis|dtype|"
    r"metadata|input|output|binding|executor|cache|runtime|plugin)\b|"
    r"\b(?:division by zero|zero divisor|non[- ]?convergen\w*|singular|"
    r"overflow|underflow|poisoned|out of bounds|missing [A-Za-z]|"
    r"unsupported[- ](?:operation|dtype|conversion|backend|extension|rule))\b"
    r")",
)
# PR #593 generated ``# Errors`` bodies from one template::
#
#     Returns an error when <subject> (<category list>).
#
# The parenthetical only names an error *class* (``a backend failure``,
# ``a shape or index mismatch``, ``an invalid-configuration failure``, ...),
# never the concrete variant a caller can match on. Such a sentence is still
# accepted by ``CONCRETE_ERROR_RE`` because it mentions ``mismatch`` or
# ``failure``; this pattern rejects it instead so a regeneration pass cannot
# reintroduce the boilerplate. A sentence that names a concrete variant
# (``GseError::MissingCenter``) is actionable even when it also carries a
# class label, so it is exempt.
GENERATED_BOILERPLATE_START_RE = re.compile(r"^(?:It also )?[Rr]eturns an error when\b")
GENERATED_BOILERPLATE_PAREN_RE = re.compile(r"\(([^()]*)\)")
# Category labels use only words, spaces, commas, dots, slashes and hyphens; a
# colon, bracket or backtick marks an explanation or a concrete variant.
GENERATED_BOILERPLATE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ,./\-]*$")
GENERATED_BOILERPLATE_TAIL_RE = re.compile(
    r"\b(?:failure|mismatch|replacement|split|value|state|coefficient|"
    r"overflow|underflow|dtype|operation|construction|bounds|gradient)$"
)
# Concrete variant path (``Error::Variant``) or a Rust type reference.
GENERATED_BOILERPLATE_VARIANT_RE = re.compile(
    r"::|\b[A-Z][A-Za-z0-9]*Error\b"
)
# Split at a real sentence boundary only, so ``i.e.``/``e.g.`` inside a label
# does not hide a wrapped generated sentence.
GENERATED_BOILERPLATE_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\[])")


def is_category_label(content: str) -> bool:
    """Return whether a parenthetical is a bare error-category list."""
    label = content.strip()
    if not GENERATED_BOILERPLATE_LABEL_RE.fullmatch(label):
        return False
    return GENERATED_BOILERPLATE_TAIL_RE.search(label) is not None


def generated_boilerplate_sentence(section: str) -> str | None:
    """Return the first generated ``# Errors`` sentence in ``section``.

    Matching is done on the whitespace-normalized section text so a sentence
    wrapped across ``///`` lines is still recognized.
    """
    normalized = re.sub(r"\s+", " ", section)
    for sentence in GENERATED_BOILERPLATE_SENTENCE_RE.split(normalized):
        if not GENERATED_BOILERPLATE_START_RE.match(sentence):
            continue
        if GENERATED_BOILERPLATE_VARIANT_RE.search(sentence):
            continue
        if any(
            is_category_label(match.group(1))
            for match in GENERATED_BOILERPLATE_PAREN_RE.finditer(sentence)
        ):
            return sentence
    return None


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    function: str
    reason: str


def rust_files(root: Path) -> list[Path]:
    excluded = {".git", ".codegraph", "target"}
    paths: list[Path] = []
    for path in root.rglob("*.rs"):
        if any(part in excluded for part in path.parts):
            continue
        paths.append(path)
    return sorted(paths)


def strip_code(line: str) -> str:
    """Remove line comments and string literals, keeping brace/terminator text."""
    code = re.sub(r"//.*$", "", line)
    code = re.sub(r'"(?:\\.|[^"\\])*"', "", code)
    return code


def brace_delta(line: str) -> int:
    # This is a documentation audit, not a Rust parser. Removing comments and
    # string literals avoids the common false positives in examples while
    # keeping the trait-range tracking deterministic and dependency-free.
    code = strip_code(line)
    return code.count("{") - code.count("}")


def preceding_docs(lines: list[str], index: int) -> tuple[str, bool, int]:
    docs: list[str] = []
    hidden = False
    # 0-based index of the first line consumed as part of the doc block; stays
    # at ``index`` when the declaration has no preceding docs.
    first = index
    cursor = index - 1
    while cursor >= 0:
        stripped = lines[cursor].strip()
        if DOC_RE.match(lines[cursor]):
            docs.append(lines[cursor])
        elif (match := DOC_ATTRIBUTE_RE.match(lines[cursor])) is not None:
            # Procedural-macro source commonly uses #[doc = "..."] because
            # quote! cannot emit a literal /// block. Treat that attribute as
            # the generated rustdoc it represents so macro-generated Result
            # APIs are held to the same documentation contract.
            docs.append(f"/// {ast.literal_eval(match.group(1))}")
        elif ATTRIBUTE_RE.match(lines[cursor]):
            hidden = hidden or "doc(hidden)" in stripped
        elif stripped == "" and docs:
            break
        else:
            break
        first = cursor
        cursor -= 1
    return "\n".join(reversed(docs)), hidden, first


def function_signature_lines(lines: list[str], index: int) -> list[str]:
    """Collect a function declaration's lines up to and including ``{`` or ``;``.

    Comments and string literals are stripped before testing for the
    terminator so ``// compatibility;`` inside a signature cannot truncate it.
    """
    parts: list[str] = []
    for line in lines[index : index + 80]:
        parts.append(line)
        code = strip_code(line)
        if ";" in code or "{" in code:
            break
    return parts


def function_signature(lines: list[str], index: int) -> str:
    return " ".join(function_signature_lines(lines, index))


def function_name(signature: str) -> str:
    match = re.search(r"\bfn\s+([A-Za-z_][\w]*)", signature)
    return match.group(1) if match else "<anonymous>"


def doc_section(doc: str, heading: re.Pattern[str]) -> str | None:
    if not doc:
        return None
    cleaned = [line.split("///", 1)[-1].strip() for line in doc.splitlines()]
    start = next((index for index, line in enumerate(cleaned) if heading.match(line)), None)
    if start is None:
        return None
    body: list[str] = []
    for line in cleaned[start + 1 :]:
        if NEXT_HEADING_RE.match(line):
            break
        body.append(line)
    return "\n".join(body).strip()


def error_section(doc: str) -> str | None:
    return doc_section(doc, ERROR_HEADING_RE)


def deferred_section(doc: str) -> str | None:
    return doc_section(doc, DEFERRED_HEADING_RE)


def audit_file(
    path: Path,
    changed_lines: set[int] | None = None,
    result_re: re.Pattern[str] = RESULT_RE,
) -> list[Finding]:
    lines = path.read_text(encoding="utf-8").splitlines()
    findings: list[Finding] = []
    trait_depth: int | None = None
    pending_trait = False
    depth = 0

    for index, line in enumerate(lines):
        if TRAIT_RE.match(line):
            pending_trait = True
        if pending_trait and "{" in strip_code(line):
            # Trait scope opens at the declaration's brace and closes when the
            # brace depth falls back below it, so a multiline declaration
            # (bounds on a later line) cannot leak trait scope into following
            # private functions.
            pending_trait = False
            trait_depth = depth + 1

        is_public = bool(PUBLIC_FN_RE.match(line))
        is_trait_method = trait_depth is not None and bool(TRAIT_FN_RE.match(line))
        if is_public or is_trait_method:
            signature_lines = function_signature_lines(lines, index)
            docs, hidden, doc_start = preceding_docs(lines, index)
            if changed_lines is not None:
                # Audit the declaration when its signature changed, but also
                # when any line of its preceding doc block changed: a doc-only
                # edit can reintroduce boilerplate without touching the
                # signature.
                signature_lines_changed = set(
                    range(index + 1, index + 1 + len(signature_lines))
                )
                doc_lines_changed = set(range(doc_start + 1, index + 1))
                if not (signature_lines_changed | doc_lines_changed) & changed_lines:
                    depth += brace_delta(line)
                    if trait_depth is not None and depth < trait_depth:
                        trait_depth = None
                    continue
            signature = " ".join(signature_lines)
            if result_re.search(signature):
                if not hidden:
                    section = error_section(docs)
                    if section is None:
                        findings.append(
                            Finding(path, index + 1, function_name(signature), "missing # Errors")
                        )
                    elif generated_boilerplate_sentence(section) is not None:
                        findings.append(
                            Finding(
                                path,
                                index + 1,
                                function_name(signature),
                                "# Errors uses generated boilerplate; name the concrete "
                                "variants or conditions",
                            )
                        )
                    elif not CONCRETE_ERROR_RE.search(re.sub(r"\s+", " ", section)):
                        findings.append(
                            Finding(
                                path,
                                index + 1,
                                function_name(signature),
                                "# Errors does not name a concrete variant or condition",
                            )
                        )
                    elif (
                        path.name == "traced.rs"
                        and DEFERRED_HINT_RE.search(docs)
                        and deferred_section(docs) is None
                    ):
                        findings.append(
                            Finding(
                                path,
                                index + 1,
                                function_name(signature),
                                "deferred validation must be documented under # Deferred errors",
                            )
                        )

        depth += brace_delta(line)
        if trait_depth is not None and depth < trait_depth:
            trait_depth = None

    return findings


def audit(
    root: Path,
    paths: list[Path] | None = None,
    changed_lines: dict[Path, set[int]] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    selected = paths if paths is not None else list(rust_files(root))
    result_re = result_type_re(list(rust_files(root)))
    for path in selected:
        if not path.is_file():
            continue
        lines = changed_lines.get(path) if changed_lines is not None else None
        findings.extend(audit_file(path, lines, result_re))
    return findings


def changed_rust_lines(root: Path, revision: str) -> dict[Path, set[int]]:
    """Return added Rust source lines between ``revision`` and ``HEAD``."""

    diff = subprocess.run(
        [
            "git",
            "diff",
            "--unified=0",
            f"{revision}...HEAD",
            "--",
            "*.rs",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result: dict[Path, set[int]] = {}
    current: Path | None = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = (root / line.removeprefix("+++ b/")).resolve()
            result.setdefault(current, set())
            continue
        if current is None or not line.startswith("@@"):
            continue
        match = re.search(r"\+(\d+)(?:,(\d+))?", line)
        if match is None:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        result[current].update(range(start, start + count))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--changed-from",
        metavar="REV",
        help="audit only Rust files changed between REV and HEAD",
    )
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    root = args.root_dir.resolve()
    paths = [((root / path).resolve() if not path.is_absolute() else path) for path in args.paths]
    changed_lines = None
    if args.changed_from:
        changed = subprocess.run(
            ["git", "diff", "--name-only", f"{args.changed_from}...HEAD", "--", "*.rs"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        paths = [
            (root / path).resolve()
            for path in changed
            if (root / path).is_file()
        ]
        changed_lines = changed_rust_lines(root, args.changed_from)
    audit_paths = paths if args.changed_from or paths else None
    findings = audit(root, audit_paths, changed_lines)
    if findings:
        print("public Result APIs with incomplete error documentation:", file=sys.stderr)
        for finding in findings:
            print(
                f"- {finding.path.relative_to(root)}:{finding.line}: "
                f"{finding.function}: {finding.reason}",
                file=sys.stderr,
            )
        print(
            "Each public Result API must document concrete variants/conditions "
            "under `# Errors`; traced symbolic APIs must also explain deferred "
            "validation where applicable.",
            file=sys.stderr,
        )
        return 1
    print("public-error-docs-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
