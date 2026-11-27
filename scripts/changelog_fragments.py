#!/usr/bin/env python3
"""Changelog fragments: validation and release assembly (#1503).

Every PR that changes user-visible behavior adds ONE file,
``changelog.d/<PR-number>-<kind>.md`` (``-2``, ``-3`` ... before ``.md`` for a
second entry of the same PR and kind), holding one entry in the prose style of
CHANGELOG.md.  Nothing edits CHANGELOG.md directly, so two PRs never share an
insertion point.  A release concatenates the fragments into a new
``## <version>`` section and deletes them.

  changelog_fragments.py check [--base REF]   validate (changed files vs REF, else all)
  changelog_fragments.py assemble VERSION     print the section to stdout
  changelog_fragments.py release VERSION      insert it into CHANGELOG.md, delete fragments

The file is self-contained and has no dependency outside the standard library,
so it can be copied to another repository as is (edit KINDS to match).
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

# kind -> heading, in the order the headings appear in a release section.
KINDS = {
    "behavior": "Behavior changes",
    "added": "Added",
    "breaking": "Breaking Rust API changes",
    "diagnostics": "Diagnostics",
    "performance": "Performance",
    # `### Fixed` is a heading `CHANGELOG.md` already uses (v0.3.0 has one), and the migration could not represent
    # an entry under it — which is how this was found: main's newest entry landed under `### Fixed` and the checker
    # rejected it as an unknown kind. A kind list that omits a heading the file already uses is not a vocabulary.
    "fixed": "Fixed",
}
FRAGMENT_DIR = "changelog.d"
LEGACY_ORDER = "legacy-order.txt"  # one-time ordering hint written by the migration
NAME_RE = re.compile(r"^(\d+)-([a-z]+)(?:-([2-9]|[1-9]\d+))?\.md$")  # no -1: it would collide with the bare name
LIST_MARK_RE = re.compile(r"^([-*+]|\d+[.)])\s")
UNRELEASED_RE = re.compile(r"^##\s+Unreleased\b", re.I | re.M)


def validate_name(name):
    m = NAME_RE.match(name)
    if not m:
        return None, f"{name}: name must be <PR-number>-<kind>[-<n>].md (n >= 2)"
    if m.group(2) not in KINDS:
        return None, f"{name}: unknown kind '{m.group(2)}' (known: {', '.join(KINDS)})"
    return (int(m.group(1)), m.group(2), int(m.group(3) or 1)), None


def validate_body(name, text):
    errs = []
    lines = text.rstrip("\n").split("\n")
    if not text.strip():
        return [f"{name}: empty body"]
    if not lines[0].startswith("- ") or not lines[0][2:].strip():
        errs.append(f"{name}: body must start with a '- ' entry")
    if any(LIST_MARK_RE.match(l) for l in lines[1:]):
        errs.append(f"{name}: one entry per fragment (found a second top-level list item)")
    if any(l and not l[0].isspace() for l in lines[1:]):
        errs.append(f"{name}: continuation lines must be indented two spaces")
    if any(l.startswith("#") for l in lines):
        errs.append(f"{name}: no headings in a fragment; the kind in the file name picks the section")
    return errs


def load(root):
    """Return [(key, name, text)] for every fragment, validated."""
    d = root / FRAGMENT_DIR
    out, errs = [], []
    for p in sorted(d.glob("*.md")) if d.is_dir() else []:
        if p.name == "README.md":
            continue
        key, e = validate_name(p.name)
        if e:
            errs.append(e)
            continue
        text = p.read_text()
        errs += validate_body(p.name, text)
        # A fragment ending in a blank line keeps it: a loose list stays loose.
        out.append((key, p.name, text.rstrip("\n") + ("\n" if text.endswith("\n\n") else "")))
    return out, errs


def legacy_order(root):
    p = root / FRAGMENT_DIR / LEGACY_ORDER
    return p.read_text().split() if p.exists() else []


def assemble(root, version):
    frags, errs = load(root)
    if errs:
        raise SystemExit("\n".join(errs))
    legacy = legacy_order(root)
    rank = {n: i for i, n in enumerate(legacy)}
    missing = [n for n in legacy if not (root / FRAGMENT_DIR / n).exists()]
    if missing:
        raise SystemExit(f"{LEGACY_ORDER} names missing fragments: {missing}")
    parts = [f"## {version}\n"]
    for kind, heading in KINDS.items():
        mine = [f for f in frags if f[0][1] == kind]
        # New fragments first, newest PR on top (the file's existing
        # convention); then fragments pinned by the one-time legacy order.
        mine.sort(key=lambda f: (f[1] in rank, rank.get(f[1], 0), -f[0][0], f[0][2]))
        if mine:
            parts.append(f"### {heading}\n\n" + "\n".join(f[2] for f in mine) + "\n")
    if len(parts) == 1:
        raise SystemExit("no fragments to assemble")
    return "\n".join(parts)


def release(root, version):
    section = assemble(root, version)
    cl = root / "CHANGELOG.md"
    text = cl.read_text()
    if UNRELEASED_RE.search(text):
        raise SystemExit("CHANGELOG.md has a '## Unreleased' section; unreleased entries live in changelog.d/")
    m = re.search(r"^## ", text, re.M)
    pos = m.start() if m else len(text)
    cl.write_text(text[:pos] + section + "\n" + text[pos:])
    for p in (root / FRAGMENT_DIR).glob("*.md"):
        if p.name != "README.md":
            p.unlink()
    (root / FRAGMENT_DIR / LEGACY_ORDER).unlink(missing_ok=True)


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout


def git_has(root, ref, path):
    return subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{ref}:{path}"], capture_output=True).returncode == 0


def check(root, base, pr=None):
    errs = []
    frags, e = load(root)
    errs += e
    if UNRELEASED_RE.search((root / "CHANGELOG.md").read_text()):
        errs.append("CHANGELOG.md: contains '## Unreleased'; add a changelog.d/<PR>-<kind>.md fragment instead")
    if base:
        cl_changed = git(root, "diff", "--name-only", f"{base}...HEAD", "--", "CHANGELOG.md").split()
        deleted = git(root, "diff", "--name-only", "--diff-filter=D", f"{base}...HEAD", "--", FRAGMENT_DIR).split()
        pinned = git_has(root, base, f"{FRAGMENT_DIR}/{LEGACY_ORDER}")
        migrating = not pinned and (root / FRAGMENT_DIR / LEGACY_ORDER).exists()
        if cl_changed and not deleted and not migrating:
            errs.append("CHANGELOG.md: edited directly; only a release (which consumes fragments) changes it. Add changelog.d/<PR>-<kind>.md")
        if pinned:
            old = git(root, "show", f"{base}:{FRAGMENT_DIR}/{LEGACY_ORDER}").split()
            now = legacy_order(root)
            if [n for n in old if n in now] != now:
                errs.append(f"{LEGACY_ORDER}: may only lose entries (at release), not gain or reorder them")
        # The number belongs to the PR that created the file; a later PR may fix its wording.
        changed = git(root, "diff", "--name-only", "--diff-filter=A", f"{base}...HEAD", "--", FRAGMENT_DIR).split()
        for f in changed:
            name = Path(f).name
            if name in (LEGACY_ORDER, "README.md") or name in legacy_order(root):
                continue  # migrated entries keep the number of the PR that wrote them
            if pr is not None and (k := validate_name(name)[0]) and k[0] != pr:
                errs.append(f"{name}: PR number {k[0]} does not match this PR (#{pr})")
    for n in legacy_order(root):
        if not (root / FRAGMENT_DIR / n).exists():
            errs.append(f"{LEGACY_ORDER}: names missing fragment {n}")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "assemble", "release"])
    ap.add_argument("version", nargs="?")
    ap.add_argument("--base")
    ap.add_argument("--pr", type=int)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if a.cmd == "check":
        errs = check(root, a.base, a.pr)
        for e in errs:
            print("changelog:", e, file=sys.stderr)
        sys.exit(1 if errs else 0)
    if not a.version:
        ap.error("VERSION required")
    if a.cmd == "assemble":
        sys.stdout.write(assemble(root, a.version))
    else:
        release(root, a.version)


if __name__ == "__main__":
    main()
