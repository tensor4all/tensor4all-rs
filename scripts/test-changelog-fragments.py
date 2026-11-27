#!/usr/bin/env python3
"""Tests for scripts/changelog_fragments.py: a bad fragment fails, a good one passes."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import changelog_fragments as cf  # noqa: E402

GOOD = "- `crate`: does a thing for users (#7).\n  A continuation line.\n"


def repo(files, changelog="# Changelog\n\n## v0.3.0\n"):
    d = Path(tempfile.mkdtemp())
    (d / "CHANGELOG.md").write_text(changelog)
    (d / "changelog.d").mkdir()
    for n, t in files.items():
        (d / "changelog.d" / n).write_text(t)
    return d


class Check(unittest.TestCase):
    def test_good_fragment_passes(self):
        self.assertEqual(cf.check(repo({"7-added.md": GOOD}), None), [])

    def test_unknown_kind(self):
        errs = cf.check(repo({"7-bogus.md": GOOD}), None)
        self.assertTrue(any("unknown kind" in e for e in errs), errs)

    def test_missing_pr_number(self):
        for name in ("added.md", "x-added.md", "-added.md"):
            self.assertTrue(cf.check(repo({name: GOOD}), None), name)

    def test_empty_body(self):
        errs = cf.check(repo({"7-added.md": "\n  \n"}), None)
        self.assertTrue(any("empty body" in e for e in errs), errs)

    def test_body_must_be_one_entry(self):
        self.assertTrue(cf.check(repo({"7-added.md": "no bullet\n"}), None))
        self.assertTrue(cf.check(repo({"7-added.md": GOOD + "- second\n"}), None))
        self.assertTrue(cf.check(repo({"7-added.md": "### Added\n" + GOOD}), None))

    def test_direct_unreleased_edit(self):
        d = repo({}, "# Changelog\n\n## Unreleased (next minor)\n\n- x\n\n## v0.3.0\n")
        self.assertTrue(any("Unreleased" in e for e in cf.check(d, None)))

    def test_pr_number_must_match_pr(self):
        d = repo({"7-added.md": GOOD})
        for c in (["init", "-q", "-b", "main"], ["add", "-A"],
                  ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"]):
            subprocess.run(["git", "-C", str(d), *c], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-qb", "pr"], check=True)
        (d / "changelog.d" / "9-fixed.md").write_text(GOOD)  # unknown kind too
        (d / "changelog.d" / "8-added.md").write_text(GOOD)
        for c in (["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "pr"]):
            subprocess.run(["git", "-C", str(d), *c], check=True)
        errs = cf.check(d, "main", pr=7)
        self.assertTrue(any("8-added.md" in e and "does not match" in e for e in errs), errs)
        # A migrated fragment (listed in legacy-order.txt) keeps its original PR number.
        (d / "changelog.d" / "legacy-order.txt").write_text("8-added.md\n")
        errs = cf.check(d, "main", pr=7)
        self.assertFalse(any("8-added.md" in e for e in errs), errs)


class Assemble(unittest.TestCase):
    def test_groups_by_kind_newest_pr_first_and_release_deletes(self):
        d = repo({"3-added.md": "- three\n", "5-added.md": "- five\n", "4-behavior.md": "- four\n",
                  "5-added-2.md": "- five b\n"})
        out = cf.assemble(d, "v1.0")
        self.assertEqual(out, "## v1.0\n\n### Behavior changes\n\n- four\n\n### Added\n\n- five\n- five b\n- three\n")
        cf.release(d, "v1.0")
        self.assertEqual((d / "CHANGELOG.md").read_text(), "# Changelog\n\n" + out + "\n## v0.3.0\n")
        self.assertEqual(list((d / "changelog.d").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
