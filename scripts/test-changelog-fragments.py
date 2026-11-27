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


    def _pr(self, files, edit):
        d = repo(files)
        git = lambda *c: subprocess.run(["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@t", *c], check=True)
        git("init", "-q", "-b", "main"); git("add", "-A"); git("commit", "-qm", "base"); git("checkout", "-qb", "pr")
        edit(d); git("add", "-A"); git("commit", "-qm", "pr")
        return cf.check(d, "main", pr=7)

    def test_direct_changelog_edit_fails_even_below_a_release_heading(self):
        errs = self._pr({}, lambda d: (d / "CHANGELOG.md").write_text("# Changelog\n\n## v0.3.0\n- stray (#9)\n"))
        self.assertTrue(any("edited directly" in e for e in errs), errs)

    def test_release_may_edit_changelog(self):
        self.assertEqual(self._pr({"3-added.md": "- three\n"}, lambda d: cf.release(d, "v1.0")), [])

    def test_legacy_order_cannot_gain_entries(self):
        errs = self._pr({"3-added.md": "- a\n", "legacy-order.txt": "3-added.md\n"},
                        lambda d: (d / "changelog.d" / "legacy-order.txt").write_text("3-added.md\n7-added.md\n"))
        self.assertTrue(any("may only lose" in e for e in errs), errs)

    def test_second_list_item_unindented_continuation_and_n1(self):
        self.assertTrue(cf.check(repo({"7-added.md": GOOD + "* second\n"}), None))
        self.assertTrue(cf.check(repo({"7-added.md": GOOD + "flush left\n"}), None))
        self.assertTrue(cf.check(repo({"7-added-1.md": GOOD}), None))


class Assemble(unittest.TestCase):
    def test_legacy_order_pins_order_and_new_fragments_come_first(self):
        d = repo({"9-added.md": "- nine\n", "3-added.md": "- three\n", "5-added.md": "- five\n",
                  "legacy-order.txt": "3-added.md\n9-added.md\n"})
        (d / "changelog.d" / "12-added.md").write_text("- twelve\n")
        self.assertEqual(cf.assemble(d, "v1"), "## v1\n\n### Added\n\n- twelve\n- five\n- three\n- nine\n")

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
