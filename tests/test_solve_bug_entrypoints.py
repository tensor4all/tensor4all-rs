import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_CODEX = REPO_ROOT / "ai" / "run-codex-solve-bug.sh"
RUN_CLAUDE = REPO_ROOT / "ai" / "run-claude-solve-bug.sh"
PROMPT = REPO_ROOT / "ai" / "solve_bug_issue.md"
CREATE_PR = REPO_ROOT / "scripts" / "create-pr.sh"
MONITOR_PR = REPO_ROOT / "scripts" / "monitor-pr-checks.sh"


class SolveBugEntrypointTests(unittest.TestCase):
    def test_prompt_contract(self) -> None:
        self.assertTrue(RUN_CODEX.is_file(), msg=f"missing file: {RUN_CODEX}")
        self.assertTrue(RUN_CLAUDE.is_file(), msg=f"missing file: {RUN_CLAUDE}")
        self.assertTrue(PROMPT.is_file(), msg=f"missing file: {PROMPT}")

        prompt = PROMPT.read_text(encoding="utf-8")
        self.assertIn("effectively no open bug or bug-like issues", prompt)
        self.assertIn("bash scripts/create-pr.sh", prompt)
        self.assertIn("bash scripts/monitor-pr-checks.sh", prompt)

    def test_run_codex_help(self) -> None:
        result = subprocess.run(
            ["bash", str(RUN_CODEX), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--prompt", result.stdout)
        self.assertIn("--run-dir", result.stdout)
        self.assertIn("--text", result.stdout)

    def test_run_claude_help(self) -> None:
        result = subprocess.run(
            ["bash", str(RUN_CLAUDE), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--prompt", result.stdout)
        self.assertIn("--run-dir", result.stdout)
        self.assertIn("--text", result.stdout)

    def test_create_pr_help(self) -> None:
        result = subprocess.run(
            ["bash", str(CREATE_PR), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--base", result.stdout)
        self.assertIn("--title", result.stdout)
        self.assertIn("--no-auto-merge", result.stdout)

    def test_monitor_pr_help(self) -> None:
        result = subprocess.run(
            ["bash", str(MONITOR_PR), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--interval", result.stdout)

    def test_create_pr_refuses_main_without_gh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".git").mkdir()
            (root / "scripts").mkdir()
            (root / "ai").mkdir()
            (root / "bin").mkdir()
            (root / "README.md").write_text("temp", encoding="utf-8")
            (root / "AGENTS.md").write_text("temp", encoding="utf-8")
            (root / "scripts" / "create-pr.sh").write_text(
                CREATE_PR.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "scripts" / "monitor-pr-checks.sh").write_text(
                MONITOR_PR.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "ai" / "solve_bug_issue.md").write_text("temp", encoding="utf-8")
            (root / "ai" / "run-codex-solve-bug.sh").write_text("temp", encoding="utf-8")
            (root / "ai" / "run-claude-solve-bug.sh").write_text("temp", encoding="utf-8")
            (root / "bin" / "git").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "if [[ \"$1\" == \"status\" && \"$2\" == \"--short\" ]]; then\n"
                "  exit 0\n"
                "fi\n"
                "if [[ \"$1\" == \"branch\" && \"$2\" == \"--show-current\" ]]; then\n"
                "  printf 'main\\n'\n"
                "  exit 0\n"
                "fi\n"
                "printf 'unexpected git invocation: %s\\n' \"$*\" >&2\n"
                "exit 1\n",
                encoding="utf-8",
            )
            (root / "bin" / "git").chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{root / 'bin'}:{env['PATH']}"
            result = subprocess.run(
                ["bash", "scripts/create-pr.sh"],
                cwd=root,
                text=True,
                capture_output=True,
                env=env,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to create a PR from main", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
