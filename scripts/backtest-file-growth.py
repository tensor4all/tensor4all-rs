#!/usr/bin/env python3
"""Replay the opt-in #917 check on the latest PR contributions merged into a ref.

Reads Git history only: no network, API key, checkout, code execution or comments.
Uses each landing commit and its first parent, so results describe the landed
contribution (including merge conflict resolutions), not the original PR head.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path


REVIEWER_PATH = Path(__file__).with_name("repository-rules-review.py")
spec = importlib.util.spec_from_file_location("repository_rules_review", REVIEWER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"unable to load {REVIEWER_PATH}")
reviewer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reviewer
spec.loader.exec_module(reviewer)


def backtest(
    base: str, count: int, soft_limit: int, large_limit: int, large_min_delta: int = 50,
) -> list[dict]:
    """Apply the same check to PRs on first-parent history, newest landing first."""
    history = reviewer.run_git([
        "log", "--first-parent", "--format=%H%x09%P%x09%s", base,
    ])
    rows = []
    for entry in history.splitlines():
        commit, parents, title = entry.split("\t", 2)
        match = re.search(r"^Merge pull request #(\d+)\b|\(#(\d+)\)$", title)
        if not match or not parents:
            continue
        parent = parents.split()[0]
        findings = reviewer.file_growth_findings(
            parent, commit, soft_limit=soft_limit, large_limit=large_limit,
            large_min_delta=large_min_delta,
        )
        rows.append({
            "number": int(match.group(1) or match.group(2)),
            "title": title,
            "base": parent,
            "head": commit,
            "verdict": reviewer.reconcile_verdict(findings),
            "findings": [finding.to_dict() for finding in findings],
        })
        if len(rows) == count:
            return rows
    raise ValueError(f"requested {count} merged PRs but found only {len(rows)} at {base}")


def markdown_report(
    rows: list[dict], base: str, soft_limit: int, large_limit: int, large_min_delta: int = 50,
) -> str:
    """Render all verdicts and file-level evidence without claiming a false-positive rate."""
    warned = sum(bool(row["findings"]) for row in rows)
    lines = [
        "# File growth dry-run backtest (#917)", "",
        f"Pinned history tip: `{base}`. Latest {len(rows)} PRs on first-parent history.",
        f"Provisional limits: soft {soft_limit}, already-large {large_limit} lines.",
        f"Already-large reminders require at least {large_min_delta} net added lines.",
        f"Warnings in **{warned}/{len(rows)} PRs**; all growth-only verdicts pass.", "",
        "Counts cover whole files, including inline tests. Conventional test/generated",
        "paths and generated headers are excluded, as documented in",
        "[File Growth Dry Run](https://github.com/tensor4all/tensor4all-rs/blob/main/REPOSITORY_RULES.md#file-growth-dry-run).",
        "Results describe landed contributions, not original pre-merge PR heads.",
        "Warning frequency is not a measured false-positive rate.", "",
        "| PR | Growth verdict | File-size reminders |", "|---|---|---|",
    ]
    for row in rows:
        details = "<br>".join(
            f"`{item['file']}`: {item['summary']}".replace("|", "\\|")
            for item in row["findings"]
        ) or "None"
        lines.append(
            f"| [#{row['number']}](https://github.com/tensor4all/tensor4all-rs/pull/{row['number']}) "
            f"| {row['verdict']} | {details} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--soft-limit", type=int, default=1000)
    parser.add_argument("--large-limit", type=int, default=2000)
    parser.add_argument(
        "--large-min-delta", type=int, default=50,
        help="Minimum net added lines for already-large files (default: 50; 0 allows any growth)",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    if args.count <= 0 or not 0 < args.soft_limit <= args.large_limit:
        parser.error("require count > 0 and 0 < soft-limit <= large-limit")
    if args.large_min_delta < 0:
        parser.error("--large-min-delta must be nonnegative")
    base = reviewer.run_git(["rev-parse", "--verify", f"{args.base}^{{commit}}"]).strip()
    rows = backtest(base, args.count, args.soft_limit, args.large_limit, args.large_min_delta)
    print(markdown_report(
        rows, base, args.soft_limit, args.large_limit, args.large_min_delta,
    ), end="")
    if args.output_json:
        args.output_json.write_text(json.dumps({
            "base": base, "soft_limit": args.soft_limit,
            "large_limit": args.large_limit, "large_min_delta": args.large_min_delta,
            "prs": rows,
        }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
