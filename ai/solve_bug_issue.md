You are running one iteration of an automated bug-fix workflow for this repository.

Your job is to:

1. Inspect open bug and bug-like issues.
2. Choose exactly one actionable workstream.
3. If open bug / bug-like issues are effectively zero, terminate cleanly with no code changes and no PR creation. In other words, if there are effectively no open bug or bug-like issues, leave the repository untouched and summarize that no worthwhile bug or bug-like issue was available.
4. Otherwise, fix the issue or close it only when it is clearly irrelevant, duplicate, or already fixed.
5. Use the repository-local PR helpers:
   - `bash scripts/create-pr.sh`
   - `bash scripts/monitor-pr-checks.sh`
6. Continue until merge or a defined stop condition.
7. Restore the local checkout as part of normal cleanup.

Hard rules:

- Work from the latest `origin/main`.
- Use a dedicated worktree only after selecting a real candidate.
- Do not touch unrelated work.
- Stop after two core fix attempts.
- Do not ask the user questions.
- Do not create a PR if there is nothing actionable to fix.

When you terminate cleanly without action, summarize that no worthwhile bug or bug-like issue was available and that no code was changed.
