#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="main"
TITLE=""
BODY_FILE=""
AUTO_MERGE=1
DRAFT=0

usage() {
  cat <<'EOF'
Usage: bash scripts/create-pr.sh [options]

Options:
  --base BRANCH          Base branch for the pull request (default: main)
  --title TITLE          Pull request title (defaults to the latest commit subject)
  --body-file PATH       Markdown body file to pass to gh pr create
  --no-auto-merge        Do not enable auto-merge after PR creation
  --draft                Create the PR as a draft
  --help                 Show this help text
EOF
}

log() {
  printf '%s\n' "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "missing required command: $1"
    exit 1
  fi
}

require_clean_tree() {
  if [[ -n "$(git status --short)" ]]; then
    log "working tree is not clean"
    exit 1
  fi
}

ensure_body_file() {
  if [[ -n "$BODY_FILE" ]]; then
    return
  fi

  BODY_FILE="$(mktemp)"
  trap 'rm -f "$BODY_FILE"' EXIT
  {
    printf '## Summary\n\n'
    git log --format='- %s' "${BASE_BRANCH}..HEAD" 2>/dev/null || true
    printf '\n## Verification\n\n'
    printf -- '- `cargo fmt --all`\n'
    printf -- '- `cargo clippy --workspace`\n'
    printf -- '- `cargo nextest run --release --workspace`\n'
  } >"$BODY_FILE"
}

run_required_checks() {
  cargo fmt --all
  cargo clippy --workspace
  cargo nextest run --release --workspace
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      BASE_BRANCH="$2"
      shift 2
      ;;
    --title)
      TITLE="$2"
      shift 2
      ;;
    --body-file)
      BODY_FILE="$2"
      shift 2
      ;;
    --no-auto-merge)
      AUTO_MERGE=0
      shift
      ;;
    --draft)
      DRAFT=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      log "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

current_branch="$(git branch --show-current)"
if [[ -z "$current_branch" ]]; then
  log "not on a named branch"
  exit 1
fi
if [[ "$current_branch" == "main" || "$current_branch" == "master" ]]; then
  log "refusing to create a PR from ${current_branch}"
  exit 1
fi

require_clean_tree
require_command gh

run_required_checks
if [[ -n "$(git status --short)" ]]; then
  log "verification modified the working tree; commit those changes before creating a PR"
  exit 1
fi

if [[ -z "$TITLE" ]]; then
  TITLE="$(git log -1 --format=%s)"
fi

ensure_body_file
if git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' >/dev/null 2>&1; then
  git push
else
  git push -u origin "$current_branch"
fi

create_args=(pr create --base "$BASE_BRANCH" --title "$TITLE" --body-file "$BODY_FILE")
if [[ "$DRAFT" -eq 1 ]]; then
  create_args+=(--draft)
fi

pr_url="$(gh "${create_args[@]}")"
log "$pr_url"

if [[ "$AUTO_MERGE" -eq 1 ]]; then
  gh pr merge --auto --squash --delete-branch "$pr_url"
fi

log "monitoring required PR checks every 30 seconds"
bash scripts/monitor-pr-checks.sh "$pr_url" --interval 30
