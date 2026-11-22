#!/usr/bin/env bash
# Every crate whose types cross a crate boundary in this graph must resolve to
# one version: two semver-compatible copies are two incompatible types that
# cannot meet (E0308). The manifests use compatible ranges rather than exact
# pins so a dependency's patch fix can be locked (R-git-workflow-58); this check
# is what stops a range from silently splitting an identity.
#
# `cargo tree -d` lists each duplicated crate as an unindented `name vX.Y.Z`
# line with its dependents indented under it, so only those top-level lines are
# matched.
set -Eeuo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

# The CPU provider stack, the one dense-library identity (colmat, formerly
# t4a-faer), and the tensor/engine crates
# that pass their own values into it.
identities='strided[-a-z]*|threadteam|tcontract[-a-z]*|cpueinsum[-a-z]*|tlinalg[-a-z]*|t4a-faer[-a-z]*|colmat[-a-z]*|elem-simd-math|tenferro[-a-z]*|hataori[-a-z]*|t4a-cubecl[-a-z]*|t4a-cubek[-a-z]*|cubecl[-a-z]*|cubek[-a-z]*'

# A resolution error must fail this check too, so the tree is built first.
tree=$(cargo tree --locked -d -e normal)
duplicates=$(grep -E "^(${identities}) v" <<<"$tree" || true)
if [[ -n "$duplicates" ]]; then
    printf '%s\n' "$duplicates" >&2
    echo "check-single-versions: duplicate dependency identities in the normal graph" >&2
    exit 1
fi
echo "check-single-versions: one version per dependency identity"

# The feature-configuration packages under tests/feature-configurations/ are
# separate workspaces with their own committed lock, which the tree check above
# never sees. Each must resolve every registry and git package to an entry of
# the root lock (same name, version and source), so a pin bump that refreshes
# only the root lock fails here instead of leaving a nested lock on the old
# graph (#1017). The PyO3 crate (crates/tensor4all-py) is its own workspace
# root with dependencies the root graph does not have (pyo3, numpy), so only
# its entries for the identities above must match the root lock.
lock_entries() {
    awk '/^\[\[package\]\]/ { name = ""; version = "" }
         /^name = /    { name = $3 }
         /^version = / { version = $3 }
         /^source = /  { print name, version, $3 }' "$1" | sort -u
}
identity_entries() {
    lock_entries "$1" | grep -E "^\"(${identities})\" " || true
}
root_entries=$(lock_entries Cargo.lock)
stale=0
report_missing() {
    if [[ -n "$2" ]]; then
        printf '%s: not in the root Cargo.lock:\n%s\n' "$1" "$2" >&2
        stale=1
    fi
}
for lock in tests/feature-configurations/*/Cargo.lock; do
    [[ -e "$lock" ]] || continue
    report_missing "$lock" "$(comm -23 <(lock_entries "$lock") <(printf '%s\n' "$root_entries"))"
done
py_lock=crates/tensor4all-py/Cargo.lock
report_missing "$py_lock" "$(comm -23 <(identity_entries "$py_lock") <(printf '%s\n' "$root_entries"))"
if (( stale )); then
    echo "check-single-versions: a nested workspace lock has drifted from the root lock" >&2
    exit 1
fi
echo "check-single-versions: nested workspace locks match the root lock"

