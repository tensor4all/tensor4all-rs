# Tree patching milestones and M6 readiness

## Decisions

The [roadmap](../design/tree-adaptive-patching-roadmap.md) distinguishes
shipped interpolation lanes/MPI from the remaining contraction and validation
exits. The existing M6 branch is synchronized with main `097e2927`, preserving
the original proposal rather than creating a competing design.

The [M6 issue assessment](../design/tree-patched-contraction.md#issue-dependencies-and-implementation-gates)
finds no open issue that must close before the first serial outcome slice.
Claimed cache work and queued approximate-sum work remain with their existing
owners. Adaptive overlap handling, outcome semantics and sibling-merge error
accounting remain design gates; this update does not approve implementation.

The [outcome review candidate](../design/treetn-contraction-outcome.md)
keeps `abort_at_rank` separate from the truncation cap, reports stored link
dimensions, preserves context and caller-RNG semantics, and limits the first
early abort to deterministic general-tree zip-up. Report collection must not
add allocations to ordinary contraction. Post-hoc reporting comes first so
all methods retain their current dispatch and validation.

The old sibling-merge proof incorrectly treated nested parent/child residuals
as disjoint. Its claimed global squared-sum bound is withdrawn. The proposal
now requires a conservative multi-level ledger or a final residual check,
including rollback/resource semantics, before merging can claim an allowance.
The related reconstruction engine and scheduler were checked: they measure
final disjoint regions or add level bounds linearly, respectively. Their
existing accounting is a reuse reference, not another confirmed instance of
the proposal's error.

## Verification conclusions and constraints

Milestone dispositions are checked against public API inventories, current
source/tests, merged PR records and issue comments. The relevant source seams
are `adaptive_interpolation/{fan_out,mpi}.rs`,
`patching.rs::contract_adaptive_with_selection`,
`partitioned_tree_tn.rs::{refined_for_contract,contract_pairs}`, and
`subdomain_tree_tn.rs::exact_sum`. The remaining generic-path interpolation
determinism tests are still ignored; the upstream default-planner fix does not
establish downstream parity by itself.

The analytical adaptive baselines cover real/complex degree-three trees,
same-ID indices distinguished by prime level, cancelling contributions,
threshold equality, uncapped and exhausted-order results, two split strategies
and two exact sum engines. Ordinary and selected outputs are checked against
known values and exact projector layouts; empty/unmatched selections retain
input validation. The three baseline tests pass without changing production
code or any existing tolerance.
Two temporary mutations confirm the baselines exercise the intended paths:
strict `>` cap comparisons fail both numerical baselines, and dropping the
selection fails all three tests. Production source was restored byte-for-byte.

Local link/anchor consistency and the deterministic repository-rules preview
are checked, and strict Clippy passes for the changed integration target with
dependency linting disabled. The crate's library, integration and routing-oracle
tests pass under `cargo test`; Nextest is unavailable. Integration has 257
passing tests and 12 existing ignored tests, including the three new baselines.
MPI and performance measurements are not rerun; historical
verification is attributed to its PR. No public API, publication or issue
assignment changes are part of this work.
