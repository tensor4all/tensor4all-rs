# TreeTN square linsolve: MPO external-index handling

Source of the still-live rationale from the deleted `plan/linsolve-mpo.md`
(tensor4all-rs#566 Task 11 debris removal).

## Scope

`crates/tensor4all-treetn/src/linsolve/square` solves
`(a₀ + a₁ A) x = b` over tree tensor networks where input and output spaces
coincide (`V_in = V_out`).

## Conventions (implemented)

- The local problem is `⟨ref|H|x⟩` vs `⟨ref|b⟩` — the reference state is the
  conjugated bra and `b` is the ket. See the `ProjectedState` rustdoc in
  `crates/tensor4all-treetn/src/linsolve/square/projected_state.rs` and the
  implementation of `relative_linear_system_residual` in
  `crates/tensor4all-treetn/src/linsolve/square/mod.rs`.
- Operators with internal input/output indices are supported through explicit
  per-site `IndexMapping { true_index, internal_index }` maps; both input and
  output maps must be provided together.
- External-index handling: global topology and site-dimension compatibility is
  validated up front (`validate_linsolve_inputs` in `square/mod.rs`); the exact
  local external-index identity precheck (init vs RHS, `<ref|b>` conventions)
  happens inside `SquareLinsolveUpdater` (see the index-structure mismatch
  diagnostics in `crates/tensor4all-treetn/src/linsolve/square/updater.rs`).
- Single-node systems use one-site ALS with `LinsolveOptions::with_nsite(1)`.
  The default two-site mode still rejects them rather than silently returning
  the initial guess. See [One-site ALS linear solves](linsolve-one-site-als.md).

## Verification

- Mapped identity solve and solution reproduction:
  `crates/tensor4all-treetn/tests/linsolve.rs`
  (`test_square_linsolve_with_mappings_identity`, one-site rejection test).
- Matching/mismatching MPO-like structures:
  `crates/tensor4all-treetn/tests/linsolve_mpo_xb.rs`.

- One-site ALS dense references, rank limits, and transport/cache regressions:
  `crates/tensor4all-treetn/tests/linsolve_one_site.rs`.
