# Local solver selection in TreeTN and MPS sweeps

## Scope and integration

The scope selects GMRES (default), CG or BiCGSTAB in one-site ALS and
two-site MALS. The work is built on a merge of the prerequisite PR heads:

- CG #1147: `f3748aee862ccf403986cd6ae907f39cef22bab9`.
- BiCGSTAB #1148: `714223b2f5af81c4ca5be5dabf3ff5c250fc7e76`.
- ALS #1149: `9e9d246b2d953407a910195be71ec0d4b81b6d48`.

The heads merge without conflicts into tree
`18fe16ceac9bc538710d6ee536d6a6b310be97ca`, recorded as an octopus merge
commit. The solver-selection changes are a separate commit on top of it, so the
branch can be rebased onto main once the prerequisites land.

## Implementation decisions

- `LocalSolver::{Gmres, Cg(CgOptions), BiCgStab(BiCgStabOptions)}` reuses native
  configurations. Both options types expose `with_local_solver`; the MPS facade
  re-exports the same enum, `CgOptions` and `BiCgStabOptions`, and provides a
  borrowing getter. The enum is `#[non_exhaustive]`.
- GMRES retains its existing affine projected-Hessenberg path and controls.
  Other choices reject changed `gmres_*` controls and apply the effective
  affine operator through existing vector-space operations, skipping zero terms
  and the scaling copy when `a0 = 0, a1 = 1`. The MPS facade takes its GMRES
  defaults from TreeTN so the two cannot disagree.
- CG requires the effective affine operator to be HPD; no automatic selection
  or fallback is added. Directional checks do not certify the whole operator.
- Native static validation is extracted into `CgOptions::validate` and
  `BiCgStabOptions::validate`, also called by their core entry points.
  `tensor4all_treetn::LinsolveOptions::validate_local_solver` owns all local
  solver validation and runs before shortcut/zero-sweep returns. The MPS facade
  calls it on the converted options and reports failures as
  `TensorTrainError::TreeTn`, as it already does for invalid update modes; a
  `KrylovError` stays in the source chain. a0 = a1 = 0 is rejected for every
  solver.
- The identity-only closed-form shortcut is solver-independent: two-site MALS
  returns rhs/a0 for every choice; one-site ALS sweeps to keep the initial bond
  spaces. Outer true residuals still decide sweep convergence.
- Local dispatch preserves typed errors and callback source chains. Trace output
  names the algorithm and distinguishes relative from absolute residual values.
- Existing GMRES option-conversion tests moved to their new common owner.
  No assertion tolerances, dependencies or lint configuration changed.

## Validation

- Prerequisite baseline: 203 focused Krylov/linsolve tests passed.
- Changed-crate Nextest (`--lib --tests`): 2,377 passed, 10 existing skips.
  One unchanged GSE AD test was slow under concurrent validation.
- Doctests: 644 passed (403 core, 204 treetn, 37 itensorlike).
- Optimized regressions: 13 TreeTN integration groups and 8 local-dispatch unit
  groups passed. Both native option-validation regressions passed in debug.
- Tests exercise real/complex HPD and general affine systems, both sweep modes,
  chains and branching trees, explicit/inferred index mappings, zero/tiny RHS,
  exact/zero initial guesses, local versus sweep tolerances, exhausted/zero
  budgets, selected versus inactive invalid controls, CG curvature/Hermitian
  errors, BiCGSTAB breakdown and callback source-chain preservation.
- Formatting, public-error docs, complete API inventory and deterministic
  repository rules (including file growth) passed. Rustdoc completed with the
  five previously observed core/TreeTN warnings.
- Strict Clippy stops at 43 existing core `nonminimal_bool` findings. The whole
  changed-crate check passes with only that known lint suppressed. The new
  test's needless-borrow finding was fixed and its 8 unit tests rerun.
- Connected-source self-review checked affine dispatch, scalar/vector-space
  operations, iteration meanings, true residual ownership, typed error chains,
  and preservation of exact QR, SVD and environment paths. Production paths
  introduce no full dense materialization. Moved tests retain helper coverage.

Local builds used Rustup stable 1.96 and the existing shared Cargo target. No
full-workspace, alternate-backend, binding or performance-benchmark checks were
run. Preconditioning, rank enrichment, core affine convenience APIs, compression
inside CG/BiCGSTAB and automatic fallback remain out of scope.

## Review follow-up

- a0 = a1 = 0 is rejected during option validation for every local solver.
- The identity-only closed form no longer depends on the selected solver.
- TreeTN and itensorlike Nextest: 1,194 passed, 7 skipped; doctests: 241
  passed; Clippy on both crates reported no new findings.

## Integration with merged prerequisites

After #1147, #1148 and #1149 merged, the solver-selection change was rebased
onto main. The integration conflicts kept both the newer CG failure-path tests
and the option-validation test, and kept `GmresToleranceMode` alongside the
`LocalSolver` exports.

Validation on main `8b756648` with the final change:

- TreeTN `--lib linsolve`: 30 passed. TreeTN integration `linsolve`: 66 passed.
- itensorlike `--lib`: 137 passed.
- Doctests: 8 TreeTN `linsolve`, 37 itensorlike and 21 core `krylov` passed.
- Clippy on TreeTN and itensorlike (`--lib --tests`): no findings.
- Deterministic repository rules with file growth: pass.

Full-workspace, release, alternate-backend, binding and benchmark checks were
not run; local disk space did not allow full builds.
