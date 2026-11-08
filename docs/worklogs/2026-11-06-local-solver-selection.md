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

## Review responses on #1199

- Iteration-sized Krylov workspaces in core GMRES, Lanczos and Krylov `expm`
  are allocated with `try_reserve_exact`; a `max_iter` (or
  `gmres_restart_dim`) whose workspace cannot be allocated returns
  `KrylovError::InvalidOptions` instead of aborting. `usize::MAX - 1` is
  tested at the core, local-dispatch and `square_linsolve` levels.
- `# Errors` sections of `square_linsolve`, `validate_local_solver`, the MPS
  `linsolve` function and method, and core GMRES/Lanczos list the conditions a
  caller can observe.
- `square_linsolve` rejects a negative or non-finite `convergence_tol`, so
  every invalid option is an error.
- `LocalSolver::Cg` and the design note state that CG's Hermitian check cannot
  fire for real scalars; a test records CG non-convergence on a real
  non-symmetric operator that BiCGSTAB and GMRES solve, and indefinite
  operators fail with `NonPositiveCurvature` through TreeTN and the MPS wrapper.
- The `a1 = 1` shortcut recognizes real and complex spellings of one.
- The CG docs do not promise non-convergence for real non-symmetric
  operators: CG can converge when the Krylov space suits it, for example
  `A = [[2, 1], [0, 2]]` with `b = (2, 0)`.
- `square_linsolve` rejects a single index mapping before the identity-only
  shortcut, which skips the mappings when no residual is requested.
- The MPS `linsolve` documents that it does not report non-convergence and
  points to `square_linsolve` for `converged` and the residual.

The counts below are from local runs (not CI), on macOS, of the commands listed
with each count; they are not reproducible from the diff alone.

Validation on the branch merged with main `e38a7498`:

- `cargo test -p tensor4all-core --lib krylov`: 121 passed, 553 filtered out
- `cargo test -p tensor4all-treetn --lib linsolve`: 33 passed, 630 filtered out
- `cargo test -p tensor4all-treetn --test integration linsolve`: 71 passed, 305 filtered out
- `cargo test -p tensor4all-itensorlike --lib`: 139 passed
- Doctests: `cargo test -p tensor4all-core --doc krylov` 21 passed (383 filtered out),
  `cargo test -p tensor4all-treetn --doc linsolve` 8 passed (197 filtered out),
  `cargo test -p tensor4all-itensorlike --doc` 37 passed.
- `cargo clippy -p tensor4all-core -p tensor4all-treetn -p tensor4all-itensorlike --lib --tests`: no findings
- `cargo clippy -p tensor4all-treetn -p tensor4all-itensorlike --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc`: no findings
- `python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run --file-growth`: pass

Full-workspace, release, alternate-backend, binding and benchmark checks were
not run locally; hosted CI covers them.
