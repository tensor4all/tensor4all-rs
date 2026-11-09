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

## Test coverage

Tests exercise real/complex HPD and general affine systems, both sweep modes,
chains and branching trees, explicit/inferred index mappings, zero/tiny RHS,
exact/zero initial guesses, local versus sweep tolerances, exhausted/zero
budgets, selected versus inactive invalid controls, CG curvature/Hermitian
errors, BiCGSTAB breakdown and callback source-chain preservation.

Preconditioning, rank enrichment, core affine convenience APIs, compression
inside CG/BiCGSTAB and automatic fallback remain out of scope.

## Review follow-up

- a0 = a1 = 0 is rejected during option validation for every local solver.
- The identity-only closed form no longer depends on the selected solver.

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
- `validate_local_solver` rejects a NaN or infinite `a0` or `a1`, which could
  otherwise reach the identity-only closed form unchecked.
- Both identity-only closed forms (two-site MALS and the local GMRES of
  one-site ALS) divide through `AnyScalar::reciprocal_factors` instead of
  forming `1 / a0`, so a subnormal `a0` such as `3 * 2^-1070`, or a complex
  `a0` whose `|a0|^2` overflows or underflows, gives the finite quotient.
  Core tests cover: a complex quotient whose magnitude exceeds `f64::MAX`
  (`0.51 * f64::MAX / (0.35 + 0.35i)`) keeps finite components;
  `from_bits(2) / 3` gives `from_bits(1)`, and a quotient just above `2^-1075`
  (`from_bits(0x224d82c07cc53ee0) / from_bits(0x657d82c07cc53edf)`) rounds up
  to `from_bits(1)` rather than to zero; results with a real `a0` match the
  direct reciprocal bit for bit after scaling by `2^±600` and `2^±1000`;
  `2^600 / (2^600 + 2^-420 i)` and `2^400 / (2^400 + 2^-600 i)` keep the
  imaginary parts `-2^-1020` and `-2^-1000`; `1 + NaN i` is rejected. A
  purely imaginary `a0` is divided exactly (`2^-1074 / i = -2^-1074 i`); for
  other complex `a0`, a sweep of subnormal quotients checks the absolute error
  bound of `9 * 2^-1074` (`|a0| < 1/2`) or `3 * 2^-1074`. A divisor
  component that scaling down would flush is applied unscaled
  (`2^1023 / (1 + 2^-1074 i)` keeps `-2^-51 i`), and a tracked complex
  divisor, purely imaginary included, gives the gradient of the direct
  reciprocal.
- `TreeTN::scale_mut` applies each factor to one node, so the range guarantee
  is for that node's stored entries, not for the represented values.
- `square_linsolve` validates `max_bond_dim` and the SVD policy before the
  identity-only shortcut.
- The `a0 = 0`, `a1 = 0` and `a1 = 1` arithmetic shortcuts apply only to
  untracked coefficients; a test differentiates the local solution with
  respect to tracked `a0` and `a1` for every local solver. A two-dimensional
  case (`A = diag(2, 3)`, `b = (1, 2)`, tracked `a1 = 0`) pins the gradient of
  the one-step iterate, `-8.4`, which differs from the exact `-8`.
- Core `gmres_affine` rejects a non-finite `a0` or `a1` before the zero-RHS
  and closed-form returns, and its closed form reports the measured residual
  of the returned solution (`2^-1074 / (0.3 + 0.2 i)` rounds to zero and is
  not converged).
- `validate_local_solver` and the option, mapping and site-count checks at the
  start of `square_linsolve` return `TreeTNOperationError::InvalidConfiguration`.
- Every solver-owned Krylov workspace (basis, Hessenberg columns, Givens
  rotations, projected residual and matrix, back-substitution result) is
  reserved with `try_reserve_exact`. Allocations made by the operator callback
  and by vector operations, and the dense eigendecomposition of the projected
  matrix in the backend, are outside the solver and still abort on failure.

Local validation runs the focused commands below; their results for each
pushed head are listed in the PR's Verification section, and hosted CI on the
head is the independent check.

- `cargo fmt --all -- --check`
- `cargo test -p tensor4all-core --lib krylov` and `--lib any_scalar`
- `cargo test -p tensor4all-treetn --lib linsolve` and `--test integration linsolve`
- `cargo test -p tensor4all-itensorlike --lib`
- Doctests: `cargo test -p tensor4all-core --doc krylov` and `--doc reciprocal_factors`,
  `cargo test -p tensor4all-treetn --doc linsolve`, `cargo test -p tensor4all-itensorlike --doc`
- `cargo clippy -p tensor4all-core -p tensor4all-treetn -p tensor4all-itensorlike --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc`
- `python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run --file-growth`

Full-workspace, release, alternate-backend, binding and benchmark checks were
not run locally. Hosted PR CI runs the workspace `--lib --bins --tests` and
doctests on Linux and macOS (Accelerate); the non-default feature matrix runs
on a schedule, and the Python bindings (outside the workspace) and benchmarks
are not run by PR CI.
