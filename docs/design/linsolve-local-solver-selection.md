# Local solver selection in TreeTN and MPS sweeps

## Public behavior

`tensor4all_treetn::LinsolveOptions` and `tensor4all_itensorlike::LinsolveOptions`
select the solver for each projected local equation with
`with_local_solver(LocalSolver)`. `LocalSolver` is defined in
`tensor4all-treetn` and re-exported by `tensor4all-itensorlike`, together with
`CgOptions` and `BiCgStabOptions`. Its variants are:

- `LocalSolver::Gmres` (the default), configured by the `gmres_*` controls of
  `LinsolveOptions`;
- `LocalSolver::Cg(CgOptions)`;
- `LocalSolver::BiCgStab(BiCgStabOptions)`.

CG and BiCGSTAB use the core option types directly, so their `rtol`, `atol` and
`max_iter` keep the meaning they have in `tensor4all_core::krylov`. A zero
native iteration budget still checks the initial residual.

Each solver's settings have exactly one owner: the `gmres_*` fields for GMRES,
the variant's options for CG and BiCGSTAB. Neither is a view of the other.
With CG or BiCGSTAB selected the `gmres_*` controls would have no effect, so
they must keep their default values; a changed value is an error rather than a
silently ignored setting.

Invalid options are always errors; no value is clamped or replaced. The affine
coefficients `a0` and `a1` must be finite, and not both zero. Zero
CG/BiCGSTAB tolerances are valid because those solvers stop when the residual
is at most the threshold. `gmres_tol` must be positive because GMRES stops when
the residual is strictly below it, so zero could never be met.

Every choice works with one-site ALS and two-site MALS, affine coefficients,
explicit and inferred index mappings, and real and complex data. The selector
changes only the local equation solve; QR transport, SVD splitting and
environment updates do not depend on it. Sweep residuals and convergence
reporting are unchanged.

CG requires the effective operator `a0 I + a1 A` to be Hermitian
positive-definite. The core CG checks (Hermitian consistency along search
directions, positive curvature) report violations as errors, but they do not
certify that the whole operator is HPD. The Hermitian-consistency check
measures the imaginary part of the directional curvature, which is zero for
real scalars, so a real non-symmetric operator is not detected as such. CG
then reports a non-positive-curvature error if a visited curvature is not
positive; otherwise it may still converge when the Krylov space happens to
suit CG (for example `A = [[2, 1], [0, 2]]` with `b = (2, 0)`), and otherwise it
does not converge and the sweep reports `converged = false` with its true
residual. GMRES and BiCGSTAB accept
general operators. There is no automatic solver selection and no fallback to
another solver.

### Behavior changes for existing callers

- `tensor4all_treetn::square_linsolve` rejects `a0 = a1 = 0` during option
  validation, for every solver, before the inputs are checked. It was already
  an error in both sweep modes, raised later by the identity-only closed form
  with a different message.
- `tensor4all_treetn::square_linsolve` rejects `gmres_tol` that is not finite
  and positive (for example `0.0`), and `convergence_tol` that is negative or
  not finite. Both were previously accepted by TreeTN; the MPS facade already
  rejected them.
- `tensor4all_itensorlike::linsolve` reports invalid GMRES and local solver
  options as `TensorTrainError::TreeTn` instead of `OperationError`.
- A `gmres_restart_dim` (or core GMRES/Lanczos `max_iter`) whose Krylov
  workspace cannot be allocated returns `KrylovError::InvalidOptions` instead
  of aborting on a capacity overflow.
- The identity-only closed form returns a finite `rhs / a0` for `a0` whose
  largest component is below `2^-500` or above `2^500`, where it previously
  formed an infinite `1 / a0` (or a complex reciprocal from an out-of-range
  `|a0|^2`) and returned `inf` or `NaN` entries. Other `a0` are unchanged: one
  multiplication by `1 / a0`.

## Implementation

GMRES solves the affine system through its projected Hessenberg problem, so
the local operator passed to it stays unshifted. CG and BiCGSTAB receive a
matrix-free callback that applies `a0 x + a1 A_local x` with
`TensorVectorSpace` operations. The callback skips a zero `a0` or `a1` term and
returns `A_local x` without a scaling copy when `a0 = 0` and `a1` is exactly
one, whether given as a real or a complex scalar. No dense operator is formed.

`LinsolveOptions::validate_local_solver` is the single owner of local solver
validation. `square_linsolve` calls it before the zero-sweep and identity-only
shortcuts, and the MPS `linsolve` calls it on the converted TreeTN options and
reports failures as `TensorTrainError::TreeTn`. It rejects `a0 = a1 = 0` for
every solver, invalid active GMRES controls, changed `gmres_*` controls with CG
or BiCGSTAB, and invalid native options. The native static checks are
`CgOptions::validate` and `BiCgStabOptions::validate`, which the core `cg` and
`bicgstab` entry points also call. RHS-dependent thresholds are checked inside
the solvers. Typed errors keep their source chains, so a `KrylovError` can be
recovered from a TreeTN or MPS error.

A local solve that stops at its iteration budget returns its best iterate; the
outer true residual decides sweep convergence. Numerical breakdown, CG
curvature and Hermitian-consistency errors, and callback errors propagate. The
trace output enabled by `T4A_LINSOLVE_TRACE_LIMIT` names the selected solver
and whether the reported local residual is relative or absolute.

The identity-only closed form does not depend on the local solver: two-site
MALS returns `rhs / a0` for every choice, and one-site ALS sweeps so the
solution stays in the initial bond spaces. The closed form multiplies by
`1 / a0` when the largest component of `a0` is in `[2^-500, 2^500]`. Outside
that range it normalizes `a0` by an exact power of two: for a small `a0` the
power of two is applied first, with a factor of four withheld so that every
component of the intermediate stays below the largest component of the
quotient; for a large `a0` the
reciprocal is applied first and the power of two last, in one step, so a
subnormal quotient is rounded once. Each entry gets the same two roundings as
a direct reciprocal multiplication.

## Tests

The tests run the actual dispatch rather than option storage:

- default GMRES matches explicitly selected GMRES;
- all solvers on real and complex HPD systems, through TreeTN and the MPS
  wrapper, with a non-identity operator;
- GMRES and BiCGSTAB on non-Hermitian systems;
- affine shifts and scales, including `a0 = 0` with `a1 = 1` and `a1 != 1`;
- one-site and two-site sweeps on chains and branching trees;
- zero and tiny right-hand sides, exact and zero initial guesses;
- exhausted and zero native budgets, and local versus sweep tolerances;
- invalid selected options, and changed inactive GMRES controls;
- CG errors on negative-definite, indefinite and non-Hermitian operators, CG
  non-convergence on a real non-symmetric operator that BiCGSTAB and GMRES
  solve, and BiCGSTAB breakdown;
- an unallocatable GMRES restart length and invalid convergence tolerances;
- residual units and callback error source chains.

## Not covered

Preconditioning, rank enrichment, compression inside CG or BiCGSTAB, additional
solvers, automatic fallback and performance benchmarks are outside this design.
