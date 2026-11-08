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
native iteration budget still checks the initial residual. With CG or BiCGSTAB
selected, the `gmres_*` controls must keep their default values; a changed
value is an error rather than a silently ignored setting.

Every choice works with one-site ALS and two-site MALS, affine coefficients,
explicit and inferred index mappings, and real and complex data. The selector
changes only the local equation solve; QR transport, SVD splitting and
environment updates do not depend on it. Sweep residuals and convergence
reporting are unchanged.

CG requires the effective operator `a0 I + a1 A` to be Hermitian
positive-definite. The core CG checks (Hermitian consistency along search
directions, positive curvature) report violations as errors, but they do not
certify that the whole operator is HPD. GMRES and BiCGSTAB accept general
operators. There is no automatic solver selection and no fallback to another
solver.

## Implementation

GMRES solves the affine system through its projected Hessenberg problem, so
the local operator passed to it stays unshifted. CG and BiCGSTAB receive a
matrix-free callback that applies `a0 x + a1 A_local x` with
`TensorVectorSpace` operations. The callback skips a zero `a0` or `a1` term and
returns `A_local x` without a scaling copy when `a0 = 0` and `a1 = 1`. No dense
operator is formed.

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
solution stays in the initial bond spaces.

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
- CG non-HPD errors and BiCGSTAB breakdown;
- residual units and callback error source chains.

## Not covered

Preconditioning, rank enrichment, compression inside CG or BiCGSTAB, additional
solvers, automatic fallback and performance benchmarks are outside this design.
