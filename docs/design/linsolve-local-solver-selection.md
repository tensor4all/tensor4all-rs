# Local solver selection in TreeTN and MPS sweeps

## Public behavior

Local solver selection builds on one-site ALS and the core CG and BiCGSTAB
solvers. No public core affine wrappers are introduced.

TreeTN and itensorlike LinsolveOptions gain a shared LocalSolver type and
with_local_solver(...) builder. The default is LocalSolver::Gmres. The
variants are Gmres, Cg(CgOptions), and BiCgStab(BiCgStabOptions), using the existing
core option types directly. The MPS facade re-exports the same selector and
provides a getter. This avoids duplicate tolerance and iteration-budget fields.

GMRES keeps its current gmres_* controls and tolerance convention. They apply
only when GMRES is selected. CG and BiCGSTAB use their native rtol, atol and
max_iter controls, with their existing iteration/cycle meanings. Zero native
iteration budgets retain the initial residual check behavior.

All choices work with one-site ALS and two-site MALS, affine coefficients,
explicit/inferred index mappings, real/complex data, and existing sweep residual
and convergence reporting. The selector changes only the local equation solve;
QR transport, SVD splitting and environment synchronization retain their roles.

CG requires the effective operator a0 I + a1 A to be Hermitian positive-definite.
The core directional checks and errors are retained; they are not a certificate
that the whole operator is HPD. GMRES and BiCGSTAB can handle general operators.
There is no automatic solver selection or silent fallback.

## Implementation choices

Keep GMRES' existing affine projected-Hessenberg path unchanged. For CG and
BiCGSTAB, form a matrix-free callback applying a0*x + a1*A_local*x through the
existing TensorVectorSpace operations; skip zero terms. No dense production
materialization or new dependency is needed. New core affine convenience APIs
are deferred.

Validate selected options at the common owner, before zero-sweep/identity
shortcuts. Expose the existing CG/BiCGSTAB static option checks as validate()
methods on their core option types and use those same checks inside the core
entry points, rather than duplicating them in downstream crates. RHS-dependent
threshold/range validation stays inside the solvers. Inactive GMRES controls do
not reject a CG/BiCGSTAB solve. Keep typed errors and their source chains.

Local nonconvergence continues to provide the best available local solution;
the existing outer true residual decides sweep convergence. Numerical breakdown
and callback errors propagate. Trace output identifies the selected algorithm
and the residual convention correctly.

## Scope and validation

Change the shared options/exports, local solve dispatch, MPS forwarding and
validation, user documentation and bundled usage skill. Add tests exercising
actual dispatch, not just option storage: default GMRES parity; all solvers on
real/complex HPD systems; GMRES/BiCGSTAB on nonsymmetric/non-Hermitian systems;
affine shifts/scales; both sweep modes and branching trees; zero RHS; exact/zero
initial guesses; tiny RHS; exhausted and zero native budgets; selected vs
inactive invalid options; CG non-HPD errors; BiCGSTAB breakdown; wrapper parity;
residual and source-chain preservation. Preserve existing test tolerances.

The identity-only closed-form shortcut does not depend on the local solver:
two-site MALS returns rhs/a0 for every choice, and one-site ALS sweeps so the
solution stays in the initial bond spaces. Coefficients a0 = a1 = 0 are
rejected during option validation for every solver.

## Deferred

Preconditioning, rank enrichment, compression inside CG/BiCGSTAB, performance
benchmarks, extra solvers, and automatic fallback remain separate follow-ups.
