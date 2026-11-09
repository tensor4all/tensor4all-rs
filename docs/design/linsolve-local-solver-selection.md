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
- An identity-only solve (`a1 = 0`) returns a finite `rhs / a0` for a real
  `a0` below `2^-500` or above `2^500`, and for a complex `a0` keeps the
  quotient components that `1 / a0` would flush (within the subnormal error
  bound given under Implementation), in two-site MALS and in
  one-site ALS with local GMRES. It previously formed `1 / a0` (or a complex
  reciprocal from an out-of-range `|a0|^2`) and returned `inf`, `NaN` or a
  flushed component. A real `a0` in
  `[2^-500, 2^500]` is unchanged: one multiplication by `1 / a0`. The
  guarantee holds for the stored entries of the scaled node, not for the
  represented values: in an extreme gauge (entries `2^-1000` on one node and
  `2^1000` on another) that node can underflow or overflow although the
  represented quotient is finite.
- `square_linsolve` rejects a zero `max_bond_dim` and an SVD policy whose
  threshold is negative or not finite before the identity-only shortcut, as
  `InvalidConfiguration`; these were previously accepted when the shortcut
  skipped the truncation.
- The `a0 = 0`, `a1 = 0` and `a1 = 1` arithmetic shortcuts in the local
  solver, in core `gmres_affine` and in the identity-only check apply only to
  untracked coefficients, so a gradient-tracked coefficient with one of these
  values stays in the graph.
- Gradients through a solve are those of the returned Krylov iterate, not of
  the exact solution. A solver that stops after `n` steps differentiates its
  `n`-step iterate, which misses directions outside that Krylov space: for
  `A = diag(2, 3)`, `b = (1, 2)`, `a0 = 1` and a tracked `a1 = 0`, every
  solver stops after one step with a zero residual and gives
  `d sum(x) / d a1 = -8.4`, while the exact value is `-sum(A b) = -8`.
  Implicit differentiation of the solve, which would give the exact
  derivative, is not implemented.

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
solution stays in the initial bond spaces; with GMRES each local `a0 x = b`
is again solved in closed form. Both closed forms multiply by the factors of
`AnyScalar::reciprocal_factors`:

- a real `a0` in `[2^-500, 2^500]`: the single factor `1 / a0`;
- another real `a0`: an exact power of two `2^j`, then `1 / (a0 * 2^j)`, with
  `|a0 * 2^j|` in `[1/4, 1/2)` for a small `a0` (every scaled entry stays below
  the quotient) and in `[2^64, 2^65)` for a large one (scaling down loses bits
  only for a quotient below `2^-1086`, where they cannot change the result).
  Every component then equals `fl(x * fl(1 / a0))` with the reciprocal rounded
  as if the exponent range were unbounded, also for a subnormal quotient;
- an untracked purely imaginary `a0 = b i`: the exact factor `-i`, then the
  factors of the real `b` (a tracked one takes the general path, whose
  recorded expression depends on both components);
- another complex `a0`, with `|a0 * 2^j|` in `[1/4, 1/2)` up to
  `|a0| = 2^201` and in `[2^64, 2^65)` above: `2^j` and `conj(a0 * 2^j)` when
  `j >= 0`, then the real `1 / |a0 * 2^j|^2`. When `j < 0` below `2^201`, the
  factors are `conj(a0 * 2^j)` and the real `2^j / |a0 * 2^j|^2` if scaling
  `a0` down is exact, and `2^j`, the unscaled `conj(a0)` and
  `2^j / |a0 * 2^j|^2` if it would flush a component. Above `2^201`, two
  powers of two with product `2^(2j)` surround `conj(a0)`, followed by
  `1 / |a0 * 2^j|^2`. A component of `1 / a0` can underflow while the matching
  component of the quotient is representable (`2^600 / (2^600 + 2^-420 i)`
  has imaginary part `-2^-1020`), and so can a component of a scaled-down
  `a0` (`2^1023 / (1 + 2^-1074 i)` has imaginary part `-2^-51`), so neither
  is formed on its own. A complex factor `w` can grow a component of a complex
  value to `|x| |w|`, about `1.4 * f64::MAX` for `x = f64::MAX * (1 + i)` and
  `|w|` near one, so every complex factor applied before the last one has
  `|w| <= 1/2`; no intermediate component then exceeds the largest component
  of the value or of the quotient. The cost is that the intermediate before
  the last factor is smaller than `q`, so in the subnormal range it is
  rounded more coarsely than `q`. The absolute error is then up to
  `9 * 2^-1074` (`|a0| < 1/2`) or `7 * 2^-1074` (otherwise), and a quotient
  component that small can round to zero.

The last product multiplies by a rounded factor, so a quotient within a few
units in the last place of `f64::MAX` can round to infinity:
`from_bits(0x5a5edcf89af5f111) / from_bits(0x1a5edcf89af5f112)` is
`f64::MAX` when divided directly but infinite through the factors, as it is
for `x * fl(1 / d)` with an unbounded exponent range.

The factors are `f64` scalars, and `scale` converts a scalar to the storage
precision of single-precision (`f32`, `Complex32`) entries. A factor outside
the `f32` range would become zero or infinite there: dividing `2^100` by
`2^200 + 2^200 i` needs `2^-199`, although the quotient `2^-101 - 2^-101 i`
is a normal `Complex32`. Both identity-only closed forms therefore reject a
single-precision right-hand side (core `gmres_affine`) or network (TreeTN,
when any node is single precision) when a factor fails
`AnyScalar::is_in_single_precision_range`, instead of returning a wrong
result. `TensorVectorSpace::is_single_precision` reports the storage
precision. Dividing in `f64` and converting once would avoid the error but
needs a cast in the tensor layer, which these closed forms do not have.

A right-hand side whose norm overflows (`b = (f64::MAX, f64::MAX)`) would make
every relative residual zero and report convergence. Every GMRES entry point
therefore returns `NumericalBreakdown` for a right-hand side whose norm is not
finite, as CG and BiCGSTAB do, and `relative_linear_system_residual`, which the
`square_linsolve` sweeps use, returns an error for such an rhs.

The coefficients themselves are converted the same way wherever they are
applied: in the closed-form residual `b - a0 x`, in every GMRES, CG and
BiCGSTAB step, and in `relative_linear_system_residual`. So for
single-precision vectors core `gmres_affine` rejects an `a0` or `a1` outside
the `f32` range as `InvalidOptions`, and `square_linsolve` and
`relative_linear_system_residual` reject it as `InvalidConfiguration` when a
node of their networks is single precision. `2^100 / (2^128 + 2^128 i)` has
factors in the `f32` range, but `a0` is not, so it is rejected.

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
