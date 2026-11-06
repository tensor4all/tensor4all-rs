# Matrix-free conjugate gradients

`tensor4all_core::krylov::cg` solves `A x = b` for a caller-supplied, fixed
Hermitian positive-definite operator. It uses `TensorVectorSpace`, matching
the abstraction used by GMRES, without materializing a dense operator.
The implementation is derived from
[KrylovKit's CG](https://github.com/Jutho/KrylovKit.jl/blob/d51fa25adfa4cda5530d8e3d42eedfb288cca447/src/linsolve/cg.jl),
with upstream copyright and license retained in the core crate.

## Public contract

- `cg(apply_a, &b, &x0, &CgOptions)` returns `CgResult<T>` or `KrylovError`.
- `CgOptions` contains `max_iter`, `rtol`, `atol`, and `hermitian_tol`.
  Defaults are `100`, `1e-10`, `0`, and `1e-10` respectively. Tolerances must
  be finite and nonnegative. Zero iterations evaluates the initial residual.
- Convergence means `||b-Ax|| <= max(atol, rtol * ||b||)`. The result records
  the **absolute true residual**, completed CG steps, and convergence flag.
  This matches the absolute residual convention of Hermitian Lanczos rather
  than GMRES's default relative-residual diagnostic.
- A zero RHS follows the same validation and initial-residual path as any
  other RHS; a nonzero initial guess is not discarded.
- Iteration exhaustion returns an approximate solution with the true residual
  and `converged = false`, unless the final true residual meets the threshold.
- Invalid options, detected non-Hermitian curvature, nonpositive curvature,
  nonfinite/degenerate recurrence arithmetic, and tensor/operator failures
  have typed error variants. Underlying errors keep their source chains.

## Numerical and abstraction boundaries

The short recurrence retains a fixed number of vectors. When its residual
estimate meets the stopping threshold, recompute `b-Ax`. If that residual
does not meet the threshold, replace the recursive residual and restart the
search direction within the existing step budget. Always recompute the true
residual at iteration exhaustion, too.

Residuals, search directions, and operator images are normalized by exact
powers of two before inner products. This keeps reductions in range for
`f32`/`Complex32` as well as `f64`/`Complex64`. Scaling is applied in chunks
of at most 126 binary exponents: every scale factor is normal and finite in
all four scalar types. The physical solution and true-residual diagnostics
retain their original units.

For scaled vectors `r`, `p`, and `q`, let the physical residual be
`R = 2^er r`, the physical search direction be `P = 2^(er+ep) p`, and
`A p = 2^ea q`. With `rho = r^H r` and `h = Re(p^H q)`, the usual CG step is
`x += 2^(er-ep-ea) (rho/h) p` and `r -= 2^(-ep) (rho/h) q`.
After dividing the new working residual by `2^k`, its direction recurrence is
`p_new = r_new + 2^(k+ep) (rho_new/rho) p`, with `er += k`.
These transformations change the working scale, not the linear system or the
CG iteration budget. Coefficients remain in `AnyScalar` arithmetic.

HPD directional inner products and recurrence coefficients are mathematically
real. Their real projections and arithmetic use `AnyScalar`, preserving
backend metadata where supported. Norm, curvature, and stopping decisions are
explicit scalar control-flow boundaries; differentiability is not promised.

The Hermitian check uses the directional Rayleigh quotient
`q = p^H A p / p^H p` and bounds `|Im(q)|` by
`hermitian_tol * max(1, |q|)`. The absolute floor admits cancellation roundoff,
as in the existing projected-Hermitian helpers; it is evaluated from scaled
vectors without forming an overflowing quotient. Positive definiteness is
checked on the visited directions. Neither check
certifies an arbitrary matrix-free operator. The HPD property remains a
caller precondition. Single-precision callers should loosen the residual
and Hermitian tolerances to their working precision.

Preconditioning, affine convenience APIs, compression-aware CG, and selection
of CG inside TreeTN sweeps are separate extensions.

## Shared norm boundary

`IdxTensor::norm` finishes the LASSQ accumulator directly, while
`norm_squared` still squares its result. `TensorVectorSpace::norm` for
`IdxTensor` delegates to the stable norm, and `BlockTensor::norm` combines
block norms with `hypot`. The explicit CPU-session norm shares the same
LASSQ accumulation and direct finish. This preserves norms such as `1e-200`
and `1e200` even though their squares cannot be represented in `f64`.
Other vector-space implementations should override the trait's default
square-root-of-squared-norm calculation when they need that range.
