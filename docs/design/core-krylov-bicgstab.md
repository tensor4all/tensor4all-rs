# Matrix-free BiCGSTAB

`tensor4all_core::krylov::bicgstab` solves `A x = b` for a fixed linear
operator, including real nonsymmetric and complex non-Hermitian systems.
It uses `TensorVectorSpace` and a fixed number of vectors. The implementation
is derived from [KrylovKit.jl's BiCGStab](https://github.com/Jutho/KrylovKit.jl/blob/d51fa25adfa4cda5530d8e3d42eedfb288cca447/src/linsolve/bicgstab.jl),
with the upstream MIT notice retained in the core crate.

## Public contract

- `bicgstab(apply_a, &b, &x0, &BiCgStabOptions)` returns
  `Result<BiCgStabResult<T>, KrylovError>` for `T: TensorVectorSpace`.
- Options are `max_iter = 100`, `rtol = 1e-10`, `atol = 0` by default.
  Tolerances must be finite and nonnegative; their derived stopping threshold
  must be finite. Single-precision callers should use a tolerance appropriate
  to their precision, such as `rtol = 1e-5`.
- Convergence requires `||b-Ax|| <= max(atol, rtol * ||b||)`.
  The result contains `solution`, `iterations`, `residual_norm` (absolute true
  residual), and `converged`, matching CG's diagnostic convention.
- One iteration is one consumed BiCGSTAB cycle. Convergence at the
  intermediate BiCG step counts as that cycle. Residual-replacement restarts
  consume the same original cycle budget. Zero cycles check the initial
  residual only. A zero-norm RHS returns the zero vector after the initial
  residual confirms that `x0` and `b` share a vector space.
- Iteration exhaustion returns the final true residual and convergence flag.
  Invalid options use `InvalidOptions`; unusable recurrence arithmetic uses
  `NumericalBreakdown` with the cycle and quantity; tensor/operator failures
  use `Operation` and preserve their source chains.

## Residuals and breakdown

Use conjugating inner products for the complex recurrence. Check `b-Ax`
before accepting convergence at both the intermediate and stabilized steps,
and at cycle exhaustion. If an estimate is optimistic, replace the residual
and restart the shadow residual and search direction together. This costs
one cycle, including when the restart happens at the intermediate step.

Check rho, the alpha denominator, stabilization denominator, omega, direction
coefficients, vector norms, and physical residual estimates for unusable
values. Finite arithmetic cannot always represent a required physical update;
that condition is a numerical breakdown. Before returning a detected recurrence
breakdown, evaluate the current true residual so an already reached solution
can still succeed. Errors from this verification preserve their source chains.
BiCGSTAB can break down even for an invertible operator; GMRES is an alternative.

Callback errors are wrapped as `Operation` before recurrence recovery. This
keeps a nested solver's `NumericalBreakdown` distinct from BiCGSTAB's own
breakdowns; callback failures are propagated without retrying or suppressing
their original typed source.

## Scaled recurrence

Shared private vector helpers implement exact power-of-two scaling in chunks
of at most 126 exponents, finite and normal for all four scalar types. CG uses
the same helpers with its recurrence unchanged. The vector-space implementation
must provide a norm preserving the required range; `IdxTensor` and `BlockTensor`
do so through the stable norm paths described in [the CG design](core-krylov-cg.md#shared-norm-boundary).

Let `R = 2^er r` be the physical residual, and normalize `p` before applying
`A p = 2^ev v`. The shadow residual is fixed within a recurrence and normalized
when that recurrence starts. With `rho = shadow^H r`, `sigma = shadow^H v`,
and `alpha = rho/sigma`, the intermediate updates are
`x_half = x + 2^(er-ev) alpha p` and `s_work = r - alpha v`.

Normalize `s_work = 2^es s` and `A s = 2^et t`. With
`omega = (t^H s)/(t^H t)`, update
`x = x_half + 2^(er+es-et) omega s` and `r_work = s - omega t`.
Normalize `r_work = 2^ek r_new`; the next physical residual exponent is
`er_new = er + es + ek`. Define `rho_new = shadow^H r_new`,
`gamma = (rho_new/rho) alpha`, and `beta = gamma/omega`.

Evaluate gamma as a combined product and quotient: outside the safe exponent
range, normalize all three scalar factors by powers of two before multiplying
and dividing, then restore the combined exponent. This prevents an overflowing
or underflowing intermediate quotient from destroying a representable gamma.
The ordinary-range path retains the original evaluation order. Scalar
normalization preserves `AnyScalar` metadata.

The next direction is proportional to
`r_new + 2^(et-ev) beta p - gamma v`. Before combining these three terms,
divide them by the common power `2^max(0, et-ev)`, so all explicit vector
scaling exponents are nonpositive. Direction normalization in the next cycle
removes that common scale.

These transformations avoid squared physical norms and overflowing physical
scalar ratios. Coefficients remain in `AnyScalar` arithmetic, including their
imaginary parts. Norm, range, and stopping decisions are explicit scalar control-flow
boundaries; a differentiable solve is not promised. The solver materializes
no dense operator. Preconditioning, affine convenience APIs, compression-aware
variants, and TreeTN solver selection are separate extensions.

Native complex scalar division is handled in the shared tensor backend. Its
real and imaginary products carry separate binary exponents, so denominator
squares and intermediate numerators stay in range without erasing a small
component beside a large one. Nonfinite inputs and zero divisors retain the
existing native classification. The separately implemented traced division
path already handles the reviewed tiny/huge cases and retains its graph.
