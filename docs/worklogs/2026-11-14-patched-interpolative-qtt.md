# Patched interpolative QTT construction

## Decisions

- Follow the prefix-bearing patch representation of #1276: a bare vector of
  local trains loses the original boxes. The construction remains independent
  of TreeTN; conversion belongs in the executable tutorial.
- Share multiscale core assembly and SVD sweeps. Explicit batch sampling and
  session execution avoid a duplicated interpolation algorithm or implicit
  process-global backend use. Each batch names its kind (Chebyshev nodes of an
  interval, or exact last-core grid points), so the residual check never infers
  a box from its points.
- Accept on the per-value interpolation residual and use the rank cap to trigger
  splitting; the cap never truncates. The SVD discarded-weight bound is
  reported per patch as `truncation_bound` but stays out of acceptance (review
  round, below).
- Both entries run on caller-supplied engines; neither creates a backend.
- Preserve parent endpoints when splitting dyadic boxes. Recomputing the upper
  endpoint as two additions of the half length can change it by an ulp, leaving
  a gap or overlap in non-binary-exact physical domains.

## Verification conclusions and constraints

- Real and complex 1D/2D fixtures establish localized refinement, grid coverage,
  disjointness and error behavior. Lane layouts from one to four lanes and one
  to three threads per engine reproduce prefixes, estimates and cores exactly.
- Serial, token-range and external/Rayon callbacks agree. A separate process
  verifies that serial and lane construction leave the global backend unopened.
- The 48-bit untruncated fixture remains a compact TT.
- The complex 2D conversion/Fourier example has input residual 1.332e-15 and
  Fourier residual 1.041e-14 against a sixteen-point direct oracle. Rustdoc,
  guide, boundary and panic checks passed. One pre-existing redundant-link
  rustdoc warning remains in tensorbackend's backend.rs.
- SVD compression also reuses constructor validation before its single-site
  shortcut, covering invalid boundaries or adjacent bonds after site replacement.
- MPI and automatic thread sizing remain out of scope.

## Review round (#1307)

- macOS (Accelerate) failed the `f64::MAX` overflow case: whether
  `sum_i w_i * MAX` overflows in the validation GEMM depends on the provider's
  summation order (faer overflowed, Accelerate did not). The fixture now keeps
  node values at `MAX / 1024` and puts `-MAX` only at the test points, so the
  residual is about `(1 + 2^-10) MAX` under every provider.
- Adding the Frobenius truncation bound to the residual compared a whole-patch
  quantity, growing like the square root of the patch's grid size, with a
  per-value tolerance. With default options a constant 2D function at 30 bits
  exhausted the 4096-patch budget. With acceptance on the residual alone, a
  constant and a smooth 2D function at 30 and 40 bits give one patch each.
- `parallel` now enables only the provider-neutral lane crates. It composes with
  the platform default (Accelerate on macOS) and with system BLAS; the previous
  forced faer selection made `features = ["parallel"]` fail to build on macOS.
  Bit-identical lane layouts are asserted on faer, where each lane's kernels
  stay within its thread budget.
- The per-axis validation stencil is tracked in #1310.

The durable contract is in [the design](../design/patched-interpolative-qtt.md).
No throughput or scaling claim is made; comparisons use identical accuracy
settings. A source-blind documentation usability audit remains optional.
