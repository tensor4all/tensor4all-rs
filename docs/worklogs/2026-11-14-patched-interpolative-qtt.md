# Patched interpolative QTT construction

## Decisions

- Follow the prefix-bearing patch representation of #1276: a bare vector of
  local trains loses the original boxes. The construction remains independent
  of TreeTN; conversion belongs in the executable tutorial.
- Share multiscale core assembly and SVD sweeps. Explicit batch sampling and
  session execution avoid a duplicated interpolation algorithm or implicit
  process-global backend use.
- Include the SVD discarded-weight bound in acceptance and use the rank cap to
  trigger splitting. This prevents a compressed but inaccurate patch from
  appearing converged merely because its rank was capped.
- Preserve parent endpoints when splitting dyadic boxes. Recomputing the upper
  endpoint as two additions of the half length can change it by an ulp, leaving
  a gap or overlap in non-binary-exact physical domains.

## Verification conclusions and constraints

- Real and complex 1D/2D fixtures establish localized refinement, grid coverage,
  disjointness and error behavior. Lane layouts from one to four lanes and one
  to three threads per engine reproduce prefixes, estimates and cores exactly.
- Serial, token-range and external/Rayon callbacks agree. A separate process
  verifies that serial and lane construction leave the global backend unopened.
- The 48-bit untruncated fixture remains a compact TT. The extensive Frobenius
  compression bound can cause extra refinement on very large grids; this is a
  documented conservative tradeoff, not a throughput claim.
- The complex 2D conversion/Fourier example has input residual 1.332e-15 and
  Fourier residual 1.041e-14 against a sixteen-point direct oracle. Rustdoc,
  guide, boundary and panic checks passed. One pre-existing redundant-link
  rustdoc warning remains in tensorbackend's backend.rs.
- SVD compression also reuses constructor validation before its single-site
  shortcut, covering invalid boundaries or adjacent bonds after site replacement.
- Native/faer plus system BLAS is rejected at compile time. The Accelerate
  feature chain enables the same upstream exclusion; macOS execution was not
  available locally. MPI and automatic thread sizing remain out of scope.

The durable contract is in [the design](../design/patched-interpolative-qtt.md).
No throughput or scaling claim is made; comparisons use identical accuracy
settings. A source-blind documentation usability audit remains optional.
