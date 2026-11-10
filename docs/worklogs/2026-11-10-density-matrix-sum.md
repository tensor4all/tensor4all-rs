# Density-matrix sums without summed input bonds

## Decisions

- Reuse sum-fit pair-overlap environments and the canonicalization traversal;
  retain subtree density eigenvectors in one pass toward the center. This
  supplies the cross terms missing from a sum of individual densities and
  avoids both an exact direct sum and redundant canonicalization sweeps.
- Expose the discarded density weight with the network so a binding cap is
  visible in the absolute error estimate. The [design record](../design/density-matrix-sum.md)
  distinguishes the exact-arithmetic bound from density-matrix roundoff.
- Keep native eigenvectors in their original scalar kind. The narrow-type
  regression exposed a core eigenvalue readback bug, fixed at that shared
  helper rather than widening algorithm tensors. Shared summability validation
  now also rejects equal index identities with different dimensions before
  either density or exact-sum tensor construction.

## Verification conclusions and constraints

- The six density tests cover interference, known discarded weights/caps,
  complex branching trees with site-free nodes, native scalar kinds, exact
  one-shot rank comparisons and a 65-site guard. The native eigenpair residual
  regression and all seven new tests have observed failing mutations; every
  injection was reverted. The shared dimension mismatch was also observed
  failing before its correction.
- A broader run passed 2,420 core/TreeTN tests before the shared dimension
  correction; the final TreeTN run passed all 1,135 tests. Three affected
  rustdoc examples pass. Independent source review found no remaining blocker.
- Clippy completed with the error/panic documentation gates enabled and no
  warnings in changed code. Strict warning denial is blocked by existing
  boolean-simplification warnings elsewhere in the two crates. Full workspace
  validation and alternate-provider/hardware checks remain hosted-CI work.
- Rustdoc builds with existing unrelated link warnings; the new API's links
  and asserted examples validate. The guide change is prose only and does not
  change any existing guide snippet or its API.
- The performance comparison used all smooth, random and mixed families
  of #1107 at `(R, N) = (8, 8), (12, 16), (12, 64)`, three fresh processes per
  method and case, alternating method order. Compare median operation time
  and median process peak RSS; preserve every result. Both methods must
  satisfy the requested error budget. No speedup threshold is required:
  this is a new selectable method, with costs reported even when slower.
- The benchmark separates target construction and dense-reference verification
  from operation time. RSS includes construction and verification. The
  reference is deliberately limited to at most 16 binary sites. Release
  builds and all CPU-library thread counts are pinned to one.
- Host: Apple M3 Pro, default macOS Accelerate provider. macOS CPU affinity
  and exclusive host access are unavailable, so these are diagnostic timing
  and RSS observations, not an isolated-host performance guarantee. Load
  averages and all 54 process samples are retained in the
  [experiment record](../../benchmarks/results/2026-11-10-density-sum.md).
  This chat ran no concurrent build or test while measuring.
- At `R=12, N=64`, density summation reached the same ranks and cutoff-scale
  errors with 53–55% lower process peak RSS, but took 6.5–8.9 times longer in
  this diagnostic run. At `N=16` random input, density used slightly more
  memory. No universal performance or thousand-term scalability claim follows.
- The review also identified a pre-existing dimension-validation concern in
  `IdxTensor::replace_indices`: the supplied old index's dimension is not
  checked against the matched axis. The shared sum validation prevents this
  route here; a separate review of replacement APIs is recommended.
