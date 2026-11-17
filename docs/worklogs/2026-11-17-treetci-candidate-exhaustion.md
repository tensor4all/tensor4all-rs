# TreeTCI candidate exhaustion above rank one

## Decisions

- Generalize the unmeasured-bond diagnostic to every rank at dimension-one
  endpoints, retaining the existing singleton check everywhere. A unit
  endpoint has no physical variation to probe beyond neighboring pivots;
  exhausting its candidates forces zero error even if both full subtrees
  admit a larger rank. Reaching the full configuration count on either side
  remains valid.
- Keep candidate generation and pivot retention unchanged. Growing through a
  dimension-one chain site would require a separate candidate-generation
  strategy; retaining arbitrary pivot history does not guarantee that the
  missing directions will be found. Reporting `DegenerateCandidateSet` is the
  explicit alternative accepted by issue #1352.
- Physical endpoints retain the ordinary sampled criterion. Exhausting their
  local expansion can be legitimate, as the separable-window regressions of
  #1159 demonstrate; a blanket exhaustion rule would misclassify them.
- Keep rank-cap precedence and the existing three-iteration stopping window.
  The engine adapter already maps the diagnostic to `ErrorNotMeasured`.
  See the [stopping contract](../design/treetci-termination.md).

## Verification conclusions and constraints

- The unmodified Linux reproducer returns `Converged` with a maximum error of
  0.1539213025 for seed 4. Environment: Ubuntu 22.04.5 under WSL2, x86_64,
  rustc 1.98.1, default faer/native CPU provider. `Cargo.lock` is byte-identical
  to the issue's pinned `ffc229ad` (tenferro `32a4e98c`, hataori `9d655157`);
  relevant TreeTCI/core/TreeTN/backend sources are unchanged at the base.
- [shinaoka's macOS result](https://github.com/tensor4all/tensor4all-rs/issues/1352#issuecomment-6938274547)
  reaches rank 16 for seed 4. Tests therefore permit an accurate converged run
  as well as an unmeasured stop instead of fixing platform-dependent ranks.
- Regression coverage uses the issue's small Gaussian chain, real and complex
  scalars, and global search enabled and disabled. Each result is materialized
  once and compared against all 256 reference entries. Cached global-search
  readout is compared with that dense contraction around the unit site.
- Edge-level coverage distinguishes exhausted rank-two candidates, a measured
  lower-rank residual, and a legitimate full-rank bipartition in both edge
  orientations. Existing rank-one junction tests remain relevant.
- This change qualifies a stopping diagnostic; it does not reduce the true
  approximation error or turn sampled convergence into a global error bound.
