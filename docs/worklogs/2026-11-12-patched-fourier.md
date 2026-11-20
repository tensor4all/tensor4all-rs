# Patched Fourier transforms on selected indices

## Decisions

- Implement issue #1275 revision 2 in quanticstransform behind an optional
  `partitioned` feature, returning the scheduler’s lazy result directly.
- Keep operator bit-node names independent of state-node names at the owning
  reconstruction boundary. Compose per-axis operators on a separated chain and
  use exact restructuring to the state-path order.
- Reconstruction error excludes operator construction error and floating-point
  roundoff; test these separately. Exact restructuring can increase bonds beyond
  the cap used for each one-axis Fourier operator.

## Verification conclusions and constraints

- Initial support is limited to selections lying on one path; branching
  selections and parallelization remain out of scope.
- Initial validation covers 1D/2D; three axes and unequal axis lengths are deferred.
- Issue #1276 owns patched interpolation and is assigned separately. Its
  remaining-bit TTs need prefix one-hot sites restored, fused bit levels split,
  and full shared index identities/topology established before constructing
  subdomains. `insert_onehot_site_in_treetn_chain` supplies the fixed-bit insertion
  at the existing simplett bridge; merely attaching a prefix projector fails
  `SubDomainTreeTN` validation when those indices are missing.
- The optional transform-to-partition dependency would form a cycle through
  partitionedtreetn's old QFT test dev-dependency. Move that unchanged integration
  suite to quanticstransform behind `partitioned`, rather than weakening the
  crate-boundary gate. The root book-test dependency enables the feature in
  workspace CI; standalone feature validation runs it explicitly.
- The first unbounded exact round-trip fixture exhausted resources. The bounded
  fixture uses the existing rank-goal scheduler with tight tolerance and passes;
  no existing test tolerance was changed.
- Transform crate: all 225 ordinary tests and 49 doctests passed, including
  all 39 relocated QFT tests. Partitioned crate: 124 unit tests and 229 integration
  tests passed (13 pre-existing ignored tests), as did its 129 doctests. New assertions use a fixed `2e-10` roundoff allowance; no existing
  tolerance was changed.
- All 12 new tests failed under at least one of four injected mutations:
  removing normalization, misclassifying invalid layout errors, erasing the
  reported error bound, or accepting caller-provided coordinate groups. Mutations
  were restored before final checks.
- The 2D grouped/interleaved operator fixtures retained maximum bond dimensions
  2 and 8 respectively. These are representation observations, not performance
  measurements or general bounds.
- Default features, optional provider forwarding and the explicit faer feature
  configuration were verified. Clippy, public error docs, crate boundaries,
  repository-rules dry run and the complete mdBook harness passed. The mdBook
  harness used the shared debug target's dependency path explicitly.
- No full workspace suite, alternate BLAS/CUDA execution, MPI, publication,
  or performance validation is claimed.

## Review follow-up: large validation (PR #1277)

- Oracle: a shifted periodic Gaussian defined by a finite Fourier series (12
  modes), so its normalized discrete DFT is known exactly; no continuous-FT
  quadrature error enters. 35 seeded sample points per case; inputs are built by
  interpolative QTT and never materialized on the full grid.
- Release build, `RAYON_NUM_THREADS=1`, 8 GiB address-space limit, 4 dyadic
  input patches per 2D case (2 in 1D), `rtol = 1e-6`, rank goal 8, output depth 1:

  | case | input bond | operator bond | operator s | scheduler s | sample error | reconstruction bound | peak RSS |
  |---|---|---|---|---|---|---|---|
  | R=10 1D | 8 | 12 | 0.0016 | 0.032 | 4.1e-14 | 7.9e-9 | 41 MB |
  | R=10 2D grouped | 8 | 12 | 0.0028 | 0.30 | 8.7e-14 | 7.7e-9 | 70 MB |
  | R=20 1D | 8 | 12 | 0.0034 | 0.25 | 2.2e-13 | 3.5e-10 | 92 MB |
  | R=20 2D grouped | 8 | 12 | 0.0068 | 1.53 | 4.2e-13 | 6.9e-9 | 124 MB |
  | R=10 2D interleaved | 50 | 144 | 0.014 | OOM | — | — | >14 GiB |
  | R=20 2D interleaved | 49 | 144 | 0.059 | OOM | — | — | >8 GiB |

- Interleaved operator construction originally composed the axes in grouped
  order and restructured by site swaps. Exact swaps at R=10 failed above 4 GiB
  (3.6 GiB RSS before the failed allocation); capping swap ranks at 16 kept
  memory bounded but gave sample error 1.1e-4. An exact R=6 dense check showed
  the interleaved Schmidt ranks are the products of the per-axis ranks (8 per
  axis, 59 interleaved), so no small cap can be accurate.
- Monotone layouts (each axis increasing or decreasing along the chain) are now
  built directly: each axis's core is embedded with the other axes' bonds passed
  through. This is exact without factorization; R=10 interleaved takes 0.02 s and
  47 MB, R=20 0.07 s and 99 MB, with bond 144 = 12 x 12. The restructure-options
  API added during the review was removed again because it no longer bounds
  anything for these layouts. Non-monotone layouts keep the exact swap route.
- The remaining interleaved limit is the scheduler's exact reference
  application (`ReconstructionTarget::from_subset_operator`), whose bonds reach
  input bond x operator bond. It is tracked in #1279; the heavy suite covers
  1D and 2D grouped only.
