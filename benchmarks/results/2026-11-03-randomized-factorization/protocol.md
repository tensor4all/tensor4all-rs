# RSVD diagnostic comparison protocol (revision 2)

This is an internal comparison of decomposition strategies, not a guarded
baseline/candidate promotion or a parallel speedup claim. Shared-host activity
is recorded but not fully controlled. The implementation revision will be
resolved from git into the run manifest before either measured process starts.

- Release build, workspace lockfile, default Linux faer provider.
- One process per thread budget: 1 thread on CPU 8, and 8 threads on CPUs 8-15.
  Observer commands use CPUs 0-3. No concurrent local build or test during timing.
- Shapes are square local two-site matrices with `n = chi * d^2`, for
  `(chi,d) = (32,2), (64,2), (128,2), (32,4)`. Fixed output rank is `chi`.
- Every matrix is the same seeded Gaussian-QR rotation of the known spectrum
  `sigma[j] = 0.9^j`; seed 1108. The standalone Lanczos driver receives the
  identical column-major values via a binary file, not a regenerated fixture.
- Compare deterministic SVD, RSVD (oversampling 8, one power iteration), and
  faer's existing deterministic Lanczos partial SVD (tolerance `1e-10`).
- One second of time-based priming, one discarded decomposition warmup and
  three timed samples per strategy/cell. Keep all samples; report medians.
- Exclude fixture construction, input file I/O, process startup and residual
  checks. Include decomposition output and workspace allocation. Lanczos's
  fixed start vector is prepared outside the timer; RSVD Gaussian sampling is
  inside its timer, as part of its composite.
- Every timed sample checks the full Frobenius reconstruction residual against
  the known spectral tail. RSVD must be within 5% of the rank-optimal error;
  SVD and Lanczos must be within 0.1%. A failed accuracy gate invalidates the
  comparison; do not relax it or selectively retry cells.
- Report `(gemm + qr) / total` from `RsvdResult::timings` in the same CSV as
  the SVD/Lanczos results. These timings include frontend dispatch overhead.
- The standalone Lanczos executable links the exact faer/dyn-stack artifacts
  identified in Cargo's JSON build output. No tensor4all library gains a
  direct faer dependency. It is a reference arm, not a production dispatch.
- Neither the synthetic spectrum nor shared-host timings establishes the best
  default for production MPOs. Environment projection and adaptive error-based
  RSVD rank selection remain outside this change.


The first attempted comparison is retained under [failed-lanczos-64](failed-lanczos-64/manifest.json).
Its 64x64 cell reaches faer's public driver's `max_dim < min(m,n)` assertion:
`partial_svd` clamps the default maximum to 64, but `partial_svd_imp` requires
it to be strictly smaller. That attempt is invalid and supplies no performance
conclusion. Revision 2 doubles the entire chi grid, before measurement, to fit
the existing driver's supported projection dimensions. It repeats every method
and thread budget with the same error gates; no favorable cell is substituted.

Both executables check their actual backend/Par worker budget before timing;
the core harness also prints each worker CPU mask. A clamped budget is an error.
