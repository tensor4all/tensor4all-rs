# MatrixLUCI scalar trait coverage (#1300)

## Decisions

- Sealed `tensor4all_core::MatrixLuciScalar` and made
  `matrix_luci_factors_from_blocks_seeded` a required method, deleting the
  forwarding default body that only existed to keep the trait non-breaking for
  out-of-crate implementors.
- Why: the nightly coverage gate attributed the whole per-file deficit to that
  never-instantiated default (#1300). The trait dispatches into crate-internal
  kernels, so a foreign implementation would have to reimplement the rook/LUCI
  kernels itself; none exists in this workspace or through the C API. Sealing
  records that boundary, and the CHANGELOG lists it as a breaking change.
- Rejected: pinning the file at 74 in `coverage-thresholds.json` (PR #1301) —
  a threshold change needs explicit owner approval, and this repository fixes
  coverage drops in code. Also rejected: a test-only mock implementor, which
  would need the four public supertraits plus ~200 lines of scaffolding to keep
  a dead default alive.

## Verification conclusions and constraints

- No behaviour change for `f32`, `f64`, `Complex32`, `Complex64`: the four impls
  and the public `matrix_luci_factors_from_blocks_seeded` facade are untouched,
  and core's existing tests already cover every remaining line in the file.
- The deleted body was a forwarding call to
  `matrix_luci_factors_from_blocks`, which stays exercised by those same tests;
  no shared helper lost its only exerciser.
- Measured on this host with the workflow's pinned 1.99.1 toolchain and
  `CARGO_BUILD_JOBS=4`: `scripts/check-coverage.py` passes 304/304 files, and
  `crates/tensor4all-core/src/matrixluci/scalar.rs` moves from 38/51 (74.5%) to
  38/38 (100%). `book-tests` was excluded (no HDF5 dev package here, no
  passwordless sudo), so its `partitioned` feature was supplied explicitly to
  keep the same file set. The remaining unverified part is the hosted run.
