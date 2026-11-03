# Randomized factorization

Issue [#1108](https://github.com/tensor4all/tensor4all-rs/issues/1108) places
index-aware randomized decompositions in `tensor4all-core`. They compose the
existing backend's matrix products, thin QR and small SVD; no direct faer
dependency or second linear algebra implementation is introduced.

## Fixed-rank contract

For an index split with matrix dimensions `m` and `n`, the range finder uses
`ell = min(rank + oversampling, m, n)` Gaussian columns. It orthogonalizes both
sides of every power iteration. Its returned Q has `ell` orthonormal columns;
use zero oversampling when a projector must have exactly `rank` columns.
RSVD retains `min(rank, m, n)` singular triplets of the projected matrix.

The range finder and RSVD never consult global truncation defaults. Threshold
policies return `FactorizeError::UnsupportedTruncationPolicy` when selecting
RSVD through `FactorizeOptions` or contraction options. The discarded singular values of the projected matrix
omit the range-finding residual and therefore cannot certify a requested
full-input tolerance. Adaptive rank selection requires a separately validated
a posteriori error estimator.

## Randomness and execution

Low-level entries consume a caller's RNG directly. A seeded convenience
operation creates one ChaCha8Rng and advances it across all local splits and
sweeps. It does not draw a seed from a supplied RNG or reset the stream per
site. Tensor index identities remain fresh even when numerical samples repeat.

Execution routing follows the existing decomposition boundaries: global
eager, caller-owned execution context, and caller-held plain explicit session.
The latter refuses tracked operands instead of discarding differentiation
metadata. Gaussian sketches are constants and are constructed in the selected
execution route. Validation precedes RNG consumption, including on shortcuts.

## Measurement contract

The comparison harness includes deterministic full SVD and Lanczos partial SVD
on the same inputs and reports end-to-end RSVD time plus the GEMM/QR share.
This provides a quantitative reference before revisiting placement in
`tenferro-linalg`; a second tenferro consumer is the other revisit condition.
No implicit fallback to deterministic SVD is permitted for unsupported policies.

## Projection scope

The range finder exposes `Q` for explicit insertion of `Q Q†` on a selected
index grouping. R2 environment projection is not valid for the current
site-canonical Fit and remains tied to the future inverse-canonical parallel
Fit. Dense two-site RSVD does not eliminate materialization of that local
two-site tensor; no full-network dense materialization is introduced.
