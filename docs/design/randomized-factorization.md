# Randomized factorization

Issue [#1108](https://github.com/tensor4all/tensor4all-rs/issues/1108) places
index-aware randomized decompositions in `tensor4all-core`. They compose the
existing backend's matrix products, thin QR and small SVD; no direct faer
dependency or second linear algebra implementation is introduced.
The supported scalar types are `f64` and `Complex64`; `f32` and `Complex32`
(c32) are explicitly unsupported.

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

## Expected error used by the numerical regression

The following exact-arithmetic bound is derived from the Gaussian argument
in [Halko, Martinsson and Tropp (2011), Theorem 9.1 and Proposition 10.2](https://tropp.caltech.edu/papers/HMT11-Finding-Structure.pdf).
It includes the final truncation of the projected SVD to rank `k`.
It is an expectation over real Gaussian sketches, not an error guarantee for
an individual seed or a public truncation policy.

Let `ell = k + p`, choose the intermediate rank `r = k + floor(p / 2)`,
and let `p' = ell - r > 1`. Assume `ell <= min(m, n)` and positive leading
`r` singular values. Write `T_a(t) = sum_{j > a} sigma_j^t` and
`H_r(q) = sum_{j <= r} sigma_j^(-4q)`. For `q` power iterations and the
returned rank-`k` approximation `Ahat_k`,

```text
E ||A - Ahat_k||_F^2
    <= T_k(2) + T_r(2) + T_r(4q + 2) H_r(q) / (p' - 1).
```

To obtain the range term, partition the Gaussian sketch in the right singular
basis at rank `r` into `Omega_1` and `Omega_2`. The sampled range is that of
`Y = A (A† A)^q Omega`. Multiplying `Y` by
`Omega_1^+ Sigma_1^(-2q)` (where `+` is the Moore–Penrose inverse) constructs
an approximation with an exact leading `r x r` singular-coordinate block and
lower-left error block
`F = Sigma_2^(2q+1) Omega_2 Omega_1^+ Sigma_1^(-2q)`.
The squared range residual is at most `T_r(2) + ||F||_F^2`.
Gaussian independence and
`E (Omega_1 Omega_1^T)^(-1) = I / (p' - 1)` give the displayed range term,
with `T` the ordinary transpose for the real Gaussian sketch.

Finally, orthogonality splits the squared residual into the range residual
plus the discarded projected singular values. Since orthogonal projection
cannot increase any singular value, that last squared tail is at most
`T_k(2)`. This proves the bound including rank truncation.

For the original `24 x 24` fixture `sigma_j = 0.6^(j-1)`, `k=5`, `p=8`,
`q=1`, this gives an RMS error at most `1.0086405662` times the optimal
rank-5 Frobenius residual. The existing `1.02` fixed-seed regression threshold
is retained above that expected RMS reference; it is not presented as a
per-draw theoretical bound. No existing test tolerance is relaxed.

The rank/error regression additionally uses randomly rotated fast (`0.6`)
and slow (`0.95`) geometric spectra, ranks `2, 5, 9`, both canonical directions,
and the same eight seeds at each rank. It checks the dense SVD residual against
the known spectral tail, decreasing RSVD residuals as rank grows, and the
ensemble mean squared residual against the expression above. This finite
ensemble is a deterministic regression, not a proof of a probabilistic claim.
