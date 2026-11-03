# Randomized factorization diagnostics

The current runs contain 72 decomposition samples and 96 Fit/Zipup samples;
all passed their predeclared accuracy gates. Actual worker budgets were read
back before timing. These are internal algorithm comparisons on the same
library stack, with an independent faer Lanczos reference. They are shared-host
diagnostics, not a guarded performance promotion or a parallel scaling claim.
See [local validation](validation.md) for regression checks and limitations.

Every timed result is checked against a whole-result oracle outside its timer.
The library default is one power iteration. The contraction comparison uses
two; its one-iteration attempt failed the error gate and is retained.

Decomposition source: `46d878a359e3a5f85f1673a3ecd9c9a3b65055c4`; [manifest](decomposition-manifest.json).
The [protocol](protocol.md) records shapes, seeds, thread settings, repetitions
and error gates. RSVD uses oversampling 8 and one power iteration.

## Decomposition, 1 thread(s)

Median milliseconds, three samples per cell.

| Strategy | 128x128, rank 32 | 256x256, rank 64 | 512x512, rank 128 | 512x512, rank 32 |
|---|---:|---:|---:|---:|
| Core deterministic SVD | 1.530 | 8.248 | 55.604 | 56.539 |
| Core RSVD | 0.719 | 2.666 | 14.861 | 3.347 |
| faer Lanczos reference | 1.152 | 7.488 | 51.553 | 5.202 |

## Decomposition, 8 thread(s)

Median milliseconds, three samples per cell.

| Strategy | 128x128, rank 32 | 256x256, rank 64 | 512x512, rank 128 | 512x512, rank 32 |
|---|---:|---:|---:|---:|
| Core deterministic SVD | 1.570 | 8.968 | 55.262 | 54.867 |
| Core RSVD | 0.689 | 2.307 | 14.052 | 2.595 |
| faer Lanczos reference | 1.311 | 6.192 | 53.794 | 2.917 |

## Decomposition error and GEMM/QR attribution

Errors are full Frobenius residuals divided by the input norm. The known
geometric spectral tail supplies the optimal rank reference, agreeing with
deterministic SVD. Projected singular values are not used as an error estimate.

| Threads | chi | d | RSVD relative error | RSVD / optimal error | GEMM+QR / RSVD total |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 2 | 0.034358 | 1.000616 | 56.3% |
| 1 | 64 | 2 | 0.0011807629 | 1.001480 | 61.5% |
| 1 | 128 | 2 | 1.3927307e-06 | 1.001904 | 68.4% |
| 1 | 32 | 4 | 0.03454201 | 1.005975 | 78.0% |
| 8 | 32 | 2 | 0.034358 | 1.000616 | 50.9% |
| 8 | 64 | 2 | 0.0011807629 | 1.001480 | 54.9% |
| 8 | 128 | 2 | 1.3927307e-06 | 1.001904 | 54.2% |
| 8 | 32 | 4 | 0.03454201 | 1.005975 | 66.8% |

## Fit and Zipup, two power iterations

Contraction source: `5c7369e4685c216fe7504b556412f6a286f60ce6`; [manifest](contraction-manifest.json).
The [contraction protocol](contraction-protocol.md) uses small bounded exact
dense oracles. Random MPO residuals can be large even at matched accuracy.
These timings compare two strategies in the same library; no winner is
promoted from these shared-host diagnostics.

| Threads | Method | chi | d | Sites | SVD ms | RSVD ms | SVD relative error | RSVD relative error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | zipup | 16 | 2 | 6 | 4.475 | 4.760 | 0.700950 | 0.700030 |
| 1 | fit | 16 | 2 | 6 | 11.392 | 14.355 | 0.434145 | 0.434558 |
| 1 | zipup | 32 | 2 | 6 | 8.859 | 9.711 | 0.637568 | 0.658724 |
| 1 | fit | 32 | 2 | 6 | 32.715 | 38.872 | 0.265329 | 0.266071 |
| 1 | zipup | 16 | 4 | 4 | 28.617 | 20.888 | 0.828146 | 0.834725 |
| 1 | fit | 16 | 4 | 4 | 134.779 | 97.188 | 0.767106 | 0.773278 |
| 1 | zipup | 32 | 4 | 4 | 47.388 | 30.544 | 0.828230 | 0.841376 |
| 1 | fit | 32 | 4 | 4 | 517.039 | 472.650 | 0.720650 | 0.732088 |
| 8 | zipup | 16 | 2 | 6 | 5.802 | 9.487 | 0.700950 | 0.700030 |
| 8 | fit | 16 | 2 | 6 | 25.122 | 32.374 | 0.434145 | 0.434476 |
| 8 | zipup | 32 | 2 | 6 | 10.448 | 17.676 | 0.637568 | 0.658724 |
| 8 | fit | 32 | 2 | 6 | 60.404 | 73.639 | 0.265329 | 0.266071 |
| 8 | zipup | 16 | 4 | 4 | 30.314 | 23.474 | 0.828146 | 0.834725 |
| 8 | fit | 16 | 4 | 4 | 99.083 | 61.155 | 0.767106 | 0.773278 |
| 8 | zipup | 32 | 4 | 4 | 50.163 | 34.814 | 0.828230 | 0.841376 |
| 8 | fit | 32 | 4 | 4 | 245.423 | 202.500 | 0.720650 | 0.732088 |

The d=2 cases are slower with this RSVD configuration; a faster local SVD
does not establish an application speedup. The d=4 cases benefit on this grid.
The GEMM/QR share gives a quantitative reference before reconsidering placement
in tenferro-linalg, not evidence that moving the composite would improve it.

Retained earlier runs: [unsupported 64x64 Lanczos case](failed-lanczos-64/manifest.json),
[supported grid before budget readback](prior-without-budget-readback/README.md),
and [one-iteration contraction accuracy failure](contraction-power1/manifest.json).
