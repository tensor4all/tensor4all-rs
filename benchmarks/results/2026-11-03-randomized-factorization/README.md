# Randomized decomposition diagnostics

All 72 timed decompositions passed the declared accuracy gates. These are
internal decomposition strategies on the same library stack, with a standalone
faer Lanczos reference. Medians of three samples; milliseconds. Shared-host
diagnostics do not establish a general performance or parallel scaling claim.

Measured implementation: `b3bf3d6d8b3ba98232edb164189cfe9cbfa06031`. The [manifest](decomposition-manifest.json)
records binary hashes, provider versions, thread settings, affinity and load.
The [protocol](protocol.md) and [invalid first attempt](failed-lanczos-64/manifest.json)
remain part of the evidence.

## 1 thread(s)

| Strategy | 128x128, rank 32 | 256x256, rank 64 | 512x512, rank 128 | 512x512, rank 32 |
|---|---:|---:|---:|---:|
| Core deterministic SVD | 1.707 | 8.251 | 53.328 | 55.264 |
| Core RSVD | 0.710 | 2.631 | 14.748 | 3.296 |
| faer Lanczos reference | 1.114 | 7.058 | 48.305 | 5.063 |

## 8 thread(s)

| Strategy | 128x128, rank 32 | 256x256, rank 64 | 512x512, rank 128 | 512x512, rank 32 |
|---|---:|---:|---:|---:|
| Core deterministic SVD | 1.538 | 8.500 | 54.044 | 53.684 |
| Core RSVD | 0.718 | 3.128 | 11.139 | 3.220 |
| faer Lanczos reference | 1.263 | 6.575 | 49.991 | 3.132 |

## Error and GEMM/QR attribution

Every error is a full Frobenius reconstruction residual divided by the input
norm. The known spectral tail provides the rank-optimal reference; it agrees
with deterministic SVD. No error estimate from the projected SVD is used.

| Threads | chi | d | RSVD relative error | RSVD / optimal error | GEMM+QR / RSVD total |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 2 | 0.034358 | 1.000616 | 54.7% |
| 1 | 64 | 2 | 0.0011807629 | 1.001480 | 61.8% |
| 1 | 128 | 2 | 1.3927307e-06 | 1.001904 | 69.0% |
| 1 | 32 | 4 | 0.03454201 | 1.005975 | 78.6% |
| 8 | 32 | 2 | 0.034358 | 1.000616 | 51.0% |
| 8 | 64 | 2 | 0.0011807629 | 1.001480 | 52.8% |
| 8 | 128 | 2 | 1.3927307e-06 | 1.001904 | 55.5% |
| 8 | 32 | 4 | 0.03454201 | 1.005975 | 67.6% |

The GEMM/QR share is a quantitative reference for revisiting placement in
tenferro-linalg, rather than evidence that moving the composite would help.
The fixture has a prescribed geometric spectrum; production MPO spectra and
contraction/environment costs require their own measurements.
