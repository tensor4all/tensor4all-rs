# Matrix transpose paired results

All final gates pass for source `f251a533074aa8a6b1f6f7a06deb62dc5cb2e6f1` against `54d162dad9b6280150ffb71c4629b966c219c42e`. Nine complete paired suites, CPU 2, all provider/Rayon threads 1, release/default faer, AMD Ryzen 9 6900HX under WSL2. All dtype bit-pattern and benchmark value checks pass. Affinity is valid and steal fraction is zero. Binary checksums and host observations are in the [raw record](2026-10-07-matrix-transpose.json).

Large-square (256 through 1024) transpose geometric mean time ratio: **0.5314** (about 46.9% less time). This describes the recorded helpers and public rrLU orientation conversions, not complete factorization or a workspace-wide speedup.

| Operation | Dtype | Shape | Baseline µs | Candidate µs | Paired ratio (95% CI) |
|---|---|---|---:|---:|---|
| micro | f32 | 1×4096 | 3.687 | 0.157 | 0.042 (0.041–0.047) |
| micro | f32 | 8×8 | 0.063 | 0.064 | 1.016 (1.000–1.111) |
| micro | f32 | 16×16 | 0.192 | 0.192 | 1.000 (0.989–1.016) |
| micro | f32 | 32×32 | 0.724 | 0.720 | 0.993 (0.968–1.004) |
| micro | f32 | 64×64 | 2.818 | 2.823 | 0.994 (0.988–1.028) |
| micro | f32 | 128×128 | 15.727 | 11.426 | 0.720 (0.711–0.740) |
| micro | f32 | 256×256 | 109.299 | 51.993 | 0.472 (0.454–0.483) |
| micro | f32 | 512×512 | 669.165 | 251.460 | 0.374 (0.373–0.380) |
| micro | f32 | 1024×1024 | 3909.912 | 2696.492 | 0.678 (0.664–0.700) |
| micro | f32 | 1024×64 | 55.285 | 39.069 | 0.708 (0.692–0.732) |
| micro | f32 | 64×1024 | 151.984 | 46.992 | 0.307 (0.291–0.317) |
| micro | f32 | 4096×32 | 119.875 | 90.529 | 0.754 (0.720–0.814) |
| micro | f32 | 32×4096 | 327.460 | 94.108 | 0.287 (0.278–0.300) |
| micro | f64 | 1×4096 | 3.707 | 0.474 | 0.127 (0.126–0.132) |
| micro | f64 | 8×8 | 0.065 | 0.067 | 1.031 (1.015–1.062) |
| micro | f64 | 16×16 | 0.220 | 0.226 | 1.036 (1.005–1.037) |
| micro | f64 | 32×32 | 0.756 | 0.765 | 1.014 (1.000–1.021) |
| micro | f64 | 64×64 | 3.391 | 3.417 | 1.007 (0.974–1.038) |
| micro | f64 | 128×128 | 27.714 | 9.651 | 0.345 (0.338–0.394) |
| micro | f64 | 256×256 | 143.256 | 50.542 | 0.352 (0.345–0.369) |
| micro | f64 | 512×512 | 840.015 | 494.552 | 0.585 (0.574–0.595) |
| micro | f64 | 1024×1024 | 4814.679 | 3683.612 | 0.751 (0.730–0.780) |
| micro | f64 | 1024×64 | 88.038 | 70.960 | 0.825 (0.805–0.853) |
| micro | f64 | 64×1024 | 155.100 | 45.546 | 0.297 (0.280–0.299) |
| micro | f64 | 4096×32 | 153.961 | 105.178 | 0.685 (0.680–0.699) |
| micro | f64 | 32×4096 | 344.305 | 86.600 | 0.250 (0.242–0.261) |
| micro | c32 | 1×4096 | 3.752 | 0.488 | 0.131 (0.123–0.143) |
| micro | c32 | 8×8 | 0.062 | 0.062 | 0.984 (0.968–1.033) |
| micro | c32 | 16×16 | 0.196 | 0.201 | 1.020 (1.005–1.031) |
| micro | c32 | 32×32 | 0.767 | 0.778 | 1.001 (0.986–1.030) |
| micro | c32 | 64×64 | 3.375 | 3.402 | 1.014 (0.984–1.021) |
| micro | c32 | 128×128 | 27.122 | 10.066 | 0.365 (0.350–0.393) |
| micro | c32 | 256×256 | 145.068 | 51.804 | 0.357 (0.341–0.373) |
| micro | c32 | 512×512 | 803.567 | 450.552 | 0.563 (0.556–0.574) |
| micro | c32 | 1024×1024 | 4442.737 | 3281.337 | 0.740 (0.714–0.751) |
| micro | c32 | 1024×64 | 87.804 | 70.905 | 0.822 (0.797–0.864) |
| micro | c32 | 64×1024 | 154.892 | 45.867 | 0.301 (0.284–0.302) |
| micro | c32 | 4096×32 | 154.823 | 104.730 | 0.677 (0.669–0.691) |
| micro | c32 | 32×4096 | 332.291 | 88.056 | 0.264 (0.246–0.274) |
| micro | c64 | 1×4096 | 4.063 | 0.995 | 0.246 (0.234–0.250) |
| micro | c64 | 8×8 | 0.067 | 0.068 | 1.015 (0.971–1.046) |
| micro | c64 | 16×16 | 0.239 | 0.241 | 1.008 (1.000–1.025) |
| micro | c64 | 32×32 | 0.822 | 0.831 | 1.010 (0.995–1.019) |
| micro | c64 | 64×64 | 6.478 | 6.414 | 0.989 (0.970–1.019) |
| micro | c64 | 128×128 | 35.338 | 11.754 | 0.332 (0.320–0.350) |
| micro | c64 | 256×256 | 206.235 | 99.797 | 0.492 (0.472–0.495) |
| micro | c64 | 512×512 | 919.713 | 530.879 | 0.572 (0.546–0.603) |
| micro | c64 | 1024×1024 | 8262.817 | 5377.781 | 0.653 (0.648–0.664) |
| micro | c64 | 1024×64 | 131.202 | 113.749 | 0.873 (0.847–0.894) |
| micro | c64 | 64×1024 | 188.585 | 48.331 | 0.256 (0.240–0.269) |
| micro | c64 | 4096×32 | 258.429 | 260.584 | 1.011 (0.998–1.484) |
| micro | c64 | 32×4096 | 347.709 | 96.562 | 0.282 (0.270–0.291) |
| rrlu_phase | f64 | 1024×8 | 6.753 | 4.645 | 0.693 (0.678–0.702) |
| rrlu | f64 | 1024×8 | 6.960 | 4.892 | 0.698 (0.689–0.721) |
| rrlu_phase | f64 | 1024×32 | 29.549 | 20.071 | 0.673 (0.665–0.725) |
| rrlu | f64 | 1024×32 | 29.865 | 20.783 | 0.694 (0.676–0.719) |
| rrlu_phase | f64 | 4096×32 | 154.644 | 104.743 | 0.680 (0.664–0.709) |
| rrlu | f64 | 4096×32 | 156.262 | 104.134 | 0.667 (0.661–0.677) |
| rrlu_phase | f64 | 4096×128 | 1025.190 | 627.286 | 0.598 (0.540–0.756) |
| rrlu | f64 | 4096×128 | 1018.564 | 536.377 | 0.528 (0.507–0.581) |

The [protocol](2026-10-07-matrix-transpose-protocol.md) records tuning, thresholds, complete-case statistics and validity gates. The [need-gate CSV](2026-10-07-matrix-transpose-need.csv) and [initial tile-tuning CSV](2026-10-07-matrix-transpose-tuning.csv) are exploratory. [Rejected experiments](2026-10-07-matrix-transpose-rejected.jsonl) retain all eight complete suites and their failed/inconclusive gates. [Version 9](2026-10-07-matrix-transpose-confirmation9.json) passes all gates; the final run repeats the complete experiment against the committed formatted source.

Copy the shared benchmark source and thin example into the baseline worktree (the baseline does not yet contain this harness). Reproduce the size ladder and public conversion fixture with `cargo run --release -p tensor4all-core --example benchmark_matrix_transpose` in baseline and candidate worktrees. Preserve the fixed thread counts and process affinity from the protocol; alternate complete suites nine times. The dependency-free [pair runner](../rust/run_matrix_transpose_pairs.py) accepts `--baseline`, `--candidate`, and `--output` paths plus `--cpu 2`, fixes all provider/Rayon thread counts, and records every case and gate. `--tune` runs the separate original exploratory tile variants.

The 2 MiB tall threshold and tile dimensions are measured CPU policies, not portable optimums. Tiny nanosecond-scale cases establish the declared non-regression gate; they are not promoted as throughput claims. No GPU or system-BLAS claim is made.
