# Matrix transpose experiment protocol

Baseline: `54d162dad9b6280150ffb71c4629b966c219c42e`. Candidate source and
benchmark commit IDs will be recorded with the results. Source:
`benchmarks/rust/benchmark_matrix_transpose.rs`; release profile, default faer
provider, AMD Ryzen 9 6900HX / Linux under WSL2, CPU affinity 2.
`RAYON_NUM_THREADS`, `BLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` are all 1.

Need gate before implementation: compare the two backend transpose phases with
complete public `RrLU::transpose` orientation conversion. The matrices are the
same unpermuted factors obtained through public APIs, setup excluded. Require
at least 5% of conversion time attributable to transpose in every case; this
does not establish its share of a complete TCI solve.

Tuning is a separate exploratory experiment: tiles 16/32/64 on f64 square
64/128/256/512/1024 and both orientations of 1024x64 and 4096x32. Choose the
lowest geometric mean relative to naive among large cases. Freeze tile and
fallback threshold before the confirmatory experiment. The confirmation uses
all 13 size-ladder shapes from the benchmark, all four float/complex kinds,
and all four public rrLU conversion cases. No case may be omitted.

Confirmation: 9 paired complete-suite invocations, alternating baseline-first
and candidate-first, fixed per-case repeat count from the benchmark. Statistic:
median candidate/baseline ratio per case, 95% bootstrap confidence intervals
(10,000 resamples, seed 738). Primary: large-square (>=256) transpose geometric
mean ratio <=0.85. Non-regression: every other helper and public conversion
case ratio <=1.05; correctness must be exact. Cases too short for that timing
resolution are inconclusive, never reported as a speedup.

Validity: all processes must complete, numerical checks must pass, /proc/stat
steal delta <1%, no measured process may migrate outside CPU 2, and median
unchanged 8x8 control ratio within [0.9,1.1] for each dtype. Record load and
available memory before/after. A failed validity gate makes the entire
experiment inconclusive; any rerun repeats the complete suite.

## Frozen tuning decision

The need gate measured transpose-phase / complete-orientation ratios of
0.9683, 0.9765, 0.9531, and 1.0277 (separately timed phases, so near-unity
ratios have measurement noise). This establishes the requested phase share
for orientation conversion, not for full factorization.

Exploratory large-case geometric means: tile16 0.7148, tile32 0.7309,
tile64 0.7702. Freeze tile16. The 64x64 case regressed with every tile,
while 128x128 and larger squares improved. Tall skinny cases also regressed,
while their wide transposes improved. Freeze dispatch to tiled only for more
than 4096 elements, at least 2 input rows, and input rows <= columns. Retain
the existing simple traversal for small/tall/single-row inputs. No tuning
results are counted as confirmatory evidence. Raw tuning and need-gate
records are retained with the final results.

## Correction after first confirmation

The first confirmation passed primary and validity gates but failed
non-regression for Complex64 single-row/tall cases and 32x32 complex cases.
Retain it as `2026-10-07-matrix-transpose-first.json` rather than omit those
cases. Correct the dispatch/code organization: clone the flat buffer for
single-axis inputs (transpose preserves their flat order), inline the small
wrapper/simple traversal, and keep the blocked kernel out of that hot-path
body. Retain tile16/4096/shape policy, primary metric, thresholds and complete
case list. Repeat the full nine-pair confirmation after that correction.

## Correction after second confirmation

The forced-inline correction worsened generated simple-path code and failed
unchanged 8x8 control validity as well as multiple non-regression cases. Retain
all results as `2026-10-07-matrix-transpose-second.json`; they are inconclusive.
Remove forced inlining and express the simple traversal with source-column
iterators, preserving contiguous reads while avoiding repeated source bounds
checks. Keep the direct single-axis clone and the frozen blocked dispatch.
Repeat the complete experiment with all original gates and cases.
