# Matrix-valued chain TCI2 (#1463): joint vs separate fits

Pilot for the review discussion on #1463, **not** an acceptance threshold:
synthetic targets, one Apple M3, no universal factor is claimed, x86 numbers
are welcome. Folding (`L+2 -> L` cores) is measured in the addendum at the end.

## Setup

- Measured revision `5bc274641f9fb1399dbefcce51c920d80ec79504` (branch `feat/1463-tci2-matrix-valued`, merged onto
  upstream `de053a07`, i.e. after #1485 and #1484). The runner builds
  `git archive HEAD` in `target/issue-1463/` and removes it afterwards.
- Apple M3, 8 cores, 16 GiB, macOS 15.6.1 ([host.txt](host.txt)); Rust 1.100.0
  ([rustc.txt](rustc.txt)), `cargo build --locked --release --jobs 2`.
  `RAYON_NUM_THREADS=OPENBLAS_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1`, `nice -n 15`,
  one case at a time, machine otherwise in normal use.

Reproduce (one `results.csv`, about 5 minutes of cases plus the build):

```sh
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  nice -n 15 python3 docs/experiments/issue-1463/run.py
```

## What is compared

`F(x)` has `rows*cols` entries on `L = 12` binary sites (4096 points). Targets
(`f64`, seeded): *corr* `F = sum_{k<2} a_k(x) U_k V_k^T` (entries share two
functions of `x`), *unrel* (every entry its own oscillating, damped function).
*burn* adds a busy loop of about 50 us to every whole-matrix evaluation, *cheap*
does not.

- **A** `crossinterpolate2_matrix`: one train on `[rows, x_1..x_L, cols]`.
- **B** the same problem through `crossinterpolate2` with a hand-written
  whole-matrix cache. Asserted identical to A (ranks, errors, source calls).
- **C** `rows*cols` scalar fits, each its own cache (source call = whole matrix).
- **D** the same fits sharing one whole-matrix cache. Asserted identical to C.

Every fit uses the absolute tolerance `1e-8 * max|F|`, `max_iter = 20`, no bond
cap, seed 1463, the largest entry as initial pivot. Error = max over 2048 sampled
points and all entries, relative to `max|F|`. "sep" is the maximum bond over the
component fits (C = D) and the summed size of their trains. Median of 3 timed
repetitions after 1 warm-up round, wall time of the whole fit including caches.

## Results

Full pivoting:

| target | shape | cost | source calls A / C / D | max bond A / sep | err A / sep | TT bytes A / sep | ms A / B / C / D |
|---|---|---|---|---|---|---|---|
| corr r=2 | 2x3 | cheap | 1734 / 8032 / 3381 | 12 / 8 | 2.23e-9 / 4.88e-9 | 12320 / 23664 | 10 / 11 / 33 / 32 |
| corr r=2 | 2x3 | burn | 1734 / 8032 / 3381 | 12 / 8 | 2.23e-9 / 4.88e-9 | 12320 / 23664 | 97 / 98 / 437 / 201 |
| corr r=2 | 8x8 | cheap | 1917 / 91435 / 4095 | 12 / 8 | 1.99e-9 / 9.36e-9 | 12688 / 279872 | 16 / 17 / 373 / 327 |
| corr r=2 | 8x8 | burn | 1917 / 91435 / 4095 | 12 / 8 | 1.99e-9 / 9.36e-9 | 12688 / 279872 | 111 / 113 / 4958 / 534 |
| unrel | 2x3 | cheap | 2677 / 8936 / 3713 | 25 / 9 | 5.72e-9 / 6.71e-9 | 40008 / 28000 | 25 / 30 / 42 / 41 |
| unrel | 2x3 | burn | 2677 / 8936 / 3713 | 25 / 9 | 5.72e-9 / 6.71e-9 | 40008 / 28000 | 159 / 161 / 485 / 221 |
| unrel | 8x8 | cheap | 3611 / 95785 / 4096 | 78 / 9 | 8.48e-9 / 7.39e-9 | 387712 / 305200 | 230 / 254 / 450 / 366 |
| unrel | 8x8 | burn | 3611 / 95785 / 4096 | 78 / 9 | 8.48e-9 / 7.39e-9 | 387712 / 305200 | 406 / 434 / 5268 / 565 |

Rook pivoting:

| target | shape | cost | source calls A / C / D | max bond A / sep | err A / sep | TT bytes A / sep | ms A / B / C / D |
|---|---|---|---|---|---|---|---|
| corr r=2 | 2x3 | cheap | 2062 / 8330 / 3377 | 12 / 8 | 2.25e-9 / 5.38e-9 | 12320 / 23808 | 19 / 20 / 54 / 54 |
| corr r=2 | 2x3 | burn | 2062 / 8330 / 3377 | 12 / 8 | 2.25e-9 / 5.38e-9 | 12320 / 23808 | 124 / 125 / 472 / 224 |
| corr r=2 | 8x8 | cheap | 2140 / 98091 / 4096 | 12 / 8 | 1.78e-9 / 1.26e-8 | 12688 / 279664 | 27 / 27 / 619 / 581 |
| corr r=2 | 8x8 | burn | 2140 / 98091 / 4096 | 12 / 8 | 1.78e-9 / 1.26e-8 | 12688 / 279664 | 134 / 143 / 5617 / 787 |
| unrel | 2x3 | cheap | 3000 / 9787 / 3762 | 25 / 9 | 6.76e-9 / 8.29e-9 | 39400 / 27824 | 42 / 46 / 60 / 57 |
| unrel | 2x3 | burn | 3000 / 9787 / 3762 | 25 / 9 | 6.76e-9 / 8.29e-9 | 39400 / 27824 | 195 / 198 / 557 / 254 |
| unrel | 8x8 | cheap | 4011 / 103956 / 4096 | 78 / 9 | 9.35e-9 / 7.34e-9 | 387328 / 306016 | 1506 / 1560 / 767 / 680 |
| unrel | 8x8 | burn | 4011 / 103956 / 4096 | 78 / 9 | 9.35e-9 / 7.34e-9 | 387328 / 306016 | 1619 / 1709 / 5986 / 858 |

## Three effects

- **Shared sampling (C to D).** One cache keyed on `x` cuts source calls 2.4x (2x3)
  and 22-25x (8x8, about `rows*cols`) with identical fits. It pays in wall time
  only when the callback is expensive: C/D is 1.0-1.2 for a cheap callback and
  2.1-2.2 (2x3) / 7.0-9.3 (8x8) with 50 us.
- **Joint fit (D to A).** Ranks are higher (12 vs 8 for corr, 25 / 78 vs 9 for
  unrel). For correlated entries it still wins: 8x8 storage 22x smaller, fit
  time 4.8-20x (Full) and 5.9-22x (Rook) faster than D; 2x3 storage 1.9x smaller,
  fit time 1.8-3.2x. For unrelated entries it loses on storage (the joint fit stores about 1.3-1.4x more bytes), but
  not on time except Rook 8x8 (D/A 0.45-0.53): it is 1.3-1.6x faster in the other
  six cases, where the separate fits need many more sweeps in total.
- **Native entry point (A vs B).** A is not slower than the hand-written adapter
  in any case (A/B 0.85-1.00), so the overhead is not visible at this noise level.

## Break-even rule

One joint train stores and reads out about `r_joint^2` numbers per site, `rows*cols`
separate trains `rows*cols * r_sep^2`: the joint fit wins when
`r_joint < sqrt(rows*cols) * r_sep`. Fit time scales like `r^3`, giving
`r_joint < (rows*cols)^(1/3) * r_sep`. Storage rule against the data: corr 2x3
12 < 2.45*8 (wins, 1.9x), corr 8x8 12 < 8*8 (22x), unrel 2x3 25 > 2.45*9 (loses,
the joint fit stores 1.4x more), unrel 8x8 78 > 8*9 (loses, 1.3x more): right in all four. The fit-time rule
predicts losses for unrel and wins for corr; the data agree only for corr and
Rook 8x8 unrel, since sweep counts and per-sweep costs differ between the fits. Which side an application is on depends on its
target, which is why the choice should be documented rather than defaulted.

## After #1485

An earlier run of this pilot (before #1485/#1484) had many component fits stall
on `max_iter` just above the tolerance (#1482, up to 18 of 64 fits at 8x8, 582
half-sweeps in total for corr 8x8), inflating C and D wall time. #1485 fixed the
alternation upstream; the numbers above are after it. The stall counters were
dropped from the harness. Compared with that run, source calls and wall time
fell for A, C and D (corr 8x8 Full cheap: A 69 to 16 ms, D 617 to 327 ms), the
joint advantage grew for correlated targets, and the conclusions (shared sampling
carries the call saving, joint wins only with shared structure) did not change.

## Limits

Synthetic targets; one machine and compiler; the 50 us callback is a busy loop,
not a real source; the stop error is sampled, not a bound (final errors land at
`1e-9` to `1.3e-8` of `max|F|`); 3 timed repetitions, so differences of a few
percent are noise; x86 pending.

## Addendum: folded layout (`MatrixTCI2Result::to_folded`)

Folding is about the **output shape** (matrix indices as open legs of the end
cores, `L+2 -> L` cores), not about speed: it contracts the row core into the
first and the column core into the last physical core, represents the same
function (no new samples, no truncation, one contraction per end at conversion)
and changes the readout by 0.98-1.05x, noise level. Measured on the four step-0
cases at `f64`, Full pivoting, cheap callback, fit A, `L = 12`
([folding.csv](folding.csv); `python3 docs/experiments/issue-1463/run.py fold`,
same host and flags as above, release, one thread, `nice`). Bytes are core data
only (`retained_bytes()` of each layout). Readout is the whole matrix at 1024
points (every fourth of the 4096), median of 3 rounds after one warm-up round.

Three readouts, so that slab extraction and folding are not conflated:

- **old**: the library `MatrixTCI2Result::evaluate`, which extracts every slab
  element by element through `Tensor3Ops::slice_site` (checked per-element
  indexing);
- **fast L+2**: harness-local; the harness copies the L+2 cores once into raw
  column-major buffers and extracts slabs by contiguous copies (bit-identical
  to the library readout, asserted on all 1024 points);
- **folded**: `FoldedMatrixTT::evaluate`, which stores raw column-major cores
  (agrees with the L+2 readout to `1e-10 max|F|`, asserted).

| target | shape | max bond | bytes L+2 | bytes folded | folded / L+2 | ms old | ms fast L+2 | ms folded | old / fast | fast / folded |
|---|---|---|---|---|---|---|---|---|---|---|
| corr r=2 | 2x3 | 12 | 12320 | 12304 | 0.999 | 6.36 | 3.22 | 3.08 | 1.97 | 1.05 |
| corr r=2 | 8x8 | 12 | 12688 | 13200 | 1.040 | 8.99 | 5.56 | 5.48 | 1.62 | 1.01 |
| unrel | 2x3 | 25 | 40008 | 39904 | 0.997 | 18.37 | 8.56 | 8.36 | 2.15 | 1.02 |
| unrel | 8x8 | 78 | 387712 | 386688 | 0.997 | 229.00 | 133.43 | 135.88 | 1.72 | 0.98 |

- **Folding effect (fast L+2 to folded):** 0.98-1.05x, noise of three
  repetitions. Folding removes only 2 of the `L+1 = 13` slab products.
- **Apparent speedup:** folded versus the library L+2 readout is 1.6-2.2x, but it
  is a side effect of slab extraction, not of folding: the folded type stores raw
  column-major cores, the library L+2 readout reads element by element. Making the
  library L+2 readout equally fast today would need a duplicate column-major copy
  of every core, because `Tensor3` has no zero-copy column-major view; that is
  proposed separately in #1513. The library L+2 readout is therefore unchanged.
- **Storage:** barely moves (the end cores are a small part of 12 interior cores)
  and grows only for corr 8x8, by 4.0%, where `rows * d_1 * r_1` exceeds
  `rows * r_0 + r_0 * d_1 * r_1`. `L+2` stays the default.

One machine, three repetitions, plain-loop readout: indicative, not a rule.
