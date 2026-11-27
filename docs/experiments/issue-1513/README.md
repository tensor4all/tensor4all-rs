# Slab readout: borrowed column-major buffer (#1513, step 2)

Evidence for the immutable `Tensor::as_col_major_slice` borrow and the in-place
slab reads built on it. **Acceptance (maintainer, #1513):** unrel 8x8 readout at
least 1.5x faster than the pre-#1516 path, every output bit-identical.
Measured on one Apple M3 only ([host.txt](host.txt), [rustc.txt](rustc.txt));
the x86 run is for the maintainer (command below).

## Revisions

| name | what | commit |
|---|---|---|
| pre-1516 | `Tensor3Ops::slice_site` reads element by element | `62c3b606^` = `1ad44e07` |
| 1516 | `slice_site` copies `right` contiguous runs of `left` elements | `62c3b606` |
| step2 | readout reads slabs in place from the borrowed buffer | the branch tip of this PR |

`run.py` builds each from `git archive` in `target/issue-1513/` (one target dir
per revision: archive timestamps would otherwise let cargo reuse another
revision's workspace crates), runs the binaries one at a time under `nice`,
rotating the order per pass, and deletes the workspace. [revisions.txt](revisions.txt)
records the resolved commits (the step2 entry is a local pre-squash commit; its library sources equal the PR tip).

## Reproduce (also on x86)

```sh
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  nice -n 15 python3 docs/experiments/issue-1513/run.py   # commit first; about 15 min incl. 3 builds
```

Defaults: 3 processes per revision and case, each the median of 7 timed rounds
after one warm-up round; the table reports the median of the three. `run.py`
fails if any revision's output hash differs from another's. Release build,
`--locked --jobs 2`, one thread.

## Readout: 1024 points, `L = 12`, `f64`, Full pivoting, cheap callback

The fits are those of [issue-1463](../issue-1463/README.md) (same targets, seed,
tolerance); the timed part is `MatrixTCI2Result::evaluate` on every fourth of the
4096 points, whole matrix per point ([readout.csv](readout.csv)). `hash` is FNV-1a over the
`to_bits()` of all 1024 output matrices: identical across the three revisions.

| target | shape | ms pre-1516 | ms 1516 | ms step2 | pre-1516 / step2 | 1516 / step2 | ms folded (step2) |
|---|---|---|---|---|---|---|---|
| corr r=2 | 2x3 | 6.88 | 3.33 | 2.54 | 2.71 | 1.31 | 2.54 |
| corr r=2 | 8x8 | 9.42 | 5.77 | 4.92 | 1.91 | 1.17 | 4.90 |
| unrel | 2x3 | 19.87 | 8.44 | 7.74 | 2.57 | 1.09 | 7.68 |
| unrel | 8x8 | 242.29 | 137.47 | 133.75 | **1.81** | 1.03 | 132.74 |

Unrel 8x8 (max bond 78) is dominated by the `m x 78 x 78` products, so the
borrow adds only 3% on top of #1516 there; most of the 1.81x is step 1. Folded
versus L+2 stays within noise (readout of the folded layout reads its own buffer
the same way).

## Micro-benchmark (ns per slab, `slice_site` on pilot core shapes)

`left x site x right` at site dimension 2 ([micro.csv](micro.csv)). `copy then sum` /
`borrow then sum` run the same sum over the slab on a fresh copy (`slice_site`)
and on the borrowed buffer (strided, in place); the sum is the stand-in
consumer, a real one is the strided gemm of `evaluate`.

| shape | slice_site pre-1516 | slice_site 1516 | copy then sum | borrow then sum |
|---|---|---|---|---|
| 1x2x12 | 74.4 | 40.6 | 44.7 | 7.3 |
| 12x2x12 | 658.7 | 36.0 | 121.9 | 66.5 |
| 25x2x25 | 2794.3 | 101.4 | 527.2 | 396.9 |
| 78x2x78 | 26307.7 | 755.7 | 4970.9 | 4270.5 |
| 78x2x1 | 376.5 | 46.7 | 91.6 | 31.9 |

The slab copy is a small, shrinking fraction of the arithmetic that consumes
it, which is why the end-to-end gain of the borrow over #1516 is modest while the
gain over the element-wise path is large.

One machine, indicative, not a rule; timings are not asserted anywhere in tests.
