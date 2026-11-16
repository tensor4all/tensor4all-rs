# Interpolative QTT TreeTN migration measurements

Local diagnostic for [#1334](https://github.com/tensor4all/tensor4all-rs/issues/1334)
against SimpleTT baseline `1647bc664aed86fcbd37fa7708aa7699c56b87a8`.
Both builds used the release profile, Rust 1.98.1, faer, and a caller-owned
one-thread CPU context on an Apple M3 Pro running macOS 27.0.1. Thread-count
environment variables were set to one. The host did not provide CPU affinity
for the probes, so the measurements are diagnostic.

Each case used three pairs of separate processes, alternating the order of
the baseline and candidate. Each process primed for one second, then ran 20
serial patched constructions. The first call was excluded from the
timings. Each process checked the engine budget, a single accepted patch,
and an estimated absolute error at most `1e-8`. The SVD tolerance was
`1e-12` with no rank cap. Ordinary construction and lane scaling were
outside the measurement.

- **Real 1D** used `1 + 0.1x + 0.7x²` on `[0, 1)`, with 18 bits and degree 10.
- **Complex 2D** used
  `(1 + 0.2x − 0.3y + 0.4xy + 0.5y³) + i(x + y + 0.05xy²)`
  on `[0, 1)²`, with 12 bits per axis and degree 7.

Timers bracketed the same three phases inside the production serial patch
body. Construction includes sampling, the interpolation stencil, core
assembly and output publication. Compression covers the local sweep.
Validation includes selecting and packing the sampled points, constructing
the cache, batch readout and comparison with the child interpolants.
The table reports pooled phase medians in microseconds over 57 calls per
row. Peak RSS is the maximum process-wide value from `/usr/bin/time -l`
over the three launches.

| Case | Implementation | Construction (µs) | Compression (µs) | Cached validation (µs) | Peak RSS (MiB) | Largest error estimate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Real 1D | SimpleTT | 56.25 | 380.25 | 75.21 | 6.44 | 1.52e-12 |
| Real 1D | TreeTN | 109.83 | 634.54 | 292.58 | 9.59 | 1.51e-12 |
| Complex 2D | SimpleTT | 1996.17 | 18633.38 | 3300.96 | 16.92 | 3.94e-12 |
| Complex 2D | TreeTN | 610.29 | 16260.29 | 1211.58 | 17.42 | 3.94e-12 |

Per-process median total phase time ranged from 0.48 to 0.60 ms for SimpleTT
and from 0.91 to 2.25 ms for TreeTN on the 1D case. It ranged from 23.78 to
24.52 ms for SimpleTT and from 17.74 to 18.36 ms for TreeTN on the 2D case.
TreeTN took longer on the small 1D case and used more peak memory in both
cases. All three phases took less time on the complex 2D case. The initial
sandboxed launch was discarded after peak RSS collection failed. The table
uses the complete paired run with that telemetry available.

The [raw paired capture](2026-11-15-interpolative-qtt-treetn.json) includes every
call from the complete run, peak RSS per process, and the Git blob IDs of the
candidate sources that define the timed phases.
