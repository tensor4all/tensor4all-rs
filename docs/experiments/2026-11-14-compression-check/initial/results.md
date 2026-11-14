# Compression-check timing results

Same-library internal comparison; see [the predeclared protocol](../protocol.md).

Validity: **VALID**. Primary candidate/reference geometric mean: **1.326**.
Decision: **FAIL**. All six cases checked bitwise-identical residuals before timing.

| D | Sites | Degree | Rank | Reference µs/call | Candidate µs/call | Candidate/reference (95% CI) |
|---|---|---|---|---|---|---|
| 1 | 4 | 3 | 4 | 5.275 | 10.781 | 2.067 (2.030–2.105) |
| 1 | 12 | 6 | 4 | 39.191 | 63.660 | 1.616 (1.596–1.622) |
| 2 | 4 | 3 | 4 | 136.208 | 130.786 | 0.962 (0.952–0.980) |
| 2 | 12 | 3 | 4 | 431.581 | 643.012 | 1.497 (1.474–1.512) |
| 3 | 4 | 2 | 4 | 1013.835 | 914.393 | 0.905 (0.888–0.914) |
| 3 | 10 | 2 | 4 | 2501.775 | 3128.025 | 1.250 (1.193–1.267) |

Provider: native/faer, f64, one backend thread, CPU 14 (SMT sibling 15 reserved).
The reference and candidate use identical fixtures, one binary, release optimization,
and alternating paired samples. Setup and SVD are excluded. Every raw pair,
including regressions, is retained in [samples.csv](samples.csv).

Source pins, binary hash, thread settings and host-noise gates are in
[metadata.json](metadata.json); [raw output](raw-output.txt) includes the correctness gate.
The reference is copied from `eaa8f7d462707b5809894768c4e7ce260bff8075`; the measured candidate is
`39bfaa0f2deb76d997ca784d855a7699233a77d1`. This measures compression checking only, not an
end-to-end interpolation speedup. No vendor or LAPACK source was consulted.
