# Compression-check timing results

Same-library internal comparison; see [the predeclared protocol](protocol.md).

Validity: **VALID**. Primary candidate/reference geometric mean: **0.691**.
Decision: **PASS**. All six cases checked bitwise-identical residuals before timing.

| D | Sites | Degree | Rank | Reference µs/call | Candidate µs/call | Candidate/reference (95% CI) |
|---|---|---|---|---|---|---|
| 1 | 4 | 3 | 4 | 5.267 | 4.600 | 0.866 (0.858–0.873) |
| 1 | 12 | 6 | 4 | 40.710 | 36.109 | 0.885 (0.874–0.906) |
| 2 | 4 | 3 | 4 | 137.914 | 65.179 | 0.471 (0.469–0.473) |
| 2 | 12 | 3 | 4 | 449.772 | 350.012 | 0.777 (0.770–0.779) |
| 3 | 4 | 2 | 4 | 1009.487 | 522.886 | 0.511 (0.506–0.524) |
| 3 | 10 | 2 | 4 | 2589.489 | 1944.963 | 0.759 (0.752–0.769) |

Provider: native/faer, f64, one backend thread, CPU 14 (SMT sibling 15 reserved).
The reference and candidate use identical fixtures, one binary, release optimization,
and alternating paired samples. Setup and SVD are excluded. Every raw pair,
including regressions, is retained in [samples.csv](samples.csv).

Source pins, binary hash, thread settings and host-noise gates are in
[metadata.json](metadata.json); [raw output](raw-output.txt) includes the correctness gate.
The reference is copied from `eaa8f7d462707b5809894768c4e7ce260bff8075`; the measured candidate is
`3409ae3745cf1bb510d178723006ea924e0f7ef9`. This measures compression checking only, not an
end-to-end interpolation speedup. No vendor or LAPACK source was consulted.

The [initial complete experiment](initial/results.md) failed the same primary
gate (1.326). The refined full-left TTCache traversal removed the two-sided
mapper and persistent-prefix overhead; all six cases improved under the unchanged
protocol. Both sets of raw samples remain available.
