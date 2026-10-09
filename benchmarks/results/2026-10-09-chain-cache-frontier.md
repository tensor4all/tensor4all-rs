# Chain evaluator cache-frontier controls

Related: [#671](https://github.com/tensor4all/tensor4all-rs/issues/671).

The earlier default TreeTN warm route retains messages around a vertex and
still contracts the center core. TTCache retains both sides of a bond split.
The new controls explicitly request the public around-split hint, with both
erased and typed outputs. The typed route matches the API form used by
TreeACI coordinate scans; this fixture does not reproduce an SGW sweep.

## Protocol

All routes use the same deterministic 16-site f64 chain, local dimension 2,
bonds 64/128/256, midpoint 8 and 8 left by 8 right coordinate combinations
(seed 646). Every cold and warm TreeTN route checks all 64 values against
independent TTCache results with the existing scale-aware tolerance 1e-12.
Setup and oracle conversion are outside timing; native API output remains
inside timing. Cold evaluators are fresh; warm evaluators reuse messages.

Measured on 2026-10-09, AMD Ryzen 9 6900HX under a virtualized Linux host,
CPU affinity 2, Rust/Cargo 1.98.1, release/default CPU backend, diagnostics
disabled, all Rayon/BLAS/Tenferro thread variables fixed to one. Criterion
uses 10 samples, 1 second warm-up and 1 second measurement per case.
Library sources are baseline `0e1c49ba`; the benchmark-only branch has the
additional controls. Merged `fd98c3af` has identical Core, TreeTN, SimpleTT
and tensorbackend sources. The benchmark and reproduction command are
linked from [the benchmark catalog](../README.md#chain-evaluator-cache-frontier-controls-671).

This is a single-revision, descriptive comparison, not a preregistered
regression gate or a before/after production speedup. Compiler and other
numerical jobs were not run concurrently with these measurements.

## Results

Microseconds per 64-point batch, point estimate and 95% bootstrap interval.
Values come from Criterion slope estimates (mean when no slope exists).

| Route | Bond 64 | Bond 128 | Bond 256 |
| --- | ---: | ---: | ---: |
| ttcache_cold | 2761.340 [2742.881, 2780.806] | 11031.558 [11007.044, 11064.088] | 45842.305 [45676.944, 46016.999] |
| ttcache_warm | 9.150 [9.133, 9.171] | 11.330 [11.305, 11.365] | 16.903 [16.870, 16.951] |
| treetn_cold | 1177.702 [1171.740, 1182.674] | 4449.547 [4430.117, 4482.924] | 23338.434 [23141.034, 23539.747] |
| treetn_warm | 362.471 [361.601, 363.562] | 1043.323 [1040.241, 1046.233] | 3783.629 [3767.312, 3801.477] |
| treetn_around_split_cold | 1129.066 [1123.933, 1134.561] | 3936.886 [3933.479, 3940.595] | 17438.721 [17374.036, 17488.639] |
| treetn_around_split_warm | 150.560 [150.147, 151.009] | 177.039 [176.435, 177.924] | 294.436 [293.150, 295.775] |
| treetn_typed_around_split_cold | 1069.951 [1065.628, 1074.123] | 3883.349 [3867.699, 3897.292] | 17392.666 [17356.520, 17448.254] |
| treetn_typed_around_split_warm | 87.980 [87.756, 88.233] | 115.017 [114.208, 116.586] | 216.230 [215.452, 216.780] |

At bond 256, default warm TreeTN is approximately 3784 us, hinted erased
294 us and hinted typed 216 us, versus TTCache 16.9 us. The selected cache
frontier and output API account for a substantial part of this comparison;
a residual gap remains. These measurements do not separate component-key
construction, cache lookup, validation, dispatch and remaining contraction
costs, so they do not classify the residual as either necessary work or a
bug. Cold TreeTN also performs different batched contraction work from
TTCache. No library algorithm, numerical tolerance, public API, dependency
or coverage threshold changes in this benchmark patch. #671 stays open.
