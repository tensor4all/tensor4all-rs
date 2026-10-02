# Issue #792: cached global pivot readout

## Decisions

- Keep one `TreeTNCachedEvaluator` per global pivot search and evaluate the full candidate batch together. This shares subtree environments across candidate points; splitting by varied site was slower on the recorded workloads.
- Keep `TreeTN::evaluate` as the pointwise reference path in tests so fixed-seed searches can compare the chosen pivots against the cached readout.

## Verification conclusions and constraints

- On current HEAD `d93b3e5e`, a paired release rerun with the same `ChaCha8Rng` and benchmark source produced identical evaluation counts, rank and error histories, pivot fingerprints, sampled-result fingerprints, and sampled relative errors between pointwise and cached readouts. The three cases measured median paired speedups of 47.00x, 25.10x, and 28.29x. Full measurements, binary hashes, and host conditions are in the [benchmark report](../benchmarks/results/2026-09-30-treetci-global-search.md#rerun-on-current-head-with-chacha8rng).
- The `tensor4all-treetci` package suite passed: 55 unit tests, 20 integration tests, and 31 doctests.
- A current-HEAD diagnostic compared both readouts on every candidate batch in 36 global searches across the three fixtures. Readout differences reached `1.45e-15` relative to the largest `|tt|`; the nearest best-error threshold was `9.11e-8` away. No per-start winner, threshold acceptance, or final pivot order changed. Detailed margins are in the [benchmark report](../../benchmarks/results/2026-09-30-treetci-global-search.md#current-head-readout-rounding-diagnostic).
- The benchmark covers three deterministic fixtures with three paired process runs. It is a post-implementation exploratory rerun, not a predeclared promotion gate or a confidence-interval study. The margin diagnostic supports these workloads but does not establish stability for inputs deliberately constructed near the acceptance threshold.
