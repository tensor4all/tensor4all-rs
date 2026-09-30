# TreeTCI global pivot search readout (issue #792)

Date: 2026-09-30. Baseline: `8868e816` (production code of `origin/main`
`dcc91f58` plus the benchmark). Candidate: `48ab63bc` (one
`TreeTNCachedEvaluator` and one `evaluate_batched` call per search). Both
binaries were built from the same benchmark source and lockfile with
`cargo build --release -p tensor4all-treetci --example benchmark_global_search`
and kept in separate paths. Runs:

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
BLAS_NUM_THREADS=1 taskset -c 2 ./<baseline|candidate>
```

Order: baseline, candidate, baseline, candidate. Each run reports the median of
three repeats. Machine: 16-CPU WSL2 host shared with other jobs; the 1-minute
load average was 8.5 to 12.6 during the runs. Absolute times on this host are
about 1.7 to 1.9 times higher than in an earlier baseline run at lower load
(`chain_cos_129` median 21.8 s), so compare the ratios rather than the absolute
values. This is a single paired comparison, not a promotion-gate experiment.

## Matched accuracy

For every case the two builds print identical evaluation counts, rank
histories, error histories (compared as `f64` bit patterns), pivot-set
fingerprints (all `ijset_history` entries and the final `ijset`) and sampled
result fingerprints. The timings are therefore at matched accuracy.

| case | sites | evaluations | ranks | sampled max rel. error |
|---|---|---|---|---|
| `chain_cos_129` | 129 | 40,776 | 2, 2, 2, 2 | 5.56e-9 |
| `quantics_chain_r20` | 20 | 18,301 | 4, 8, 8, 8 | 1.14e-8 |
| `tree_3x10_plus_centre` | 31 | 48,973 | 4, 9, 9, 9 | 7.39e-9 |

## Wall time (median of three runs, seconds)

| case | baseline pair 1 | candidate pair 1 | baseline pair 2 | candidate pair 2 | speed-up |
|---|---|---|---|---|---|
| `chain_cos_129` | 38.16 | 0.682 | 37.13 | 0.708 | 52-56x |
| `quantics_chain_r20` | 1.055 | 0.0432 | 1.030 | 0.0360 | 24-29x |
| `tree_3x10_plus_centre` | 2.541 | 0.0790 | 2.501 | 0.0771 | 32x |

## Readout rounding

A temporary diagnostic (not committed) evaluated both readouts on every search
of the same three cases in a separate run. The cached readout contracts in a different order, so 35-96% of
the candidate values differ from `TreeTN::evaluate` in their bits, by at most
4.6e-15 relative to the largest `|tt|` of the search. No per-start choice
changed. The smallest margins of the decisions that affect the result, relative
to the largest `|tt|`, were:

- best versus runner-up distinct candidate within a start whose best error
  passes the threshold: 1.5e-5;
- between consecutive accepted pivots in the cross-start ordering: 5.0e-5;
- between a start's best error and the acceptance threshold: 8.6e-8.

Starts whose best error stays below the threshold had near-ties down to 9.4e-16,
but those starts return no pivot. The pointwise `TreeTN::evaluate` is not
bit-reproducible across calls on one tree (78 of 200 values differed in a
check), while the cached readout reproduced its values bit for bit.
