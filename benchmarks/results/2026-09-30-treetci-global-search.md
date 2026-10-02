# TreeTCI global pivot search readout (issue #792)

Date: 2026-09-30. Baseline: `8868e816` (production code of `origin/main`
`dcc91f58` plus the benchmark). Candidate: `48ab63bc` (one
`TreeTNCachedEvaluator` and one `evaluate_batched` call per search). Both are
pre-rebase commits and are not on the rebased branch; their trees are
`eae982919639` (`8868e816`) and `e646b0871bc5` (`48ab63bc`). The rebased
equivalents are `abff46e9` and `4f3bf03d`, measured again below. Both
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
check), while the cached readout reproduced its values bit for bit within one
process. Reproducibility across threads or processes was not checked here.

## Rerun on the #793 base

After the branch was rebased onto `9ad67f2c` (#793), the comparison was
repeated with the same protocol. The baseline was `abff46e9`: the benchmark on
`9ad67f2c`, with the pointwise readout. The candidate was `4f3bf03d`: the
cached readout. Both use `StdRng`. Each commit was built in its own detached
worktree with its own target directory. The three binaries of this rerun had
distinct SHA-256 hashes: baseline `48e93eb2...`, candidate `312a56a3...`,
`dcc91f58` baseline `6418ca0c...`.

For every case the two builds again print identical evaluation counts, rank
histories, error bit patterns, pivot fingerprints and sampled-result
fingerprints. The values are the same as in the tables above, so #793 did not
change the result of this workload.

| case | baseline run 1 | candidate run 1 | baseline run 2 | candidate run 2 | speed-up |
|---|---|---|---|---|---|
| `chain_cos_129` | 17.28 | 0.371 | 17.16 | 0.375 | 46x |
| `quantics_chain_r20` | 0.420 | 0.0174 | 0.405 | 0.0155 | 24-26x |
| `tree_3x10_plus_centre` | 0.965 | 0.0322 | 0.949 | 0.0329 | 29-30x |

Times are medians of three repeats, in seconds. `chain_cos_129` was also run
with the `dcc91f58` baseline (`8868e816`) in the same session: 16.30 s and
16.66 s, which is not faster than `abff46e9`. So #793 did not change the cost
of the pointwise readout.

The branch also switches the global search from `StdRng` to the named
`ChaCha8Rng`. That changes the random starting points for a fixed seed, and
with them the recorded pivots and error histories in the unit tests. The timing
comparison above predates that switch. A paired rerun on the resulting HEAD is
recorded below. The readout-rounding margins in the previous section were
measured on the historical `dcc91f58`/`StdRng` base. A separate diagnostic on
current HEAD follows. Whether the RNG switch changes result quality is tested
in [the last section](#rng-switch-stdrng-versus-chacha8rng-200-seeds).

### Excluded run: validity failure

An earlier rerun on this base (2026-09-30) is excluded as a validity failure.
Its baseline was meant to be the pointwise readout: commit `f8a6754c` (tree
`4810c076`), `9ad67f2c` with the benchmark harness and a `ChaCha8Rng` change.
Its candidate was commit `9d38f361` (tree `6aabff89`), with the cached readout.
Both binaries were built one after the other into the same shared
`CARGO_TARGET_DIR` and copied out. Three pairs ran, baseline first. Median
seconds:

| case | baseline 1 | candidate 1 | baseline 2 | candidate 2 | baseline 3 | candidate 3 |
|---|---:|---:|---:|---:|---:|---:|
| `chain_cos_129` | 0.3804 | 0.3825 | 0.4041 | 0.4293 | 0.4286 | 0.4167 |
| `quantics_chain_r20` | 0.0190 | 0.0258 | 0.0222 | 0.0246 | 0.0254 | 0.0231 |
| `tree_3x10_plus_centre` | 0.0442 | 0.0476 | 0.0481 | 0.0497 | 0.0528 | 0.0449 |

Binary SHA-256 for pairs 1 and 2: baseline
`c0f3e2f2e0603616bbb25bb02a25e25a04f9075df591dd7f77d2ad8e19bcb5af`, candidate
`53455de1d971cc9412d6527bbb8cf457018d4da4730edfe86fca04285ea0a02e`. Both were
rebuilt before pair 3, which ran baseline
`a286282b23e2d45370d763f86aa075cf1ae3735c23bacaf1087656f3216ed255` and candidate
`ff361802bcee66283605efba89b776cfd8beb579cbbdbbad2b590f895ba3249a`.

Two facts confirm that the baseline binary did not measure the pointwise
readout. First, its `chain_cos_129` times, 0.38-0.43 s, match the cached readout
and not the pointwise one. Second, the clean rebuild above gave 16-17 s for the
pointwise baseline: each commit was built in its own worktree with its own
target directory, and the binaries had distinct hashes. The pointwise rerun on
the later `ChaCha8Rng` head, below, also takes 17 s. The mechanism that put the
wrong code into the excluded baseline was not identified; the shared target
directory is the methodological fault. None of its numbers are used.

## Rerun on current HEAD with `ChaCha8Rng`

Date: 2026-10-02. Candidate: `d93b3e5e`. The pointwise baseline was built from
the same source and lockfile, with only the production dispatch in
`find_global_pivots` changed to call `TreeTN::evaluate` instead of
`cached_batched_readout`. Both builds therefore use `ChaCha8Rng`, the same
benchmark source, options, fixtures, and sampling seed. The pointwise change
was made in a temporary detached worktree and is not a branch commit. To
rebuild that baseline, change the readout passed by `find_global_pivots` in
`crates/tensor4all-treetci/src/globalpivot.rs` at `d93b3e5e` to the pointwise
`TreeTN::evaluate` (the same readout as `pointwise_readout` in the unit
tests):

```diff
-    search_with_readout(state, evaluate, params, cached_batched_readout)
+    search_with_readout(state, evaluate, params, |treetn, site_indices, candidates| {
+        Ok(treetn.evaluate(site_indices, candidates)?)
+    })
```

This reconstruction follows the description above; the exact patch file of
that run was not kept.

Both used `cargo build --locked --release -p tensor4all-treetci --example
benchmark_global_search`, separate target directories, and the same release
profile. Rust: `1.98.1`; Cargo: `1.98.1`. Machine: WSL2, AMD Ryzen 9 6900HX, 16
logical CPUs. Runs were pinned to CPU 2 with all Rayon, OMP, OpenBLAS, MKL, and
BLAS thread counts set to 1. The complete three-case suite ran in this order:
pointwise, cached, cached, pointwise, pointwise, cached. Each process reports
the median and minimum of three timed repetitions per case.

For this rerun, the validity limit was a maximum 1-minute load average of 4.0
at each process start and end. The maximum observed value was 1.99; all six
processes exited successfully. CPU frequency was unavailable through the
container's cpufreq interface. The acceptance check for this rerun was exact
agreement in evaluation counts, rank and error histories, pivot and sample
fingerprints, sampled relative errors, and a median speedup above 1x for every
case. All checks passed. This remains an exploratory rerun, not a promotion-gate
experiment: the optimization predates a formally predeclared performance gate,
and three process pairs do not provide confidence intervals.

The median and minimum times below are seconds. Speed-up is the baseline median
divided by the candidate median for each pair.

| case | pair | pointwise min / median | cached min / median | speed-up |
|---|---:|---:|---:|---:|
| `chain_cos_129` | 1 | 15.8857 / 17.5417 | 0.3521 / 0.3673 | 47.76x |
| `chain_cos_129` | 2 | 15.7097 / 17.0861 | 0.3548 / 0.3635 | 47.00x |
| `chain_cos_129` | 3 | 15.6517 / 16.9524 | 0.3628 / 0.3703 | 45.78x |
| `quantics_chain_r20` | 1 | 0.4634 / 0.4769 | 0.0175 / 0.0179 | 26.64x |
| `quantics_chain_r20` | 2 | 0.4676 / 0.4693 | 0.0182 / 0.0187 | 25.10x |
| `quantics_chain_r20` | 3 | 0.4571 / 0.4616 | 0.0186 / 0.0189 | 24.42x |
| `tree_3x10_plus_centre` | 1 | 1.1179 / 1.1185 | 0.0384 / 0.0395 | 28.32x |
| `tree_3x10_plus_centre` | 2 | 1.1019 / 1.1063 | 0.0407 / 0.0415 | 26.66x |
| `tree_3x10_plus_centre` | 3 | 1.0969 / 1.1003 | 0.0377 / 0.0389 | 28.29x |

The median of the three paired speed-ups is 47.00x for `chain_cos_129`,
25.10x for `quantics_chain_r20`, and 28.29x for `tree_3x10_plus_centre`.
All six process runs produced identical values in these result fields:

| case | evaluations | ranks | error-history bits | pivot fingerprint | sample fingerprint | sampled relative error |
|---|---:|---|---|---|---|---:|
| `chain_cos_129` | 42,145 | 2, 2, 2, 2 | `3e37fe61bc24f6ec`, `3e28000000000020` × 3 | `3d8b6a37988f7d2a` | `40988ea4f9613ee4` | 5.56e-9 |
| `quantics_chain_r20` | 17,946 | 4, 8, 8, 8 | `3e3215cebb358024`, `3e39734e02d66482`, `3e3d2cda0d8c8845`, `3e3bcbb3b6fbf3d1` | `45ccbee3c3dcd8ab` | `1f20dd0fd11cf18b` | 1.28e-8 |
| `tree_3x10_plus_centre` | 50,253 | 4, 9, 9, 9 | `3c74215e5dade614`, `3e3b0280f17db7ca`, `3e3cb7f908e32455`, `3e3cb7bf7da8b4cf` | `1360104c504748d5` | `c579b4d6d8bf8ed1` | 7.30e-9 |

Binary SHA-256: pointwise `01505ed18052fa028833811444ca64a2628052d2855f3bcb4c14374187660153`,
cached `55d1488529b23c914a7d098bbbb33af48f5bb164b5a22090ccd7ee5d4a309f1d`.


## Current-HEAD readout rounding diagnostic

Candidate: `d93b3e5e` with `ChaCha8Rng`. A temporary instrumentation build
compared the cached readout and `TreeTN::evaluate` on the exact candidate batch
of every global search. It ran the complete benchmark suite (three internal
repetitions per case); timing output from this instrumented build is ignored.
There were 12 search calls per case, 36 total. The instrumentation was removed
and the branch source and normal release example binary were restored after
collection.

Readout differences and decision margins below are normalized by the largest
`|tt|` value from either readout within each search. The differing-value range
is the minimum and maximum fraction of candidate values whose `f64` component
bits differed across the 12 calls. Threshold margin is the minimum distance of
either readout's per-start best error to the acceptance threshold, including
starts that returned no pivot. The accepted-threshold column includes only
per-start best errors above the threshold. Runner-up margins compare distinct
candidate points for starts whose best error passed the threshold; cross-start
margins compare adjacent accepted pivots in output order.

| case | candidates per search | bit-different values | max relative readout difference | min threshold margin | min accepted threshold margin | min winner/runner-up margin | min cross-start margin | decision changes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `chain_cos_129` | 1,290 | 44.9%-87.1% | 1.776361e-15 | 9.443152e-8 | 1.135005e0 | 5.987673e-2 | 5.839336e-3 | none |
| `quantics_chain_r20` | 200 | 71.5%-93.0% | 1.077057e-15 | 9.110857e-8 | 1.489425e-4 | 1.569439e-5 | 4.323341e-6 | none |
| `tree_3x10_plus_centre` | 310 | 85.2%-97.4% | 1.451450e-15 | 2.098997e-7 | 1.107742e-3 | 1.808576e-4 | 7.665757e-4 | none |

No per-start winner, threshold acceptance, or final pivot order changed in any
of the 36 search calls. The closest threshold decision was about `8.46e7` times
farther from the threshold than the maximum readout difference for that case.
This supports the recorded workloads; it does not prove decision stability for
arbitrary inputs deliberately constructed near the acceptance threshold.

## RNG switch: `StdRng` versus `ChaCha8Rng` (200 seeds)

Date: 2026-10-02. Question: does switching the global search seed RNG from
`StdRng` to `ChaCha8Rng` change result quality? A fixed seed gives different
starting points under the two RNGs, so fixed-seed trajectories change. In
`rand` 0.9, which the workspace uses, `StdRng` is ChaCha12, so both are ChaCha
stream ciphers that differ only in round count.

Setup. Both binaries were built in release mode from `3b109925` with the cached
readout and a scratch-only harness patch. The patch reads the TreeTCI seed from
the environment and prints one JSON line per case. The `StdRng` binary also
replaced `ChaCha8Rng::seed_from_u64` with `rand::rngs::StdRng::seed_from_u64`
in `find_global_pivots`. Neither patch is committed. Each binary ran the three
benchmark cases for seeds 1 to 200 (tolerance `1e-8`, `normalize_error`,
default options otherwise), with the RNG order alternating per seed, pinned to
CPU 2 with one thread for Rayon and BLAS. That gives 600 paired runs. The host
was heavily loaded (1-minute load average 23-108), so no timing from this run
is used. Evaluation counts, ranks and errors are deterministic for a given
binary and seed and do not depend on load.

Criterion, fixed before the runs. For every case, all four must hold:

1. the `ChaCha8Rng` median sampled relative error lies inside the `StdRng`
   interquartile range (IQR);
2. the `ChaCha8Rng` median evaluation count lies inside the `StdRng` IQR;
3. false-convergence parity: either both RNGs or neither have a run whose
   sampled relative error exceeds `10 * tol = 1e-7`;
4. the worst sampled relative errors are of the same order (their `log10`
   values differ by less than 1).

The verdict is quality-neutral if all four hold in all three cases.

Results. Median [IQR] over 200 seeds. The sampled error is the maximum
relative error on the benchmark's 2000 fixed sample points. Paired differences
are `ChaCha8Rng` minus `StdRng` for the same seed.

| case | RNG | evaluations | sampled rel. error | worst error | iterations | final rank |
|---|---|---|---|---:|---|---|
| `chain_cos_129` | `StdRng` | 40,928 [40,730, 41,290] | 5.56e-9 [5.56e-9, 1.11e-8] | 1.11e-8 | 4 (all) | 2 (all) |
| `chain_cos_129` | `ChaCha8Rng` | 41,065 [40,763, 41,532] | 5.56e-9 [5.56e-9, 1.11e-8] | 1.11e-8 | 4 (all) | 2 (all) |
| `quantics_chain_r20` | `StdRng` | 18,116 [17,862, 18,327] | 1.30e-8 [1.26e-8, 1.57e-8] | 1.89e-8 | 4 (all) | 8 (all) |
| `quantics_chain_r20` | `ChaCha8Rng` | 18,088 [17,872, 18,330] | 1.31e-8 [1.26e-8, 1.57e-8] | 1.82e-8 | 4 (all) | 8 (all) |
| `tree_3x10_plus_centre` | `StdRng` | 50,172 [49,373, 50,850] | 7.33e-9 [6.65e-9, 8.21e-9] | 1.03e-8 | 4 (all) | 9 (all) |
| `tree_3x10_plus_centre` | `ChaCha8Rng` | 50,140 [49,308, 50,979] | 7.40e-9 [7.19e-9, 8.23e-9] | 1.06e-8 | 4 (all) | 9 (all) |

| case | evaluations: mean (median) paired difference | 95% bootstrap CI of the mean | more / fewer evaluations | `log10` error: mean paired difference | higher / lower error |
|---|---|---|---|---:|---|
| `chain_cos_129` | +157 (+127), +0.38% | [-10, +327] | 122 / 77 | -0.017 | 91 / 108 |
| `quantics_chain_r20` | +12 (+27.5), +0.07% | [-55, +80] | 105 / 95 | +0.003 | 103 / 97 |
| `tree_3x10_plus_centre` | +20 (-56.5), +0.04% | [-211, +249] | 98 / 102 | +0.009 | 113 / 87 |

Criterion check: all four conditions hold in every case. No run of either RNG
had a sampled error above `1e-7`, and every run stopped after 4 iterations at
the same final rank. The pivot fingerprint differed between the RNGs for every
seed, so the comparison is between genuinely different trajectories.

Uncertainty notes:

- Overlapping seeds. A run with seed `s` uses search seeds `s` to `s + 3`, so
  neighbouring seeds share three of four search seeds and the 200 runs are not
  independent. The subset `s % 4 == 1` (50 runs with disjoint search seeds)
  gives the same picture. Mean evaluation differences there are +116
  (`chain_cos_129`, 95% bootstrap CI [-297, +536]), +60 ([-86, +208]) and +127
  ([-365, +615]). On this subset, the only paired test with p < 0.05 is the
  `chain_cos_129` error sign test, 16 / 34, p = 0.015, which favours `ChaCha8Rng` by 0.08
  decades.
- Multiple tests. On all 200 seeds, the `chain_cos_129` evaluation sign test
  gives p = 0.0017 (122 / 77), but the paired sign-flip test of the mean gives
  p = 0.066, and the bootstrap interval of the mean includes zero. The effect
  is +0.38% of the median and not significant on the independent subset
  (p = 0.89). The other sign tests and sign-flip tests have p > 0.07. The
  p-values are uncorrected across 3 cases x 2 metrics x 2 subsets.
- Scope. Three deterministic workloads at one tolerance. The comparison
  establishes no detectable quality difference on them, not equivalence for
  every target function.

Verdict: quality-neutral. The switch to `ChaCha8Rng` stays. It changes
fixed-seed trajectories, so recorded pivots and error histories change, but it
showed no detectable change in accuracy or convergence, and no change in
cost beyond a +0.38% evaluation difference on `chain_cos_129` that is not
significant on the independent-seed subset.

Binary SHA-256: `ChaCha8Rng`
`c91ffdd398d16b1cca2b5fa12d2c4974ca6fee316555a7dbb6f81ee3e4d0c58c`, `StdRng`
`ef562c6a2d623c623893d71e1747d47eea4e9b7b1c531e3ebd33e209f8bbe3d6`. The raw
per-run output (1,200 rows) is not committed; the tables above contain every
quantity the criterion uses.
