# Merge-refine schedule measurements (issue #752)

Date: 2026-09-16. Build: `fd244019fab9853bb593fa426ed4b87fa3f46c70` (recorded in the JSON lines via
`T4A_BENCH_GIT_COMMIT`). Machine: single workstation, CPU affinity pinned to core 2,
Rayon/OMP/OpenBLAS/MKL thread counts forced to one. Command:

```bash
T4A_BENCH_GIT_COMMIT=$(git rev-parse HEAD) \
  cargo build --release -p tensor4all-partitionedtreetn --example benchmark_merge_refine
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 2 ./target/release/examples/benchmark_merge_refine \
  > benchmarks/results/2026-09-16-merge-refine-schedule.jsonl
```

Raw output: [`2026-09-16-merge-refine-schedule.jsonl`](./2026-09-16-merge-refine-schedule.jsonl).
Every elapsed time is the median of three runs. `rtol = 1e-6` and
`target_bond_dim = 8` are held fixed across every approximate method.
`deviation` is the maximum absolute difference from the dense exact merge-refine
trajectory, which the crate tests establish as the normalized DFT; `bound` is the
measured `error_bound` that method reported. The input is a chain MPS whose ranks
are maximal for its bit count, split into its `2^bits` dyadic coordinate leaves.

## Measurements

| bits | family | method | ms | retained rank | transient rank | regions | terms | additions | compressions | bound | deviation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | smooth | sum_then_transform | 0.18 | 8 | - | - | - | - | - | - | 1.26e-15 |
| 3 | smooth | merge_refine_exact | 12.71 | 64 | 64 | 8 | 8 | 24 | 0 | 0.00e+00 | 0.00e+00 |
| 3 | smooth | merge_refine_adaptive | 20.61 | 8 | 16 | 1 | 8 | 8 | 8 | 0.00e+00 | 9.93e-16 |
| 3 | smooth | merge_refine_adaptive_truncated_apply | 26.40 | 1 | 2 | 1 | 8 | 8 | 0 | 1.09e-14 | 3.58e-15 |
| 3 | smooth | greedy_reconstruct | 23.77 | 1 | - | 1 | 8 | - | - | 1.74e-14 | 2.81e-15 |
| 3 | spike | sum_then_transform | 0.15 | 8 | - | - | - | - | - | - | 0.00e+00 |
| 3 | spike | merge_refine_exact | 10.48 | 64 | 64 | 8 | 8 | 24 | 0 | 0.00e+00 | 0.00e+00 |
| 3 | spike | merge_refine_adaptive | 19.29 | 8 | 16 | 1 | 8 | 8 | 8 | 0.00e+00 | 0.00e+00 |
| 3 | spike | merge_refine_adaptive_truncated_apply | 24.50 | 1 | 2 | 1 | 8 | 8 | 0 | 7.04e-16 | 4.75e-16 |
| 3 | spike | greedy_reconstruct | 5.19 | 1 | - | 1 | 1 | - | - | 1.65e-15 | 7.35e-16 |
| 3 | comb | sum_then_transform | 0.15 | 8 | - | - | - | - | - | - | 6.21e-17 |
| 3 | comb | merge_refine_exact | 12.38 | 64 | 64 | 8 | 8 | 24 | 0 | 0.00e+00 | 0.00e+00 |
| 3 | comb | merge_refine_adaptive | 18.17 | 8 | 16 | 1 | 8 | 8 | 8 | 0.00e+00 | 6.89e-17 |
| 3 | comb | merge_refine_adaptive_truncated_apply | 26.04 | 1 | 2 | 1 | 8 | 8 | 0 | 1.87e-15 | 9.56e-16 |
| 3 | comb | greedy_reconstruct | 10.14 | 1 | - | 1 | 1 | - | - | 4.26e-15 | 1.43e-15 |
| 4 | smooth | sum_then_transform | 0.25 | 16 | - | - | - | - | - | - | 1.23e-15 |
| 4 | smooth | merge_refine_exact | 362.02 | 256 | 256 | 16 | 16 | 64 | 0 | 0.00e+00 | 0.00e+00 |
| 4 | smooth | merge_refine_adaptive | 64.80 | 2 | 32 | 2 | 16 | 32 | 16 | 2.61e-14 | 3.58e-15 |
| 4 | smooth | merge_refine_adaptive_truncated_apply | 86.49 | 1 | 2 | 1 | 16 | 16 | 0 | 3.04e-14 | 6.20e-15 |
| 4 | smooth | greedy_reconstruct | 79.66 | 1 | - | 1 | 16 | - | - | 4.89e-14 | 6.62e-15 |
| 4 | spike | sum_then_transform | 0.23 | 16 | - | - | - | - | - | - | 0.00e+00 |
| 4 | spike | merge_refine_exact | 334.92 | 256 | 256 | 16 | 16 | 64 | 0 | 0.00e+00 | 0.00e+00 |
| 4 | spike | merge_refine_adaptive | 66.71 | 1 | 32 | 2 | 16 | 32 | 16 | 1.53e-15 | 3.92e-16 |
| 4 | spike | merge_refine_adaptive_truncated_apply | 77.22 | 1 | 2 | 1 | 16 | 16 | 0 | 1.60e-15 | 6.94e-16 |
| 4 | spike | greedy_reconstruct | 14.58 | 1 | - | 1 | 1 | - | - | 1.41e-15 | 4.91e-16 |
| 4 | comb | sum_then_transform | 0.24 | 16 | - | - | - | - | - | - | 2.23e-16 |
| 4 | comb | merge_refine_exact | 334.92 | 256 | 256 | 16 | 16 | 64 | 0 | 0.00e+00 | 0.00e+00 |
| 4 | comb | merge_refine_adaptive | 68.21 | 1 | 32 | 2 | 16 | 32 | 16 | 5.15e-15 | 1.09e-15 |
| 4 | comb | merge_refine_adaptive_truncated_apply | 77.70 | 1 | 2 | 1 | 16 | 16 | 0 | 4.44e-15 | 1.60e-15 |
| 4 | comb | greedy_reconstruct | 27.63 | 1 | - | 1 | 4 | - | - | 5.66e-15 | 7.86e-16 |
## What these numbers show

- **Structural work matches the design exactly.** The exact trajectories perform
  `additions = 2^bits * bits` (24 at three bits, 64 at four) with
  `applied_operator_count = 2^bits`, and hold `2^bits` items at every level. The
  adaptive and truncated-apply runs stop refining once the rank goal is met, so they
  perform only `additions = 2^bits` (8 at three bits, 16 at four) with
  `stopped_regions > 0`.
- **Runtime is rank-driven, and the operation counts do not predict its sign.** At
  three bits the adaptive runs perform three times fewer additions (8 versus 24) but
  are *slower* than the exact trajectory (18-26 ms versus 10-13 ms), because the
  extra truncation work outweighs the smaller retained ranks at that size. At four
  bits the same policy performs four times fewer additions (16 versus 64) and is
  about five times *faster* (65-87 ms versus 335-362 ms), where the exact
  trajectory's retained rank reaches `256` and the adaptive runs stay at or below the
  goal (`1`-`2`). The retained rank, not the operation count, explains the timing.
- **Accuracy is unaffected by the rank policy.** Every method's deviation from the
  exact trajectory is at roundoff level (`<= 6.7e-15` here) and every reported bound
  covers it. The truncated-apply runs additionally charge their measured application
  error, which raises their bound without changing their deviation.
- **The whole-state application baseline is far cheaper on these inputs.** Applying
  the complete transform to the whole input at once takes `0.15-0.25 ms` and keeps
  the output rank at `8`/`16`, because a three- or four-bit input chain MPS is a
  small object. On this input family the schedule is not a runtime win: its purpose
  is to avoid first assembling a global output sum when the input is a *patched*
  representation whose patch transforms have high rank.

## Limits of this evidence

- Two bit counts, three deterministic input families, one machine, one thread, three
  repeats. These are single-machine observations limited to the tested cases; no
  constant-rank or speedup claim is made for arbitrary data.
- The dyadic-leaf contract of this iteration accepts only coordinate-restricted
  input leaves, which are low-rank for the tested states. The intended regime - a
  patched QTT whose patch transforms are high-rank objects - needs the nonuniform
  geometry that remains follow-up work, so no comparison against that regime is
  offered here.
- The greedy engine is measured on the same subset-operator target and the same
  rank goal and tolerance. It is competitive on these small cases because the whole
  output domain is small; the schedule's advantage is expected in the regime above.
- Operator-construction error (for example `FourierOptions::tolerance`) is outside
  every reported bound; the bounds cover application deviation and scheduling
  compression only.
