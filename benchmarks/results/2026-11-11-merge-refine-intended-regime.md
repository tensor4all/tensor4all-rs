# Merge-refine schedule in the intended regime (issue #777 item F4)

Date: 2026-11-11. Checkout: `72f8e7785dfbeb24797419f807171bdd02a4b037` (`origin/main`), plus the
benchmark harness of this PR. The JSON lines were built without
`T4A_BENCH_GIT_COMMIT`, so their `build_commit` field says `unrecorded`; the SHA above is the
checkout the recorded binary was built from, and rebuilding with the variable set only changes that
field.

This page answers item F4 of #777: measure `schedule_merge_refine` against `sum_then_transform` on the
same input in the regime the plan names (20-30 output bits, few input patches, localized input whose
patch transforms are high rank), which the 2026-09-16 3-4 bit run could not reach. The question is
whether the schedule's structural work savings (`2^d` applications, `2^d * t` additions, never a sum
over the whole output domain) show up as runtime, or whether runtime is set by the ranks.

## Input

There is no chirp generator in the tree, so the harness adds one
(`benchmarks/rust/benchmark_merge_refine_intended_regime.rs`, `chirp()`): a Gaussian-windowed
quadratic chirp on `N = 2^R` grid points,

```
w(x) = exp(-((x - N/2) / (N/6))^2) * exp(2*pi*i*chirp_turns*(x/N)^2)
```

with `chirp_turns = 256` for every cell except the rank control below. The window localizes the state
(the two outer quarters of the domain carry `||w|| = 1.07` each, the two inner ones `20.6` each), and
the chirp makes its patch transforms high rank: the dense vector is turned into a chain MPS by a
truncated left-to-right TT-SVD at a relative per-value threshold of `1e-8`, which gives an input bond
dimension of `27` from 12 bits on and a whole-state application rank of `324`. For comparison, the
2026-09-16 runs used coordinate leaves of a smooth/spike/comb state and reached an applied rank of
`8`-`16`, so this input's patch transforms are 20-40x higher rank.

The input partition is the dyadic one of depth `d = 2` (the geometry of finding 4): four input
patches, each fixing the first two of the `R` selected bits. The schedule then runs `2^d = 4`
applications, and level `t = 1..d` merges one input bit while fixing output bit `r_t`; the run stops
after `t = 2` output bits, i.e. it returns four output regions covering `2^(R-2)` coordinates each
(`--output-depth 2`). That is the caller shape the schedule exists for: a few patches in, a coarse
output out, no global output assembly.

## Host and protocol

- AMD EPYC 7713P, `nproc` = 64, 995 GiB RAM, Linux 6.8.0-147-generic, rustc 1.99.1, governor
  `schedutil`, `~` 113 GiB free before and after (the largest single run wrote no files).
- Measured process pinned to one L3 domain, cores `8-15`
  (`/sys/devices/system/cpu/cpu8/cache/index3/shared_cpu_list` = `8-15`), the agent pinned outside it
  to `24-31` (`taskset -apc 24-31 <pi pid>`). `mpstat -P 8,9,10,11,12,13,14,15 1 3` reported 100%
  idle on that domain before the sweep started.
- Threads: `RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
  VECLIB_MAXIMUM_THREADS=8`. `/usr/bin/time -v` reports 199-770% CPU per cell, so the pool really ran
  with the requested workers; the low end (199-231% at 26 bits) is the memory-bound input build, and
  the schedule part of every cell stayed at 600-770%.
- The host is shared and was occupied by other jobs throughout (`keep.py` GPU wallclock recorder, two
  `claude` sessions of other users, the kache daemon); `cat /proc/loadavg` read `5.02 4.77 4.00`
  during the sweep and `1.01 3.47 4.45` after it, and the 8-core domain was 100% idle when the sweep
  started. Repeat-to-repeat variation on the slowest exact cells reached ~25% (see the note under the
  R scan), which is why the page makes no percentage-level claim; no cell was re-run to pick a better
  number.
- One method per process, so `/usr/bin/time -f "wall_s %e cpu_pct %P maxrss_kb %M"` gives that
  method's peak RSS. `repeats = 3` (median of three) for `R <= 16`, `repeats = 1` for `R >= 20`
  because a repeat costs seconds to minutes there; the repeat count is a CSV column.
- The deviation-enabled cells were measured in two passes, because the harness later grew the two L2
  columns that carry the ledger comparison. The tables use the second pass; the two passes differ by
  3-27% from 12 bits up and by 2.5x in the sub-second 8-bit cell. The 26-bit cells and the R scan were
  run once with `--no-deviation`, so the code added between the passes was not executed in them.
- Per-level rows are separate runs at `output_depth = 0..t`, not instrumentation: the schedule has no
  per-level timing hook and the library is unchanged. Each run splits its own global allowance over
  its own level count (`level_share(remaining, levels_left, ..)`), so a shallower run may accept a
  truncation a deeper run rejects. In the exact cells of this page the trajectory performs no
  compression and drops nothing at any depth, so its prefix runs are the same trajectory; in the
  adaptive cells they are not, and the page treats an adaptive `delta` as "the time to ask for one more
  output bit" rather than as a level duration (see the per-level section).
- Commands:

```bash
T4A_BENCH_GIT_COMMIT=$(git rev-parse HEAD) \
  cargo build --release -j8 -p tensor4all-benchmarks --example benchmark_merge_refine_intended_regime

# one cell; run it three times, once per method, so peak RSS is attributable
RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
VECLIB_MAXIMUM_THREADS=8 taskset -c 8-15 timeout 1800 \
/usr/bin/time -f "wall_s %e cpu_pct %P maxrss_kb %M" \
  ./target/release/examples/benchmark_merge_refine_intended_regime \
  --bits 24 --input-depth 2 --output-depth 2 --goal 8 --rtol 1e-6 \
  --chirp-turns 256 --repeats 1 <--baseline-only | --schedule-only --only exact | --schedule-only --only adaptive>

# the single-thread cell: same command with a one-core mask and a one-worker pool
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 taskset -c 8 <same binary and arguments>

python3 benchmarks/rust/collect_merge_refine_intended_regime.py \
  --dir target/f4 --out benchmarks/results/evidence/2026-11-11-merge-refine-intended-regime.csv
```

Raw per-cell CSV: [`evidence/2026-11-11-merge-refine-intended-regime.csv`](./evidence/2026-11-11-merge-refine-intended-regime.csv)
(one row per cell, method, kind and output depth, with that process's wall clock, CPU share and peak
RSS). No run in this page exceeded 30 minutes; the longest was the 24-bit exact cell at `542 s` wall.

## The sweep over R

`d = 2`, `output_depth = 2`, `goal = 8`, `rtol = 1e-6`, `chirp_turns = 256`, so the input rank is 27
from 12 bits on. `ms` is the run's own elapsed time (median of `repeats`), not the process wall clock.
`peak RSS` is that method's process peak, which includes the harness's input construction and the dense
deviation check (see "Limits"); `retained` and `transient` are the report's `max_bond_dim` and
`max_transient_bond_dim`. `--` means the cell ran without a dense reference (`R = 26`).

| R | method | ms | peak RSS GiB | retained rank | transient rank | regions | terms | applications | additions | compressions | error_bound | max deviation | L2 residual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | sum_then_transform | 1.2 | 0.02 | 192 | -- | 1 | 1 | 1 | 0 | 0 | 0.00e+00 | 3.24e-14 | 1.59e-13 |
| 8 | merge_refine_exact | 126.0 | 0.15 | 768 | 768 | 4 | 4 | 4 | 8 | 0 | 0.00e+00 | 3.27e-14 | 1.59e-13 |
| 8 | merge_refine_adaptive | 122.5 | 0.04 | 8 | 384 | 2 | 2 | 4 | 10 | 10 | 4.85e-14 | 3.36e-14 | 1.62e-13 |
| 12 | sum_then_transform | 5.0 | 0.05 | 324 | -- | 1 | 1 | 1 | 0 | 0 | 0.00e+00 | 2.33e-12 | 2.00e-11 |
| 12 | merge_refine_exact | 440.5 | 0.95 | 1296 | 1296 | 4 | 4 | 4 | 8 | 0 | 0.00e+00 | 2.33e-12 | 2.00e-11 |
| 12 | merge_refine_adaptive | 273.6 | 0.15 | 19 | 648 | 2 | 2 | 4 | 10 | 10 | 4.83e-07 | 2.02e-08 | 2.61e-07 |
| 16 | sum_then_transform | 5.3 | 0.09 | 324 | -- | 1 | 1 | 1 | 0 | 0 | 0.00e+00 | 5.36e-12 | 7.57e-11 |
| 16 | merge_refine_exact | 3283.3 | 1.32 | 1296 | 1296 | 4 | 4 | 4 | 8 | 0 | 0.00e+00 | 5.36e-12 | 7.57e-11 |
| 16 | merge_refine_adaptive | 1309.0 | 0.57 | 19 | 648 | 2 | 2 | 4 | 10 | 8 | 1.92e-06 | 7.70e-08 | 1.09e-06 |
| 20 | sum_then_transform | 6.5 | 0.69 | 324 | -- | 1 | 1 | 1 | 0 | 0 | 0.00e+00 | 2.58e-11 | 5.11e-10 |
| 20 | merge_refine_exact | 3804.9 | 2.23 | 1296 | 1296 | 4 | 4 | 4 | 8 | 0 | 0.00e+00 | 2.58e-11 | 5.11e-10 |
| 20 | merge_refine_adaptive | 1531.4 | 0.77 | 19 | 648 | 2 | 2 | 4 | 9 | 7 | 7.78e-06 | 2.56e-07 | 4.64e-06 |
| 22 | sum_then_transform | 4.5 | 1.72 | 324 | -- | 1 | 1 | 1 | 0 | 0 | 0.00e+00 | 5.15e-11 | 1.07e-09 |
| 22 | merge_refine_exact | 4673.9 | 6.63 | 1296 | 1296 | 4 | 4 | 4 | 8 | 0 | 0.00e+00 | 5.15e-11 | 1.07e-09 |
| 22 | merge_refine_adaptive | 1390.0 | 1.95 | 19 | 648 | 2 | 2 | 4 | 9 | 7 | 1.56e-05 | 5.15e-07 | 9.17e-06 |
| 24 | sum_then_transform | 4.5 | 6.38 | 324 | -- | 1 | 1 | 1 | 0 | 0 | 0.00e+00 | 1.03e-10 | 2.88e-09 |
| 24 | merge_refine_exact | 3763.4 | 24.71 | 1296 | 1296 | 4 | 4 | 4 | 8 | 0 | 0.00e+00 | 1.03e-10 | 2.88e-09 |
| 24 | merge_refine_adaptive | 1479.4 | 6.89 | 20 | 648 | 2 | 2 | 4 | 9 | 7 | 2.87e-05 | 7.91e-07 | 1.76e-05 |
| 26 | sum_then_transform | 4.2 | 10.01 | 324 | -- | 1 | 1 | 1 | 0 | 0 | 0.00e+00 | -- | -- |
| 26 | merge_refine_exact | 3574.5 | 10.01 | 1296 | 1296 | 4 | 4 | 4 | 8 | 0 | 0.00e+00 | -- | -- |
| 26 | merge_refine_adaptive | 1574.5 | 10.01 | 20 | 648 | 2 | 2 | 4 | 9 | 7 | 5.68e-05 | -- | -- |

27 bits and beyond were not run: the harness constructs the input from a dense `2^R` vector and then
runs a truncated TT-SVD on it, so every added bit doubles the input construction and the process peak
(each 26-bit cell here: 65 s wall, 10.0 GiB, of which the schedule contributes a few MB of stored
terms). The time budget for this task was spent at 26 bits. The schedule's own times are flat between
20 and 26 bits, so a point at 27-30 would not change the conclusion; no 30-bit number is reported
because none was measured.

Reading the table:

- **The structural counts are flat in R, and that is the whole saving.** At every bit count the
  schedule performs 4 applications, 8 additions (exact) or 9-10 (adaptive, where an operand that does
  not fit the goal stays a separate term), 32-40 projections, and returns 4 regions (exact) or 2
  (adaptive, two of which stopped refining); the whole-state application performs one application over
  `2^R` coordinates. Nothing about that changes between 8 and 26 bits.
- **The runtime is not flat.** Exact 126.0 ms -> 3574.5 ms and adaptive 122.5 ms -> 1574.5 ms while
  the whole-state application stays at 1.2-6.5 ms. The gap is 105x (exact) / 102x (adaptive) at 8
  bits, 88x / 55x at 12, and then 586x-1033x / 236x-374x from 16 bits on. It does not close, and the
  jump between 12 and 16 bits is much larger than the extra sites explain (see the scan below).
- **The retained rank explains the exact/adaptive gap, not the operation counts.** The exact
  trajectory is rank-additive: each level doubles the merged rank, 324 -> 648 -> 1296, and the
  compression probes never fire (`compressions == 0`). The adaptive trajectory truncates at level 1
  (`compressions = 7`, retained rank 19-20, transient 648) and is 2.5x faster than the exact one at
  24 bits with 2 terms in 2 regions instead of 4 in 4. Truncation buys time by lowering the rank; it
  does not lower the count of applications, additions or projections.
- **Per bit, the exact trajectory's cost is far above linear between 8 and 16 bits.** The scan below
  holds `d`, `t`, the input and the reported ranks fixed and only moves `R`:

| R | exact ms | retained rank | transient rank | RSS GiB |
|---|---|---|---|---|
| 8 | 57.0 | 768 | 768 | 0.09 |
| 10 | 273.3 | 1248 | 1248 | 0.38 |
| 11 | 381.2 | 1296 | 1296 | 0.47 |
| 12 | 473.0 | 1296 | 1296 | 0.53 |
| 13 | 639.9 | 1296 | 1296 | 0.60 |
| 14 | 1061.8 | 1296 | 1296 | 0.66 |
| 15 | 2067.7 | 1296 | 1296 | 0.73 |
| 16 | 3279.4 | 1296 | 1296 | 0.87 |
| 18 | 3598.6 | 1296 | 1296 | 0.96 |
| 20 | 3600.4 | 1296 | 1296 | 1.08 |

  (cells `rscan08`-`rscan20`, `--no-deviation`, `repeats = 1`, same command as above with
  `--only exact`). The reported `max_bond_dim` and `max_transient_bond_dim` are the same 1296 from 11
  to 20 bits (768 at 8, 1248 at 10), and the counts are identical too (4 applications, 8 additions,
  same regions and terms), yet the time changes 9.4x. So neither statistic the report exposes explains
  the growth between 8 and 16 bits: something per item grows with the coordinates in a region up to 16
  bits and then saturates, while from 16 bits on the level-2 cost (next section, 3.0 s) is the
  rank-1296 work. Both ranks above are maxima, not full rank profiles, and
  `schedule.rs::drop_negligible` already canonicalizes through `SubDomainTreeTN::norm`, so this page
  records the shape and does not attribute it; the probe accounting of finding 5 is where to look.

Two of these cells move by up to 25% between runs on this host (the 22-bit exact cell measured 3675 ms
in an earlier pass and 4673.9 ms here, its neighbours at 20 and 24 bits both ~3.6-3.8 s). The
observations above are differences of a factor 2-10, not percentages; no percentage-level claim is
made from this page.

## Per level at the largest R with a dense reference (24 bits)

`--output-depth 2`, so each row is a run stopped after that many output bits; `cumulative` is that
run's median time and `delta` its difference from the previous row's (cell `r24`).

`delta` is a level duration only for the exact trajectory, which performs no compression
(`compressions = 0`) and drops nothing at any depth here, so its three runs are the same trajectory.
In the adaptive rows each depth is an independent run whose own level count splits the global
allowance (`level_share(remaining, levels_left, ..)`), so a shallower run can truncate differently
from a deeper one; its `delta` can even be negative (`-94 ms` at 16 bits, `-45 ms` at 22 bits, and
`+68 ms` at 26 bits for the same level). Read the adaptive column as "time to ask for this many output
bits", not as a level duration.

| depth | method | cumulative ms | delta ms | retained rank | transient rank | regions | terms | additions | compressions | dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | merge_refine_exact | 16.5 | 16.5 | 324 | -- | 1 | 4 | 0 | 0 | 0 |
| 1 | merge_refine_exact | 729.0 | 712.5 | 648 | 648 | 2 | 4 | 4 | 0 | 0 |
| 2 | merge_refine_exact | 3763.4 | 3034.4 | 1296 | 1296 | 4 | 4 | 8 | 0 | 0 |
| 0 | merge_refine_adaptive | 16.2 | 16.2 | 324 | -- | 1 | 4 | 0 | 0 | 0 |
| 1 | merge_refine_adaptive | 1472.9 | 1456.7 | 17 | 648 | 2 | 4 | 4 | 4 | 0 |
| 2 | merge_refine_adaptive | 1479.4 | 6.5 | 20 | 648 | 2 | 2 | 9 | 7 | 2 |
| -- | sum_then_transform | 4.5 | 4.5 | 324 | -- | 1 | 1 | 0 | 0 | 0 |

Level 0 (`16 ms`, four complete-transform applications) is the only level whose cost is proportional
to the structural work; it is also the only one the whole-state alternative does not have to pay. The
two merging levels cost 0.7 s and 3.0 s for four and eight additions respectively, because each merge
carries a rank-648 or rank-1296 item through the probe. The adaptive trajectory spends its time in the
same place: asking for two output bits instead of one leaves the time unchanged (1472.9 -> 1479.4 ms
at 24 bits; the difference is negative at some sizes), and asking for one bit instead of zero costs the
whole 1.46 s. Nearly the entire adaptive cost is the truncating probe of the single merging level,
i.e. exactly the cost finding 5 is about.

## Controls

Back at 16 bits, `d = 2`, `t = 2` unless the row says otherwise. Same `goal = 8`, `rtol = 1e-6`; the
input rank is set by `--chirp-turns` through the input TT-SVD threshold.

| cell | sum_then_transform ms | exact ms | adaptive ms | exact retained / transient | adaptive retained / transient | applications | additions (exact / adaptive) |
|---|---|---|---|---|---|---|---|
| rank `turns = 16` (input rank 10) | 1.4 | 401.7 | 298.2 | 480 / 480 | 8 / 240 | 4 | 8 / 9 |
| rank `turns = 256` (input rank 27) | 5.3 | 3283.3 | 1309.0 | 1296 / 1296 | 19 / 648 | 4 | 8 / 10 |
| rank `turns = 2048` (input rank 64) | 20.5 | 8539.6 | 4895.4 | 3072 / 3072 | 64 / 1536 | 4 | 8 / 9 |
| rank `turns = 8192` (input rank 128) | 166.8 | 17577.4 | 10055.6 | 6144 / 6144 | 128 / 3072 | 4 | 8 / 10 |
| input depth `d = 1` (2 patches) | 4.7 | 245.2 | 651.9 | 648 / 648 | 19 / 648 | 2 | 2 / 2 |
| input depth `d = 2` (4 patches) | 5.3 | 3283.3 | 1309.0 | 1296 / 1296 | 19 / 648 | 4 | 8 / 10 |
| input depth `d = 3` (8 patches) | 5.4 | 6352.8 | 2508.1 | 1296 / 1296 | 17 / 648 | 8 | 16 / 18 |
| output depth `t = 1` | 5.6 | 518.9 | 1330.4 | 648 / 648 | 17 / 648 | 4 | 4 / 4 |
| output depth `t = 2` | 5.3 | 3283.3 | 1309.0 | 1296 / 1296 | 19 / 648 | 4 | 8 / 10 |
| output depth `t = 3` | 5.5 | 2795.0 | 1277.1 | 1296 / 1296 | 20 / 648 | 4 | 8 / 9 |

- **Input rank control.** Every method's time moves with the rank and nothing else changes: from input
  rank 10 to 64 the exact trajectory goes 401.7 -> 8539.6 ms (21x) and the whole-state application
  1.4 -> 20.5 ms (15x); from rank 27 to 128 the schedule goes 3283.3 -> 17577.4 ms (5.4x exact) and
  the baseline 5.3 -> 166.8 ms (31x). Because the baseline grows faster with rank, the gap at 16 bits
  narrows from 619x / 247x (rank 27) to 105x / 60x (rank 128). The `turns = 8192` row is the
  production-like arm - input rank 128, near the `chi ~ 200` the shared rules call realistic - and it
  is the arm closest to a regime where the comparison could flip; even there the schedule is 60x
  slower for the same structure.
- **Input-depth control.** Doubling the patch count doubles the schedule's applications (2/4/8) and
  moves its exact time 245.2 -> 3283.3 -> 6352.8 ms: the 13.4x step from `d = 1` to `d = 2` also
  doubles the merged rank (648 -> 1296), while `d = 2` -> `d = 3` doubles only the patch count and
  costs 1.9x. This is the one place where a structural count and the time move together.
- **Output-depth control.** The adaptive time is flat in `t` (1330 -> 1309 -> 1277 ms) because its
  single truncating probe dominates; the exact time jumps from 518.9 ms at `t = 1` to 3.3 / 2.8 s at
  `t = 2` / `t = 3` and then barely moves. Neither shape closes the gap to 5.3 ms.

## Error ledger against a measured error

The reference is the dense normalized DFT of the *represented* input, computed in the operator's
documented convention (selected bits most significant first, frequency bit `t` on selected position
`t`, i.e. the bit-reversed transform), with a radix-2 FFT in the harness. Two checks that the
convention and the reference are right:

1. The whole-state application, densified and reordered, deviates from that reference by an L2 of
   `2.0e-11` at 12 bits, `5.1e-10` at 20 bits and `2.9e-09` at 24 bits (the `sum_then_transform` rows
   above) - roundoff, since both are the same operator applied to the same state.
2. The input itself: the TT-SVD reconstruction deviates from the analytic chirp by at most `3.1e-08`
   at 12-26 bits, and its norm by `1e-15` relative.

`error_bound` is an L2 bound over the computed regions, so the deviation to compare with it is the L2
residual, not the maximum coordinate error (which is smaller for the same vector and is reported
separately above). The residual also contains the roundoff floor of the comparison itself - the
operator application plus the FFT reference - which the bound excludes by design; the `roundoff floor` column is the L2 residual of the exact run of the same
cell, i.e. what this comparison reports when nothing is truncated.

| R | cell | error_bound | L2 residual (adaptive) | relative L2 | roundoff floor (exact L2) | bound / residual |
|---|---|---|---|---|---|---|
| 8 | r08 | 4.85e-14 | 1.62e-13 | 2.2e-14 | 1.59e-13 | 0.3 |
| 12 | r12 | 4.83e-07 | 2.61e-07 | 8.9e-09 | 2.00e-11 | 1.9 |
| 16 | r16 | 1.92e-06 | 1.09e-06 | 9.4e-09 | 7.57e-11 | 1.8 |
| 20 | r20 | 7.78e-06 | 4.64e-06 | 9.9e-09 | 5.11e-10 | 1.7 |
| 22 | r22 | 1.56e-05 | 9.17e-06 | 9.8e-09 | 1.07e-09 | 1.7 |
| 24 | r24 | 2.87e-05 | 1.76e-05 | 9.4e-09 | 2.88e-09 | 1.6 |

The ledger holds and it is tight, not generous: from 12 bits up the bound covers the L2 residual by
1.6-1.9x, and the relative residual is the requested `rtol = 1e-6` times the preimage scale, i.e. a
constant `~9e-9` of `||w||` at every size. At 8 bits the bound (`4.85e-14`) is *below* the raw
residual (`1.62e-13`) because the roundoff floor of the comparison (`1.59e-13`) alone exceeds it;
removing that floor in quadrature leaves `3e-14`, which the bound covers. In the exact cells
`error_bound` is exactly zero and the residual equals the floor, as it should. Both rows and the
floor grow with R because `rtol` is relative to the preimage scale and `||w||` grows like `2^(R/2)`
for this windowed input. The input's own TT-SVD error (`3e-8`) is not part of the ledger, as expected:
the bound covers the schedule's truncation, not the caller's input representation.

## Single-thread cell

One cell repeats the 16-bit point with `RAYON_NUM_THREADS=1` and one pinned core (`taskset -c 8`), so
the pool size is visible in the CPU share:

| method | 1T ms | peak RSS GiB | CPU share |
|---|---|---|---|
| sum_then_transform | 6.6 | 0.09 | 99% |
| merge_refine_exact | 14309.3 | 1.31 | 99% |
| merge_refine_adaptive | 5417.4 | 0.56 | 99% |

Everything scales by 4.1-4.4x against the 8T cell above and no ordering changes, so the 8T numbers are
not a threading artifact; the schedule's disadvantage grows to 2170x (exact) / 820x (adaptive) at 1T,
which tells the same story as the rank control: its cost sits in work that does not parallelize as
well as one operator application.

## What these numbers support

**The rank side predicts the runtime; the structural side does not.**

- "Structural work savings" predicts that a schedule doing 4 applications and 8-10 additions instead
  of one application over `2^R` coordinates, with no global output sum, should be cheaper. It is not,
  at any tested R: 105x slower at 8 bits, 850x at 26 bits (exact), 102x and 374x for the adaptive
  runs. The saving in counts is real and R-independent; it has no runtime effect, because the
  alternative is one application over the same object, which costs `O(R * chi^2 * chi_F^2)` and grows
  only linearly in R.
- "Rank-dependent runtime" predicts the times that were measured. The exact trajectory's rank doubles
  per level (324 -> 648 -> 1296), the adaptive one pays one rank-648 -> 17 truncation probe and then
  runs cheaply, and both time and gap follow the input rank: from input rank 10 to 128 the schedule
  and the baseline move in the same direction while the counts stay identical, and the gap narrows
  from 619x / 247x to 105x / 60x at 16 bits. A structural account predicts no such dependence at all.
- **Where the two cannot be separated:** the 8-16 bit growth is not explained by the counts (identical)
  nor by the reported rank statistics (identical maxima from 11 bits on), so the cost sits in work the
  report does not expose. `schedule.rs::drop_negligible` canonicalizes through
  `SubDomainTreeTN::norm`, and the compression probe canonicalizes again; locating the term needs the
  accounting of finding 5. This page records its shape (it doubles per added bit up to ~16 bits, then
  saturates) without assigning a cause.
- **What this does not decide:** the baseline here applies the operator once to the whole state, which
  is the cheapest possible reference for an input that is itself a single rank-27 chain. The
  schedule's own premise is a patched representation whose *assembly* is the expensive part: the
  alternative there is to apply the operator per patch and then build and keep the sum, which pays the
  same four applications the schedule pays, and the schedule's endorsement rests on that assembly
  being the larger cost. That baseline is not measured here, so whether the schedule wins against it
  remains open. What is decided is that it does not win against the whole-state application at 8-26
  bits for an input rank of 10-128, and that its cost is rank-driven rather than structural. The
  2026-09-16 finding therefore stands in the intended regime.

## Limits

- One machine, one shared host with other tenants' jobs running (recorded above), one input family,
  one window width, one rank goal (`8`) and one tolerance (`1e-6`). Repeat-to-repeat variation reached
  ~25% on the slowest exact cells.
- The RSS column is the *process* peak and includes the harness's dense `2^R` reference and its dense
  comparison against all retained terms; at 24 bits that check alone dominates the 24.7 GiB of the
  exact cell (a dense `to_dense` of a rank-1296 24-site tree materializes bond x 2 x `2^24`
  intermediates). The schedule's own stored representation is the `logical_parameters` column of the
  CSV: `6e2`-`4.6e4` complex entries for the adaptive runs and `2.9e7`-`4.9e8` for the exact ones,
  against `~5.0e6`-`5.6e6` for the whole-state result. The 26-bit cells (`--no-deviation`) all peak at
  10.0 GiB, which is the harness's dense input construction, not the schedule.
- The input rank reaches 27 in the R sweep because the sweep's linear cost is what makes 26 bits
  affordable; the rank control reaches 128 at 16 bits. The shared rules ask benchmark workloads to sit
  near `chi ~ 200`, and this chirp/window family saturates at 142 even at `turns = 16384`, so the
  closest cell to that target is the `turns = 8192` row. A rank-200 input at 20+ bits is not measured.
- `output_depth = 2` is one point in the caller's space (4 output regions). Larger depths raise the
  exact trajectory's cost per level and leave the adaptive trajectory's single probe unchanged, so
  they do not change the sign of the result; they are not measured beyond `t = 3`.
