# F3 / #670: the G0 stage's worker-count diagnosis

Package F3 of [tensor4all-rs#859](https://github.com/tensor4all/tensor4all-rs/issues/859), answering
[#670](https://github.com/tensor4all/tensor4all-rs/issues/670). The report from the downstream SGW pipeline
(`lingrui96/gw-rs`) is a *symptom*: the `G0` stage - `crossinterpolate2` with global pivots on a small tree -
became slower on a faster machine, and the report says explicitly that the dominant cost is **not**
established: nested parallelism, over-parallelization of small contractions, or missing reuse. It also notes
that `RAYON_NUM_THREADS=1` is not a usable fix, because it serializes the later `Pi`/`W`/`Sigma` stages too,
and that the stages need independently controllable budgets.

This record delivers the first thing that report needs - a controlled measurement of whether the G0-shaped
path benefits from more workers - and states what remains open.

## 1. Measured: more workers make this path slower

`crates/tensor4all-treetci/benches/g0_stage_budget.rs` (`harness = false`), run once per worker count because
the process-global context takes its count from `RAYON_NUM_THREADS` when it first initializes. The workload is
the G0 shape: a linear chain of 8 sites, local dimension 4, a tensor-product integrator (`1 + sum of per-site
table values`), `max_bond_dim = 16`, `max_iter = 30`, **global pivots enabled**, three repetitions per run,
release build, on the local AMD EPYC 7713P:

| workers (env / observed) | repetition 1 | repetition 2 |
| ---: | ---: | ---: |
| 1 / 1 | 5.31 ms | 5.69 ms |
| 4 / 4 | 7.98 ms | 7.68 ms |
| 64 / 64 | 8.06 ms | 7.58 ms |

- **The path is about 40% slower with four or more workers than with one**, and adding more beyond four does
  not change that. The ranks (`[2, 2, 2]`) and the final error (`1.48e-16`) are identical in every run, so
  this is a scheduling effect, not a numerical one.
- The observed worker counts match the requested ones, so the comparison rests on a measured schedule rather
  than on the environment variable alone.
- This is a `G0`-*shaped* workload in this repository, not the downstream `R=10` run: the report's absolute
  times come from a different model, tree size and oracle cost. What it establishes is the direction - this
  path over-parallelizes - which is one of the three mechanisms the report left open.

## 1.1 Measured: an ablation of the global pivot search, with the process's own thread and CPU figures

The same bench now also reports, per repetition, the peak number of process threads sampled while the stage ran
and the process's own CPU ticks across the call, and it can run the stage with the global pivot search disabled
(`T4A_G0_GLOBAL_PIVOTS=0`). The seed is pinned, so the two configurations differ in the search and not in the
random stream. 16 sites, local dimension 4, three repetitions per run:

| configuration | elapsed | peak process threads | CPU ticks (10 ms) |
| --- | ---: | ---: | ---: |
| global pivots on, 1 worker | 23.5 / 19.0 / 19.2 ms | 2 | 2 |
| global pivots on, 4 workers | 25.2 / 25.2 / 25.4 ms | 6 | 3 / 4 / 5 |
| global pivots on, 64 workers | 27.5 / 25.8 / 26.3 ms | 66 | 7 / 9 / 10 |
| global pivots off, 1 worker | 3.47 / 2.52 / 2.51 ms | 2 | 0 (below tick resolution) |
| global pivots off, 4 workers | 5.96 / 4.90 / 4.72 ms | 6 | 1 / 2 / 0 |

- **Disabling the search removes about 85-90% of this stage's end-to-end time at one worker** (19-24 ms to
  2.5-3.5 ms) and about 77-81% at four workers. This is an *ablation*, not a measurement of the search alone:
  disabling it also removes any pivots it would have accepted, so the remaining time covers initialization,
  candidate generation, oracle evaluation, LUCI factorization and assembly, and the final materialization -
  not only "sweep contractions", which an earlier revision of this record said.
- **The ablation did the same sweep work in both configurations**: every run reports `Converged` with the same
  complete ranks history (`[2, 2, 2]`) and the same error history, and the ranks history has one entry per
  iteration, so the iteration counts match. That does not establish identical per-edge ranks, pivot sets or
  candidate work, and the search itself draws random starts, so it is a strong but not complete check.
- **Larger worker budgets were slower in every displayed repetition**, in both configurations (19-24 ms to
  25-28 ms with the search; 2.5-3.5 ms to 4.7-6.0 ms without it), so the effect is not confined to the search.
- **The extra CPU is not buying wall time**: the process's own CPU ticks rise from 2 to 7-10 per 20-27 ms run
  as the budget grows, while the wall time rises too, and most of a 64-core machine stays idle. That is
  *consistent with* scheduling and pool overhead - wake-ups, spin, and repeated pool entry on tiny operands -
  rather than useful parallelism; it does not attribute the cost to work inside any particular operation, and
  process-wide totals cannot say which workers did what. Nested tasks can also reuse one pool, so the peak
  thread count does not distinguish nesting from repeated entry.
- The CPU figures come from `USER_HZ` ticks (10 ms granularity) and are reported as raw tick deltas next to the
  conversion, with a ratio only when the delta can support one; the ~3 ms runs are reported as below that
  resolution rather than as zero. The thread count is a sampled maximum over an interval that approximately
  covers the call, and the sampler's own thread is included.
- The 16-site `G0` shape is still small. Whether a downstream `R=10` run with a more expensive oracle behaves
  the same **may differ, and is not measured here**.

## 2. What remains open

- **Which mechanism dominates, now narrowed but not settled.** §1.1 shows the global pivot search is ~90% of
  the stage and that both it and the sweep contractions get slower with more workers, while CPU stays mostly
  idle. That is consistent with pool wake-up and spin overhead on tiny operands rather than with useful
  parallelism. It does not yet separate nested parallelism inside the SRC/probe contractions from
  per-contraction pool wake-ups, nor does it break the search into materialization versus evaluation.
- **Where the budget can be controlled, with the inventory.** `tensor4all-treetci` has no execution-context
  parameter at all (`src/api.rs`'s entry points take options, pivots, a graph and a proposer), and its
  per-sweep factorization goes through `matrix_luci_factors_from_matrix` in
  `tensor4all-treetci/src/update.rs:63`. That helper lives in `tensor4all-core`, which already offers
  session-scoped variants (`matrix_luci_factors_from_matrix_owned_in`), but the operations it calls reach the
  process-global context through `tensor4all-tensorbackend/src/backend.rs`: **twelve production
  `with_default_session` call sites** in that file (lines 228, 615, 636, 692, 743, 1054, 1107, 1153, 1206, 1245,
  1392), whose worker count is the process-global context's. An earlier revision of this record pointed at the
  TreeTN SRC contraction helpers instead, and that was wrong: the `default_cpu_execution_context()`
  occurrences in `treetn/contraction/src_chain.rs` and `src_probe.rs` are inside their `#[cfg(test)]` modules,
  and the production SRC path already has a context-scoped entry (`contract_src_with_rng_in`).
  The recommended order for giving `G0` its own budget, none of it in this record:
  1. a context-taking entry for the LUCI factorization that wraps the existing session-scoped helper with the
     caller's own session, so the core seam stays one function wide;
  2. thread that entry, as an option rather than a new requirement, from `crossinterpolate2` into
     `update.rs`'s factorization;
  3. measure what is left of the budget gap, then give the remaining ops on the path context-taking twins
     from the twelve sites - the full-pivot-LU, SVD and QR entries the TCI path actually reaches, not all
     twelve blindly.
- **The downstream workaround stays valid.** The child-process isolation the report describes achieves a
  stage-local budget today; nothing here replaces it.

## 3. Evidence and limits

- The bench prints the requested and observed worker counts, every repetition's stage time, the resulting
  ranks and the final error, so a run is self-describing and the two configurations are compared on the same
  workload in the same binary.
- Limits: one machine, one tree shape, release build, a cheap integrator, and (for the CPU figures) `USER_HZ`
  at 10 ms granularity. No per-operation or per-worker attribution, no CUDA path, and no downstream
  reproduction.
