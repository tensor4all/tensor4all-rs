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
  The recommended order for giving `G0` its own budget, and what steps 1-2 delivered:
  1. a context-taking entry for the LUCI factorization that wraps the existing session-scoped helper with the
     caller's own session, so the core seam stays one function wide - **delivered** (`crossinterpolate2_in`,
     `crossinterpolate2_with_rng_in`, and the context threaded through the optimizer, the per-edge update and
     the factorization; see `docs/design/859-f3-tci-context.md`);
  2. thread that entry, as an option rather than a new requirement, from `crossinterpolate2` into
     `update.rs`'s factorization - **delivered**, with the legacy entry unchanged;
  3. measure what is left of the budget gap, then give the remaining ops on the path context-taking twins
     from the twelve sites - the full-pivot-LU, SVD and QR entries the TCI path actually reaches, not all
     twelve blindly.

  **Where the remaining cost reaches the process-global context, corrected.** An earlier revision of this
  section said the pivot search's readout ends in the bridge's `einsum_native_tensors`. **That was wrong**, and
  a pre-review of the design built on it falsified the claim against the source:

  - the search materializes the current approximation once per search (`treetci/src/globalpivot.rs`, through
    `to_treetn`) and reads it with a `TreeTNCachedEvaluator` (`tensor4all-treetn`);
  - for the homogeneous `f64`/`Complex64` single-physical-index trees `to_treetn` produces - which is what the
    benchmark above measures - the cached evaluator takes **raw message kernels** (`cached_evaluator.rs`),
    slicing host data directly and using `mat_mul_owned` for larger grouped messages; it never reaches
    `TensorLike::contract`, and a fully warm hinted query answers from cached columns with no backend
    contraction at all;
  - where it does take the generic cached path (`f32`/`c32`, heterogeneous trees, multi-index layouts), it
    contracts through `contract_with_options`, which reaches the bridge's `einsum_native_tensor_reads` - a
    different function from the borrowed-tensor entry - and that path also selects physical indices and stacks
    tensors through their eager runtimes;
  - `contract` is an associated function on `TensorContractionLike`, not a method of the blanket-implemented
    `TensorLike`, so the design's proposed override was not even expressible.

  So the readout is not one seam, and its cost is **not yet attributed**. What the inventory does show, with a
  demonstrated caller on this stage's path, is the **final materialization's nonzero full-pivot solve**
  (`treetci/src/materialize.rs`, `T::solve_right_full_piv_lu`) - reached once per pivot search and again for the
  final tree - which goes to the process-global session through
  `tensor4all-tensorbackend/src/backend.rs`'s solve. That is a different entry from the LUCI helper step 3a
  threaded, so it was the next candidate; it has since been threaded and **relocating it showed no clear
  end-to-end benefit** on this workload (`docs/design/859-f3-materialization-context.md` §6). That is a
  placement result, not an attribution: the solve's share and the remaining cost's location are both still
  open, and the next measurement has to accumulate per-component times rather than compare placements.

  **The step-3 measurement says the factorization's budget is not where the gap is.** With the process-global
  context fixed at one worker, giving the factorization its own context makes the stage *slower* rather than
  faster: 22.5-23.9 ms with no caller context against 24.0-25.4 ms with four caller workers, while the
  penalty the global budget itself causes is about 4 ms (25.3-26.7 ms at four global workers). Peak process
  threads do grow with the caller budget (2, 3 and 6 at one, one and four caller workers), so the caller's
  context really is used - and every configuration converges with identical ranks and errors. A same-count
  comparison (four global, four caller) shows no win from the owned factorization either. What remains is the
  evaluator, the global pivot search's readout and the final materialization, which step 3 has to place before
  any further threading: the factorization is a small part of this stage, exactly as §1.1's ablation said.
- **The downstream workaround stays valid.** The child-process isolation the report describes achieves a
  stage-local budget today; nothing here replaces it.

## 2.1 Thread-count precedence, and what a downstream caller should set

The report asks for two things this section answers directly: the thread-count precedence, and recommended
settings. Both are grounded in the entries that exist today, not in proposals.

**The two budgets, and which one an operation takes.**

| Budget | Set by | Applies to |
| --- | --- | --- |
| process-global | `RAYON_NUM_THREADS`, read when the process-global tenferro CPU context first initializes | every operation with no caller context: the legacy `crossinterpolate2`/`optimize_with_proposer` entries, and the ops on this path that still have no context-taking twin (§2's inventory) |
| caller-owned | `CpuExecutionContext::from_backend(CpuBackend::with_threads(n))`, passed to an `_in` entry | only the operations threaded through that context: the per-edge factorization, the optimizer, and the materialization solve (`crossinterpolate2_in`, `optimize_with_proposer_in`, `to_treetn_in`) |

An operation with a context-taking twin takes the **caller's** context and never the global one; an operation
without a twin still takes the global one. The stage's effective parallelism is therefore the *larger* of the
two for its un-threaded remainder, which is why the caller's context bounds the stage only partially today.

**Measured, `sites = 8` and `sites = 14`, `local_dim = 4`, global pivots on, the bench's pinned options
(`seed = 20_261_011`, `tolerance = 1e-8`, `max_iter = 30`, `max_bond_dim = 16`)** (`cargo bench -p
tensor4all-treetci --bench g0_stage_budget`; each row is one process, because the global context fixes its
worker count at first initialization). The `sites = 8`, one-worker row reproduces §3's end-to-end figure
(6.18 ms) within noise, so the two sections measure the same workload:

| configuration | `sites = 8` (3 reps) | `sites = 14` (1 rep) | peak process threads |
| --- | --- | --- | --- |
| global 1 worker | 7.41 / 6.23 / 6.32 ms | 15.25 ms | 2 |
| global 4 workers | 9.16 / 7.72 / 7.67 ms | - | 6 |
| global 64 workers (this machine) | 9.01 / 7.63 / 7.50 ms | 21.14 ms | 66 |
| global 64 workers, caller context at 1 worker | 7.38 / 6.20 / 6.29 ms | 18.25 ms | 67 |
| global pivots **off**, global 1 / 64 workers | 1.80 / 3.30 ms | - | 2 / 66 |

Three things follow, and they are the answer to the report:

1. **More workers make this path slower, not faster** (at `sites = 14`, 21.1 ms at 64 workers against 15.2 ms at
   one, with CPU ticks 7 against 1: more CPU for worse wall time is the over-parallelization signature).
2. **A one-worker caller context recovers the whole gap at `sites = 8`** (6.20-6.29 ms, matching the global
   one-worker run) while the process-global budget stays at 64 for the later `Pi`/`W`/`Sigma` stages - which is
   exactly the independence the report says it needs. At `sites = 14` it recovers only part of it (18.25 ms
   against 15.25 at one worker and 21.14 at 64), because the global pivot search's readout and the final
   materialization still run on the global context; §3's component split places that remainder in the search's
   readout (92% of a search at one worker) and the materialization machinery (0.12 ms per call at one worker,
   doubling at four), which is the attribution §2 asked for.
3. **Peak process threads are not a measure of the stage's own parallelism.** The global pool is created even
   when the stage runs on a caller context (67 peak threads in the two-budget row), so a thread count that
   grows with the global budget does not mean the stage used it.

**Recommended downstream settings.**

- Pass a caller-owned context with **one worker** to the G0 stage (`crossinterpolate2_in` with
  `CpuBackend::with_threads(1)`), and leave the process-global budget alone for the later stages. This is the
  operation-local control the report asks for, and it needs no child process for the threaded share.
- **Do not** set `RAYON_NUM_THREADS=1` process-wide to fix G0: it serializes `Pi`/`W`/`Sigma` too, which the
  report already established.
- Keep the child-process isolation described in the report for the **residual** - the search's readout and the
  final materialization - until §2's attribution is done and those ops have context-taking twins. Today the
  caller context does not cover them, so a process-level budget is still the only bound for that share.
- Treat a stage-local context as a **bound, not a speed-up**: on this workload the caller's factorization
  budget was never the bottleneck (§2's step-3 measurement), so the win comes from *lowering* the budget, not
  from owning the factorization.


## 3. Evidence and limits

- The bench prints the requested and observed worker counts, every repetition's stage time, the resulting
  ranks and the final error, so a run is self-describing and the two configurations are compared on the same
  workload in the same binary.
- Limits: one machine, one tree shape, release build, a cheap integrator, and (for the CPU figures) `USER_HZ`
  at 10 ms granularity. No per-operation or per-worker attribution, no CUDA path, and no downstream
  reproduction.

## 3. Measured: the stage's cost is the search's readout, and the worker penalty is in the session-bound machinery

`crates/tensor4all-treetci/benches/stage_components.rs` (`harness = false`) splits the same G0-shaped stage using
only the public API: it grows a state with the stage's own optimizer (unmeasured fixture), then measures the
caller's oracle directly (ns per point), a materialization of that state (`to_treetn`, its oracle points counted),
a global pivot search of that state at the run's own search budget (`find_global_pivots`), and the end-to-end run
with and without the search. Release build, three repetitions, medians, local AMD EPYC 7713P, 8-site chain,
local dimension 4, `max_bond_dim = 16`, `max_iter = 30`, the state converged to ranks `[2, 2, 2]` in 3 iterations,
default search budget (`nsearch = 5`, `max_nglobal_pivot = 5`):

| component (median) | 1 worker | 4 workers | change |
| --- | ---: | ---: | ---: |
| caller's oracle | 27.4 ns/point | 27.2 ns/point | - |
| materialize the state: total | 0.124 ms | 0.243 ms | +96% |
| - of which machinery (solve + publication + tree) | 0.120 ms | 0.240 ms | +100% |
| - of which the caller's oracle | 0.004 ms | 0.004 ms | - |
| one search at the run's budget: total | 1.787 ms | 1.906 ms | +7% |
| - of which the readout (`total - oracle - materialization`) | 1.657 ms | 1.669 ms | +1% |
| the whole stage, search on | 6.180 ms | 7.485 ms | +21% |
| the whole stage, search off (ablation) | 1.023 ms | 1.974 ms | +93% |

Derived from those numbers (the search's contribution is the difference of the two stage rows, divided by the
per-search figure):

- **The global pivot search is ~83% of the stage at one worker and ~74% at four**: `6.180 - 1.023 = 5.157 ms`
  against `1.787 ms` per search gives **2.9 searches**, i.e. one per non-final sweep of a run that converged in
  3 iterations - the same count at four workers (`5.511 / 1.906 = 2.9`).
- **Within a search the readout dominates**: 92% of it at one worker, 88% at four. Three searches put it at
  ~80% (one worker) / ~67% (four) of the whole stage.
- **The session-bound machinery is small but not negligible**: a materialization is ~0.12 ms of which ~0.004 ms
  is the caller's oracle; with one per search plus the final one, that is ~0.5 ms, ~8% of the one-worker stage
  (~13% at four). The caller's own function is ~1.5% of the stage.
- **The worker penalty is in the machinery, not in the readout.** Doubling the worker count doubles the
  materialization's machinery and doubles the non-search part of the stage (`0.962 -> 1.913 ms`), while the
  readout is flat (`1.657 -> 1.669 ms`). The stage's +21% is the sum of a doubled machinery and the extra work
  each search's materialization now does.

That is consistent with both placement results earlier in this record: threading the factorization (step 3a) or
the materialization solve (step 3d) into a caller-owned context changed nothing end-to-end, and by this
attribution it could not - the penalty follows the work into whichever context performs it, and the work it
follows is ~8-13% of the stage rather than the readout that dominates.

**Caveats, stated because these are differences of measured quantities rather than internal instrumentation:**
the oracle share assumes the caller's per-point cost is constant across stages; the readout figure absorbs
whatever else a search does; the component search and materialization are measured on the converged state while
the run's earlier searches act on smaller states; and this is one machine, one shape and one search budget.
What the numbers establish is the ordering and the worker-count dependence, not an exact time budget.
