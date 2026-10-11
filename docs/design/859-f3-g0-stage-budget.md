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

## 2. What remains open

- **Which mechanism dominates.** This measurement shows that extra workers hurt, but not whether the cost is
  nested parallelism inside the SRC/probe contractions, per-contraction pool wake-ups on tiny operands, or
  the global pivot search's materialization and evaluation. CPU utilization and peak-thread measurements, plus
  a per-stage breakdown (pivot search vs materialization vs evaluation vs factorization), would separate them.
- **Where the budget can be controlled.** The G0 path reaches the process-global context from
  `tensor4all-treetn`'s SRC contraction arms and their helpers (`treetn/contraction.rs` and its
  `contraction/src_chain.rs` and `src_probe.rs` modules call `default_cpu_execution_context()` in many
  places), and `tensor4all-treetci` has no execution-context parameter at all. Giving `G0` its own budget
  therefore needs a control surface threaded from the TCI entry point down to those call sites, or an
  explicitly scoped budget override - not just a documentation change. That work is **not** in this record.
- **The downstream workaround stays valid.** The child-process isolation the report describes achieves a
  stage-local budget today; nothing here replaces it.

## 3. Evidence and limits

- The bench prints the requested and observed worker counts, every repetition's stage time, the resulting
  ranks and the final error, so a run is self-describing and the two configurations are compared on the same
  workload in the same binary.
- Limits: one machine, one tree shape, three repetitions, release build, and a cheap integrator. No CPU
  utilization or peak-thread instrumentation, no CUDA path, and no downstream reproduction.
