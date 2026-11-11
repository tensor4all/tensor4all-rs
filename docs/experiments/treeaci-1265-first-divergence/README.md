# First divergence of the #1265 candidate prototype

The R9/seed-2026 error regression is localized to an adaptive Guard feedback path.
Replacing **only the second Guard proposal batch** swaps the final numerical
trajectory in both directions. Subsequent Guard searches run normally.
This explains how the regression is produced; it does not establish an accepted
algorithm change. The candidate prototype remains **not promoted**.

The fixture PR [#1266](https://github.com/tensor4all/tensor4all-rs/pull/1266) is
merged. Its base `5909b184` and main merge `74935265` have identical source trees.
The rejected algorithm patch is preserved here only as an experimental input.
It is not part of the default dispatch through this record.

## Controlled workload and validity

Use R9, nblock, T=0.01, seed 2026, relative tolerance 1e-4, nonbinding rank cap
4000, default Guard, and 50 nominal sweeps. The fixed validation set contains the
same 5,000 importance samples and 1,094 peak-aware points as the preceding study.
The numerical route uses Complex64 and an explicit CPU session. Builds are
release because the workload is impractically slow without optimization.
There is no performance claim.

Both natural traced runs exactly reproduce their frozen uninstrumented
maximum-rank histories, local-error histories, Guard counts, termination and
final held-out statistics. Both self-replay controls also reproduce those
values exactly. Internal sample IDs are compared through binary physical
coordinates, not allocation-dependent identifiers.

Complete outcomes and caveats are in [results.json](results.json); the case list
is in [protocol.json](protocol.json). The preceding four-case study is retained
in [prior-four-case-results.json](prior-four-case-results.json), with its frozen
[protocol](prior-four-case-protocol.json). All 14 measured cells are retained,
including four incomplete diagnostic runs whose unchanged final accuracy
assertions fail. Diagnostic timings are deliberately excluded from conclusions.

## Earliest causal changes

After Guard 1 the distinct membership differs on cuts 12 and 47. On cut 47,
24→22, baseline's four slots contain only two distinct physical samples:
`[50331648, 16777216, 50331648, 16777216]` in the binary coordinate encoding.
The query-pool prototype holds the distinct samples
`[50331648, 16777216, 33554432]`. This supplies a child needed by a parent.

In sweep 2, update 24, cut 44 (21→22), the first distinct candidate grid and
selected-pivot changes occur:

| Quantity | Baseline | Query-pool prototype |
|---|---:|---:|
| Local matrix shape | 16 × 8 | 16 × 6 |
| Distinct columns | 4 | 6 |
| Selected rank | 4 | 6 |
| Selected cross 2-norm condition | 24.9734 | 50.5620 |
| Local cross reconstruction max residual | 7.11e-15 | 8.88e-15 |

The common columns agree **exactly**. The six singular values of the new matrix
are approximately `[212.86, 78.10, 34.79, 21.03, 11.77, 5.76]`.
Restricting it to the baseline's four distinct columns restores rank four.
The extra directions are far above the unchanged LU threshold; the normalized
Schur maximum after four prototype pivots is 0.384, versus tolerance 1e-4.
The first rank difference therefore reflects additional query information,
not a nearly singular first cross. This does not rule out conditioning problems
in later matrices. See [condition.json](condition.json).

The complete validation error immediately after update 24 is the same to
roundoff (importance L2 9.77808813). The next child-edge update, cut 46, makes it
10.13545524 in baseline and 9.68925119 in the prototype. After sweep 2 it is
9.64149428 versus 9.19469196. These early iterates are inaccurate in both arms;
they are localization measurements, not accepted numerical results.

## Why Guard 2 changes

Actual random starts agree in the first two Guard calls. Two previously known
points remain unresolved in baseline but are repaired in the prototype:

| Grid point (kx, ky, n) | Baseline absolute residual | Prototype absolute residual |
|---|---:|---:|
| (448, 31, 255) | 2.57879050 | 6.10e-14 |
| (448, 63, 256) | 1.21003588 | 2.36e-14 |

The Guard threshold at that call is 0.03183084. Removing these two repaired
points from unresolved evidence is correct. Baseline retains them and offers
three new points; the prototype offers five new points. Its changed approximation
also changes the residual-guided walks from the same random starts.
[guard-residuals.json](guard-residuals.json) retains all starts, known residuals
and thresholds for these two calls.

## Counterfactual replay

| Implementation | Source of Guard proposals | Importance L2 after 50 sweeps |
|---|---|---:|
| Baseline | Natural baseline | 0.0024271927264 |
| Prototype | Natural prototype | 0.0037736277172 |
| Baseline | Full prototype replay | 0.0037736277172 |
| Prototype | Full baseline replay | 0.0024271927264 |
| Baseline | Prototype through Guard 2, then natural | 0.0037736277172 |
| Prototype | Baseline through Guard 2, then natural | 0.0024271927264 |
| Baseline | Prototype through Guard 3, then natural | 0.0037736277172 |
| Prototype | Baseline through Guard 3, then natural | 0.0024271927264 |

All complete runs end at MaxSweeps/50. Every crossed replay also exactly matches
the source's maximum-rank history, local-error history and final validation
statistics. In the full crossed replays, the complete selected-pivot state
matches the source after sweep 4. Guard 1 is identical, so the prefix-two
intervention actually changes only Guard 2. Later Guard calls are natural in
the prefix experiments; their results are not forced to hide late residuals.

Replay overrides the proposed coordinates and therefore the next retained
Guard evidence, not the random generator or the operator. This is a causal
ablation, not a permissible production strategy: replaying known coordinates
would specialize the algorithm to one workload.

## Decisions and limits

The query/output-basis separation is not the cause of an independent numerical
conversion error in these observations. The earlier axis-bound prototype and
the query-pool prototype have identical rank/error/Guard histories, evaluation
counts and held-out statistics for all four preceding R9/R10 seed-42/2026 cases.
Only their diagnostic timings differ. Restoring a useful child changes the
adaptive proposal path, and a single subsequent proposal batch is sufficient
to select the observed final trajectory.

There is no evidence here for changing LU tolerances or imposing rank floors.
The original per-case accuracy gates still fail, including the R9 seed-2026
endpoint increase of 55.47%. Keep the prototype unpromoted and keep Guard search
quality or resolved-point reuse proposals separate from the #1265 closure diff.
Any such proposal needs a new matched study; retaining repaired points as
unresolved failures would be incorrect.

Local Reset/LU updates control residuals on changing local query grids. These
runs do not provide a monotone global-error guarantee. The separate #784
counterexamples already show that positive truncation and finite Guard searches
cannot guarantee a mathematically exact solution. No stopping rule is changed
or reinterpreted by this investigation.

Related diagnostic code was checked: `validate::check_nesting` checks selected
samples against query membership, so it does not by itself certify nesting
against actual selected child pivots when query pools exceed the basis.
The prototype's per-commit regression uses actual selected children instead.
Inspect that diagnostic distinction before extending its use to the new pools.

## Reproduction

The bundle is Linux-oriented (`taskset`) and requires the repository toolchain.
Its numerical fixture derives from the repository's `g0.rs` at `9a0f3861`, with
the documented M6a frequency arithmetic order; the importance sampler is the
same one retained by the preceding #784 investigation. `probe.rs` and `g0.rs`
are frozen experimental inputs. The optional condition analysis uses NumPy's
SVD through its public API; no vendor implementation was inspected.

`reproduce.py` creates disposable worktrees pinned to `5909b184`, applies the
rejected [candidate.patch](candidate.patch) only in the candidate worktree, and
adds private diagnostic hooks there. It preserves binaries, source hashes,
traces and every numerical result. It removes only the disposable worktrees it
created and does not modify the caller's checkout. The output directory must
be new. Specify a reusable Cargo target explicitly to avoid a cold dependency
build; otherwise artifacts go under the experiment output.

```sh
python3 docs/experiments/treeaci-1265-first-divergence/reproduce.py \
  --out-dir target/1265-divergence-reproduction \
  --target-dir target --cpu 14 --focus
```

The full runs validate all 5,000 importance samples and 1,094 peak-aware points.
Per-edge focus traces normally use the first 256 fixed importance points and
all peak-aware points; sweep-2 updates 23–26 use all 5,000 importance points.
The incomplete five-sweep probes may perform a final cleanup pass after Guard
and intentionally fail the unchanged final accuracy assertions. Their failed
exit codes remain in the records; they are not numerical acceptance runs.

Analyze the emitted matrices and compare semantic traces with:

```sh
python3 docs/experiments/treeaci-1265-first-divergence/condition.py \
  target/1265-divergence-reproduction
python3 docs/experiments/treeaci-1265-first-divergence/compare.py \
  baseline-natural candidate-natural target/1265-divergence-reproduction
```

`--build-only` validates both overlays and produces the fixed validation file
without running the long study. `--skip-pass-errors` omits intermediate held-out
readouts while preserving the final validation and all matrix/pivot metadata.
The original full raw traces, failed probes, compiler logs and binary hashes
remain in the task-owned local `target/issue1265/divergence/` directory; they are
not duplicated in this source bundle.
