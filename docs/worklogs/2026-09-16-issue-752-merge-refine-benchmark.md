# Issue #752: merge-refine schedule measurements

Session: 2026-09-16 (fifth batch). Base: `fd244019` (squash of #758). Branch:
`feat/752-merge-refine-benchmark`. Covers the plan's step 6 evidence for the
scheduling work: "benchmark evidence distinguishes structural work savings from
rank-dependent runtime".

## What was added

- `benchmarks/rust/benchmark_merge_refine.rs`, included by a thin
  `crates/tensor4all-partitionedtreetn/examples/benchmark_merge_refine.rs`, as the
  repository benchmark layout requires. It measures, for three-bit and four-bit
  inputs on three deterministic families (smooth, spike, comb):
  the exact merge-refine trajectory, the adaptive trajectory at a rank goal, the
  adaptive trajectory with a truncating application, the greedy reconstruction on
  the same subset-operator target, and applying the complete transform to the whole
  input at once.
- Each row records the schedule's structural counters, retained and transient
  ranks, stored parameters, median elapsed time over three runs, the reported
  bound, and the deviation from the dense exact trajectory, as JSON lines.
- `benchmarks/results/2026-09-16-merge-refine-schedule.{jsonl,md}`: the recorded
  output and its analysis, with the exact command, the pinned-core single-thread
  protocol, and the limitations. `benchmarks/README.md` documents the entry point.

## Findings

- The structural counters match the design exactly: `2^d` applications and
  `2^d * d` additions for the exact trajectory, `2^d` additions with
  `stopped_regions > 0` for an adaptive stop, and `2^d` items per level.
- The adaptive runs perform three times fewer additions than the exact trajectory
  yet are *slower* at three bits (about 18-26 ms versus 12-13 ms) and about five
  times *faster* at four bits (65-87 ms versus 362-816 ms), where the exact
  trajectory's retained rank reaches `256` and the adaptive runs stay at or below the
  goal. The operation count therefore does not predict the timing at all: the
  retained rank does, which is exactly the distinction the plan asks the evidence to
  make. The first draft of this analysis claimed a uniform speedup; the recorded
  numbers were re-checked against a fresh run and the claim was corrected - which is
  also why no range is quoted without the bit count.
- Accuracy is unaffected: every method's deviation from the exact trajectory is at
  roundoff level and every reported bound covers it.
- The whole-state application baseline is far cheaper on these inputs. That is
  reported rather than hidden: the schedule's purpose is to avoid the global output
  sum for *patched* inputs whose patch transforms are high-rank, and the dyadic-leaf
  contract of this iteration accepts only coordinate leaves, which are low-rank for
  the tested states.

## Limits

Two bit counts, three deterministic families, one machine, one thread, three
repeats; no constant-rank or speedup claim for arbitrary data. This is a
single-revision comparison, not a baseline/candidate regression gate. The intended
patched-input regime needs the nonuniform geometry that remains follow-up work;
automatic padding still awaits a domain/embedding contract decision.
