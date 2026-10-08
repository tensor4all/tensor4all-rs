# TreeTCI target memo experiment protocol

Recorded before executing candidate benchmarks. Issue #802 already supplies
end-to-end need evidence (repeated expensive oracle evaluations); this is an
issue-driven implementation rather than a static-audit candidate.

## Builds and timing boundaries

Baseline: `f11e30d847fadb3739aa2cc35f1bd0951186254d` TreeTCI/core sources,
built from the #849 worktree whose changes only affect tensorci. Immutable
binary: `/tmp/t4a-burn3-global-baseline`. Candidate: the committed
`fix/treetci-bounded-evaluation` revision recorded in the result manifest before
execution, with clean tracked sources and source SHA-256 hashes. Both use
`cargo build --release -p tensor4all-treetci --features tensor4all-core/default`
and the default faer provider. No new dependencies or profile overrides.

Hardware: AMD Ryzen 9 6900HX, virtualized Linux x86-64, 16 logical CPUs.
Pin the complete child process (therefore inherited threads) to CPU 2.
Set `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `BLIS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`.
No concurrent owned builds/tests/benchmarks. Record load average and
`/proc/stat` CPU-2 steal ticks around each process. Load must be <=8 and steal
fraction <=2%; otherwise the entire experiment is INCONCLUSIVE.

## Complete paired suite

Seven rounds, alternating baseline/candidate order. No warm-up exclusions,
selective retries, replacement cases or threshold adjustments.

1. Existing default-disabled path: unchanged source
   `benchmarks/rust/benchmark_treetci_global_search.rs`; all three cases
   `chain_cos_129`, `quantics_chain_r20`, `tree_3x10_plus_centre`.
   Each invocation reports the median of its three internal runs.
   Match evaluation counts, rank/error histories, pivot/sample fingerprints
   and sampled accuracy exactly. Candidate/baseline time-ratio bootstrap
   upper 95% confidence bound must be <=1.10 in every case.
2. Opt-in memo: same candidate binary, disabled (`plain`) versus enabled
   (`memo`, 256 MiB logical payload), source
   `benchmarks/rust/benchmark_treetci_memo.rs`. All eight cases:
   cheap/expensive x chain/branch x size 8/16 (chain sites 8/16;
   site-free-centre branches have three arms of 4/8, sites 13/25).
   Synthetic expensive target adds 1,024 black-box sine steps per evaluation
   to the same deterministic rank-three analytic target. Timing includes the
   whole high-level call, cache construction and drop, including final
   materialization; output sampling is outside timing.
   The primary metric is the geometric mean of the four expensive-case
   paired time ratios. Its upper 95% bootstrap confidence bound must be <=0.80.
   All expensive cases must evaluate <=50% of the uncached requests.
   All eight cases must preserve request counts, rank/error histories,
   termination and sample bits exactly, return analytic samples within 1e-7,
   retain <=256 MiB logical payload and report no dropped inserts.
   Cheap-case timing is reported in full but is informative: memo is opt-in,
   and no unconditional speedup is claimed.

Resample paired rounds 10,000 times with RNG seed 802. Report medians and
95% percentile bootstrap intervals for every case. Relative population
standard deviation of paired ratios must be <=20% per case; otherwise the
whole paired experiment is INCONCLUSIVE. Report process max RSS separately
from logical payload (RSS includes runtime/output/scratch); do not equate them.

A failure of correctness or a non-regression/primary gate means the candidate
is not promoted on this experiment. Retain every round, failure and noise
observation. Any rerun must repeat the complete suite under this protocol.
The synthetic oracle measures the memo mechanism for costly deterministic
callbacks; it is not a TreeTNCachedEvaluator throughput claim. Four scalar
kinds and borrowed mutable TreeTN oracles are covered by correctness tests.
