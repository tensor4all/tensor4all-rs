# Cached batch memory experiment protocol

Release, default CPU faer backend, AMD Ryzen 9 6900HX / Linux WSL2, CPU 2.
All RAYON/BLAS/OMP/OPENBLAS/MKL thread variables are 1. Source:
`benchmarks/rust/benchmark_cached_batch_memory.rs`, deterministic ChaCha8 seed
123456789. Input is a three-leaf branched tree with a 2-value hub, 128-value
leaves and uniform advertised bond rank. Its known analytic values validate
all output points (relative error <1e-12).

Complete case list: ranks 8/17/33 crossed with points 256/1024/4096. Each case
runs legacy (unlimited batch and caches), unchunked (finite default caches),
and bounded (default chunk of 16, finite caches). Fresh process per case to
avoid retained allocator arenas from contaminating peak RSS. Same binary and
native kernels for all modes; this isolates policies, not revision-wide
speedups. Core runtime wrapping is fixed in all modes. Record exact benchmark
commit and binary checksum with results.

Three paired runs per case/mode, alternating legacy/bounded order; complete
suite, no selective retries. Statistics: median peak RSS and elapsed time per
case, median bounded/legacy ratios, 95% bootstrap intervals (10,000 resamples,
seed 805). Primary memory gate: at rank 33 / 4096 points, bounded RSS <=25% of
legacy RSS. Throughput gate: each bounded/legacy median <=1.1; any regression
must be recorded and resolved, not omitted. Observe unchunked separately to
attribute effects of cache budgets versus transient chunking.

Validity: exact case list, all numerical checks pass, all runs finish, no
outside-affinity threads, /proc/stat steal fraction <1%, no concurrent build or
benchmark from this task. Record host load/available memory. Inconclusive if
any validity gate fails; rerun the full suite under the same protocol.

The largest legacy case fits within this host's 19 GiB RAM. This protocol does
not repeat the downstream 20k/rank33 stress case that approached that limit.

## Investigation and revised confirmatory experiment

The original raw-center experiment failed the memory primary gate (RSS ratio
0.999 at rank33/4096) and the small-rank throughput gate. It is retained in
`2026-10-07-cached-batch-memory-raw-first.json`. That fixture uses a streaming
raw-center kernel and does not exercise the reported native batched branch
intermediate. Its taskset-startup affinity probe also made the validity result
inconclusive; the benchmark itself begins after taskset sets the affinity.

A separate exploratory suite adds a site-free hub centered at a leaf, forcing
the generic branch contraction while preserving an analytic result. It retains
the original raw fixture, every size, and fixed limits16/64/256. The sampled
wrapper-startup affinity check is corrected by setting affinity before exec;
all final measurements check every observed thread after that boundary.
Exploratory records are retained, not promoted as confirmatory evidence.

Freeze automatic sizing before fresh confirmation: `None` picks at most 256
points, further limited by a 32 MiB largest-logical-local-tensor scalar-buffer
estimate. Explicit `Some(n)` selects the requested positive limit, including
whole-batch `Some(usize::MAX)`. This avoids fixed16 overhead for small tensors
while limiting the large intermediates. The estimate is not a total-RSS bound.

Final confirmation runs BOTH fixtures, all nine rank/point cases, legacy and
automatic bounded mode, seven pairs each with alternating order, in fresh
processes. Statistic remains median paired ratios, bootstrap95 intervals.
Primary: generic rank33/4096 RSS ratio <=0.25. Every case's median throughput
ratio must be <=1.1. Correctness and original validity gates remain in force.
The fixture correction is explicit; the failed original experiment is not
rewritten as success. Final binary checksum and code commit are recorded.
