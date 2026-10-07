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
