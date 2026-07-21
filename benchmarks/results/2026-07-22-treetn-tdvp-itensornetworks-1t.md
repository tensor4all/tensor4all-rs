# TreeTN TDVP vs ITensorNetworks.jl, strict single-thread rerun

Date: 2026-07-22

Purpose: rerun the 2026-06-27 TreeTN TDVP comparison
(`2026-06-27-treetn-tdvp-itensornetworks.md`) under strictly verified
single-thread conditions, and identify why the Rust implementation is faster
on the star topology.

Two corrections to the earlier setup motivated this rerun:

1. The earlier Julia command set `BLAS_NUM_THREADS=1`, but Julia's OpenBLAS
   reads `OPENBLAS_NUM_THREADS`. The earlier Julia timings may therefore have
   run with multi-threaded BLAS. This rerun sets `JULIA_NUM_THREADS=1` and
   `OPENBLAS_NUM_THREADS=1` and verifies `BLAS.get_num_threads() == 1` in the
   benchmark process environment.
2. The earlier run used a Linux machine. This rerun records the platform
   explicitly (Apple M5 Max, macOS 25.5.0) so future comparisons are not mixed
   across machines.

The physical setup is unchanged: alternating product state evolved under the
Pauli Heisenberg Hamiltonian `sum_(i,j in E) X_i X_j + Y_i Y_j + Z_i Z_j`
for `t = 0.08` in 4 two-site TDVP steps of `dt = 0.02`, `order = 2`,
`maxdim = 32`, ITensors-compatible relative discarded squared singular-value
tail cutoff `1e-12`, Hermitian Krylov exponentials with `krylovdim = 30`,
`tol = 1e-12`. Accuracy is the dense-vector L2 error against an exact dense
Hermitian eigendecomposition reference.

## Environment

- Machine: Apple M5 Max, macOS (Darwin 25.5.0)
- Rust: `benchmark_tdvp` release build, default faer-backed tensor backend.
  The per-apply profile below shows GEMM time is negligible at these tensor
  sizes, so the faer-vs-OpenBLAS backend choice does not affect this
  comparison.
- Julia: 1.12.5, ITensorNetworks.jl at commit `9a4c4a7` (v0.21.1,
  2026-05-08). This is the newest upstream commit that still provides the
  `ttn(OpSum, ...)` constructor used by the benchmark script; later commits
  (#355/#356) remove that API.
- Thread verification: `BLAS.get_num_threads() == 1`,
  `Threads.nthreads() == 1` under the exact benchmark environment.

## Commands

Rust:

```bash
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 \
cargo run -p tensor4all-treetn --example benchmark_tdvp --release -- 8 4 3 0.02
```

Julia (bench project with `Pkg.develop` of the pinned ITensorNetworks.jl
checkout, plus `Graphs`, `ITensors`, `TensorOperations`, `KrylovKit`):

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
julia --project=$T4A_ITN_BENCH_PROJECT \
  benchmarks/julia/benchmark_tdvp_itensornetworks.jl 8 4 3 0.02
```

## Results (1 thread)

| Implementation | Topology | Norm | L2 error | Mean ms | Min ms | Max ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Rust TreeTN TDVP | chain | 0.999999999998 | 1.375e-5 | 104.585 | 99.150 | 107.514 |
| ITensorNetworks.jl | chain | 0.999999999998 | 1.375e-5 | 101.034 | 90.188 | 117.843 |
| Rust TreeTN TDVP | star | 1.000000000000 | 3.999e-4 | 1739.476 | 1736.142 | 1741.698 |
| ITensorNetworks.jl | star | 1.000000000000 | 7.643e-14 | 5414.554 | 5373.677 | 5458.630 |

Summary: on the chain control the two implementations are equal within noise.
On the star topology Rust is 3.1x faster at strictly one thread. The earlier
Linux run showed 5.3x; part of that gap was likely the unpinned Julia BLAS
threading (multi-threaded BLAS hurts at these tiny tensor sizes) plus the
different machine.

## Why Rust wins on the star: profile evidence

Both implementations spend essentially the whole star runtime applying the
projected Hamiltonian inside the local Krylov exponentials, and both perform
almost the same number of applies. The difference is the cost per apply.

Rust (`T4A_PROFILE_TDVP=1`, 1 repeat, total 1.647 s):

- `evolve local` (Krylov exponentials): 1.626 s = 98.7% of total
- `projected apply` (H·v): 1.552 s = 94.2% of total, 752 calls
- everything else (canonicalization moves, factorize, plan, environment
  bookkeeping): < 2%
- per apply: 1.552 s / 752 = 2.06 ms

Julia (`Profile` flat samples, one run, 28 630 samples total):

- `KrylovKit.exponentiate`: 99.7% of samples
- `optimal_map` (projected H·v): 95.6% of samples
- inside the apply, by exclusive share: `permutedims` 21.6%, allocation
  (`similar`) 16.5%, contraction-sequence search 3.4%, remainder NDTensors
  pairwise-contraction dispatch and kernel overhead; BLAS GEMM does not
  register at these sizes
- apply count (instrumented wrapper run): 802 calls
- per apply: 0.956 x 5.415 s / 802 = 6.45 ms

So the whole 3.1x gap reduces to a 3.1x per-apply cost ratio
(6.45 ms vs 2.06 ms) at nearly identical apply counts (802 vs 752).

The star's center vertex has degree 7, so a two-site projected apply at the
center contracts on the order of 9 small tensors (6 leaf environments, 2
operator tensors, the local state) pairwise. Every pairwise step carries
per-contraction bookkeeping: index-set matching, output allocation, and a
layout permutation. The numeric work is negligible (chi <= 32, d = 2), so
per-contraction overhead dominates and the implementation with cheaper
per-contraction machinery wins. The Rust side benefits from its multi-tensor
`T::contract(&refs)` path and lighter per-contraction bookkeeping; the Julia
side pays `permutedims` + allocation + dynamic dispatch per pairwise ITensor
contraction. The chain control confirms this reading: with only 2
environments per region (4-5 tensors per apply), the overhead share shrinks
and the two implementations tie.

Notably, the contraction-sequence search (`alg="optimal"`) is only ~3% of
Julia's samples, ruling out the a-priori plausible hypothesis that repeated
sequence optimization at the high-degree center explains the gap.

## Caveat: accuracy divergence against the 2026-06-27 run

At the same nominal parameters, ITensorNetworks.jl at `9a4c4a7` reaches
L2 error 7.6e-14 with final `maxlinkdim = 2` on the star, while both Rust and
the older ITensorNetworks checkout used on 2026-06-27 give 3.999e-4. Upstream
therefore changed two-site TDVP behavior on non-chain trees somewhere before
`9a4c4a7`, and the Rust implementation currently reproduces the older, less
accurate behavior. This does not affect the timing conclusion above (the
newer Julia code does less numerical work per bond, chi = 2, yet is still
3.1x slower), but it means the "matching accuracy" parity claim from
2026-06-27 no longer holds against current upstream.

The changing commit was identified by targeted testing of the range
`88f1b8c..9a4c4a7`: the transition happens exactly at `c6d7e64` ("Upgrade
ITensorNetworks to use DataGraphs v0.4.0 and NamedGraphs v0.11.0", #317).
Its parent `9ed5949` gives 3.999e-4; `c6d7e64` gives 7.64e-14. The
improvement therefore comes from a change in graph traversal ordering in the
NamedGraphs/DataGraphs dependencies (which determines the post-order DFS
sweep order over the tree), not from any change in the TDVP solver code
itself, which is unchanged over this range. Implication for tensor4all-rs:
on branching trees the sweep edge ordering alone moves the TDVP error by ten
orders of magnitude at identical bond dimensions, and `TdvpRegionPlan`
currently reproduces the pre-#317 ordering. Determining the NamedGraphs
v0.11 ordering and adopting it is tracked as a follow-up.

## Notes

- Rust benchmark source: `benchmarks/rust/benchmark_tdvp.rs` via
  `crates/tensor4all-treetn/examples/benchmark_tdvp.rs`.
- Julia benchmark source: `benchmarks/julia/benchmark_tdvp_itensornetworks.jl`.
- The Julia apply count was measured with a wrapper solver that counts
  operator evaluations; the wrapper adds significant overhead, so its timing
  is discarded and only the count (802) is used. The per-apply time is
  computed from the uninstrumented run times the profile share.
- Rust reported `sweeps_completed=4`, `local_updates=104`,
  `max_krylov_iterations=8` on both topologies, matching the 2026-06-27 run,
  and identical L2 errors to the 2026-06-27 run on both topologies.
