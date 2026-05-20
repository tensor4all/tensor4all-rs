# TensorTrain Inner Product Bottleneck Check

Date: 2026-05-19

This note records a focused check of `TensorTrain::inner` performance.  The
benchmark source is `benchmarks/rust/benchmark_tt_ops.rs`, included by
`crates/tensor4all-itensorlike/examples/benchmark_tt_ops.rs`.

## Main Finding

`TensorTrain::inner` already conjugates one site tensor at a time and contracts
it immediately.  It does not first build a fully conjugated MPS.  A full-MPS
`conj()` costs only about 0.18-0.20 ms for `L=32, chi<=16`, while the inner
product costs about 1.3 ms, so whole-MPS conjugation is not the primary
bottleneck in the current Rust implementation.

The dominant cost for small bond dimensions is fixed per-site contraction
overhead.  For `L=32`, the inner product performs 32 tensor conjugations and 63
small pairwise contractions.  ITensorMPS.jl is much faster in this regime, which
suggests that tensor4all-rs is paying more overhead per small contraction
(index/spec preparation, tensor wrapper dispatch, and tenferro eager einsum
entry) rather than doing asymptotically more arithmetic.

## Rust Command

```bash
RAYON_NUM_THREADS=1 \
cargo run -p tensor4all-itensorlike --example benchmark_tt_ops --release -- \
  --L 32 --chis 4,8,16 --warm-up-time 0.05 --measurement-time 0.15 \
  --min-samples 5 --no-zipup
```

Additional runs used `--chis 32,64` and `--L 8/64 --chis 4`.

## Julia Command

```bash
BLAS_NUM_THREADS=1 \
julia --project=benchmarks/julia benchmarks/julia/benchmark_tt_ops.jl \
  --L 32 --chis 4,8,16 --warm-up-time 0.05 --measurement-time 0.15 \
  --min-samples 5 --blas-threads 1
```

Additional runs used `--chis 32,64`.

## Median Timings

| Case | Rust median ms | Julia median ms | Ratio |
| --- | ---: | ---: | ---: |
| `L=32, chi=4` inner | 1.29 | 0.205 | 6.3x |
| `L=32, chi=8` inner | 1.30 | 0.225 | 5.8x |
| `L=32, chi=16` inner | 1.35 | 0.353 | 3.8x |
| `L=32, chi=32` inner | 2.07 | 1.10 | 1.9x |
| `L=32, chi=64` inner | 6.89 | 5.44 | 1.3x |

The gap shrinks as bond dimension grows, supporting the fixed-overhead
hypothesis.

## Rust Variants

For `L=32`:

| Variant | chi=4 ms | chi=8 ms | chi=16 ms | Meaning |
| --- | ---: | ---: | ---: | --- |
| current `TensorTrain::inner` | 1.29 | 1.30 | 1.35 | production path |
| sitewise pair, no `sim_internal_inds` | 1.29 | 1.29 | 1.36 | same chain algorithm without internal-index simulation |
| sitewise 3-array contract, no simulation | 1.17 | 1.12 | 1.22 | combines `env * conj(A_i) * B_i` per site |
| sitewise binary through generic `contract(&[a,b])` | 1.53 | 1.62 | 1.57 | slower, generic binary entry has too much overhead |
| full MPS `conj()` only | 0.18 | 0.18 | 0.19 | not dominant |
| per-site `conj()` loop only | 0.19 | 0.20 | 0.21 | not dominant |

For `chi=32/64`, the 3-array variant is neutral or slower, so changing
`TensorTrain::inner` to that unconditionally would not be a clean improvement.

## Next Optimization Targets

- Reduce fixed overhead in `TensorDynLen` pairwise contractions for small dense
  tensors.
- Avoid materializing conjugated payloads by carrying a conjugation flag down
  into tenferro where possible.
- Keep the MPS algorithm sitewise; building a conjugated MPS is unnecessary.
- Do not introduce a TensorTrain-specific shortcut unless the same improvement
  can be expressed as a general tensor contraction optimization.

## Follow-up: tenferro Raw Comparison

After adding an experimental eager-einsum contraction-plan cache in local
`tenferro-rs`, `TensorTrain::inner` improved substantially for small bond
dimensions.  The remaining gap was checked against the in-tree tenferro
benchmarks:

```bash
RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
cargo bench -p tenferro --bench mps_inner_product_eager -- eval_local_path \
  --warm-up-time 0.05 --measurement-time 0.25 --sample-size 10

RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
cargo bench -p tenferro --bench mps_inner_product -- eval_only \
  --warm-up-time 0.05 --measurement-time 0.25 --sample-size 10

RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
cargo run -p tensor4all-itensorlike --example benchmark_tt_ops --release -- \
  --L 32 --chis 4,8,16,32,64 --warm-up-time 0.05 \
  --measurement-time 0.25 --min-samples 10 --inner-only
```

| Case (`L=32,d=2`) | tensor4all `TensorTrain::inner` | tenferro eager local path | tenferro traced eval | Remaining gap vs eager |
| --- | ---: | ---: | ---: | ---: |
| `chi=4` | 0.731 ms | 0.327 ms | 0.313 ms | 2.24x |
| `chi=8` | 0.744 ms | 0.327 ms | 0.322 ms | 2.28x |
| `chi=16` | 0.876 ms | 0.361 ms | 0.347 ms | 2.43x |
| `chi=32` | 1.721 ms | 1.161 ms | 1.066 ms | 1.48x |
| `chi=64` | 6.529 ms | 6.198 ms | 5.912 ms | 1.05x |

The same fixture was also measured with ITensorMPS.jl:

```bash
JULIA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
julia --startup-file=no --project=benchmarks/julia \
  ../tenferro-rs/scripts/bench-itensormps-inner-product.jl \
  --L 32 --d 2 --chis 4,8,16,32,64 --warm-up-time 0.05 \
  --measurement-time 0.25 --min-samples 10 --blas-threads 1
```

Julia versions in this run:

- Julia 1.12.5
- ITensors 0.6.23
- ITensorMPS 0.2.6

| Case (`L=32,d=2`) | tensor4all `TensorTrain::inner` | tenferro eager local path | tenferro traced eval | ITensorMPS.jl `inner` | tensor4all / ITensorMPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chi=4` | 0.731 ms | 0.327 ms | 0.313 ms | 0.199 ms | 3.67x |
| `chi=8` | 0.744 ms | 0.327 ms | 0.322 ms | 0.232 ms | 3.22x |
| `chi=16` | 0.876 ms | 0.361 ms | 0.347 ms | 0.303 ms | 2.89x |
| `chi=32` | 1.721 ms | 1.161 ms | 1.066 ms | 1.067 ms | 1.61x |
| `chi=64` | 6.529 ms | 6.198 ms | 5.912 ms | 6.912 ms | 0.94x |

The high-bond case is now essentially arithmetic-bound and close to tenferro.
The small-bond gap is a fixed-cost problem.

Additional profiling of `TensorTrain::inner` showed `sim_internal_inds()` is not
the bottleneck: even with profiling overhead, it was only about 0.01-0.04 ms for
`L=32`.  The visible costs are:

- explicit per-site `conj()` payload materialization, about 0.2-0.5 ms depending
  on measurement noise and bond dimension;
- generic `TensorDynLen::contract_pair` dispatch into eager einsum for 63 tiny
  contractions.

The tenferro eager benchmark avoids both costs: it calls
`dot_general_with_conj` directly, so conjugation is passed as a flag into the
backend and the generic einsum label-analysis path is bypassed.

Clean design implication: the next upstream-quality improvement should be a
general tensor contraction API that can carry per-operand conjugation flags
through the normal `TensorDynLen`/tenferro path.  `TensorTrain::inner` can then
use that general API instead of materializing `bra.tensor(i).conj()`.  This is
not a TensorTrain-specific native shortcut; it is the same semantic operation
expressed without an unnecessary payload copy.

## Follow-up: Read-only Tensor Input

After introducing the general read-only tensor input path (`TensorRead` /
`TensorView`) in local `tenferro-rs` and wiring tensor4all compact payload
contractions through it, the same one-thread TT inner benchmark was rerun:

```bash
RAYON_NUM_THREADS=1 cargo run -q -p tensor4all-itensorlike \
  --example benchmark_tt_ops --release -- \
  --L 32 --zipup-L 10 --chis 4,8,16,32,64 \
  --measurement-time 0.25 --min-samples 10 --inner-only
```

| Case (`L=32,d=2`) | tensor4all `TensorTrain::inner` median | Previous median |
| --- | ---: | ---: |
| `chi=4` | 0.708 ms | 0.731 ms |
| `chi=8` | 0.710 ms | 0.744 ms |
| `chi=16` | 0.803 ms | 0.876 ms |
| `chi=32` | 1.725 ms | 1.721 ms |
| `chi=64` | 6.465 ms | 6.529 ms |

The read-only path removes the avoidable compact-payload copy for contiguous
storage inputs, but it does not eliminate the remaining small-bond fixed cost:
`TensorTrain::inner` still performs many tiny generic contractions and still
materializes the conjugated site tensors.

The non-AD `TensorDynLen` operation benchmark was also updated for the current
`contract_pair` API and rerun:

```bash
RAYON_NUM_THREADS=1 cargo run -q -p tensor4all-core \
  --example benchmark_tensor_ops --release -- 20000 6 2 2 6
```

| Operation | Time | Per call |
| --- | ---: | ---: |
| `inner_product` | 0.313624 s | 15.681 us |
| `norm` | 0.273433 s | 13.672 us |
| `axpby` | 0.019593 s | 0.980 us |
| `conj_contract_sum` | 0.270288 s | 13.514 us |

## Follow-up: Current tenferro and Julia Comparison

The same `L=32,d=2` benchmark was rerun against current local `tenferro-rs`
and ITensorMPS.jl with one thread:

```bash
RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
cargo bench -p tenferro --bench mps_inner_product_eager -- \
  eval_local_path --warm-up-time 0.05 --measurement-time 0.25 --sample-size 10

RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
cargo bench -p tenferro --bench mps_inner_product -- \
  eval_only --warm-up-time 0.05 --measurement-time 0.25 --sample-size 10

JULIA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
julia --startup-file=no --project=benchmarks/julia \
  ../tenferro-rs/scripts/bench-itensormps-inner-product.jl \
  --L 32 --d 2 --chis 4,8,16,32,64 \
  --warm-up-time 0.05 --measurement-time 0.25 \
  --min-samples 10 --blas-threads 1
```

| Case (`L=32,d=2`) | tensor4all `TensorTrain::inner` median | tenferro eager local path | tenferro traced eval | ITensorMPS.jl `inner` | tensor4all / Julia |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chi=4` | 0.708 ms | 0.333 ms | 0.301 ms | 0.194 ms | 3.64x |
| `chi=8` | 0.710 ms | 0.328 ms | 0.309 ms | 0.219 ms | 3.24x |
| `chi=16` | 0.803 ms | 0.340 ms | 0.342 ms | 0.303 ms | 2.65x |
| `chi=32` | 1.725 ms | 1.050 ms | 0.972 ms | 1.087 ms | 1.59x |
| `chi=64` | 6.465 ms | 6.090 ms | 5.859 ms | 6.866 ms | 0.94x |

Interpretation: the high-bond case is fine; tensor4all is now slightly faster
than Julia at `chi=64`.  The remaining gap is concentrated at small bond
dimension, where fixed costs dominate.  The gap to tenferro's direct eager path
is still about 2.1-2.4x for `chi <= 16`, so the next target is not arithmetic
throughput but avoiding the tiny-contraction overhead and materialized
per-site conjugation.

## Follow-up: tenferro Binary Einsum Fast Path

Added a conservative non-AD binary einsum fast path in local `tenferro-rs`.
For two-input contractions with unique labels and at least one shared
contracting label, eager einsum now bypasses contraction-tree planning and the
generic `HashMap`/`HashSet` label-analysis path, directly lowering to
`dot_general_read`.  Repeated-label, one-sided reduction, and outer-product
cases still use the generic path.

The same one-thread tensor4all benchmark was rerun:

```bash
RAYON_NUM_THREADS=1 cargo run -q -p tensor4all-itensorlike \
  --example benchmark_tt_ops --release -- \
  --L 32 --zipup-L 10 --chis 4,8,16,32,64 \
  --measurement-time 0.25 --min-samples 10 --inner-only
```

| Case (`L=32,d=2`) | After binary fast path | Previous read-only path |
| --- | ---: | ---: |
| `chi=4` | 0.685 ms | 0.708 ms |
| `chi=8` | 0.712 ms | 0.710 ms |
| `chi=16` | 0.768 ms | 0.803 ms |
| `chi=32` | 1.646 ms | 1.725 ms |
| `chi=64` | 6.343 ms | 6.465 ms |

This helps, but does not remove the small-bond fixed-cost gap.  The remaining
visible costs are still per-site conjugation materialization and many tiny
contraction calls.  The fast path is therefore useful and generally clean, but
not the whole fix.

## Follow-up: Overhead Breakdown

Added a benchmark-only variant that precomputes the conjugated bra site tensors
outside the timed loop and then runs the same sitewise `TensorDynLen`
`contract_pair` sequence.  This isolates explicit conjugation materialization
from the contraction wrapper overhead:

```bash
RAYON_NUM_THREADS=1 cargo run -q -p tensor4all-itensorlike \
  --example benchmark_tt_ops --release -- \
  --L 32 --zipup-L 10 --chis 4,8,16,32,64 \
  --warm-up-time 0.5 --measurement-time 0.25 \
  --min-samples 10 --inner-only
```

| Case (`L=32,d=2`) | Current `TensorTrain::inner` | Preconjugated sitewise pair | Difference |
| --- | ---: | ---: | ---: |
| `chi=4` | 0.700 ms | 0.473 ms | 0.227 ms |
| `chi=8` | 0.710 ms | 0.483 ms | 0.226 ms |
| `chi=16` | 0.780 ms | 0.562 ms | 0.218 ms |
| `chi=32` | 1.668 ms | 1.429 ms | 0.240 ms |
| `chi=64` | 6.407 ms | 5.916 ms | 0.491 ms |

The same run confirmed that direct tenferro eager local path for `chi=4` is
about `0.337 ms`.  Therefore the `chi=4` gap decomposes roughly as:

- explicit per-site `conj()` materialization: about `0.23 ms`;
- remaining `TensorDynLen::contract_pair` wrapper/index/rebuild overhead over
  direct dot-general: about `0.14 ms`;
- arithmetic/backend GEMM itself is not the bottleneck at small bond dimension.

Additional aggregated profiling with `T4A_PROFILE_PAIRWISE_CONTRACT=1` showed
the non-GEMM `TensorDynLen` wrapper work is small per call but repeated 63
times: `prepare_contraction`, result axis-class computation, binary subscript
construction, and result `TensorDynLen` reconstruction together account for the
same order as the remaining gap after preconjugation.

Conclusion: the true small-bond overhead is not contraction-path search and not
GEMM.  It is the combination of (1) materialized conjugation and (2) repeatedly
calling the fully generic indexed `TensorDynLen::contract_pair` path for a known
MPS local contraction.

## Follow-up: Operand-Level Conjugation

Implemented pairwise contraction options with operand-level conjugation:
`contract_pair_with_operand_options(lhs, rhs, PairwiseContractionOptions)`.
`TensorTrain::inner` now uses `lhs_conj` / `rhs_conj` flags instead of
materializing `tensor.conj()` for each bra site.

`T4A_PROFILE_TT_INNER=1` confirms `conj_ms=0.000000` in the new
`TensorTrain::inner` path.

The same one-thread benchmark now gives:

| Case (`L=32,d=2`) | New `TensorTrain::inner` | Previous `TensorTrain::inner` | Preconjugated baseline |
| --- | ---: | ---: | ---: |
| `chi=4` | 0.461 ms | 0.700 ms | 0.473 ms |
| `chi=8` | 0.474 ms | 0.710 ms | 0.483 ms |
| `chi=16` | 0.460 ms | 0.780 ms | 0.562 ms |
| `chi=32` | 1.517 ms | 1.668 ms | 1.429 ms |
| `chi=64` | 6.218 ms | 6.407 ms | 5.916 ms |

This removes the materialized-conjugation cost from the real `inner` path.
The remaining small-bond gap to direct tenferro eager local contraction is now
mostly the generic `TensorDynLen` pairwise wrapper/index/result-rebuild cost.

## Follow-up: 2026-05-20 After Linsolve Spectator Fix

After the TreeTN linsolve spectator-node fixes, the one-thread
`TensorTrain::inner` benchmark was rerun:

```bash
RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
cargo run -q -p tensor4all-itensorlike --example benchmark_tt_ops \
  --release -- --L 32 --zipup-L 10 --chis 4,8,16,32,64 \
  --warm-up-time 0.5 --measurement-time 0.25 \
  --min-samples 10 --inner-only
```

| Case (`L=32,d=2`) | Current `TensorTrain::inner` median | Preconjugated sitewise pair median |
| --- | ---: | ---: |
| `chi=4` | 0.464 ms | 0.457 ms |
| `chi=8` | 0.476 ms | 0.457 ms |
| `chi=16` | 0.449 ms | 0.425 ms |
| `chi=32` | 1.508 ms | 1.495 ms |
| `chi=64` | 6.233 ms | 6.232 ms |

The current production path is now essentially at the preconjugated baseline,
which confirms that operand-level conjugation removed the explicit per-site
conjugation materialization cost from `TensorTrain::inner`.

The focused profiling command was also rerun for `L=32, chi=4`:

```bash
T4A_PROFILE_TT_INNER=1 RAYON_NUM_THREADS=1 \
cargo run -q -p tensor4all-itensorlike --example benchmark_tt_ops \
  --release -- --L 32 --zipup-L 10 --chis 4 \
  --warm-up-time 0 --measurement-time 0 --min-samples 1 --inner-only
```

The profiled production path still reports `conj_ms=0.000000`; profiling
overhead is visible in single-sample timings, and `contract_ms` remains the
dominant component.  The linsolve changes therefore did not reintroduce
materialized site-tensor conjugation into `TensorTrain::inner`.

### Current Comparison With Direct tenferro And ITensorMPS.jl

Direct tenferro eager local path was rerun with:

```bash
RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
cargo bench -p tenferro --bench mps_inner_product_eager -- \
  eval_local_path --warm-up-time 0.05 --measurement-time 0.25 \
  --sample-size 10
```

ITensorMPS.jl was rerun with:

```bash
JULIA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
julia --startup-file=no --project=benchmarks/julia \
  ../tenferro-rs/scripts/bench-itensormps-inner-product.jl \
  --L 32 --d 2 --chis 4,8,16,32,64 \
  --warm-up-time 0.05 --measurement-time 0.25 \
  --min-samples 10 --blas-threads 1
```

| Case (`L=32,d=2`) | tensor4all `TensorTrain::inner` | direct tenferro eager | ITensorMPS.jl `inner` | tensor4all / tenferro | tensor4all / Julia |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chi=4` | 0.464 ms | 0.328 ms | 0.208 ms | 1.41x | 2.23x |
| `chi=8` | 0.476 ms | 0.313 ms | 0.228 ms | 1.52x | 2.09x |
| `chi=16` | 0.449 ms | 0.349 ms | 0.318 ms | 1.29x | 1.41x |
| `chi=32` | 1.508 ms | 1.049 ms | 1.124 ms | 1.44x | 1.34x |
| `chi=64` | 6.233 ms | 6.162 ms | 6.878 ms | 1.01x | 0.91x |

The operand-level conjugation fix moves tensor4all close to direct tenferro at
large bond dimension and removes the old explicit-conjugation gap.  The
remaining small-bond gap is still fixed overhead above direct tenferro's local
path, plus ITensorMPS.jl's particularly low overhead for tiny contractions.

### Follow-up: TensorDynLen Wrapper/Rebuild Fixed-Cost Breakdown

The benchmark now includes two same-process raw tenferro baselines using the
same boundary-rank convention as `TensorTrain`:

- `tenferro_raw_eager_inner_t4a_shapes`: direct eager `dot_general_with_conj`
  on the same MPS tensor shapes, bypassing `TensorDynLen` indices/storage;
- `tenferro_raw_eager_inner_t4a_shapes_snapshot_outputs`: the same direct
  eager path, but cloning each intermediate output tensor to approximate the
  `TensorDynLen` output-storage snapshot.

One-thread command:

```bash
RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
cargo run -q -p tensor4all-itensorlike --example benchmark_tt_ops \
  --release -- --L 32 --zipup-L 10 --chis 4,8,16,32,64 \
  --warm-up-time 0.5 --measurement-time 0.25 \
  --min-samples 10 --inner-only
```

| Case (`L=32,d=2`) | `TensorDynLen` `inner` | raw tenferro same shapes | raw tenferro + output clone | clone-only delta | wrapper gap after clone |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chi=4` | 0.461 ms | 0.307 ms | 0.318 ms | 0.011 ms | 0.143 ms |
| `chi=8` | 0.481 ms | 0.309 ms | 0.332 ms | 0.023 ms | 0.149 ms |
| `chi=16` | 0.445 ms | 0.333 ms | 0.334 ms | 0.001 ms | 0.111 ms |
| `chi=32` | 1.494 ms | 1.138 ms | 1.183 ms | 0.045 ms | 0.311 ms |
| `chi=64` | 6.277 ms | 6.030 ms | 6.283 ms | 0.253 ms | -0.005 ms |

The raw tenferro values are intentionally measured in the tensor4all benchmark
binary, so this comparison avoids Criterion/JIT/process differences.  The
values confirm that `TensorDynLen` adds a real small-bond fixed cost, but the
large payload-copy hypothesis is not supported:

- input `Storage -> NativeTensor` materialization does not appear in the
  profiled `inner` path; `eager_cache` is hit;
- intermediate output snapshot copying is real, but only accounts for about
  `0.01-0.02 ms` at `chi=4/8`;
- at `chi=64`, output copying becomes visible (`~0.25 ms`), but arithmetic
  dominates and the full path is already essentially raw-tenferro speed.

The profiled `TensorDynLen` pairwise contraction path for a single
`L=32, chi=4` inner product reports:

| Component | Calls | Total | Per call | Bytes |
| --- | ---: | ---: | ---: | ---: |
| `dot_general_execute` | 63 | 0.987 ms | 15.7 us | 0 |
| `from_inner_axis_classes` | 63 | 0.091 ms | 1.45 us | 0 |
| `result_axis_classes` | 63 | 0.043 ms | 0.69 us | 0 |
| `operand_indices` | 126 | 0.036 ms | 0.28 us | 0 |
| `prepare_contraction` | 63 | 0.027 ms | 0.43 us | 0 |
| `from_inner_storage_snapshot` | 63 | 0.020 ms | 0.32 us | 23,440 |
| `as_native` | 126 | 0.010 ms | 0.08 us | 0 |
| `from_inner_eager_cache` | 63 | 0.010 ms | 0.16 us | 0 |

For `chi=8/32/64`, `from_inner_storage_snapshot` copies `93,456` /
`1,491,984` / `5,965,840` bytes, respectively.  Therefore the unnecessary-copy
candidate to remove is specifically the output storage snapshot in
`TensorDynLen::from_inner_with_axis_classes`; it is not the small-`chi` dominant
cost, but it is a real bandwidth cost at larger bond dimensions and it
duplicates the already-owned `EagerTensor` kept in `eager_cache`.

ITensorMPS.jl on the same deterministic MPS gives:

| Case (`L=32,d=2`) | ITensorMPS.jl `inner` |
| --- | ---: |
| `chi=4` | 0.207 ms |
| `chi=8` | 0.265 ms |
| `chi=16` | 0.331 ms |
| `chi=32` | 0.954 ms |
| `chi=64` | 5.543 ms |

Current interpretation:

- ITensorMPS.jl has much lower tiny-contraction fixed overhead than the current
  `TensorDynLen` path and even lower than the current direct tenferro eager
  path for `chi=4/8`;
- the residual tensor4all overhead is mostly the repeated generic
  index/wrapper/rebuild path around 63 tiny contractions, plus tenferro eager
  per-call execution/session overhead;
- the next clean optimization candidate is to make `TensorDynLen` results able
  to keep the eager payload as the primary representation and lazily build
  `Storage` only when a storage-backed operation actually requires it.  That
  would remove the duplicated output payload copy generally, without adding an
  MPS-specific fast path.

### Follow-up: CPU-Only Eager-Dense TensorDynLen Payload

Implemented a CPU-only first step toward `TensorDynLen = EagerTensor + layout +
indices`: dense eager contraction results now store the tenferro
`EagerTensor<CpuBackend>` directly instead of immediately snapshotting into
`Storage`.  `Storage` remains available as a materialized compatibility
snapshot through `storage()` / `to_storage()`, but dense hot-path contraction
intermediates no longer carry both copies.

The implementation keeps non-dense/structured results on the existing storage
path for now, and preserves the old eager cache there so AD through diagonal SVD
singular values continues to work.

One-thread release benchmark:

```bash
RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
cargo run -q -p tensor4all-itensorlike --example benchmark_tt_ops \
  --release -- --L 32 --zipup-L 10 --chis 4,8,16,32,64 \
  --warm-up-time 0.5 --measurement-time 0.25 \
  --min-samples 10 --inner-only
```

| Case (`L=32,d=2`) | tensor4all after eager-dense payload | tensor4all before | direct tenferro same shapes | ITensorMPS.jl `inner` |
| --- | ---: | ---: | ---: | ---: |
| `chi=4` | 0.444 ms | 0.461 ms | 0.312 ms | 0.207 ms |
| `chi=8` | 0.458 ms | 0.481 ms | 0.315 ms | 0.265 ms |
| `chi=16` | 0.396 ms | 0.445 ms | 0.332 ms | 0.331 ms |
| `chi=32` | 1.244 ms | 1.494 ms | 1.142 ms | 0.954 ms |
| `chi=64` | 6.150 ms | 6.277 ms | 5.981 ms | 5.543 ms |

The small-bond gap is still mostly generic wrapper/per-call overhead, not payload
copy.  The larger-bond cases improve more because the removed output snapshot
copy was real bandwidth work.

Validation:

```bash
cargo test --release -q -p tensor4all-core
```

passed.

### Follow-up: SmallVec Contraction Preparation

Implemented SmallVec-backed contraction preparation in `tensor4all-core`:

- `ContractionSpec` now stores axes, result indices, and result dimensions in
  `SmallVec<[T; 8]>`.
- Small contractions use the ITensor-like nested-loop matcher with no hash
  table.
- Larger contractions fall back to a hash map keyed by `IndexLike::id()`, with
  `is_contractable` still used for the final dimension/prime/direction check.
- Result construction uses boolean axis flags instead of repeated
  `axes.contains(...)` scans.

The pairwise profile confirms that preparation is not the current bottleneck:

| Section (`L=32,d=2,chi=4`) | Calls | Total | Per call |
| --- | ---: | ---: | ---: |
| `prepare_contraction` | 63 | 0.027 ms | 0.43 us |
| `result_axis_classes` | 63 | 0.035 ms | 0.55 us |
| `dot_general_execute` | 63 | 2.619 ms | 41.58 us |

The benchmark run after this change was globally slower than the earlier table,
including the direct tenferro baseline, so it should not be used as an absolute
performance regression measurement.  It does show that SmallVec cleanup removes
heap-oriented preparation code without moving the dominant cost; the remaining
gap is still in per-call eager contraction/wrapper execution.

Validation:

```bash
cargo test --release -q -p tensor4all-core
```

passed.
