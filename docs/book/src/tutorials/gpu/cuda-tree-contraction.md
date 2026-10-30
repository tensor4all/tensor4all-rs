# CUDA TreeTN Contraction

This experimental path contracts a small dense `TreeTN<IdxTensor>` on one NVIDIA GPU. Transfers are explicit: tensor4all-rs never uploads, downloads, or falls back to the CPU inside the contraction.

## Prerequisites

- A stable Rust toolchain; see [Getting Started](../../getting-started.md).
- An NVIDIA GPU and CUDA driver/toolkit compatible with the CUDA versions used by the pinned tenferro GPU stack: `tenferro-gpu` drives `t4a-cuda-kernels` and `t4a-cuda-runtime` (cuBLAS / cuTENSOR). No wider version or compute-capability support matrix is currently promised.
- The intended device must be visible as CUDA ordinal 0.
- Build with the non-default `tenferro-cuda` feature. The commands also name the `tenferro-cpu-faer` feature explicitly because the quickstart computes a CPU reference alongside the CUDA contraction.

Clone tensor4all-rs and run the checked quickstart:

```bash
git clone https://github.com/tensor4all/tensor4all-rs.git
cd tensor4all-rs
cargo run --release -p tensor4all-treetn --example cuda_quickstart \
  --features tenferro-cuda,tenferro-cpu-faer
```

The complete checked source is embedded below. It is rendered as text rather than an mdBook-tested Rust block because executing it requires CUDA hardware; the feature-gated example itself is compile-checked and run on CUDA.

```text
{{#include ../../../../../crates/tensor4all-treetn/examples/cuda_quickstart.rs}}
```

Source: [`cuda_quickstart.rs`](https://github.com/tensor4all/tensor4all-rs/blob/main/crates/tensor4all-treetn/examples/cuda_quickstart.rs).

It performs this flow:

1. Build a two-node host `TreeTN` and compute a CPU reference.
2. Create one caller-owned `CudaExecutionContext` for visible ordinal 0.
3. Upload every node with `TreeTN::upload_cuda`.
4. Contract all internal bonds with `contract_to_tensor_cuda`.
5. Verify that the result is still resident in the same CUDA context.
6. Download explicitly and assert a maximum CPU/GPU residual of at most `1e-10`.

A successful run prints the GPU name and residual, for example:

```text
device="NVIDIA A100 80GB PCIe" residual_max_abs=0.000e0
```

## Measured: where the time goes

The example reports each phase separately, which is what makes the numbers interpretable - a single wall-clock
figure would hide that this route is dominated by fixed costs rather than by the contraction. On the local A100
80GB PCIe (driver 580.126.09), release build, chain length 6, bond dimension 4, 10 iterations:

```bash
cargo run --release -p tensor4all-treetn --example cuda_tree_contraction \
  --features tenferro-cuda,tenferro-cpu-faer -- --chain-length 6 --bond-dim 4 --iterations 10
```

| phase (one run, cold: the first figure of each shape carries first-use initialisation) | time |
| --- | ---: |
| host tree construction | 4.97 ms |
| CUDA context setup | 198.28 ms |
| upload (the tree) | 313.46 ms |
| first call (JIT and caches) | 2.02 ms |
| steady contraction, synchronised | 0.176 ms / iteration |
| download | 0.086 ms |
| CPU contraction of the same tree, for reference | 0.138 ms / iteration |

Two things follow, and both are the point of separating the phases:

- **The first call of a shape is not the shape's cost.** A second, component-wise example separates the transfer
  from the initialisation that precedes it on first use:

  ```bash
  cargo run --release -p tensor4all-treetn --example cuda_component_upload \
    --features tenferro-cuda,tenferro-cpu-faer -- --chain-length 6 --bond-dim 4
  ```

  On the same machine it reports, for a six-node chain: the first per-node upload `427.42 ms` (that call carries
  the route's first-use initialisation), the remaining five `0.05–0.09 ms` each, a warmed second pass of `0.32 ms`
  in total, and the whole-network upload `0.36 ms` then `0.28 ms`. **The transfer itself is sub-millisecond for a
  chain this size**: the hundreds of milliseconds in the table above are first-use initialisation inside the
  context and the first transfer, not data movement. The table is worth reading as "what a user pays the first
  time", and the component example is worth reading as "what the transfer costs once that is paid".
- **The device is not automatically faster per operation**: 0.176 ms versus 0.138 ms per contraction here. A
  branching tree contraction of small tensors does not have the arithmetic intensity that a device kernel wants.

Each per-node call in the component example also clones the network metadata and updates one node, and each
transfer includes native duplication, a session entry and eager wrapping; the per-node figure is therefore an
upper bound on the data movement, not a measurement of it alone.

### Wait and overlap, measured with both routes warmed and doing the same work

The pending routes make a copy's *submission* and its *completion* separable, so two copies can be in flight at
once, and the three tests below print a serial figure (submit, wait, submit, wait) against a batched one (submit
both, then wait for both). Each test warms up with one throwaway transfer first and keeps both regions doing the
same work inside the timer, so neither region pays a first-use cost, and neither pays a destruction the other
does not. The figures are printed, not asserted: a ratio on a shared machine is not a correctness property.

```bash
# Two whole networks (4 nodes each), end-to-end including both host reconstructions:
cargo test -p tensor4all-treetn --features tenferro-cuda --test integration cuda_pending_download -- --nocapture

# Two 8 MiB device-to-host copies, pinned allocations and decodes included, uploads excluded:
cargo test -p tensor4all-tensorbackend --features tenferro-cuda --test cuda_async_download -- --nocapture

# Two 8 MiB host-to-device copies, staging and pinned allocations included, destinations excluded:
cargo test -p tensor4all-tensorbackend --features tenferro-cuda --test cuda_async_upload -- --nocapture
```

Measured on one local A100 80GB PCIe (driver 580.126.09), one run each. The absolute figures move with the
machine and with what else is running: repeating these three tests on the same machine moved them by tens of
percent between runs, which is the main reason they are printed by the test rather than promised here.

| workload | serial | batched | ratio |
| --- | ---: | ---: | ---: |
| two 4-node networks (TreeTN pending download) | 6.988 ms | 4.530 ms | 1.54x |
| two 8 MiB copies (tensor pending download) | 32.619 ms | 36.720 ms | 0.89x |
| two 8 MiB copies (tensor pending upload) | 26.867 ms | 28.089 ms | 0.96x |

**What the figures are.** These are three end-to-end observations on one machine, with the timer contents named
above; the tests print them rather than asserting them, and nothing here is a promise about another workload or
another machine. Two of the three batched rows came out faster than their serial row and one came out slower, so
this table does not establish that overlap pays in general: it shows that the route makes submission and
completion separable, and that whether a caller sees a gain depends on its own transfers, which it has to
measure itself. The tests also cannot separate how much of each figure is staging, allocation, submission,
polling or the wait.

Two details of the contract that the figures do not show: `CudaExecutionContext::download_pending` consumes the
device tensor (the pinned API has no way to copy one) and hands it back from `wait`, and a rejected submission
keeps only its diagnostic, because the rejected package is neither `Send` nor `Sync`. The upload route mirrors
that: `upload_pending` takes a pinned source buffer and a device destination and returns a `PendingCudaUpload`
whose `wait` hands back the written device tensor together with the source buffer, and a caller that wants to keep
its tensor should not use the route.

The SRC route has its own separated breakdown (setup, upload, warm-up, steady, download, CPU) in
`cuda_src_timing_breakdown` under `crates/tensor4all-treetn/tests/cuda_src_contraction.rs`, where its upload figure
carries the same first-use caveat.

## Using it from another project

The crates are not published to crates.io yet. Enable CUDA on both crates imported by the quickstart:

```toml
[dependencies]
tensor4all-core = { git = "https://github.com/tensor4all/tensor4all-rs", features = ["tenferro-cuda", "tenferro-cpu-faer"] }
tensor4all-treetn = { git = "https://github.com/tensor4all/tensor4all-rs", features = ["tenferro-cuda", "tenferro-cpu-faer"] }
```

Create a binary project, put the dependency entries above under `[dependencies]` in `Cargo.toml`, and copy the embedded program to `src/main.rs`:

```bash
cd ..
cargo new cuda-tree-quickstart
cd cuda-tree-quickstart
# Add the dependency entries above to Cargo.toml.
cp ../tensor4all-rs/crates/tensor4all-treetn/examples/cuda_quickstart.rs src/main.rs
cargo run --release
```

Keep one `CudaExecutionContext` for upload, contraction, synchronization, and download. Mixing host and CUDA nodes, CUDA contexts, or node dtypes returns a typed error before contraction; it does not trigger a hidden transfer or CPU fallback.

## Current limits

This is a dense full-network contraction, so output memory scales with the product of external-index dimensions. It currently supports only:

- dense, untracked `IdxTensor` nodes;
- one dtype and one CUDA context across the tree;
- visible CUDA ordinal 0;
- full contraction to one dense tensor.

CUDA SVD, QR, truncation, TreeTN-to-TreeTN contraction, zip-up/fitting/SRC,
TCI/ACI, automatic device selection, and multi-GPU execution are not yet
supported.

For timing, use the separate `cuda_tree_contraction` example, which reports context setup, upload, warm-up, steady-state GPU contraction plus synchronization, download, and CPU contraction independently:

```bash
cargo run --release -p tensor4all-treetn --example cuda_tree_contraction \
  --features tenferro-cuda,tenferro-cpu-faer
```
