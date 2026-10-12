# CUDA TreeTN Contraction

This experimental path contracts a small dense `TreeTN<IdxTensor>` on one NVIDIA GPU. Transfers are explicit: tensor4all-rs never uploads, downloads, or falls back to the CPU inside the contraction.

## Prerequisites

- A stable Rust toolchain; see [Getting Started](../../getting-started.md).
- An NVIDIA GPU and CUDA driver/toolkit compatible with the CUDA versions used by the pinned tenferro/cubecl dependencies. No wider version or compute-capability support matrix is currently promised.
- The intended device must be visible as CUDA ordinal 0.
- Build with the non-default `tenferro-cuda` feature. The commands also name the default `tenferro-cpu-faer` feature explicitly because the quickstart computes a CPU reference alongside the CUDA contraction.

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

| phase | time |
| --- | ---: |
| host tree construction | 4.97 ms |
| CUDA context setup | 198.28 ms |
| upload (the tree) | 313.46 ms |
| first call (JIT and caches) | 2.02 ms |
| steady contraction, synchronised | 0.176 ms / iteration |
| download | 0.086 ms |
| CPU contraction of the same tree, for reference | 0.138 ms / iteration |

Two things follow, and both are the point of separating the phases:

- **The fixed costs dominate**: one context plus one upload costs about 510 ms, while ten contractions cost
  1.8 ms. For a workload this small the device route is not worth it; the device only pays off when enough work
  is amortised against one setup and one transfer.
- **The device is not automatically faster per operation**: 0.176 ms versus 0.138 ms per contraction here. A
  branching tree contraction of small tensors does not have the arithmetic intensity that a device kernel wants.

**What these numbers do not cover, because this crate does not expose it yet:** the *wait* on an asynchronous
transfer and producer/consumer *overlap*. There is no stream or readiness surface on this route, so transfers are
synchronous and the example cannot tell "waiting" apart from "working"; the component-wise transfer migration that
F5 still owes is what would make those two measurable. The SRC route has its own separated breakdown (setup,
upload, warm-up, steady, download, CPU) in `cuda_src_timing_breakdown` under
`crates/tensor4all-treetn/tests/cuda_src_contraction.rs`, which is where the mixed workload's numbers live.

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
