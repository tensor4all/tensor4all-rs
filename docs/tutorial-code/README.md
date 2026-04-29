# Tensor4all QTT Tutorial Code

Runnable Rust source code for the QTT tutorials in the `tensor4all-rs` mdBook.

This directory is meant to stay small, readable, and consistent with the online
tutorial pages. The main workflow is:

1. run a Rust demo,
2. inspect the generated CSV data,
3. render plots with Julia,
4. change Rust constants or target functions,
5. rerun and compare.

## Prerequisites

- Rust and Cargo
- Julia
- CairoMakie for plotting

The Rust dependencies use local path dependencies to the workspace crates in
this checkout, so the tutorials are checked against the current
`tensor4all-rs` source tree.

Set up the Julia plotting environment from `docs/plotting/Project.toml` and
`docs/plotting/Manifest.toml` with:

```bash
julia --project=docs/plotting -e 'using Pkg; Pkg.instantiate()'
```

## First Run

Run the simplest QTT demo:

```bash
cargo run --bin qtt_function
```

Then generate its plots:

```bash
julia --project=docs/plotting docs/plotting/qtt_function_plot.jl
```

The Rust demo writes CSV files into `docs/data`. The Julia script writes plots
into `docs/plots`.

## Tutorial Order

Start with [docs/index.md](docs/index.md).

Suggested order:

1. [QTT of a scalar function](docs/tutorials/qtt_function_tutorial.md)
2. [QTT on a physical interval](docs/tutorials/qtt_interval_tutorial.md)
3. [QTTs for multivariate functions](docs/tutorials/qtt_multivariate_tutorial.md)
4. [Definite integrals](docs/tutorials/qtt_integral_tutorial.md)
5. [Sweep over bit depth](docs/tutorials/qtt_r_sweep_tutorial.md)
6. [Elementwise TreeTN product](docs/tutorials/qtt_elementwise_product_tutorial.md)
7. [Fourier transform](docs/tutorials/qtt_fourier_tutorial.md)

For terminology, see [docs/glossary.md](docs/glossary.md).

## Experiment Workflow

This repo is optimized for editing the Rust examples directly. Change constants
such as `BITS`, `TOLERANCE`, or a target function body, then rerun the demo.

Example:

```rust
const BITS: usize = 7;

fn target_function(x: f64) -> f64 {
    x.cosh()
}
```

Command-line flags are intentionally not part of the first workflow. They can be
added later if scripted sweeps become common.

## Verification

Run the local smoke check with:

```bash
./scripts/check.sh
```

The script writes demo output into `target/check/data`, not into `docs/data`.
That keeps normal verification from changing the curated tutorial artifacts.

## Refreshing Tracked Outputs

`docs/data` and `docs/plots` are tracked intentionally. They make the tutorials
easy to inspect without rerunning every workflow first.

To refresh a tutorial artifact deliberately:

```bash
cargo run --bin qtt_function
julia --project=docs/plotting docs/plotting/qtt_function_plot.jl
```

Then review the changed CSVs and plots before committing them.

## Staying Consistent with `tensor4all-rs`

Because this crate depends on the local workspace crates, updating
`tensor4all-rs` automatically updates the APIs used by these tutorials. If a
check fails after a library change, update the tutorial code or the mdBook page
that quotes it in the same branch.
