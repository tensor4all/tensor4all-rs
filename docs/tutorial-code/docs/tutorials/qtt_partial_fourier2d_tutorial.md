# 2D Partial Fourier Transform

This tutorial applies a Fourier transform to only one coordinate of a two-dimensional QTT.

The input function is

```text
f(x,t) = exp(-x^2 / 2) * cos(2 pi omega t)
```

with `omega = 3`. Only `x` is transformed:

```text
f(x,t) -> F(k,t)
```

The analytic reference is

```text
F(k,t) = sqrt(2 pi) * exp(-2 pi^2 k^2) * cos(2 pi omega t)
```

## Files in this example

The Rust side lives in:

- [`src/bin/qtt_partial_fourier2d.rs`](../../src/bin/qtt_partial_fourier2d.rs)
- [`src/qtt_partial_fourier2d_common.rs`](../../src/qtt_partial_fourier2d_common.rs)

The Julia plotting script lives in:

- [`docs/plotting/qtt_partial_fourier2d_plot.jl`](../plotting/qtt_partial_fourier2d_plot.jl)

The generated data and plots live in:

- [`docs/data/qtt_partial_fourier2d_samples.csv`](../data/qtt_partial_fourier2d_samples.csv)
- [`docs/data/qtt_partial_fourier2d_bond_dims.csv`](../data/qtt_partial_fourier2d_bond_dims.csv)
- [`docs/data/qtt_partial_fourier2d_operator_bond_dims.csv`](../data/qtt_partial_fourier2d_operator_bond_dims.csv)
- [`docs/plots/qtt_partial_fourier2d_values.png`](../plots/qtt_partial_fourier2d_values.png)
- [`docs/plots/qtt_partial_fourier2d_values.png`](../plots/qtt_partial_fourier2d_values.png)
- [`docs/plots/qtt_partial_fourier2d_error.png`](../plots/qtt_partial_fourier2d_error.png)
- [`docs/plots/qtt_partial_fourier2d_error.png`](../plots/qtt_partial_fourier2d_error.png)
- [`docs/plots/qtt_partial_fourier2d_bond_dims.png`](../plots/qtt_partial_fourier2d_bond_dims.png)
- [`docs/plots/qtt_partial_fourier2d_bond_dims.png`](../plots/qtt_partial_fourier2d_bond_dims.png)

## Figures at a glance

### Values

![](../plots/qtt_partial_fourier2d_values.png)

This figure compares the analytic reference with the QTT approximation after the partial Fourier transform.

### Error

![](../plots/qtt_partial_fourier2d_error.png)

This figure shows the absolute error on the (k, t) grid. The error is largest at the boundaries of the spatial domain due to discrete-continuous scaling differences.

### Bond dimensions

![](../plots/qtt_partial_fourier2d_bond_dims.png)

This figure compares the input and transformed state bond dimensions alongside the partial Fourier MPO bond dimensions.

## Interleaved Partial Operator

The two-dimensional QTT uses `UnfoldingScheme::Interleaved`. For variables `(x,t)`, the site order is

```text
x0, t0, x1, t1, x2, t2, ...
```

The one-dimensional Fourier operator is built on sites

```text
0, 1, 2, ...
```

The tutorial renames those operator nodes to the `x` sites:

```text
0 -> 0
1 -> 2
2 -> 4
...
```

Before application, the helper expands that x-only operator onto the full interleaved state chain by inserting identity tensors on the `t` sites. Then `align_to_state(...)` aligns the full operator's index mappings to the 2D state and `apply_linear_operator` applies it without changing the temporal dimension.

## What the example computes

The default tutorial configuration uses:

- `bits = 6`
- `2^6 = 64` points in each direction
- `64 x 64 = 4096` exported sample points
- the symmetric spatial grid `[-10, 10]`
- the temporal grid `[0, 1]`
- max bond dimension `64`
- max QTCI sweeps `20`

The Rust program then:

1. Builds an interleaved `DiscretizedGrid` for the input.
2. Builds an interleaved `DiscretizedGrid` for frequency output.
3. Calls `quanticscrossinterpolate(...)` to build the 2D QTT.
4. Builds a 1D Fourier operator and renames its nodes to the x-sites.
5. Calls `apply_linear_operator(...)` to apply only the x-part of the Fourier.
6. Collects samples on the full (k, t) Cartesian grid.
7. Writes exact values, QTT values, and absolute errors to CSV.
8. Writes bond-dimension profiles to CSV.
9. Lets Julia turn those CSV files into plots.

## Running the workflow

1. Build the QTTs and write the CSV files:

```bash
cargo run --bin qtt_partial_fourier2d --offline
```

2. Turn the CSV files into plots:

```bash
julia --project=docs/plotting docs/plotting/qtt_partial_fourier2d_plot.jl
```

3. Inspect the generated figures in `docs/plots/`.

## How the Rust code is split

The main Rust file, [`src/bin/qtt_partial_fourier2d.rs`](../../src/bin/qtt_partial_fourier2d.rs), orchestrates the workflow:

- read the default configuration
- build the input and frequency grids
- build the source QTT using `quanticscrossinterpolate`
- build the partial Fourier operator
- apply the operator to the QTT
- collect samples and bond dimensions
- print a summary
- write CSV files

The shared helper file, [`src/qtt_partial_fourier2d_common.rs`](../../src/qtt_partial_fourier2d_common.rs), contains the reusable tutorial logic:

- `PartialFourier2dConfig` and `DEFAULT_PARTIAL_FOURIER2D_CONFIG`
- `source_function` and `partial_fourier_reference`
- `build_input_grid`, `build_frequency_grid`, `build_source_qtt`
- `build_partial_fourier_operator` and `transform_x_dimension`
- `collect_samples`, `collect_bond_dims`
- `write_samples_csv`, `write_bond_dims_csv`, `write_operator_bond_dims_csv`
- `print_summary`

This split keeps the binary readable and makes the helper easy to reuse in tests.

## Key API pieces

### `DiscretizedGrid::builder(...)` with Interleaved

The binary builds two-dimensional grids with Tensor4all's grid builder:

```rust
DiscretizedGrid::builder(&[bits, bits])
    .with_variable_names(&["x", "t"])
    .with_lower_bound(&[config.x_lower_bound, config.t_lower_bound])
    .with_upper_bound(&[config.x_upper_bound, config.t_upper_bound])
    .with_unfolding_scheme(UnfoldingScheme::Interleaved)
    .include_endpoint(config.include_endpoint)
    .build()?
```

The interleaved scheme alternates bits from `x` and `t` in the tensor train.

### Node renaming for partial operators

The 1D Fourier operator has nodes named `0, 1, 2, ...`. The tutorial renames them to match the `x` sites in the 2D interleaved layout:

```rust
pub fn x_site_node_mapping(bits: usize) -> Vec<(usize, usize)> {
    (0..bits).map(|site| (site, 2 * site)).collect()
}
```

After renaming, `transform_x_dimension(...)` expands the operator across the intervening `t` sites with identity tensors, then `align_to_state(...)` aligns it to the 2D state.

### `apply_linear_operator`

This function applies the partial operator. Sites that the operator doesn't touch (the `t` sites in this case) are automatically filled with identity operators.

## Function map

| Where | Function or type | Purpose |
|---|---|---|
| `src/bin/qtt_partial_fourier2d.rs` | `main()` | runs the tutorial workflow |
| `src/qtt_partial_fourier2d_common.rs` | `PartialFourier2dConfig` | stores tutorial parameters |
| `src/qtt_partial_fourier2d_common.rs` | `DEFAULT_PARTIAL_FOURIER2D_CONFIG` | default checked-in settings |
| `src/qtt_partial_fourier2d_common.rs` | `source_function(x, t)` | input Gaussian modulated by cosine |
| `src/qtt_partial_fourier2d_common.rs` | `partial_fourier_reference(k, t)` | analytic Fourier transform |
| `src/qtt_partial_fourier2d_common.rs` | `build_input_grid` | creates 2D interleaved input grid |
| `src/qtt_partial_fourier2d_common.rs` | `build_frequency_grid` | creates 2D interleaved frequency grid |
| `src/qtt_partial_fourier2d_common.rs` | `build_source_qtt` | builds QTT using cross-interpolation |
| `src/qtt_partial_fourier2d_common.rs` | `build_partial_fourier_operator` | builds 1D Fourier and renames nodes |
| `src/qtt_partial_fourier2d_common.rs` | `transform_x_dimension` | applies partial operator |
| `src/qtt_partial_fourier2d_common.rs` | `collect_samples` | evaluates transformed QTT on grid |
| `docs/plotting/qtt_partial_fourier2d_plot.jl` | `plot_values` | renders analytic vs QTT heatmaps |
| `docs/plotting/qtt_partial_fourier2d_plot.jl` | `plot_error` | renders absolute error heatmap |
| `docs/plotting/qtt_partial_fourier2d_plot.jl` | `plot_bond_dims` | renders bond dimension comparison |
