# QTTs for multivariate functions with `tensor4all-rs`

This tutorial builds two QTT approximations of a two-dimensional function,
one with interleaved quantics bits and one with grouped quantics bits.

The function is sampled on the half-open square `[0, 1) x [0, 1)`. Which can
be changed in the GridBuilder via the config.

The main purpose is to compare how the ordering of quantics bits changes the
bond dimensions, even when the represented function values are the same.

The workflow is split into two parts:

1. **Rust** builds the two QTTs, samples them on the full Cartesian grid, and
   exports CSV data.
2. **Julia + CairoMakie** reads the CSV files and turns them into plots.

That split keeps the Rust code focused on the tensor logic and keeps plotting
code out of Rust.

## Files in this example

The Rust side lives in:

- [`src/bin/qtt_multivariate.rs`](../../src/bin/qtt_multivariate.rs)
- [`src/qtt_multivariate_common.rs`](../../src/qtt_multivariate_common.rs)

The Julia plotting script lives in:

- [`docs/plotting/qtt_multivariate_plot.jl`](../plotting/qtt_multivariate_plot.jl)

The generated data and plots live in:

- [`docs/data/qtt_multivariate_samples.csv`](../data/qtt_multivariate_samples.csv)
- [`docs/data/qtt_multivariate_bond_dims.csv`](../data/qtt_multivariate_bond_dims.csv)
- [`docs/plots/qtt_multivariate_values.png`](../plots/qtt_multivariate_values.png)
- [`docs/plots/qtt_multivariate_values.png`](../plots/qtt_multivariate_values.png)
- [`docs/plots/qtt_multivariate_error.png`](../plots/qtt_multivariate_error.png)
- [`docs/plots/qtt_multivariate_error.png`](../plots/qtt_multivariate_error.png)
- [`docs/plots/qtt_multivariate_bond_dims.png`](../plots/qtt_multivariate_bond_dims.png)
- [`docs/plots/qtt_multivariate_bond_dims.png`](../plots/qtt_multivariate_bond_dims.png)

## Figures at a glance

### Values

![](../plots/qtt_multivariate_values.png)

This figure compares the exact sampled function with the two QTT
approximations. Both layouts target the same function values, but their
quantics tensor cores are arranged differently.

### Error

![](../plots/qtt_multivariate_error.png)

This figure plots absolute error on the same Cartesian grid. It is a quick
correctness check that both layouts reconstruct the sampled function.

### Bond dimensions

![](../plots/qtt_multivariate_bond_dims.png)

This figure is the main point of the tutorial. It compares the internal
bond-dimension profiles of the two layouts.

`Interleaved` alternates bits from `x` and `y`:

```text
x1, y1, x2, y2, ...
```

`Grouped` puts all bits from one variable before moving to the next:

```text
x1, x2, ..., y1, y2, ...
```

This is the Rust/tensor4all version of the sequential-versus-interleaved
comparison in the Julia notebook. The Julia notebook calls the grouped layout
`sequential`; tensor4all-rs names the same idea `Grouped`.

## What the example computes

The target function can be changed in 

```text
fn multivariate_target(x: f64, y: f64) -> f64
```

The default tutorial configuration uses:

- `bits = 5`
- `2^5 = 32` points in each direction
- `32 x 32 = 1024` exported sample points
- the half-open grid `[0, 1) x [0, 1)`
- max bond dimension `64`
- max QTCI sweeps `20`

The Rust program then:

1. Builds an interleaved `DiscretizedGrid`.
2. Builds a grouped `DiscretizedGrid`.
3. Calls `quanticscrossinterpolate(...)` once for each grid.
4. Collects dense values on the full Cartesian grid.
5. Uses Tensor4all's cached batch evaluation path for dense QTT sampling.
6. Writes exact values, QTT values, and absolute errors to CSV.
7. Writes both bond-dimension profiles to CSV.
8. Lets Julia turn those CSV files into plots.

The checked-in data uses `bits = 5` so the repository stays lightweight.
The same code can run larger local examples; see
[Running larger local grids](#running-larger-local-grids).

## Why `DiscretizedGrid` matters here

For continuous coordinates, `quanticscrossinterpolate(...)` takes a
`DiscretizedGrid` and a callback of type `Fn(&[f64]) -> f64`.

In this tutorial the callback is:

```rust
let f = move |coords: &[f64]| -> f64 {
    multivariate_target(coords[0], coords[1])
};
```

The grid is responsible for mapping between:

- physical coordinates like `(x, y)`
- 1-based grid indices like `[17, 9]`
- quantics indices used by the tensor train

For multivariate layouts, the unfolding scheme must be attached to the grid:

```rust
DiscretizedGrid::builder(&[bits, bits])
    .with_variable_names(&["x", "y"])
    .with_bounds(0.0, 1.0)
    .with_unfolding_scheme(UnfoldingScheme::Grouped)
    .include_endpoint(false)
    .build()
```

Important detail:

When using an explicit `DiscretizedGrid`, setting
`QtciOptions::with_unfoldingscheme(...)` is not enough. The grid already owns
the quantics index order, so the scheme belongs on the grid builder.

## How the Rust code is split

The main Rust file,
[`src/bin/qtt_multivariate.rs`](../../src/bin/qtt_multivariate.rs), stays
focused on orchestration:

- read the default configuration plus optional environment overrides
- define the target function for the tutorial experiment
- build the two `DiscretizedGrid` values directly
- build the `QtciOptions` directly
- call `quanticscrossinterpolate(...)` directly for both layouts
- collect dense samples
- collect bond dimensions
- print a summary
- write CSV files

The shared helper file,
[`src/qtt_multivariate_common.rs`](../../src/qtt_multivariate_common.rs),
contains the reusable tutorial logic:

- `MultivariateTutorialConfig`
- `DEFAULT_MULTIVARIATE_CONFIG`
- `collect_samples(...)`
- `collect_bond_dims(...)`
- `write_samples_csv(...)`
- `write_bond_dims_csv(...)`
- `print_summary(...)`
- convenience grid/QTT builders used by tests and reusable examples

This split keeps the binary readable and makes the helper easy to reuse in
tests or future multivariate experiments.

## Important Rust API pieces

### `MultivariateTutorialConfig`

This struct stores the knobs for the tutorial:

- `bits`
- lower and upper domain bounds
- whether the grid includes the upper endpoint
- QTCI tolerance
- maximum bond dimension
- maximum number of QTCI sweeps

The default values are stored in `DEFAULT_MULTIVARIATE_CONFIG`.

### `DiscretizedGrid::builder(...)`

The binary builds the two-dimensional grids with the Tensor4all grid builder.

Important builder calls in this example:

- `with_variable_names(&["x", "y"])`
- `with_bounds(0.0, 1.0)`
- `with_unfolding_scheme(UnfoldingScheme::Interleaved)`
- `with_unfolding_scheme(UnfoldingScheme::Grouped)`
- `include_endpoint(false)`

The helper module also has `build_multivariate_grid(...)` for tests and reuse,
but the tutorial binary spells out the API call so the layout choice is visible.

### `QtciOptions`

The binary builds interpolation options directly:

- tolerance
- maximum bond dimension
- maximum QTCI sweep count
- number of random initial pivots
- verbosity

These options are passed unchanged to `quanticscrossinterpolate(...)`.

### `quanticscrossinterpolate(...)`

This is the tensor4all-rs constructor for a QTT on physical coordinates.

It takes:

- a `DiscretizedGrid`
- a callback `Fn(&[f64]) -> f64`
- optional starting pivots
- `QtciOptions`

Output:

- the QTT object
- a rank history
- an error history

The returned QTT object is a `QuanticsTensorCI2<f64>`.

### `collect_samples(...)`

This helper evaluates both QTTs on the full Cartesian grid and stores:

- grid indices
- physical coordinates
- exact values
- interleaved QTT values
- grouped QTT values
- absolute errors for both layouts

For dense exports it uses the underlying `TensorTrain` plus
`tensor4all_simplett::TTCache::evaluate_many(...)`. This is faster than calling
scalar `QuanticsTensorCI2::evaluate(...)` once per grid point.

### `tensor_train()` and `TTCache::evaluate_many(...)`

`QuanticsTensorCI2::tensor_train()` exposes the underlying tensor train.

`TTCache::evaluate_many(...)` then evaluates many quantics indices while
reusing cached left and right contractions.

This matters for `bits = 9`, where a full two-dimensional grid already has:

```text
512 x 512 = 262144 points
```

Evaluating those one point at a time works, but it misses Tensor4all's cached
batch path.

### `link_dims()` and `rank()`

These inspect the QTT bond dimensions:

- `link_dims()` returns the full bond-dimension profile
- `rank()` returns the largest bond dimension

The bond plot is built from `link_dims()`.

### CSV writers

`write_samples_csv(...)` writes the dense comparison table used by the value
and error plots.

`write_bond_dims_csv(...)` writes the bond-dimension profiles used by the bond
plot.

## High-level call order

The Rust side follows this sequence:

1. start with `DEFAULT_MULTIVARIATE_CONFIG`
2. apply optional environment overrides
3. build an interleaved grid
4. build a grouped grid
5. build a QTT on the interleaved grid
6. build a QTT on the grouped grid
7. batch-evaluate both QTTs on the dense Cartesian grid
8. compare both QTTs against the analytic target function
9. collect bond dimensions from both QTTs
10. print a short terminal summary
11. write CSV files
12. plot the CSV files with Julia

## Pseudocode view

This is the same workflow in compact pseudocode:

```text
config = DEFAULT_MULTIVARIATE_CONFIG
config = apply_environment_overrides(config)

target(x, y) = cos(20*pi*x*y) / 1000

interleaved_grid = build_grid(config, Interleaved)
grouped_grid = build_grid(config, Grouped)

interleaved_qtt, interleaved_ranks, interleaved_errors =
    quanticscrossinterpolate(interleaved_grid, target, options)

grouped_qtt, grouped_ranks, grouped_errors =
    quanticscrossinterpolate(grouped_grid, target, options)

interleaved_values =
    TTCache(interleaved_qtt.tensor_train()).evaluate_many(all_grid_points)

grouped_values =
    TTCache(grouped_qtt.tensor_train()).evaluate_many(all_grid_points)

samples = []
for x_index in 1:2^bits
    for y_index in 1:2^bits
        x, y = grid_coordinates(x_index, y_index)
        exact = target(x, y)
        interleaved_value = interleaved_values[x_index, y_index]
        grouped_value = grouped_values[x_index, y_index]
        push sample row with exact values and errors
    end
end

bond_dims = zip(interleaved_qtt.link_dims(), grouped_qtt.link_dims())

write_samples_csv(samples)
write_bond_dims_csv(bond_dims)

plot_with_julia(samples, bond_dims)
```

## Function map

| Where | Function or type | Purpose |
|---|---|---|
| `src/bin/qtt_multivariate.rs` | `main()` | runs the tutorial workflow |
| `src/bin/qtt_multivariate.rs` | `config_from_env()` | applies optional local run overrides |
| `src/bin/qtt_multivariate.rs` | `DiscretizedGrid::builder(...)` | builds the interleaved and grouped grids |
| `src/bin/qtt_multivariate.rs` | `QtciOptions::default().with_...` | configures QTCI accuracy and limits |
| `src/bin/qtt_multivariate.rs` | `quanticscrossinterpolate(...)` | builds each QTT |
| `src/bin/qtt_multivariate.rs` | `multivariate_target(x, y)` | analytic target function for this tutorial |
| `src/qtt_multivariate_common.rs` | `MultivariateTutorialConfig` | stores tutorial parameters |
| `src/qtt_multivariate_common.rs` | `DEFAULT_MULTIVARIATE_CONFIG` | default checked-in tutorial settings |
| `src/qtt_multivariate_common.rs` | `collect_samples(...)` | creates dense value and error rows |
| `src/qtt_multivariate_common.rs` | `collect_bond_dims(...)` | pairs the two bond profiles for CSV |
| `src/qtt_multivariate_common.rs` | `write_samples_csv(...)` | writes value and error data |
| `src/qtt_multivariate_common.rs` | `write_bond_dims_csv(...)` | writes bond-dimension data |
| `src/qtt_multivariate_common.rs` | `print_summary(...)` | prints terminal diagnostics |
| `docs/plotting/qtt_multivariate_plot.jl` | `plot_values(...)` | renders exact and QTT heatmaps |
| `docs/plotting/qtt_multivariate_plot.jl` | `plot_errors(...)` | renders absolute error heatmaps |
| `docs/plotting/qtt_multivariate_plot.jl` | `plot_bond_dims(...)` | renders the bond-dimension comparison |

## Julia mapping

The table below gives a rough translation between the Julia notebook idea and
the Rust version.

| Julia notebook concept | Rust `tensor4all-rs` equivalent |
|---|---|
| `R` | `config.bits` |
| `xis(R)` / sampled coordinate vectors | `DiscretizedGrid::grid_origcoords(...)` |
| `f(x, y)` | `multivariate_target(x, y)` |
| sequential tensor layout | `UnfoldingScheme::Grouped` |
| interleaved tensor layout | `UnfoldingScheme::Interleaved` |
| `QTT(...)` | `quanticscrossinterpolate(...)` |
| bond-dimension list | `qtci.link_dims()` |
| dense matrix reconstruction | `collect_samples(...)` |
| plotting in the notebook | plotting from CSV with Julia + CairoMakie |

Do not use `UnfoldingScheme::Fused` for the sequential comparison. In
tensor4all-rs, `Fused` combines variables at the same bit level into a single
tensor core. The Julia notebook's sequential layout has separate binary cores
grouped by variable, so `Grouped` is the closer match.

## How to read the plots

### Values

The first figure shows:

- the exact sampled function
- the interleaved QTT reconstruction
- the grouped QTT reconstruction

The visual structure should match across all three panels.

### Error

The second figure shows absolute error for both layouts.

Large isolated error regions usually mean one of three things:

- the rank cap is too small
- the QTCI sweep count is too small
- the function is harder to compress in that layout

### Bond dimensions

The third figure compares `link_dims()` for the two QTTs.

Interpretation:

- smaller values mean a more compact representation
- larger values mean the chosen layout needs more internal room
- different layouts can have very different bond profiles for the same function

## Running the workflow

1. Build the QTTs and write the CSV files:

```bash
cargo run --bin qtt_multivariate --offline
```

2. Turn the CSV files into plots:

```bash
julia --project=docs/plotting docs/plotting/qtt_multivariate_plot.jl
```

3. Inspect the generated figures in `docs/plots/`.

## Running larger local grids

The checked-in tutorial data uses `bits = 5` so the docs stay small. For a
larger local run, override the binary configuration without editing source:

```bash
QTT_MULTIVARIATE_BITS=9 TENSOR4ALL_DATA_DIR=/tmp/qtt-multivariate-bits9 cargo run --bin qtt_multivariate --offline
```

For harder functions or stricter interleaved accuracy, you can also override
the bond cap and sweep count:

```bash
QTT_MULTIVARIATE_BITS=9 \
QTT_MULTIVARIATE_MAXBONDDIM=128 \
QTT_MULTIVARIATE_MAXITER=40 \
TENSOR4ALL_DATA_DIR=/tmp/qtt-multivariate-bits9 \
cargo run --bin qtt_multivariate --offline
```

Useful environment variables:

| Variable | Default | Meaning |
|---|---:|---|
| `QTT_MULTIVARIATE_BITS` | `5` | bits per dimension |
| `QTT_MULTIVARIATE_MAXBONDDIM` | `64` | maximum QTT bond dimension |
| `QTT_MULTIVARIATE_MAXITER` | `20` | number of QTCI sweeps |
| `TENSOR4ALL_DATA_DIR` | `docs/data` | output directory for CSV files |

## What to change next

Change `multivariate_target(...)` in
[`src/bin/qtt_multivariate.rs`](../../src/bin/qtt_multivariate.rs) to
try another function of `(x, y)`.

Smooth functions usually compress more easily than functions with sharp jumps
or strong oscillations. If a new function looks inaccurate, first try increasing
`QTT_MULTIVARIATE_MAXBONDDIM`; if the rank has not hit the cap, try increasing
`QTT_MULTIVARIATE_MAXITER`.
