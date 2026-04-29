# QTT sweep over `R` for `sin(30x)`

This example shows how the QTT approximation changes when we vary the number
of quantics bits `R`.

For each `R` in `2..15` we use a grid with

```text
N = 2^R
```

sample points, build a QTT for

```text
f(x) = sin(30x)
```

and then compare the QTT values against the analytic function again.

The runtime measurement is intentionally narrow:

- only the QTT construction is timed
- sampling, CSV export, and Julia plotting are not part of the runtime number

## Files in this example

Rust:

- [`src/bin/qtt_r_sweep.rs`](../../src/bin/qtt_r_sweep.rs)
- [`src/qtt_r_sweep_utils.rs`](../../src/qtt_r_sweep_utils.rs)

Julia:

- [`docs/plotting/qtt_r_sweep_plot.jl`](../plotting/qtt_r_sweep_plot.jl)

Generated data:

- [`docs/data/qtt_r_sweep_samples.csv`](../data/qtt_r_sweep_samples.csv)
- [`docs/data/qtt_r_sweep_stats.csv`](../data/qtt_r_sweep_stats.csv)

Generated plots:

- [`docs/plots/qtt_r_sweep_samples.png`](../plots/qtt_r_sweep_samples.png)
- [`docs/plots/qtt_r_sweep_samples.png`](../plots/qtt_r_sweep_samples.png)
- [`docs/plots/qtt_r_sweep_error.png`](../plots/qtt_r_sweep_error.png)
- [`docs/plots/qtt_r_sweep_error.png`](../plots/qtt_r_sweep_error.png)
- [`docs/plots/qtt_r_sweep_runtime.png`](../plots/qtt_r_sweep_runtime.png)
- [`docs/plots/qtt_r_sweep_runtime.png`](../plots/qtt_r_sweep_runtime.png)

## Figures at a glance

### Sample plot

![](../plots/qtt_r_sweep_samples.png)

This figure overlays the analytic curve with the QTT sample points for several
selected `R` values. It is the easiest way to see how the discretization gets
denser as `R` grows.

### Error plot

![](../plots/qtt_r_sweep_error.png)

This figure shows the mean absolute error per `R`. It summarizes how closely
the QTT follows the analytic function as the number of grid points increases.

### Runtime plot

![](../plots/qtt_r_sweep_runtime.png)

This figure shows how long the QTT construction itself takes for each `R`.
Only the interpolation time is measured here, not CSV writing or plotting.

## What the Rust code does

For each `R`:

1. define `N = 2^R`
2. build a QTT with `quanticscrossinterpolate_discrete(...)`
3. measure only the QTT construction time
4. sample the QTT again on all grid points with `evaluate(...)`
5. compute the pointwise absolute error
6. store the samples and per-R summary statistics in CSV

## High-level call order

The Rust side follows this order:

1. set `R`
2. call `quanticscrossinterpolate_discrete(...)`
3. inspect the interpolation result through `qtci.evaluate(...)`
4. collect all samples for the current grid
5. compute the mean absolute error
6. save the per-point and per-R tables
7. let Julia plot the CSV files

## Pseudocode

```text
function f(x) = sin(30x)

for R in 2..15:
    N = 2^R

    start timer
    qtci = quanticscrossinterpolate_discrete(grid size = N, callback = f)
    runtime = elapsed time

    for i in 1..N:
        x = (i - 1) / N
        exact = f(x)
        qtt_value = qtci.evaluate(i)
        abs_error = |exact - qtt_value|

    mean_error = average(abs_error over the grid)
    write rows to CSV
```

## Important library functions

### `quanticscrossinterpolate_discrete(...)`

This is the main Tensor4all constructor used in the sweep.

Inputs:

- the grid size, here `2^R`
- a callback of type `Fn(&[i64]) -> f64`
- interpolation options

Output:

- a QTT interpolation object
- rank history
- error history

In this example the callback converts the discrete grid index into `x in [0,1)`.

### `evaluate(...)`

This is the library method that reads one QTT value back at a discrete grid point.

Inputs:

- one grid index

Output:

- the approximated function value at that point

### `mean_abs_error(...)`

This helper is not a Tensor4all API function. It is only used to summarize the
per-grid-point errors into one number per `R`.

## How to read the plots

### Sample plot

The first plot overlays:

- the analytic curve `sin(30x)`
- the QTT sample points for each `R`

This lets you see how the sampled approximation gets denser as `R` grows.

### Error plot

The second plot shows the mean absolute error per `R`.

### Runtime plot

The third plot shows the construction time of the QTT only.

## Julia mapping

| Rust concept | Julia view |
|---|---|
| `quanticscrossinterpolate_discrete(...)` | build the QTT from the sampled function |
| `evaluate(...)` | read back the QTT value at a grid point |
| CSV export | input to the CairoMakie plots |

## A note on the grid

For each `R`, the grid uses `N = 2^R` points.

The helper converts the discrete index into

```text
x = (i - 1) / N
```

so the sampled points lie in the interval `[0, 1)`.
