# mdBook Tutorials Design

## Goal

Add a "Tutorials" section to the mdBook documentation at
`docs/book/src/tutorials/` with 9 adapted tutorial pages from the playground
repo `sdirnboeck/rust-Tensor4all`. The tutorials show concrete, runnable
quantics tensor train examples with plots; the full executable projects remain
in the playground repo.

## Context

The mdBook currently has `[Guides]()` which explain concepts and APIs.
Tutorials are a distinct section: they show end-to-end examples without
duplicating the executable code in this repo. Each tutorial page links to the
corresponding source file in the playground repo.

Source materials:

- **Tutorial Markdown**: `docs/tutorials/` in
  [sdirnboeck/rust-Tensor4all](https://github.com/sdirnboeck/rust-Tensor4all)
  (9 `.md` files, raw URLs under `docs/tutorials/`)
- **Plots**: `docs/plots/` in the same repo (`.png` files, raw URLs under
  `docs/plots/`)
- **Existing mdBook**: `docs/book/src/` with `SUMMARY.md` navigation

## Non-Goals

- Do not copy executables, binaries, or `src/bin/*.rs` into this repo.
- Do not copy Julia plotting scripts (`docs/plotting/`), CSV data
  (`docs/data/`), or PDF plots into this repo.
- Do not write a landing/introduction page for the Tutorials section (same
  pattern as the existing `[Guides]()` stub section).
- Tutorial code blocks are compilable ` ```rust ` with hidden lines, not
  standalone rustdoc examples the reader could run in a web playground.
- Do not include `tt_basics_tutorial.md` (optional prelude, not a full
  tutorial).

## File Organization

```
docs/book/src/tutorials/
├── quantics-basics/
│   ├── qtt-scalar-function.md
│   ├── qtt_function_vs_qtt.png
│   ├── qtt_function_bond_dims.png
│   ├── qtt-physical-interval.md
│   ├── qtt_interval_function_vs_qtt.png
│   ├── qtt_interval_bond_dims.png
│   ├── qtt-definite-integrals.md
│   ├── qtt_integral_sweep.png
│   ├── sweep-bit-depth.md
│   ├── qtt_r_sweep_samples.png
│   ├── qtt_r_sweep_error.png
│   ├── qtt_r_sweep_runtime.png
│   ├── multivariate-functions.md
│   ├── qtt_multivariate_values.png
│   ├── qtt_multivariate_error.png
│   └── qtt_multivariate_bond_dims.png
└── computations-with-qtt/
    ├── elementwise-product.md
    ├── qtt_elementwise_product_factors.png
    ├── qtt_elementwise_product_product.png
    ├── qtt_elementwise_product_bond_dims.png
    ├── affine-transformation.md
    ├── qtt_affine_values.png
    ├── qtt_affine_error.png
    ├── qtt_affine_bond_dims.png
    ├── qtt_affine_operator_bond_dims.png
    ├── fourier-transform.md
    ├── qtt_fourier_transform.png
    ├── qtt_fourier_bond_dims.png
    ├── qtt_fourier_operator_bond_dims.png
    ├── partial-fourier2d.md
    ├── qtt_partial_fourier2d_values.png
    ├── qtt_partial_fourier2d_error.png
    └── qtt_partial_fourier2d_bond_dims.png
```

### Naming

- Markdown files: kebab-case (matches existing guides like `tensor-basics.md`).
- Plot files: use the original underscore names from the playground repo
  (e.g., `qtt_function_vs_qtt.png`).

### SUMMARY.md Placement

Insert before the `[Conventions]` line:

```markdown
- [Tutorials]()
  - [Quantics Basics]()
    - [QTT of a Scalar Function](tutorials/quantics-basics/qtt-scalar-function.md)
    - [QTT on a Physical Interval](tutorials/quantics-basics/qtt-physical-interval.md)
    - [Definite Integrals](tutorials/quantics-basics/qtt-definite-integrals.md)
    - [Sweep over Bit Depth](tutorials/quantics-basics/sweep-bit-depth.md)
    - [Multivariate Functions](tutorials/quantics-basics/multivariate-functions.md)
  - [Computations with QTT]()
    - [Elementwise Product](tutorials/computations-with-qtt/elementwise-product.md)
    - [Affine Transformation](tutorials/computations-with-qtt/affine-transformation.md)
    - [Fourier Transform](tutorials/computations-with-qtt/fourier-transform.md)
    - [2D Partial Fourier Transform](tutorials/computations-with-qtt/partial-fourier2d.md)
```

## Tutorial Overview

### Learning Progression

The tutorials are split into two sections with a clear learning arc:

**Quantics Basics** — build intuition for QTT fundamentals. Each tutorial
introduces one new concept, starting from bare-bones function encoding and
progressing to multiple dimensions:

1. *QTT of a Scalar Function* — simplest possible QTT: encode `f(x) = sin(30x)`,
   evaluate on grid, compare values, inspect bond dimensions.
2. *QTT on a Physical Interval* — same idea but mapped to a real interval
   `[-1, 2]` with `DiscretizedGrid`. Reader learns interval setup.
3. *Definite Integrals* — builds on #2, adds a single library call
   (`.integral()`). Shows how to extract a scalar result from a QTT.
4. *Sweep over Bit Depth* — varies `R` (number of quantics bits) from 2 to 15
   for `f(x) = sin(30x)`. Teaches the trade-off between accuracy and cost.
5. *Multivariate Functions* — two-dimensional QTT with two bit-ordering
   strategies (interleaved vs grouped). Reader learns multivariate encoding.

**Computations with QTT** — operations on existing QTTs. Assumes the reader
understands QTT basics:

6. *Elementwise Product* — build two QTTs, convert to `TreeTN`, multiply via
   `partial_contract`. Introduces QTT-on-QTT operations.
7. *Affine Transformation* — apply a 2D affine pullback `(x,y) → (x+y, y)`.
   Teaches passive coordinate transforms.
8. *Fourier Transform* — build a QTT for a Gaussian and apply the built-in
   quantics Fourier operator.
9. *2D Partial Fourier Transform* — Fourier along one coordinate of a 2D QTT.
   Most complex tutorial: combines multivariate, affine, and Fourier concepts.

### Reader Profile

Target reader: master student new to the topic. Knows basic QTT concepts
(e.g., what a tensor train is, what quantics encoding means roughly) but has
never used the `tensor4all-rs` library before. The tutorials assume no prior
experience with this library's API, types, or conventions.

The [Quantics Transform](../guides/quantics.md) guide is a suggested companion
but not a prerequisite.

## Page Template

Every tutorial page follows this structure:

| Section | Source | Action |
|---------|--------|--------|
| Title + Introduction | Original tutorial | Adapt (remove Julia mentions) |
| What the example computes | "What the example computes" | Adapt |
| Key API pieces | "Important Rust API pieces" | Adapt, use full ` ```rust ` with hidden lines |
| Figures | PNG plots | Embed with `![](./filename.png)` |
| How to read the plots | "How to read the plots" | Adapt |

The "Source code" section from the original tutorial is removed. The tutorials
are self-contained; readers write their own code based on the snippets shown.

**Removed** from originals:
- "Files in this example" (lists playground-internal paths)
- "Code organization" (module structure)
- "Pseudocode"
- Julia-specific sections (plotting commands, function tables, `julia --project=...`)

**Replaced** by mdBook navigation:
- "Suggested reading order" → mdBook prev/next buttons + SUMMARY hierarchy

## Full Page Example

Below is the complete markdown for the first tutorial page
(`qtt-scalar-function.md`). Every subsequent tutorial follows the same
structure; only the content of each section changes.

```markdown
# QTT of a scalar function

This tutorial shows a complete and beginner-friendly workflow for building
a Quantics Tensor Train (QTT) from an analytic function.

The core workflow is Rust-only: build the QTT, evaluate it, and inspect the
numerical data in the exported CSV files.

## What the example computes

The binary creates a quantics tensor train from an analytic function,
evaluates the QTT on the grid points, compares the QTT values against the
analytic function, and exports the results to CSV. It also tracks the bond
dimensions of the QTT cores.

## Key API pieces

The quantics construction uses `QuanticsTciBuilder`:

```rust
# use tensor4all_quanticstci::QuanticsTciBuilder;
# use anyhow::Result;
#
# fn f(x: f64) -> f64 { x.sin() }
#
# fn example() -> Result<()> {
#     let grid = todo!("create grid");
let mut builder = QuanticsTciBuilder::new(&grid);
builder.set_function(|x: &[f64]| Ok(f(x[0])));
let qtci = builder.build()?;
#     Ok(())
# }
```

The QTT is evaluated with the `evaluate` method:

```rust
# use anyhow::Result;
#
# fn example(qtci: &impl Evaluate) -> Result<()> {
let value = qtci.evaluate(&[x])?;
#     Ok(())
# }
```

## Figures

### Function versus QTT

![](./qtt_function_vs_qtt.png)

This figure overlays the analytic function and the reconstructed QTT values.
If the QTT approximation is working well, the sampled points should sit on top
of the curve almost perfectly.

### Bond dimensions

![](./qtt_function_bond_dims.png)

This figure shows the bond dimensions across the QTT cores. Spikes in the bond
dimension indicate sites where the function requires more entanglement —
typically near features like sharp transitions or high curvature.

## How to read the plots

- **Function vs QTT plot**: The solid line is the analytic function evaluated
  on a fine grid. The markers are the QTT values at the grid points. If the
  approximation is good, the markers lie on the line.
- **Bond dimensions plot**: Each bar shows the bond dimension of one QTT core.
  Higher bars mean more information is stored at that site. A flat profile
  indicates uniform complexity across sites.
```

The hidden lines (`# `) are invisible in the rendered HTML. They exist solely
to make the snippet compilable for CI.

## Code Block Policy

Use full ` ```rust ` blocks with hidden lines (`# ` prefix) for Rust snippets,
following the same pattern as the existing guides. Each block must compile
successfully under `mdbook test`.

The hidden boilerplate wraps each snippet:

````markdown
```rust
# use tensor4all_quanticstci::QuanticsTciBuilder;
# use anyhow::Result;
# use tensor4all_core::grid::DiscretizedGrid;
#
# fn f(x: f64) -> f64 { x.sin() }
#
# fn example() -> Result<()> {
#     let grid = DiscretizedGrid::new(0.0, 1.0, 256)?;
let mut builder = QuanticsTciBuilder::new(&grid);
builder.set_function(|x: &[f64]| Ok(f(x[0])));
let qtci = builder.build()?;
#     Ok(())
# }
```
````

The visible lines show the key API calls; the hidden lines provide the
imports and scaffolding needed for compilation. The web-based play button
will fail (same as existing guides, since dependencies aren't on the Rust
Playground), but CI's `mdbook test` compiles and validates them locally.

## Source Code Links

Each tutorial page has a "Source code" section linking to the playground repo:

```markdown
## Source code

The complete runnable project for this tutorial is available at:
<https://github.com/sdirnboeck/rust-Tensor4all/tree/main/src/bin/qtt_function.rs>
```

## Cross-References to Guides

Add cross-references when the tutorial uses a concept documented in a guide:

| Tutorial | Cross-reference |
|----------|----------------|
| `qtt-scalar-function.md` | — |
| `qtt-physical-interval.md` | — |
| `qtt-definite-integrals.md` | — |
| `sweep-bit-depth.md` | — |
| `multivariate-functions.md` | — |
| `elementwise-product.md` | — |
| `affine-transformation.md` | — |
| `fourier-transform.md` | See the [Quantics Transform](../guides/quantics.md) guide for Fourier operator details |
| `partial-fourier2d.md` | See the [Quantics Transform](../guides/quantics.md) guide for Fourier operator details |

## Tutorial–Source Mapping

| mdBook file | Source file | Plots |
|-------------|-------------|-------|
| `qtt-scalar-function.md` | `qtt_function_tutorial.md` | `qtt_function_vs_qtt.png`, `qtt_function_bond_dims.png` |
| `qtt-physical-interval.md` | `qtt_interval_tutorial.md` | `qtt_interval_function_vs_qtt.png`, `qtt_interval_bond_dims.png` |
| `qtt-definite-integrals.md` | `qtt_integral_tutorial.md` | `qtt_integral_sweep.png` |
| `sweep-bit-depth.md` | `qtt_r_sweep_tutorial.md` | `qtt_r_sweep_samples.png`, `qtt_r_sweep_error.png`, `qtt_r_sweep_runtime.png` |
| `multivariate-functions.md` | `qtt_multivariate_tutorial.md` | `qtt_multivariate_values.png`, `qtt_multivariate_error.png`, `qtt_multivariate_bond_dims.png` |
| `elementwise-product.md` | `qtt_elementwise_product_tutorial.md` | `qtt_elementwise_product_factors.png`, `qtt_elementwise_product_product.png`, `qtt_elementwise_product_bond_dims.png` |
| `affine-transformation.md` | `qtt_affine_tutorial.md` | `qtt_affine_values.png`, `qtt_affine_error.png`, `qtt_affine_bond_dims.png`, `qtt_affine_operator_bond_dims.png` |
| `fourier-transform.md` | `qtt_fourier_tutorial.md` | `qtt_fourier_transform.png`, `qtt_fourier_bond_dims.png`, `qtt_fourier_operator_bond_dims.png` |
| `partial-fourier2d.md` | `qtt_partial_fourier2d_tutorial.md` | `qtt_partial_fourier2d_values.png`, `qtt_partial_fourier2d_error.png`, `qtt_partial_fourier2d_bond_dims.png` |
