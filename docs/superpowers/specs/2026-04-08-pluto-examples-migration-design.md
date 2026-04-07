# Pluto Examples Migration Design

**Date:** 2026-04-08
**Status:** Approved
**Scope:** Port T4APlutoExamples content to tensor4all-rs mdBook guides (Rust code)

## Goal

Port the practical examples from [T4APlutoExamples](https://tensor4all.org/T4APlutoExamples/)
to the tensor4all-rs mdBook user guide, using Rust code with assertions that
verify interpolation accuracy and computation correctness.

## Pluto-to-Book Mapping

| Pluto Example | mdBook Target | Action |
|---|---|---|
| quantics1d.jl | guides/tci.md | Enrich with 1D/2D QTT examples |
| quantics2d.jl | guides/tci.md | Enrich with 2D multivariate example |
| quantics1d_advanced.jl | guides/tci-advanced.md | **New page** |
| compress.jl | guides/compress.md | **New page** |
| qft.jl | guides/qft.md | **New page** (1D + manual 2D) |
| interfacingwithitensors.jl | — | Skip (Julia-specific) |
| about_pluto.jl | — | Skip (Pluto-specific) |
| plots.jl | — | Skip (visualization-specific) |

## Content Details

### 1. guides/tci.md (Enrich)

Add practical examples corresponding to quantics1d.jl and quantics2d.jl:

**1D QTT interpolation section:**
- Define a 1D function (e.g., multi-scale oscillatory function)
- Construct QTT via `quanticscrossinterpolate_discrete`
- Evaluate at specific points and assert accuracy: `assert!((val - exact).abs() < tol)`
- Compute integral and assert against analytic result
- Check convergence: `assert!(*errors.last().unwrap() < tol)`

**2D QTT interpolation section:**
- Define a 2D function
- Construct QTT via `quanticscrossinterpolate_discrete` with 2D grid
- Evaluate and assert accuracy at multiple points
- Verify convergence

Each section starts with a link: "This example corresponds to
[Quantics TCI of univariate function](https://tensor4all.org/T4APlutoExamples/quantics1d.html)"

### 2. guides/tci-advanced.md (New)

Corresponds to quantics1d_advanced.jl.

**Content:**
- Link to Pluto example at top
- Low-level `crossinterpolate2` direct usage (bypassing QuanticsTCI wrapper)
- Manual grid construction and quantics-domain function wrapping
- Initial pivot selection strategy (choose points where |f(x)| is large)
- Using `CachedFunction` for evaluation tracking
- Converting TCI result to tensor train
- Computing integral from tensor train (sum * dx)

**All examples with assertions:**
- Evaluate TT at specific points, assert against original function
- Assert integral accuracy against analytic value

### 3. guides/compress.md (New)

Corresponds to compress.jl.

**Content:**
- Link to Pluto example at top
- Problem: compress existing multi-dimensional data (not a function)
- Approach: wrap array element access as a function for `crossinterpolate2`
- Example: 3D dataset (e.g., `cos(x_i) + cos(y_j) + cos(z_k)`)
- Run TCI compression with tolerance
- Verify accuracy: assert max error < tolerance
- Show bond dimensions of compressed representation

**Assertions:**
- `assert!((compressed_val - original_val).abs() < tol)` at multiple points
- `assert!(max_error < tol)`

### 4. guides/qft.md (New)

Corresponds to qft.jl.

**Content:**
- Link to Pluto example at top

**1D QFT (fully supported):**
- Define a 1D function (e.g., sum of exponentials)
- Construct QTT via `quanticscrossinterpolate`
- Build Fourier operator via `quantics_fourier_operator(r, options)`
- Apply operator to TreeTN via `apply_linear_operator`
- Normalize result (1/sqrt(2^R))
- Verify: compare result against discrete DFT at specific frequencies

**2D QFT (manual, using partial apply):**
- Construct 2D function in interleaved quantics encoding
- Build 1D Fourier operator
- Apply to variable-1 sites using `apply_linear_operator` partial apply
  (operator nodes are subset of state nodes; identity auto-extended)
- Apply to variable-2 sites similarly
- Verify against 2D DFT reference values
- Note: "A dedicated multivar Fourier API is not yet available.
  This example uses partial apply to achieve 2D transforms manually."

**Assertions:**
- 1D: assert Fourier coefficients match DFT at specific frequencies
- 2D: assert 2D Fourier coefficients match reference values

### 5. SUMMARY.md Update

```markdown
- [Guides]()
  - [Tensor Basics](guides/tensor-basics.md)
  - [Tensor Train](guides/tensor-train.md)
  - [Tensor Cross Interpolation](guides/tci.md)
  - [TCI Advanced Topics](guides/tci-advanced.md)
  - [Compressing Existing Data](guides/compress.md)
  - [Quantics Transform](guides/quantics.md)
  - [Quantum Fourier Transform](guides/qft.md)
  - [Tree Tensor Networks](guides/tree-tn.md)
```

## Cross-Cutting Requirements

### Assertions

Every code example MUST include assertions that verify computational correctness:
- Interpolation accuracy: evaluate at specific points, compare to exact values
- Integral accuracy: compare to analytic results
- Compression accuracy: compare compressed vs original data
- Transform accuracy: compare to reference DFT values

Pattern: `assert!((computed - exact).abs() < tolerance)`

### Pluto Example Links

Each page/section that corresponds to a Pluto example includes a link at the top:
> "This guide corresponds to the Julia [Example Name](https://tensor4all.org/T4APlutoExamples/xxx.html) notebook."

### Code Style

- All code blocks: `rust,ignore` (mdBook, not doctests)
- Use `?` operator (assume wrapping `fn main() -> Result<()>`)
- Import statements shown explicitly
- Comments explain the "why", not the "what"

## Out of Scope

- Julia code examples (Rust only)
- Dedicated multivar Fourier API (document manual approach, improve API later)
- Visualization/plotting (no Rust equivalent of Plots.jl in the book)
- ITensors.jl interfacing (Julia-specific)
