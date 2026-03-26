# Design: `matrixluci` Refactor and Extraction

**Date:** 2026-03-26  
**Scope:** Introduce a new `crates/matrixluci` crate inside this monorepo as the reusable LUCI / rrLU substrate for `matrixci`, `tensor4all-tensorci`, and future TreeTCI work.  
**Related issues:** `#332`, `#335`

## 1. Context

The current `matrixci` crate mixes three concerns:

- high-level matrix cross interpolation (`MatrixCI`)
- ACA (`MatrixACA`)
- LU-based pivot selection (`RrLU`, `MatrixLUCI`)

For TreeTCI, the blocker is not the full `matrixci` surface but the LU-based substrate:

- reusable rank-revealing LU / LUCI core
- lazy matrix element access
- block-rook pivot search
- clear benchmark targets

At the same time, it is acceptable for this refactor to be breaking. Backward compatibility is not required.

Future TreeTCI follow-up work must preserve upstream authorship metadata from
`TreeTCI.jl`, including the original `Project.toml` author entry for
Ryo Watanabe.

## 2. Goals

### 2.1 Functional goals

- Add a new `crates/matrixluci` crate in the current repository.
- Keep it in the same monorepo for now so existing repository-level workflows continue to apply:
  - `AGENTS.md`
  - `CLAUDE.md`
  - `./ai`
  - CI
  - branch protection
  - auto-merge
- Make `matrixluci` the low-level substrate for:
  - `matrixci`
  - `tensor4all-tensorci`
  - future `tensor4all-treetci`
- Support both dense and lazy candidate matrices.
- Implement block-rook search for the lazy path.
- Make pivot-only output the primary result shape.
- Keep factor reconstruction available, but as an optional helper layer rather than the primary kernel contract.

### 2.2 Performance goals

- Dense, no-truncation benchmark performance must not regress relative to direct `faer` full-pivoting LU.
- Benchmark sizes should include:
  - `32x32`
  - `64x64`
  - `100x100`
  - `128x128`
- Pass condition for dense no-truncation benchmarks:
  - `criterion` median of `matrixluci` is within `+5%` of the `faer` baseline.

## 3. Non-goals

- Porting TreeTCI in this issue.
- Preserving current public `matrixci` API compatibility.
- Introducing a separate repository now.
- Binding work (C API / Python / Julia) in this task.

## 4. High-level Architecture

### 4.1 New crate layout

Add:

- `crates/matrixluci`

Keep:

- `crates/matrixci` as a higher-level matrix CI crate rebuilt on top of `matrixluci`

The intended dependency direction is:

- `matrixluci` <- `matrixci`
- `matrixluci` <- `tensor4all-tensorci`
- `matrixluci` <- future `tensor4all-treetci`

`tensor4all-tensorci` and future TreeTCI should depend on `matrixluci` directly, not through `matrixci`.

### 4.2 Ownership model

`matrixluci` is tensor4all-owned code. Dense LU logic is based on a port of `faer` full-pivoting LU ideas and primitives, but the following remain tensor4all-owned semantics:

- rank-revealing truncation behavior
- pivot error reporting
- lazy source integration
- block-rook logic
- pivot-only result contract

## 5. Core Abstractions

### 5.1 Candidate matrix source

The substrate should operate on a matrix oracle, not on a concrete row-major matrix container.

```rust
pub trait CandidateMatrixSource<T> {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    /// Fill `out` with A[rows, cols] in column-major order.
    fn get_block(&self, rows: &[usize], cols: &[usize], out: &mut [T]);
}
```

Design choice:

- `get_block` is the required primitive.
- Any scalar `get(row, col)` helper is derived from `get_block`.
- Public block semantics are column-major.

This makes both dense and lazy paths compatible with the same API and matches the repository-wide column-major direction.

### 5.2 Pivot kernel

```rust
pub struct PivotKernelOptions {
    pub rel_tol: f64,
    pub abs_tol: f64,
    pub max_rank: usize,
    pub left_orthogonal: bool,
}

pub struct PivotSelectionCore {
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub pivot_errors: Vec<f64>,
    pub rank: usize,
}

pub trait PivotKernel<T> {
    fn factorize<S: CandidateMatrixSource<T>>(
        &self,
        source: &S,
        options: &PivotKernelOptions,
    ) -> crate::Result<PivotSelectionCore>;
}
```

Design choice:

- The primary kernel output is pivot-only.
- This avoids forcing factor reconstruction on lazy callers that only need pivot sets.
- `pivot_errors` and `last_pivot_error` are compatibility-sensitive outputs.
- Their semantics must match the current `matrixci::RrLU` behavior exactly, even where that behavior is slightly awkward.
- In particular, preserve the current distinctions between:
  - full-rank completion, where `last_pivot_error == 0`
  - tolerance-based stopping
  - `max_rank` stopping

### 5.3 Optional factor reconstruction

For high-level `MatrixCI` or debugging use, expose a separate factor reconstruction layer:

```rust
pub struct CrossFactors<T> {
    pub pivot: DenseMatrix<T>,      // P = A[I, J]
    pub pivot_cols: DenseMatrix<T>, // C = A[:, J]
    pub pivot_rows: DenseMatrix<T>, // R = A[I, :]
}
```

Derived helpers may compute:

- `C * P^{-1}`
- `P^{-1} * R`

But those are not required kernel outputs.

## 6. Dense and Lazy Backends

### 6.1 Dense backend

Dense candidate matrices should use column-major storage and be optimized for `faer` interop.

```rust
pub struct DenseMatrixSource<'a, T> {
    data: &'a [T], // column-major
    nrows: usize,
    ncols: usize,
}
```

Dense path implementation requirements:

- materialize the candidate matrix in column-major order
- use `faer` as the numerical primitive backend
- implement the rrLU / LUCI control flow in tensor4all code

### 6.2 Lazy backend

Lazy candidate matrices should expose block reads directly through `CandidateMatrixSource`.

Typical use cases:

- expensive callbacks
- candidate matrices too large or too wasteful to materialize fully
- TreeTCI edge updates

The lazy backend must support block-rook search without full dense materialization.

## 7. Kernel Implementations

### 7.1 Dense kernel

Provide a dense kernel specialized for the dense source path:

- name: `DenseFaerLuKernel` or equivalent
- implementation strategy:
  - tensor4all-owned rrLU / LUCI control flow
  - `faer` used for factorization primitives

This path defines the dense benchmark baseline for `matrixluci`.

### 7.2 Lazy block-rook kernel

Provide a lazy kernel specialized for on-demand evaluation:

- name: `LazyBlockRookKernel` or equivalent
- source access only through `get_block`
- no requirement to materialize the full candidate matrix

This is the required substrate for future TreeTCI work.

## 8. Public API Reshaping

### 8.1 `matrixluci`

`matrixluci` should export:

- source traits and source implementations
- pivot kernel traits and options
- pivot-only result types
- optional factor reconstruction helpers

It should not export a row-major `Matrix<T>`-centric API.

### 8.2 `matrixci`

`matrixci` should be rebuilt on top of `matrixluci`.

Recommended changes:

- keep `MatrixCI` as a higher-level matrix CI concept
- remove or sharply reduce incremental mutation APIs tied to the old dense matrix container
- treat pivot sets as the primary result
- make factor/materialization helpers secondary

`MatrixACA` is not a preservation target for this refactor. Since this work is explicitly breaking, removing ACA from `matrixci` is acceptable if keeping it would slow or complicate the LUCI substrate migration.

## 9. Migration Plan

Recommended order:

1. Add `crates/matrixluci`.
2. Implement the dense source, lazy source, and pivot-only core interfaces.
3. Implement dense no-truncation kernel and benchmark against `faer`.
4. Implement truncation behavior and pivot error reporting.
5. Implement lazy block-rook kernel.
6. Rebuild `matrixci` on top of `matrixluci`.
7. Migrate `tensor4all-tensorci` to depend directly on `matrixluci`.
8. Use `matrixluci` as the blocker-resolving substrate for TreeTCI.

## 10. Benchmark Requirements

Benchmark categories:

- dense / no truncation:
  - `matrixluci` vs direct `faer`
- dense / truncation enabled
- lazy source with cheap callback
- lazy source with expensive callback
- block-rook effectiveness
- end-to-end chain TCI regression benchmark

Acceptance rule for dense / no truncation:

- compare `criterion` medians
- pass if `matrixluci` median is within `+5%` of direct `faer`

## 11. Open Design Constraints Resolved

The following decisions are fixed by this design:

- breaking API changes are allowed
- the new substrate lives in the current repository for now
- the crate name is `matrixluci`
- pivot-only is the required output contract
- factor reconstruction is optional
- dense benchmark parity with `faer` is a completion criterion
- `#332` remains blocked on this substrate work

## 12. Success Condition

This design is complete when the repository has:

- a working `crates/matrixluci`
- direct consumers in `matrixci` and `tensor4all-tensorci`
- lazy + block-rook support
- benchmark evidence that dense no-truncation performance is not meaningfully worse than direct `faer`
