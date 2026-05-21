# Alternating Cross Interpolation Rust API Design

## Goal

Port the Rust public API for
[`AlternatingCrossInterpolation.jl`](https://github.com/tensor4all/AlternatingCrossInterpolation.jl)
into `tensor4all-rs`. The first scope is Rust-only: no C API, Python API, or
Tensor4all.jl binding surface is included in this design.

ACI computes elementwise operations on existing tensor trains with an
alternating two-site cross interpolation scheme. The implementation should use
the existing `tensor4all-simplett::TensorTrain`, `tensor4all-tcicore`
MatrixLUCI, and `tensor4all-tensorbackend::Matrix` abstractions.

## Attribution

Create a new crate with explicit upstream attribution:

```toml
authors = [
    "Marc Ritter <mritter@flatironinstitute.org>",
    "Hiroshi Shinaoka <h.shinaoka@gmail.com>",
    "tensor4all contributors",
]
```

The crate-level documentation or a `NOTICE` file should state:

```text
This crate ports AlternatingCrossInterpolation.jl, originally authored by
Marc Ritter <mritter@flatironinstitute.org> and contributors.
```

The upstream MIT copyright notice must be preserved:

```text
Copyright (c) 2026 Marc Ritter <mritter@flatironinstitute.org> and contributors
```

## Crate Boundary

Add a new crate:

```text
crates/tensor4all-aci
```

Primary dependencies:

- `tensor4all-simplett`
- `tensor4all-tcicore`
- `tensor4all-tensorbackend`

This keeps ACI separate from `tensor4all-simplett` core tensor-train
containers and from `tensor4all-tensorci`, whose main role is interpolation of
external functions into tensor trains.

## Public API Shape

Expose a small high-level API around simple tensor trains:

```rust
pub fn elementwise<T, F>(
    op: F,
    inputs: &[TensorTrain<T>],
    options: &AciOptions<T>,
) -> Result<AciResult<T>>
where
    F: FnMut(&[T]) -> T;

pub fn elementwise_batched<T, F>(
    op: F,
    inputs: &[TensorTrain<T>],
    options: &AciOptions<T>,
) -> Result<AciResult<T>>
where
    F: FnMut(ElementwiseBatch<'_, T>, &mut [T]) -> Result<()>;
```

`elementwise` is a convenience wrapper over the batched path. The internal
implementation should avoid maintaining two separate update algorithms.

`AciResult<T>` should contain:

- `tensor_train: TensorTrain<T>`
- `ranks: Vec<usize>`
- `errors: Vec<f64>`

`AciOptions<T>` should contain:

- `max_iters`
- `min_iters`
- `max_bond_dim`
- `tolerance`
- `scale_tolerance`
- `initial_guess`
- deterministic random initialization controls, such as `rng_seed`

## Batch Evaluation Contract

Batched elementwise callbacks receive a 2D column-major view with shape
`(n_inputs, n_points)`. For point `p` and input tensor train `k`, the local
value is stored at:

```text
values[k + n_inputs * p]
```

Represent this with an ACI-specific view wrapper:

```rust
pub struct ElementwiseBatch<'a, T> {
    values: &'a [T],
    n_inputs: usize,
    n_points: usize,
}
```

The wrapper should expose:

```rust
impl<'a, T> ElementwiseBatch<'a, T> {
    pub fn n_inputs(&self) -> usize;
    pub fn n_points(&self) -> usize;
    pub fn get(&self, input: usize, point: usize) -> T;
    pub fn as_col_major_slice(&self) -> &'a [T];
}
```

The callback writes one output value per point into `out[p]`.

This layout matches the repository-wide column-major dense boundary policy and
avoids allocation-heavy `&[Vec<T>]` APIs.

## Local Update Data Flow

For each bond update:

1. Maintain per-input left and right frame matrices.
2. Expose the local two-site candidate matrix to MatrixLUCI as a lazy block
   source through `matrix_luci_factors_from_blocks`.
3. When LUCI requests `(rows, cols)`:
   - decode each row into the left-frame row and site `b` value,
   - decode each column into the site `b + 1` value and right-frame column,
   - compute local values for every input tensor train,
   - call the batched elementwise operator,
   - fill the requested output block in column-major order.
4. Reshape MatrixLUCI left/right factors back into the two updated site cores.
5. Update the left or right frames depending on sweep direction.

The implementation must not dense-materialize the full global tensor. Dense
materialization is allowed only in small reference tests.

## Cache Policy

Use three cache levels with clear lifetimes:

1. Frames are required algorithm state, not optional cache. They persist across
   updates and are invalidated only by the sweep direction and local updates.
2. Local entry cache is scoped to one bond update. It may map candidate matrix
   entry positions to output values so repeated LUCI block requests do not
   recompute the same elementwise result. It is dropped at the end of the bond
   update.
3. `TTCache` and `CachedFunction` are not part of the ACI core. They may be used
   for tests, reference checks, or fallback diagnostics, but the production
   local update path should use frames and lazy MatrixLUCI blocks.

Do not expose detailed cache controls in the initial public API. If benchmarks
show the need, add a later `CachePolicy` option such as:

```rust
pub enum CachePolicy {
    None,
    LocalEntries { max_entries: usize },
}
```

## Error Handling

Use a crate-local typed error enum, not `anyhow::Result`, for the public Rust
API. Expected error categories include:

- empty input list
- tensor-train length mismatch
- site dimension mismatch
- invalid initial guess
- invalid options
- batched operator output length mismatch
- MatrixLUCI factorization failure
- tensor-train construction or mutation failure

Callback failures should be preserved in a public error variant without
discarding context.

## Testing

Add focused release-mode tests for:

- validation paths: empty inputs, length mismatch, site dimension mismatch,
  invalid options, invalid initial guess
- algebraic cases: constant tensor trains, addition, multiplication, and a
  small nonlinear elementwise function
- dense oracle checks on small tensor trains by materializing each full result
  once and comparing whole dense buffers
- parity between `elementwise` and `elementwise_batched`
- local cache behavior within one bond update and cache invalidation across
  bond updates
- small Julia-parity examples derived from the upstream Gaussian/product tests

Doc examples must be runnable and include assertions. User-facing docs should
state that only Rust public API is supported in the first phase; C API and
language bindings are intentionally out of scope.
