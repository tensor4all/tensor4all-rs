# Issues #486 and #490 Error and Documentation Cleanup Design

## Spec

Issues:

- [#486](https://github.com/tensor4all/tensor4all-rs/issues/486)
- [#490](https://github.com/tensor4all/tensor4all-rs/issues/490)

This chunk handles the smaller error-type and documentation cleanup that fits in
the same PR as the public-surface audit:

- Replace selected `Result<_, String>` public/internal error returns with
  appropriate typed or contextual errors.
- Remove the broad `#[allow(missing_docs)]` suppression from
  `tensor4all-core::AnyScalar` public methods by documenting the methods.
- Deduplicate repeated C API `require_layout()` error handling in
  `quanticstransform.rs`.

The issue #490 HDF5 doc-example item belongs to the #487 layering chunk, not
this chunk.

Acceptance criteria:

- `tensor4all-tensorbackend::Storage` methods listed in #486 return a
  `thiserror`-based `StorageError` instead of `String`.
- `tensor4all-itensorlike::linsolve::infer_index_mappings` uses the crate error
  type instead of plain `String`.
- `tensor4all-treetn` graph helper modules return `anyhow::Result` or a
  typed tree-network error rather than `Result<_, String>`.
- `AnyScalar` public methods have rustdoc that meets repository standards
  without using `ignore` or `no_run` examples.
- `quanticstransform.rs` has one reusable helper or macro for
  `require_layout()` error propagation.
- No new C API path discards error text.

Non-goals:

- Do not attempt to eliminate all `unwrap()`, `expect()`, or `panic!()` calls in
  the workspace. That belongs to #484 and #485.
- Do not rewrite the C API last-error architecture; preserve the current
  `set_last_error` and `run_catching` model.
- Do not relax test tolerances.

## Design

### `tensor4all-tensorbackend` Storage Errors

Add a public error enum in the backend crate, for example in a new
`storage_error.rs` module or near `storage.rs` if that is the local convention:

```rust
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("expected {expected} storage, got {actual}")]
    ScalarKindMismatch {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("storage lengths must match for {operation}: {left} != {right}")]
    LengthMismatch {
        operation: &'static str,
        left: usize,
        right: usize,
    },
    #[error("invalid structured storage: {0}")]
    InvalidStructuredStorage(String),
}
```

Use a crate-local result alias if it improves readability:

```rust
pub type StorageResult<T> = std::result::Result<T, StorageError>;
```

Update these methods first:

- `payload_f64_col_major_vec`
- `payload_c64_col_major_vec`
- `to_dense_f64_col_major_vec`
- `to_dense_c64_col_major_vec`
- `try_add`
- `try_sub`
- `axpby` or the method currently reported as `try_mul` in #486, depending on
  the current source name.

When wrapping `StructuredStorage::new` errors, preserve the original message in
`StorageError::InvalidStructuredStorage` rather than converting to an untyped
`String`.

### `tensor4all-itensorlike` Linsolve Errors

Change `infer_index_mappings` to return the local
`tensor4all_itensorlike::error::Result<SiteMappings>` or directly
`Result<SiteMappings, TensorTrainError>`.

Use existing variants where they fit:

- `TensorTrainError::InvalidStructure` for missing or incompatible site
  structure.
- `TensorTrainError::OperationError` only for residual operation failures that
  do not have a more precise variant.

Remove call-site `map_err(|e| TensorTrainError::OperationError { message: e })`
wrappers that become redundant after the helper returns the crate error type.

### `tensor4all-treetn` Graph Helper Errors

The graph helper modules are internal library infrastructure. Move
`Result<_, String>` to `anyhow::Result<_>` for:

- `node_name_network.rs`
- `named_graph.rs`
- `site_index_network.rs`

Use `anyhow::bail!`, `anyhow::ensure!`, and `.with_context(...)` to preserve
diagnostic details. Do not turn these into panics.

If a public API exposes these error types and rustdoc becomes less clear, add
function docs that describe the error conditions instead of exposing an enum
prematurely.

### `AnyScalar` Documentation

Remove `#[allow(missing_docs)]` from the `impl AnyScalar` block and document the
public methods it contains. Every doc example must run and include assertions.

Examples should use the public constructors:

- `AnyScalar::new_real`
- `AnyScalar::new_complex`

Do not match on enum variants in examples unless the method is explicitly about
variant-level behavior.

Minimum method groups to document:

- constructors and conversions: `from_value`, `from_real`, `new_real`,
  `new_complex`;
- accessors: `real`, `imag`, `abs`, `is_complex`;
- arithmetic helpers: `conj`, `sqrt`, `powf`;
- tensor conversion helpers if they are public.

### C API `require_layout()` Deduplication

Introduce a helper that preserves the existing status-code and last-error
behavior:

```rust
fn require_layout_or_status(
    layout: *const t4a_qtt_layout,
) -> Result<&'static InternalQttLayout, StatusCode> {
    match require_layout(layout) {
        Ok(layout) => Ok(layout),
        Err((code, msg)) => {
            set_last_error(&msg);
            Err(code)
        }
    }
}
```

The exact lifetime should follow the current `require_layout` signature; if the
layout borrow cannot be expressed cleanly with a helper function, use a local
macro that expands to the existing early-return pattern.

The key rule is behavior preservation: null layout still returns the same status
code and sets the same last-error message.

## Files

- Modify `crates/tensor4all-tensorbackend/src/storage.rs` and possibly
  `crates/tensor4all-tensorbackend/src/lib.rs`.
- Modify `crates/tensor4all-itensorlike/src/linsolve.rs`.
- Modify `crates/tensor4all-treetn/src/node_name_network.rs`.
- Modify `crates/tensor4all-treetn/src/named_graph.rs`.
- Modify `crates/tensor4all-treetn/src/site_index_network.rs`.
- Modify `crates/tensor4all-core/src/any_scalar.rs`.
- Modify `crates/tensor4all-capi/src/quanticstransform.rs`.
- Update `docs/api` after public API changes.

## Testing

Add tests where behavior changes from string errors to typed/contextual errors:

- Storage scalar-kind mismatch for f64/c64 payload access.
- Storage length mismatch for addition/subtraction/axpby.
- Linsolve index mapping error branch.
- TreeTN graph duplicate node, missing node, and invalid topology branches.
- C API null layout path still sets `t4a_last_error_message`.
- AnyScalar doc examples through rustdoc.

Run:

```bash
cargo test --release -p tensor4all-tensorbackend
cargo test --release -p tensor4all-itensorlike linsolve
cargo test --release -p tensor4all-treetn
cargo test --release -p tensor4all-capi quanticstransform
cargo test --doc --release -p tensor4all-core
cargo test --doc --release --workspace
cargo run -p api-dump --release -- . -o docs/api
```

## PR Handling

This chunk can close #486 if all listed non-C-API `Result<_, String>` targets
are converted. It can close the remaining #490 items if AnyScalar docs,
`require_layout()` deduplication, and the HDF5 doc-example cleanup from the
#487 chunk all land in the same PR.

#484 and #485 should be referenced only as partially addressed if the local
cleanup removes any `unwrap()`, `expect()`, or `panic!()` incidentally.
