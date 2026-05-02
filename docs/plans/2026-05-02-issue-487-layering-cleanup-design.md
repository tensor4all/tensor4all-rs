# Issue #487 Layering Cleanup Design

## Spec

Issue: [#487](https://github.com/tensor4all/tensor4all-rs/issues/487)

Downstream crates should not read or write the representation fields of
`tensor4all-core` types directly. The immediate scope is the direct field access
reported in `tensor4all-hdf5` and `tensor4all-capi`, plus the HDF5 rustdoc
example duplicated in [#490](https://github.com/tensor4all/tensor4all-rs/issues/490).

The user-visible behavior must stay unchanged:

- HDF5 round trips preserve index ID, dimension, prime level, and tags.
- C API index and tensor query functions return the same values and status
  codes as before, except any diagnostics improved by separate C API cleanup.
- Public examples show high-level APIs rather than encouraging representation
  access.

Acceptance criteria:

- `tensor4all-hdf5` no longer reads or writes `DynIndex` fields directly outside
  `tensor4all-core`.
- `tensor4all-capi` no longer reads `TensorDynLen.indices` or `DynIndex.plev`
  directly.
- The HDF5 rustdoc example compares index identity through public methods.
- New or updated tests cover same-ID indices that differ by prime level or tags
  where the changed code could otherwise collapse identity to ID-only behavior.
- `cargo test --release -p tensor4all-hdf5` and
  `cargo test --release -p tensor4all-capi` pass for the affected tests.

Non-goals:

- Do not make index fields private in this change. That is a larger public API
  hardening step and belongs after downstream crate usage is removed.
- Do not redesign tag storage or index identity semantics.
- Do not change C API signatures for this issue.

## Design

Use `IndexLike` and existing high-level methods wherever they already express
the needed data:

- `idx.dim()` for dimensions.
- `idx.plev()` for prime levels.
- `idx.tags()` for tag sets.
- `tensor.indices()` for tensor index slices.
- Full `DynIndex` equality for identity-preserving comparisons in examples and
  tests.

`DynId` currently exposes only a tuple field, and HDF5 serialization must write
the numeric ITensors-compatible ID. Add a narrow public accessor on `DynId`,
for example:

```rust
impl DynId {
    /// Return the numeric ID value used by ITensors-compatible serialization.
    pub fn value(&self) -> u64 {
        self.0
    }
}
```

This accessor is intentionally less general than exposing mutable access. It
supports stable serialization and C API query needs while keeping all callers
read-only.

For deserialization, continue constructing `DynIndex` through
`Index::new_with_tags(DynId(id), dim, tags)`, then use the existing
`set_plev(plev)` method instead of mutating `idx.plev`.

The HDF5 example should stop building ID vectors through `idx.id`. Prefer
comparing the whole index list:

```rust
assert_eq!(loaded.indices(), tensor.indices());
```

This verifies the ID and all metadata, matching repository rules that index
identity is the full `Index` value, not the ID alone.

## Files

- Modify `crates/tensor4all-core/src/defaults/index.rs` to add the `DynId`
  accessor and its rustdoc example.
- Modify `crates/tensor4all-hdf5/src/index.rs` to use `id().value()`, `dim()`,
  `plev()`, `tags()`, and `set_plev()`.
- Modify `crates/tensor4all-hdf5/src/lib.rs` to update the rustdoc example.
- Modify `crates/tensor4all-capi/src/index.rs` to use `id().value()` and
  `plev()`.
- Modify `crates/tensor4all-capi/src/tensor.rs` and any TreeTN helper path that
  directly reads `.indices` to use `indices()`.

## Testing

Add focused regression coverage rather than broad snapshots:

- HDF5 round trip with two indices that share `DynId` but differ by `plev` or
  tags, asserting the loaded indices equal the originals exactly.
- C API tensor rank/index query tests should continue to pass after switching
  to `indices()`.
- A `DynId::value()` doc example with an assertion.

Run:

```bash
cargo test --release -p tensor4all-core defaults::index
cargo test --release -p tensor4all-hdf5
cargo test --release -p tensor4all-capi
cargo test --doc --release -p tensor4all-core
cargo test --doc --release -p tensor4all-hdf5
```

## PR Handling

This chunk can close #487 and cover the HDF5 example item in #490. It may reduce
some direct representation usage counted by #484 or #485, but those umbrella
issues should not be closed by this PR.
