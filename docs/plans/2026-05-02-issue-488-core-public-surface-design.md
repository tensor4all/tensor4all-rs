# Issue #488 Core Public Surface Design

## Spec

Issue: [#488](https://github.com/tensor4all/tensor4all-rs/issues/488)

`tensor4all-core` currently re-exports several backend or implementation-detail
items as stable-looking public API. Because the project is in early development,
the cleanup should remove deprecated or accidental public surface immediately
instead of adding compatibility shims.

The cleanup must reduce public exposure without breaking internal crate
compilation:

- Remove direct public access to backend storage internals through
  `tensor4all_core::storage`.
- Stop re-exporting helper traits intended for `TensorDynLen` implementation
  details.
- Review backwards-compatibility top-level module re-exports and keep only
  modules that are intentionally part of the user-facing API.

Acceptance criteria:

- `tensor4all_core::storage::*` is no longer a public re-export path.
- `RandomScalar` and `TensorAccess` are not exported from the crate root.
- Downstream crates in the workspace compile by importing intended crates or
  using higher-level `tensor4all-core` methods.
- Rustdoc public API no longer advertises the removed internal paths after
  regenerating `docs/api`.
- README and guide snippets do not rely on removed paths.

Non-goals:

- Do not redesign `Storage` itself.
- Do not remove public `TensorDynLen::storage()` or `to_storage()` in this
  chunk; those are broader API design decisions.
- Do not add compatibility aliases. The repository rules explicitly allow
  immediate removal during early development.

## Design

Make the public API boundary explicit in `crates/tensor4all-core/src/lib.rs`.

### Storage Re-exports

Delete the public `storage` module and root-level storage re-exports that come
from `tensor4all_tensorbackend` unless an item is intentionally public at the
core level. Internal code that needs backend storage should import from
`tensor4all-tensorbackend` directly inside the crate or use local module paths.

Expected removals from the core crate root:

- `storage::{make_mut_storage, mindim, Storage, StorageKind, StructuredStorage,
  SumFromStorage}`
- `tensor4all_core::storage::*`

If workspace crates outside `tensor4all-core` require storage internals, the
preferred fix is one of:

- add an appropriate high-level `TensorDynLen` method; or
- add a direct dependency on `tensor4all-tensorbackend` only if the crate is
  intentionally working at backend-storage level.

Do not work around missing access by reaching into fields.

### TensorDynLen Implementation Helpers

Stop root re-exporting:

- `RandomScalar`
- `TensorAccess`

`TensorDynLen::random` can keep its generic bound in the implementation module.
If callers need random tensor construction, they should call
`TensorDynLen::random` with supported scalar types and should not depend on the
helper trait as public API. If rustdoc requires the trait to be public because
it appears in a public signature, prefer making the method signature use an
existing public scalar trait or moving the trait documentation into the
`tensor` module rather than exporting it from the root.

`TensorAccess` is an implementation helper. Public callers should use inherent
methods such as `TensorDynLen::indices()`.

### Backwards Compatibility Modules

Review the flattened modules:

- `tensor4all_core::direct_sum`
- `tensor4all_core::factorize`
- `tensor4all_core::qr`
- `tensor4all_core::svd`

Keep the top-level function and type re-exports that are already documented as
the intended public API, such as `svd`, `svd_with`, `SvdOptions`, `qr`,
`qr_with`, `FactorizeOptions`, and `direct_sum`. Remove only module aliases
that expose internal module structure rather than user-facing capabilities.

This keeps user-facing operations discoverable while avoiding the
`defaults/` module layout becoming a public compatibility contract.

## Files

- Modify `crates/tensor4all-core/src/lib.rs`.
- Modify downstream imports that fail after the public re-export cleanup.
- Regenerate `docs/api` if public API docs are part of the PR.
- Check `README.md`, `docs/book/src/`, and examples for stale import paths.

## Testing

Run targeted compile checks first:

```bash
cargo check --release -p tensor4all-core
cargo check --release --workspace
```

Then run the relevant public-surface checks:

```bash
cargo test --doc --release -p tensor4all-core
cargo test --release -p tensor4all-core
cargo run -p api-dump --release -- . -o docs/api
```

If any downstream crate needs a removed item, add or use a high-level API before
falling back to direct backend imports.

## PR Handling

This chunk can close #488 if the removed paths match the issue scope and the API
dump confirms the public surface changed as intended. It should be reviewed as
an API cleanup, not as a compatibility migration.
