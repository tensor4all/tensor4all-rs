# tensor4all-core API cleanup notes

## N-ary contraction API

Current direction:

- Prefer one public N-ary contraction entry point over separate binary and
  N-ary APIs.
- Remove public `contract_pair`-style convenience APIs. Binary contraction
  should be expressed as N-ary contraction with two operands.
- Prefer borrowed operands, i.e. a slice of tensor references, for the default
  API. This avoids forcing callers to move or clone tensors just to contract
  existing values.
- Keep an owned variant only as an explicit optimization path for callers that
  can transfer ownership.
- The public API should accept structural labels directly instead of building a
  string equation and parsing it again downstream.

Open naming sketch:

- `contract(&[&a, &b, &c])`
- `contract_owned(vec![a, b, c])`

Implemented cleanup steps:

- Added `contract(&[&TensorDynLen])` as the connected-network default entry.
- Added `contract_owned(Vec<TensorDynLen>)` with the same connected-network
  semantics.
- Added `contract_with_options` and `contract_owned_with_options` as the
  connected-network advanced entries while retained-index users are migrated to
  TreeTN-level APIs.
- Removed the legacy `contract_multi*` names and the temporary
  `contract_components_and_outer_product*` helpers.
- Simplified `TensorContractionLike::contract` to the connected default
  signature `contract(&[&Self])`. Tensor-edge restrictions now live only in
  concrete `ContractionOptions` APIs.

- `tenferro` now has a canonical `EinsumSubscripts { inputs, output }` payload
  for `StdTensorOp::NaryEinsum`.
- String APIs in `tenferro` remain compatibility wrappers that parse once into
  integer labels.
- `tensor4all-tensorbackend` native einsum and `tensor4all-core` AD-backed
  eager contractions now pass integer subscripts to `tenferro`; strings are
  retained only for human-readable diagnostics and path reports.

## Retained indices

Current direction:

- Remove the retained-index feature from the public contraction API unless a
  concrete production use case appears.
- Existing uses are mostly tests and C API plumbing. The feature complicates
  graph connectivity, AD behavior, and user-facing semantics.
- Site-index-aware partial contraction should remain a TreeTN-level concept
  (`contract_pairs`, `diagonal_pairs`, etc.), not a dense tensor contraction
  retained-index feature.

## Allowed tensor edges

Current direction:

- `AllowedPairs` has been removed from `tensor4all-core`.
- Normal public contraction now always considers all tensor pairs and contracts
  matching contractable indices across a connected tensor graph.
- Tensor-edge restrictions should live at the TreeTN/topology layer, where the
  graph is explicit, instead of leaking into dense `TensorDynLen` contraction.

## Connected vs disconnected contraction

Current direction:

- Prefer connected contraction as the normal public semantic. A default
  contraction over disconnected components should error because silently
  returning an outer product can hide missing links or index bugs.
- Provide an explicit outer-product/combine API for intentional disconnected
  products.
- `contract_multi*` and `contract_components_and_outer_product*` have been
  removed from the Rust core API. Callers now use `contract` /
  `contract_with_options` for connected networks and spell disconnected
  products explicitly with `outer_product`.
- The C API multi-tensor retained-index entry was renamed from
  `t4a_tensor_contract_multi` to `t4a_tensor_contract_many_retain` so the
  legacy name does not leak across the boundary.

## Outer product

Current findings:

- `outer_product` is not part of the core numerical contraction path. Current
  uses are mostly shape/topology construction helpers:
  - explicit disconnected products after contracting connected components
  - default multi-index `delta` construction from pairwise diagonals
  - TreeTN dummy links, ones tensors, bridge deltas, and trivial factorization
    boundary cases
- Structured tensor storage likely makes several of these uses unnecessary.
  For example, multi-index delta/copy tensors and adding unit-valued dummy
  axes can be represented directly instead of multiplying by separate tensors.
- Do not keep `outer_product` as a required `TensorContractionLike` method in
  the long-term public API unless a concrete generic use case remains.

Possible replacements:

- direct constructors for structured multi-index delta/copy tensors
- an explicit helper to attach unit-valued dummy axes to a tensor
- a clearly named low-level tensor product helper only where intentional

## End-of-session visibility audit

Completed:

- `ContractionSpec`, `ContractionError`, `prepare_contraction`, and
  `prepare_contraction_pairs` were removed from the top-level public re-exports
  and demoted to `pub(crate)`.
- Direct planning tests that need those helpers moved from integration tests to
  `index_ops` unit tests. Public integration tests now cover the user-facing
  index APIs only.
- The unused `result_dims` field was removed from `ContractionSpec`; result
  shape remains derived from the result indices, avoiding duplicate metadata.

Remaining public-surface cleanup candidates:

- `TensorContractionLike::outer_product` is still public because TreeTN and
  construction helpers currently use it. Long term, replace those use cases with
  explicit structured constructors or dummy-axis helpers before removing it from
  the trait.
- `contract_pair` remains as compatibility API in several examples/tests. The
  intended public direction is still `contract(&[...])` for connected
  contractions plus explicit `outer_product` for disconnected products.
