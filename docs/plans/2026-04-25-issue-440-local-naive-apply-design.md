# Issue 440 Local Naive Apply Design

## Context

`ApplyOptions::naive()` for `apply_linear_operator` currently dispatches through
the generic TreeTN `ContractionMethod::Naive`. That generic path materializes
both input TreeTNs as full dense tensors and then decomposes the result back into
a TreeTN. This violates the repository rule that production tensor-network paths
must not hide dense materialization with exponential memory use.

For `LinearOperator` application, `naive` should match the ITensorMPS-style
meaning: exact local MPO/state contraction without truncation during the local
apply step. Full dense reference behavior is not needed for this API.

## Goals

- Make `ApplyOptions::naive()` a local exact tensor-network apply algorithm.
- Do not call `contract_to_tensor()` or `to_dense()` from the production naive
  apply path.
- Preserve the generic `apply_linear_operator<T, V>` shape where practical.
- Add reusable core support for local tensor axis fusion.
- Keep dense comparisons only in small tests.

## Non-Goals

- Keep or expose a full dense reference apply mode.
- Redesign TreeTN to support multiple bond indices per edge.
- Implement approximate truncation inside the local exact contraction step.

## Core API Additions

Add product link construction to `IndexLike`:

```rust
fn product_link(indices: &[Self]) -> anyhow::Result<Self>
where
    Self: Sized;
```

For `DynIndex`, this returns a fresh `Link` index whose dimension is the checked
product of the input dimensions. This is distinct from `sim()`, which preserves
the original dimension. Future symmetry-aware index implementations can define
the product space at this abstraction boundary.

Add local axis fusion to `TensorLike`:

```rust
fn fuse_indices(
    &self,
    old_indices: &[Self::Index],
    new_index: Self::Index,
    order: LinearizationOrder,
) -> anyhow::Result<Self>;
```

`TensorDynLen` implements this as the inverse of `unfuse_index`: replace several
axes with one axis whose dimension is the product of the fused axes. The data
movement is local to one tensor. It does not materialize the full TreeTN.

`BlockTensor<T>` delegates to its blocks. `TensorTrain` can return an
unsupported-operation error because it is not used for TreeTN local tensor
storage in this path.

## Apply Algorithm

`apply_linear_operator` keeps the existing partial-operator extension and input
index transformation steps.

For `ApplyOptions::naive()` only, dispatch to a dedicated local exact apply
helper instead of the generic `contract()` dispatcher.

The helper:

1. Simulates internal indices of the transformed state and MPO independently to
   avoid accidental link ID collisions.
2. For each node, binary-contracts the corresponding state tensor and MPO tensor.
   The shared input site indices contract away. The remaining indices are state
   links, MPO links, and MPO output site indices.
3. For each TreeTN edge, creates one product link from the state bond and MPO
   bond using `IndexLike::product_link`.
4. Fuses the pair `(state_bond, mpo_bond)` into the product link on both endpoint
   tensors using `TensorLike::fuse_indices`.
5. Builds the result with `TreeTN::from_tensors`, using the same node names and
   topology encoded by the fused common link IDs.
6. Converts operator output indices back to true output indices using the
   existing output transform.

The local contraction output order conceptually follows:

```text
(l1, l1', l2, l2', ..., output_site_indices...)
```

The implementation does not need an initial explicit permute because the binary
contraction and `fuse_indices` input order define the linearization.

## Error Handling

- Return an error if state and operator topologies differ after partial operator
  extension.
- Return an error if a node is missing in either network.
- Return an error if an expected state or MPO bond is missing for an edge.
- Return an error if product dimensions overflow.
- Return an error if a tensor does not contain all indices requested by
  `fuse_indices`.

## Tests

Small correctness tests may materialize dense tensors once and compare with
`sub().maxabs()`.

Long regression tests must avoid dense comparison. They should apply an identity
chain MPO to a long TreeTN and check:

- node count and site-space structure are preserved;
- result bond dimensions are bounded by `state_bond_dim * mpo_bond_dim`;
- several fixed `evaluate()` samples match the input state.

Add direct `TensorDynLen::fuse_indices` tests covering:

- roundtrip with `unfuse_index` for column-major order;
- dimension mismatch errors;
- missing index errors;
- fusing non-adjacent axes.

## Migration

Update rustdoc for `ApplyOptions::naive()` and `ContractionMethod::Naive` so
method names do not imply that apply uses a dense full-network reference path.
If the generic TreeTN dense naive contraction remains internally for other small
tests, it must be documented as dense/reference and excluded from production
apply dispatch.
