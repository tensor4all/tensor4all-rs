# Compact Delta/Copy Tensor Design

## Problem

`TensorLike::delta()` is used to create identity, gap, and bridge tensors in
TreeTN operator composition. Its default implementation builds each pair from
`TensorLike::diagonal()` and then takes outer products. For `TensorDynLen`,
`diagonal()` currently constructs a dense identity matrix with `from_dense`.

That is numerically correct, but it violates the repository rule that known
structured tensors must not be silently represented as dense tensors in
production paths. In particular, topology-preserving bridge deltas can become
hidden `bond_dim^2` payloads even though their true representation is a copy
constraint.

## Chosen Approach

Use the existing compact storage model. `TensorDynLen` already supports
diagonal/copy storage through `from_diag`, `copy_tensor`, storage axis classes,
and structured contraction. The minimal fix is:

1. Change `TensorDynLen`'s `TensorLike::diagonal()` implementation to return
   `TensorDynLen::from_diag(vec![input, output], vec![1.0; dim])`.
2. Keep the default `TensorLike::delta()` implementation. Once `diagonal()` is
   compact, the default outer-product path should preserve independent copy
   constraints through structured axis classes such as `[0, 0, 1, 1]`.
3. Add representation tests, not only dense-value tests, so future changes
   cannot reintroduce hidden dense delta tensors.
4. Add a TreeTN apply regression that exercises a nontrivial bridge delta and
   checks that the bridge tensor is compact/structured before local naive apply.

This avoids adding a new storage kind or public API. It also keeps the generic
`TensorLike::delta()` contract unchanged.

## Alternatives Considered

### Restrict topology embedding to dimension-1 links only

This would avoid dense bridge tensors in the immediate issue, but it leaves the
core bug in place. Other production paths that call `T::delta()` would still
turn known copy structure into dense storage.

### Add a new copy/delta storage kind

This is unnecessary now. The backend already represents copy tensors as
structured storage with repeated axis classes. Adding a new storage kind would
increase API and implementation surface without a clear benefit for this issue.

## Detailed Behavior

For a single pair `(i, o)` with dimension `d`, `TensorDynLen::diagonal(&i, &o)`
returns a diagonal storage tensor:

- logical dimensions: `[d, d]`
- payload dimensions: `[d]`
- axis classes: `[0, 0]`
- storage kind: `Diagonal`

For multiple pairs, `TensorLike::delta(&[i1, i2], &[o1, o2])` returns a
structured tensor:

- logical dimensions: `[d1, d1, d2, d2]`
- payload dimensions: `[d1, d2]`
- axis classes: `[0, 0, 1, 1]`
- storage kind: `Structured` for independent copy constraints

Dense extraction with `to_vec()` remains numerically unchanged.

## Error Handling

Existing dimension checks remain:

- `diagonal(input, output)` errors if dimensions differ.
- `delta(inputs, outputs)` errors if pair counts differ or paired dimensions
  differ.

No new public errors are introduced.

## Tests

Core tests should check:

- `TensorDynLen::diagonal()` has `StorageKind::Diagonal`.
- `TensorDynLen::delta()` for one pair has diagonal storage.
- `TensorDynLen::delta()` for two independent pairs has structured storage,
  payload dimensions `[d1, d2]`, and axis classes `[0, 0, 1, 1]`.
- Dense values for delta remain correct.
- A non-contiguous bonded identity operator applied with `ApplyOptions::naive()`
  can be embedded along the state path without creating dense bridge storage.

The TreeTN regression should inspect the extended operator before final apply,
because the final local contraction may legitimately change storage layout.

## Non-Goals

- Do not add new public tensor constructors.
- Do not add a new storage kind.
- Do not change `BlockTensor` support; it currently rejects `diagonal`,
  `delta`, `outer_product`, and `contract`, so it is not part of this fix.
- Do not implement truncation/max-rank behavior for local naive apply.
