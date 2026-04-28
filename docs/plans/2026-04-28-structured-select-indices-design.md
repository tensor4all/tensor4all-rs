# Structured Select Indices Design

## Goal

Make `TensorDynLen::select_indices` work for general structured storage while
preserving compact storage metadata and avoiding full logical dense
materialization of the input tensor.

The design introduces an internal structured-zero constructor first. Selection
can then represent impossible equality constraints as compact zero tensors
instead of falling back to dense zeros.

## Background

`Storage` represents dense, diagonal, and copy-like tensors with one structured
model:

- dense: `axis_classes = [0, 1, ...]`
- diagonal: `axis_classes = [0, 0, ...]`
- general structured: repeated classes such as `[0, 1, 0, 2]`

A structured tensor with payload `P`, logical dimensions `dims`, and
`axis_classes` represents:

```text
T[i0, i1, ...] =
    P[j0, j1, ...] if all logical axes in the same class have equal indices
    0              otherwise
```

where `jc` is the common coordinate for payload class `c`.

`select_indices` fixes one or more logical axes and drops them. For structured
storage, this operation should become a payload slice plus metadata
canonicalization, not a full logical dense slice.

## Non-Goals

- Do not expose a new public API in this change. Start with `pub(crate)`
  helpers.
- Do not change the observable `select_indices` value semantics.
- Do not add support for arbitrary sparse storage.
- Do not materialize the full input logical tensor.

## Structured Zero

Add internal helpers:

```rust
TensorDynLen::structured_zeros<T>(
    indices: Vec<DynIndex>,
    axis_classes: Vec<usize>,
) -> Result<TensorDynLen>
where
    T: TensorElement + Zero + Clone;
```

and, if it keeps responsibilities cleaner:

```rust
Storage::structured_zeros<T>(
    payload_dims: Vec<usize>,
    axis_classes: Vec<usize>,
) -> Result<Storage>
where
    T: StorageScalar + Zero + Clone;
```

The helper computes compact payload dimensions from logical dimensions and
`axis_classes`. All logical axes sharing one class must have the same dimension.
The payload data is zero-filled with length `product(payload_dims)`, and strides
are compact column-major strides.

The storage kind remains derived from metadata:

- distinct classes for all logical axes may be `StorageKind::Dense`;
- all logical axes sharing one class may be `StorageKind::Diagonal`;
- mixed repeated classes are `StorageKind::Structured`.

The helper name can still be `structured_zeros`; the resulting storage kind need
not always be `Structured`.

## Selection Semantics

Selection is defined by mapping selected logical axes to payload classes.

For each selected logical axis:

1. Find the logical axis by full `DynIndex` equality.
2. Read `class_id = storage.axis_classes()[axis]`.
3. Record `(class_id, selected_position)`.

If the same `class_id` is selected more than once with different positions, the
logical slice is all zeros. Return `structured_zeros::<T>(kept_indices,
kept_axis_classes)` after canonicalizing the kept classes.

If the positions agree, that payload class is fixed to that coordinate.

## Metadata Algorithm

Given:

- input `indices`
- input `axis_classes`
- selected axes and positions

Compute:

1. `kept_axes`: logical axes not selected.
2. `kept_indices`: indices for those axes, preserving logical order.
3. `old_kept_classes`: input classes for kept axes.
4. `fixed_classes`: map from selected class to fixed coordinate.
5. `remaining_payload_classes`: old classes that appear in `old_kept_classes`.
6. Canonicalize `old_kept_classes` by first appearance to produce
   `new_axis_classes`.
7. Compute `new_payload_dims` from the kept logical dimensions and
   `new_axis_classes`.

Selected classes that do not appear in kept axes become payload slice axes that
are removed from the output payload. Selected classes that still appear in kept
axes constrain the corresponding output payload coordinate to the fixed value.

Example:

```text
axis_classes = [0, 1, 0, 2]
select logical axis 1 at p

kept classes      = [0, 0, 2]
fixed classes     = {1: p}
new axis_classes  = [0, 0, 1]
```

## Payload Algorithm

The output payload is built by iterating over the new compact payload, not over
the full output logical dense tensor.

For each new payload coordinate:

1. Map it back to old payload classes that remain in output.
2. Fill fixed selected classes from `fixed_classes`.
3. Compute the old payload offset with `payload_strides`.
4. Copy the payload value into the new compact payload.

This is `O(product(new_payload_dims))` in memory and time. It does not depend on
`product(input logical dims)` except through the compact payload dimensions.

## Scalar Types

Implement the algorithm generically over payload scalar type where possible.
`TensorDynLen` may dispatch through existing `Storage` payload accessors:

- `payload_f64_col_major_vec`
- `payload_c64_col_major_vec`

The implementation should not add scalar-specific public Rust entry points.

## AD

Initial implementation may reject tracked structured AD values if preserving AD
metadata is not yet straightforward. The important rule is to avoid silently
densifying tracked structured payloads.

If AD is supported in the same change, the selected payload should be built from
the tracked compact payload tensor rather than from snapshot storage, and the
result should keep tracked structured AD metadata.

## Error Handling

Return errors for:

- argument length mismatch;
- selected index not present;
- duplicated selected index;
- coordinate out of range;
- invalid or non-canonical axis classes;
- inconsistent dimensions within one payload class;
- unsupported scalar type.

Do not return an error for equality conflicts caused by selection positions.
Those are valid slices whose value is exactly zero.

## Testing

Tests should cover:

- structured zero helper for f64 and Complex64;
- invalid shared-class dimensions;
- select on structured `[0, 1, 0]` preserving `[0, 0]`;
- select on structured `[0, 1, 0]` with conflicting selected axes returning
  compact zero;
- select all axes returning rank-0 storage;
- same-ID different-prime indices select the exact full index;
- no full input dense materialization by using dimensions that would be too
  large to materialize densely but have small compact payload.

