# General Structured Contraction Design

> Historical note: this design predates the current repository rule that
> production paths must preserve compact structured representations and avoid
> hidden full dense materialization. Mentions of dense fallback below describe a
> temporary reference/debug path, not current production guidance.

## Goal

Implement contraction for dense, diagonal, and general structured
`TensorDynLen` values while preserving the strongest sound output structure.

The core operation is not only value computation. It must also compute the
logical-axis equivalence classes of the result, execute on compact payloads
whenever possible, and preserve the same compact layout through reverse-mode AD.

## Background

`Storage` already models every layout as structured storage:

- dense: `axis_classes = [0, 1, ...]`
- diagonal: `axis_classes = [0, 0, ...]`
- general structured: repeated classes such as `[0, 1, 0, 2]`

The current contraction path has a metadata-only propagation step for diagonal
and structured equality classes, but numeric contraction still generally goes
through materialized dense native tensors. That is acceptable as a temporary
value fallback, but it is not the intended implementation because it loses
general compact structure and makes AD gradients dense too easily.

Older `tenferro-rs` had the right architectural model:

- `StructuredTensor<T>` stored `payload`, `logical_dims`, and `axis_classes`.
- Its `Differentiable::Tangent` was `StructuredTensor<T>`, so zero tangents,
  seed cotangents, and accumulated gradients kept the same layout.
- `structured/einsum.rs` planned output axis classes with
  `plan_axis_classes_for_subscripts`.
- It normalized repeated compact labels with `normalize_payload_for_roots`
  before calling dense payload einsum.

This design ports those ideas to `tensor4all-rs` using the current
`Storage`/`TensorDynLen` representation.

## Non-Goals

- Do not silently densify tracked structured tensors in contraction AD.
- Do not implement every tensor operation as structured-native in this change.
  The target operation is contraction and the structural helpers needed by its
  forward and reverse paths.
- Do not expose scalar-specific Rust APIs. Keep Rust APIs generic over `Storage`
  and `TensorDynLen`; scalar-specific handling remains an FFI detail.
- Do not depend on tenferro strict-binary/GEMM lowering for repeated labels.
  Public tenferro eager einsum accepts repeated labels, but strict GEMM lowering
  may decline them as a fast-path applicability rule.

## Semantics

A structured tensor with payload `P`, logical dimensions `dims`, and
`axis_classes` represents the logical tensor

```text
T[i0, i1, ...] =
    P[j0, j1, ...] if all logical axes in the same class have equal indices
    0              otherwise
```

where `jc` is the shared logical index value for class `c`.

Contraction adds equality constraints between logical axes. The result structure
is determined by taking the transitive closure of:

1. each operand's existing `axis_classes`, and
2. each contracted or retained shared label relation.

The output `axis_classes` are the canonicalized equivalence-class roots of the
logical output axes. Roots that have no output axis are closed summation roots.

This rule is maximal but sound: it preserves every equality forced by the input
layouts and contraction relations, and it does not invent new equality classes
when no relation forces them.

## Metadata Algorithm

The planner works over payload-class nodes, not dense logical coordinates.

1. For every operand, create one union-find node per payload class.
2. For every logical axis, remember `(operand, logical_axis) -> payload_node`.
3. Union nodes for all contracted label occurrences.
4. Union nodes for retained shared labels as equality constraints, but do not
   remove those labels from output.
5. Validate that every merged root has one consistent dimension.
6. For each output logical axis, map its payload node to its root.
7. Canonicalize output roots in output-axis order to produce result
   `axis_classes`.
8. Record distinct output roots in first-appearance order. These become the
   compact result payload axes.

For `contract` and current `contract_multi`, "contracted" means labels that
appear in multiple inputs and are not retained. For the future API in issue
[#436](https://github.com/tensor4all/tensor4all-rs/issues/436), retained labels
come from `ContractOptions` / `ContractMultiOptions`.

## Compact Payload Algorithm

The compact payload contraction is an einsum over payload axes.

For each operand:

1. Take the operand payload axes in payload-class order.
2. Map each payload class to the global union-find root.
3. Use the root id as the compact einsum label.

If one operand maps two different payload axes to the same root, the operand
gets repeated compact labels. This is not an error. It means the payload must be
diagonalized over those payload axes before the main contraction.

There are two valid ways to handle this:

- call tenferro public eager einsum, which explicitly handles repeated labels;
- or normalize first with the old tenferro algorithm:
  repeatedly extract the diagonal of the first duplicate pair and remove one
  duplicate payload axis.

The tensor4all implementation should keep this normalization explicit in its
planner/executor. That makes repeated-label behavior local and prevents a future
strict GEMM path from accidentally receiving repeated labels.

The compact output labels are the distinct output roots in first-appearance
order. The resulting payload is wrapped in `Storage::new_structured` with:

- `payload_dims`: dimensions of those distinct output roots;
- `strides`: compact column-major strides;
- `axis_classes`: result logical `axis_classes`.

If the result has no output roots, the payload is scalar storage.

## AD Semantics

AD must preserve structure. A structured tensor's tangent space has the same
logical dimensions and the same `axis_classes` as the primal.

The old tenferro rule is the target:

```text
Tangent(StructuredTensor(payload, dims, axis_classes))
    = StructuredTensor(tangent_payload, dims, axis_classes)
```

Therefore:

- `enable_grad()` must track the compact payload, not a dense logical
  materialization.
- `grad()` must return a `TensorDynLen` with the same `axis_classes` and compact
  payload shape as the primal.
- tangent accumulation must require matching logical dims and matching
  `axis_classes`.
- contraction forward with AD must call compact payload `eager_einsum_ad` and
  wrap the compact output payload with the planned result layout.
- no tracked contraction path may call `as_native()` as a hidden densification
  step.

If a dense cotangent enters a structured reverse path, it must be compressed
back into the expected layout before accumulation. The old tenferro helper
`compress_dense_to_layout_in_ctx` did this by using einsum from logical
`axis_classes` to payload class labels. Tensor4all should add the same helper
for `Storage`.

## TensorDynLen Representation Impact

The current `eager_cache` is a dense logical `EagerTensor`. That is the wrong
cache for structured AD. Replace or split it into two concepts:

```rust
struct TensorDynLen {
    indices: Vec<DynIndex>,
    storage: Arc<Storage>,
    ad: Option<Arc<StructuredAdValue>>,
    materialized_cache: Arc<OnceLock<Arc<EagerTensor<CpuBackend>>>>,
}

struct StructuredAdValue {
    payload: EagerTensor<CpuBackend>,
    payload_dims: Vec<usize>,
    axis_classes: Vec<usize>,
}
```

The materialized cache is only a value cache. It is never the reverse-mode leaf
for structured tensors. Dense tensors are just the special case where payload
dims equal logical dims and `axis_classes = [0, 1, ...]`.

## Randomized Thought Experiments

### Diagonal by diagonal partial contraction

Input layouts:

```text
A(i, j): axis_classes [0, 0]
B(j, k): axis_classes [0, 0]
contract j
output i, k
```

The contracted `j` relation unions `A` class 0 with `B` class 0. Output roots
for `i` and `k` are equal, so result `axis_classes = [0, 0]`.

### General structured result

Input layouts:

```text
A(a, b, c): axis_classes [0, 0, 1]
B(c, d, e): axis_classes [0, 1, 1]
contract c
output a, b, d, e
```

The `c` contraction unions only `A` class 1 with `B` class 0. Output roots are
`A0, A0, B1, B1`, so result `axis_classes = [0, 0, 1, 1]`.

### Payload repeated label

Input layouts:

```text
A(i, l): dense axis_classes [0, 1]
B(j, k): diagonal axis_classes [0, 0]
contract i with j and l with k
output none
```

Both payload classes of `A` become the same global root through `B`'s diagonal
constraint. The compact payload labels for `A` are repeated. The executor must
extract the diagonal of `A`'s payload or call a repeated-label-safe public
einsum path.

### Retained batch label

Input layouts:

```text
A(b, i, j): dense [0, 1, 2]
B(b, j, k): B has j == k, axis_classes [0, 1, 1]
retain b
contract j
output b, i, k
```

The retained `b` roots remain output roots. The `j` contraction unions `A`'s
`j` root with `B`'s `j/k` root, so output `i` and `k` share a root only if
another relation forces it. Here the result layout is `[0, 1, 2]`: batched
matrix-like output, not diagonal over `i` and `k`.

## Testing Strategy

Test metadata planning separately from numeric execution:

- pairwise diagonal/diagonal partial contraction -> output diagonal;
- general `[0, 0, 1]` by `[0, 1, 1]` -> output `[0, 0, 1, 1]`;
- repeated payload labels are normalized to unique labels before the final
  payload einsum;
- dimension mismatches inside a merged root return errors;
- retained labels remain in output and are not reduced.

Then test value and AD behavior:

- compare structured compact results against dense materialization;
- verify result `storage_kind()` and `axis_classes()`;
- enable grad on diagonal and general structured inputs, contract to a scalar,
  call `backward()`, and verify `grad().storage().axis_classes()` matches the
  input layout;
- verify gradient payload values against dense whole-result comparisons.
