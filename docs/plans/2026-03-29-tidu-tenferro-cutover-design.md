# Design: tidu/tenferro/tensor4all AD Cutover

**Date:** 2026-03-29
**Status:** Approved in chat

## Goal

Replace the current `AdTensor<T>`-centered AD stack with a cleaner split:

- `tidu-rs`: generic AD engine and graph execution
- `tenferro-rs`: public tensor API, primal tensor operations, and op-specific AD adapters
- `tensor4all-rs`: downstream consumer that only follows upstream API changes and keeps its tests green

The target user-facing model is torch-like at the `tenferro::Tensor` surface, but with Rust-friendly immutable APIs.

Current state:

- `tenferro` now sources `AdTensor` / `NodeId` directly from `tenferro-internal-ad-core`
- `tenferro-internal-ad-surface` remains the transitional public-ad surface for `AdMode`, tensor wrappers, and reverse-mode entrypoints
- `tenferro-internal-ad-ops` and `tenferro-internal-ad-linalg` no longer re-export `AdTensor` from their crate roots
- `internal-ad-surface` has reduced `AdTensor` / `NodeId` / `AdTensorSnapshot` exposure to crate-private re-exports
- the typed carrier itself still exists and still appears in the internal op/linalg APIs

## Hard Constraints

- Post-cutover `tenferro-rs` must not use `tidu::expert`.
- If a `tenferro` op cannot be expressed with `tidu` high-level traits, extend `tidu` high-level traits instead of escaping to low-level APIs.
- `tenferro-rs` cutover happens in a single PR.
- `AdTensor<T>`-based public surface is removed immediately at cutover.
- `tensor4all-rs` does not add downstream AD logic or ad hoc workarounds.
- Public `elementwise` and `reduction` ops must have full conformance coverage against `chainrules` for both `f64` and `Complex64`.

The remaining design blocker is not the public helper leak. It is the fact that
`internal-ad-ops`, `internal-ad-linalg`, and `internal-ad-surface` still use
`AdTensor<T>` as the typed carrier in the internal AD implementation, including
typed result structs such as SVD/QR/LU outputs and eager AD entrypoints.

## Non-Goals

This cutover does not attempt to complete:

- full PyTorch alias/version-counter/in-place correctness
- a declarative shared `chainrules` rule DSL for by-construction tensor/scalar agreement
- full GPU-native coverage for all elementwise unary AD paths

The current elementwise unary GPU gap is tracked upstream in `tensor4all/tenferro-rs#608`.

Detailed reverse-mode migration work for removing single-tape leakage is captured in
`docs/plans/2026-03-29-tenferro-edge-based-reverse-design.md` and
`docs/plans/2026-03-29-tenferro-edge-based-reverse-plan.md`.

The next carrier-replacement staging is now fixed by
`docs/plans/2026-03-30-adtensor-carrier-replacement-design.md`:

- Stage A: `internal-ad-surface::Tensor` moves to an erased carrier
- Stage B: higher internal crate-boundary APIs move to erased carriers
- Stage C: `AdTensor<T>` is replaced inside core
- Stage D: transitional shims are deleted

So the design question is no longer open-ended. The immediate next code slice is
Stage A, not direct `AdTensor<T>` deletion.

## Repo Responsibilities

### tidu-rs

`tidu-rs` remains a generic AD engine. Its public high-level API is unified around a single op trait.

Responsibilities:

- graph construction and execution
- reverse-mode gradient accumulation
- forward-mode level management
- shared high-level op wiring
- schema validation and node lifecycle

`tidu-rs` does not define tensor kernels, einsum semantics, linalg formulas, or dtype semantics.

### tenferro-rs

`tenferro-rs` owns:

- public `Tensor`
- primal tensor operations
- op-specific `primal`, `jvp`, and `backward` implementations
- dtype/device/layout semantics
- AD surface methods such as `with_requires_grad`, `grad`, `backward`, `copy`, `detach`, `to_device`, and `to_dtype`

`tenferro-rs` implements `tidu` high-level ops for `V = tenferro::Tensor`.

### tensor4all-rs

`tensor4all-rs` only:

- tracks upstream API changes
- updates tests as needed
- preserves existing AD integration coverage, especially TreeTN tests

It does not own tape policy, graph policy, or downstream AD patches.

## tenferro Public Tensor Model

`tenferro::Tensor` remains the single public tensor handle.

Long-term internal shape:

```rust
pub struct Tensor {
    primal: /* tenferro-owned dynamic tensor payload */,
    autograd: Option<AutogradMeta>,
}
```

`AutogradMeta` is a reverse-mode header only. It may contain:

- `requires_grad`
- reverse edge / grad accumulator reference
- leaf/non-leaf state
- retained grad cache

It does not store op saved-state or derivative rules.

Forward-mode state is not stored in `AutogradMeta`. It is owned by `tidu` forward levels and attached through explicit `forward_ad` APIs.

## Public AD Surface

### Reverse Mode

The public reverse-mode model is:

- `with_requires_grad(true)` for leaf creation
- `backward()` for scalar outputs
- `backward_with_gradient(seed)` for non-scalar outputs
- `grad() -> Option<Tensor>`
- `retain_grad()`
- `zero_grad()`
- `requires_grad() -> bool`
- `is_leaf() -> bool`

Agreed semantics:

- `with_requires_grad(true)` is idempotent on an existing grad-requiring leaf
- otherwise `with_requires_grad(true)` detaches and creates a new leaf
- `grad()` returns `None` unless a grad cache exists
- leaf grads accumulate across repeated backward calls
- `zero_grad()` clears visible grad cache and is a no-op if no grad exists
- `backward()` on non-differentiable tensors is an error
- graph memory is released after `backward()` by default, while retained grad caches remain readable

### Forward Mode

The public forward-mode model is explicit:

- `forward_ad::make_dual(x, dx)`
- `forward_ad::unpack_dual(y)`

Agreed semantics:

- reverse and forward participation may coexist on the same `Tensor`
- `make_dual` on an already-dual tensor is an error in phase 1
- `unpack_dual` on a non-dual tensor returns `(primal, None)`
- `detach()` drops both reverse and forward attachments

### Copy and View Families

Two operation families are intentionally distinct.

Copy family:

- `copy()`
- `to_device(...)`
- `to_dtype(...)`

View family:

- `view(...)`
- `reshape(...)`
- `permute(...)`

Agreed semantics:

- `copy()` is a differentiable deep copy op
- `to_device(...)` is differentiable and treated as a linear transfer op
- `to_dtype(...)` is differentiable only when the target dtype is differentiable; casting to integer/bool produces a non-differentiable tensor
- `view(...)` is zero-copy when possible
- `reshape(...)` is zero-copy when possible and may materialize otherwise
- `permute(...)` is a metadata op
- view-family outputs are non-leaf op results when the input requires grad

## tidu High-Level Op API

The old `Function` / `MultiOutputOp` split is replaced by a single high-level trait, conceptually named `Op`.

Target shape:

```rust
trait Op<V: Differentiable> {
    type SavedBackward;
    type SavedJvp;

    fn primal(inputs: &[&V]) -> AdResult<Vec<V>>;
    fn input_schema(inputs: &[&V]) -> AdResult<Schema>;
    fn output_schema(inputs: &[&V], outputs: &[V]) -> AdResult<Schema>;
    fn save_for_backward(inputs: &[&V], outputs: &[V]) -> AdResult<Self::SavedBackward>;
    fn save_for_jvp(inputs: &[&V], outputs: &[V]) -> AdResult<Self::SavedJvp>;
    fn materialize_grads() -> bool { false }
    fn backward(
        saved: &Self::SavedBackward,
        grad_outputs: &[Option<V>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<V>>>;
    fn jvp(
        saved: &Self::SavedJvp,
        tangents: &[Option<V>],
    ) -> AdResult<Vec<Option<V>>>;

    fn apply(inputs: &[&Value<V>]) -> AdResult<Vec<Value<V>>> { /* default */ }
}
```

This trait remains generic in `V`.

Single-output ops are the one-output case of `Op`. `tidu-rs` should provide helper wrappers or macros so single-output ops do not need to hand-build one-element vectors.

In this note, `AdResult<T>` means the public `tidu` result type used by high-level AD APIs. It is not shorthand for `anyhow::Result<T>`.

`backward(...)` is the VJP entrypoint for the op. `tenferro` may document this as backward-mode AD, but the trait-level meaning is explicitly vector-Jacobian product.

## Schema Model

`tidu-rs` owns the schema layer.

Minimal phase-1 model:

```rust
pub struct SlotSchema {
    pub differentiable: bool,
    pub auxiliary: bool,
}

pub struct Schema {
    pub slots: Vec<SlotSchema>,
}
```

Invariants:

- `auxiliary = true` implies `differentiable = false`

Important boundary:

- dtype concepts such as integer/real/complex do not belong in `tidu::SlotSchema`
- those remain `tenferro::Tensor` semantics

This keeps `tidu` focused on graph roles instead of tensor typing policy.

The schema is runtime-driven, not compile-time static. `output_schema(inputs, outputs)` is the phase-1 replacement for dynamic output marking such as PyTorch's `mark_non_differentiable(...)`.

## Output and Input Policy

Internally, `Op` uses variable-length input and output slices.

Public `tenferro` methods remain schema-fixed:

- `x.sin()? -> Tensor`
- `x.add(&y)? -> Tensor`
- `x.svd()? -> (Tensor, Tensor, Tensor)`

The public API is fixed-schema, while `tidu` internals operate on:

- input slices: `&[&Tensor]`
- output vectors: `Vec<Tensor>`

For multi-output ops, runtime representation is one node plus multiple output slots.

`input_schema(...)` states which inputs are differentiable in principle for the op call. `input_grad_mask` in `backward(...)` states which input gradients are actually requested for the current backward pass. The former is part of op meaning; the latter is an execution-time optimization signal analogous to PyTorch's `needs_input_grad`.

Phase-1 default gradient materialization policy is non-materializing. Undefined output gradients stay `None` and are passed through to `backward(...)` and `jvp(...)` as `Option<V>`. An op may opt into zero materialization by overriding `materialize_grads()`.

## chainrules Relationship

`chainrules-rs` remains the source of truth for derivative meaning.

Rules:

- do not invent a separate derivative semantics layer inside `tidu`
- `tidu` high-level ops are wiring/execution abstractions
- `tenferro` owns op-specific adapters from tensor operations to `tidu` high-level ops

For phase 1:

- `elementwise` and `reduction` tensor-level AD may use hand-written tensor formulas in `tenferro`
- but they must be covered by conformance tests against scalar `chainrules` lifting for both `f64` and `Complex64`

The stronger by-construction unification via shared declarative rule specs is deferred.

## einsum and Structured Gradients

`einsum` remains a `tenferro` op.

Design rules:

- `einsum` primal, JVP, and backward are implemented in `tenferro`
- `tidu` only provides the op/node model
- phase 1 does not use dense-to-structured roundtrips as an AD staging mechanism
- structured gradients must be returned directly
- saved-state may include detached `Tensor` inputs and op metadata such as einsum specs

The same principle applies to `elementwise`, `reduction`, and linalg ops: phase 1 should not preserve helper designs that fundamentally depend on dense host snapshots as the core AD model.

## Testing and Verification

### tidu-rs

- add tests for the unified `Op` trait
- cover schema invariants
- cover single-output helpers
- cover non-differentiable output handling

### tenferro-rs

- migrate existing test coverage to the new `Tensor` model
- keep all existing public-op coverage
- add tests for:
  - `with_requires_grad`
  - `grad`
  - `retain_grad`
  - `zero_grad`
  - `copy` / `detach`
  - `to_device` / `to_dtype`
  - forward/reverse coexistence
  - multi-output slot semantics
- require conformance coverage for all public `elementwise` and `reduction` ops for both `f64` and `Complex64`

### tensor4all-rs

Done condition:

- existing suite remains green against the new upstream stack
- TreeTN AD integration coverage is preserved

## PR Strategy

Use stacked PRs, one repo per PR:

1. `tidu-rs`
2. `tenferro-rs`
3. `tensor4all-rs`

The final `tidu-rs` public reverse-mode API should also be unified around the
edge-based `Value` / `Op` model rather than preserving a parallel public
`expert::Tape` reverse surface.

But inside `tenferro-rs`, the cutover itself is a single PR:

- no prolonged coexistence with `AdTensor<T>`
- no fallback to `tidu::expert`
- no downstream ad hoc patching

Because the final cutover still lands as one PR, the implementation should be commit-sliced inside that PR. The preferred slicing is:

1. `tidu-rs` dependency update and new high-level `Op` scaffolding
2. `tenferro::Tensor` internal model change
3. op-family migrations (`copy`/`to_*`, view family, elementwise/reduction, einsum, linalg)
4. old `AdTensor<T>` public-surface removal
5. tests and docs refresh

The final merged diff still performs a single cutover, but the PR history should remain reviewable and bisectable.

## Deferred Follow-Up

- full alias/version-counter/in-place semantics
- declarative shared `chainrules` rule specs for by-construction tensor/scalar agreement
- complete GPU-native elementwise unary AD coverage beyond the currently tracked gap
- optional saved-tensor memory features such as clearing saved tensors on first access
