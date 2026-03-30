# Design: AdTensor Carrier Replacement

**Date:** 2026-03-30
**Status:** Drafted after refined-plan checkpoint
**Scope:** Detailed design for removing `AdTensor<T>` from higher crates without losing the current green reverse-mode baseline

## Goal

Remove `AdTensor<T>` as a public and cross-crate carrier on the path to the
approved torch-like `Tensor + edge-based reverse metadata` model.

The intended end state is:

- public `tenferro::Tensor` does not expose `AdTensor<T>` or `DynAdTensor`
- higher internal crates do not use `AdTensor<T>` in their crate-boundary APIs
- `AdTensor<T>` does not survive as the long-term core carrier
- mixed-dtype multi-output ops such as SVD still work naturally

## Non-Goals

This note does not try to:

- delete `AdTensor<T>` in one broad mechanical sweep
- complete the entire `Tensor + AutogradMeta` rewrite in one slice
- preserve the current enum-shaped public `Tensor` API for compatibility reasons

Early development rules apply: shape changes are allowed if they reduce leakage.

## Why Another Design Note Is Needed

The refined cutover plan now hits an explicit design gate.

What is already true:

- helper-free reverse joins across independently created leaves are green
- explicit backward inputs on non-leaf tensors are green
- the public tape-helper leak is gone
- `DynAdTensor` exists as a transitional erased carrier

What is still structurally true:

- `tenferro-internal-ad-surface::Tensor` still stores `AdTensor<T>` directly
- `TypedTensorRef` still reconstructs `&AdTensor<T>`
- `tenferro-internal-ad-ops` and `tenferro-internal-ad-linalg` still expose
  typed builder/result APIs based on `AdTensor<T>`

That means the current problem is no longer visibility cleanup. It is carrier
replacement.

## Current Ownership Graph

Today the ownership graph is effectively:

- `tenferro-internal-ad-core`
  - owns `AdTensor<T>`
  - owns `DynAdTensor`
- `tenferro-internal-ad-surface`
  - public `Tensor` enum payload is still `AdTensor<T>`
  - typed accessors still return views backed by `&AdTensor<T>`
- `tenferro-internal-ad-ops`
  - typed op builders consume `&AdTensor<T>` and return `AdTensor<T>`
- `tenferro-internal-ad-linalg`
  - typed eager/builders/results still use `AdTensor<T>`
  - newer dyn entrypoints exist, but typed APIs are still canonical in many places

As long as this graph remains, deleting `AdTensor<T>` is not a cleanup task.

## Design Constraints

- post-cutover `tenferro` must not use `tidu::expert`
- public `Tensor` must remain the only public tensor handle
- mixed-dtype outputs must remain expressible without fake homogenization
- structured layout information must remain attached to the tensor payload, not
  reconstructed from dense snapshots
- each transition slice must leave release tests green

## PyTorch-Informed Direction

PyTorch does not expose a typed AD carrier between `Tensor` and autograd.

The useful analogies are:

- public tensor handle owns storage and nullable autograd metadata
- reverse graph is stored as edges, not tape identity
- typed kernels operate after dtype dispatch, not through a public typed carrier

For `tenferro`, this suggests:

- public `Tensor` should stop carrying `AdTensor<T>` in its enum variants
- typed `AdTensor<T>` should stop crossing crate boundaries before it is deleted
- typed kernel dispatch should happen inside op implementations, not in public or
  cross-crate surface types

## Rejected Strategies

### 1. Keep `DynAdTensor` as the final carrier

Rejected because:

- it still mirrors the `AdTensor<T>` era
- it still encourages typed-carrier accessors such as `as_f64() -> &AdTensor<f64>`
- it does not match the approved `Tensor + edge metadata` target

`DynAdTensor` is acceptable only as a transitional cross-crate shell.

### 2. Replace `AdTensor<T>` in place everywhere first

Rejected because:

- it couples surface, ops, linalg, and core rewrites into one high-risk batch
- it makes red/green localization poor
- it is incompatible with the refined plan's design gate

## Chosen Transitional Strategy

The transition should happen in four ordered stages.

### Stage A: Public and surface carrier unification

Make `DynAdTensor` the only carrier visible above `tenferro-internal-ad-core`.

Concretely:

- `tenferro-internal-ad-surface::Tensor` stops storing `AdTensor<T>` directly
- `Tensor` becomes a wrapper over `DynAdTensor`
- public and integration tests stop matching `Tensor::F64(value)` to obtain
  `&AdTensor<f64>`
- `TypedTensorRef` survives, but it is a typed view over erased storage, not a
  disguised `&AdTensor<T>`

This is the first real boundary cut. It does not delete `AdTensor<T>`, but it
removes direct typed-carrier leakage from the public tensor wrapper.

### Stage B: Erased cross-crate crate-boundary APIs

Make `DynAdTensor` or `DynAdTensorRef` the canonical external interface of the
higher internal crates.

Concretely:

- `tenferro-internal-ad-ops` crate-boundary entrypoints use erased carriers
- `tenferro-internal-ad-linalg` crate-boundary entrypoints and result structs use
  erased carriers
- typed builders such as `*_ad(...) -> Builder<'_, T>` become internal
  implementation details, not the canonical cross-crate API

At the end of Stage B:

- higher crates may still use `AdTensor<T>` internally
- but cross-crate wiring no longer depends on it

This is the minimum acceptable condition before touching the core carrier.

### Stage C: Replace the core carrier

Only after Stages A and B are green should `AdTensor<T>` itself be replaced.

The target core shape is a dynamic, edge-aware tensor state, conceptually:

```rust
pub struct DynAutogradTensor {
    primal: DynTensor,
    tangent: Option<DynTensor>,
    reverse: Option<ReverseState>,
    flags: AutogradFlags,
}
```

with:

```rust
pub struct ReverseState {
    edge: Option<ReverseEdge<DynTensor>>,
    retained_grad: Option<DynTensor>,
}
```

and:

```rust
pub struct AutogradFlags {
    requires_grad: bool,
    is_leaf: bool,
}
```

Important consequences:

- dtype dispatch happens by downcasting `primal` / `tangent` inside op kernels
- reverse metadata is no longer stored inside a typed carrier
- `TypedTensorRef<'a, T>` becomes a borrow/view API over dynamic payload
- `DynAdTensor` becomes unnecessary once `Tensor` can own `DynAutogradTensor`
  directly

This stage is where `AdTensor<T>` actually dies.

### Stage D: Delete transitional shims

After Stage C is green:

- delete `DynAdTensor`
- delete `DynAdTensorRef`
- delete typed-carrier conversions `From<AdTensor<T>>`
- delete tests that explicitly assert `AdTensor<T>` access paths

## Typed View Design

`TypedTensorRef` should remain as a user-facing and test-facing convenience, but
its implementation should no longer require access to `&AdTensor<T>`.

Target behavior:

- `Tensor::as_f64()` returns `Option<TypedTensorRef<'_, f64>>`
- `TypedTensorRef` provides:
  - `primal()`
  - `structured_primal()`
  - `tangent()`
  - `structured_tangent()`
  - `requires_grad()`
  - `is_leaf()`
  - `node_id()` or the future reverse-handle equivalent
- none of those methods expose `AdTensor<T>`

This allows tests and downstream code to stay typed where useful without
coupling them to the old carrier.

## Mixed-Dtype Ops Are The Main Reason To Prefer Erased Boundaries

SVD is the clearest example.

For `Complex64` input:

- `u: Complex64`
- `s: f64`
- `vt: Complex64`

This is awkward if the higher crates insist on `AdTensor<T>` as the result
carrier, because one result tuple naturally contains more than one scalar type.

Erased boundary carriers make this natural:

- dynamic wrappers return `Tensor`
- internal crate-boundary results return `DynAdTensor`
- typed kernels still execute after runtime dtype dispatch

The same pressure exists for `eig`, `slogdet`, and any future mixed-dtype
output op.

## Ordered Execution Plan

The next implementation batches should follow this order exactly.

1. Convert `tenferro-internal-ad-surface::Tensor` from `AdTensor<T>` payloads to
   `DynAdTensor` payload or an equivalent erased wrapper.
2. Rewrite public/internal tests that still pattern-match on `Tensor::F64(...)`
   or otherwise extract `&AdTensor<T>` from `Tensor`.
3. Convert higher internal crate-boundary result structs and entrypoints to
   erased carriers only.
4. Re-run the ownership graph query. If `AdTensor<T>` is now local to
   `tenferro-internal-ad-core` internals, begin the actual core carrier
   replacement.
5. Replace `AdTensor<T>` with a dynamic edge-aware carrier in core.
6. Delete `DynAdTensor` once the new core carrier is in place.

## Stop Conditions

Stop and re-design before continuing if either of these happens:

- Stage A reveals that public `Tensor` ergonomics or tests fundamentally require
  direct access to `AdTensor<T>`
- Stage B reveals an unavoidable cycle that forces higher crates to expose typed
  carriers again

Both conditions would mean the intended `Tensor + edge metadata` target has not
been reduced to a viable sequence yet.

## Immediate Next Slice

The next concrete code slice should be Stage A:

- make `tenferro-internal-ad-surface::Tensor` an erased-carrier wrapper
- preserve `Tensor::as_f32/as_f64/as_c32/as_c64`
- keep behavior green under release tests
- do not begin Stage C in the same batch

That is the smallest slice that materially reduces `AdTensor<T>` leakage without
trying to delete the carrier too early.
