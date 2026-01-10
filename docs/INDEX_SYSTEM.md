# Index System Design

This document describes the index system design in tensor4all-rs-index-like, including compatibility with QSpace and ITensors.jl.

## Rust primer: What is a trait?

In Rust, a **trait** is an interface that defines a set of methods a type must provide.
It is similar to an “interface” in other languages.

- A trait can have **required methods** (must be implemented by each concrete type).
- A trait can have **default methods** (a reusable implementation that can be overridden).
- A trait can have **associated types** (placeholders for types chosen by the implementer).

This repository uses traits to make algorithms generic over different index/tensor implementations
while keeping a consistent API surface.

### Why traits matter here

We want algorithms (contraction, factorization, etc.) to work with:
- ITensors.jl-like **directionless** indices for general-purpose tensor math
- QSpace-like **directed** indices (bra/ket) for quantum tensor networks

Traits let us express these behaviors in a single API, while allowing different concrete types
to choose their own internal representation.

## Overview

The index system in tensor4all-rs-index-like is designed to support both:
- **ITensors.jl-like behavior**: Directionless indices for convenient tensor operations
- **QSpace compatibility**: Directed indices (bra/ket) for quantum tensor networks

## QSpace Overview

QSpace (extern/qspace-v4-pub) is a MATLAB/Octave library for tensor networks with quantum number symmetries. In QSpace:

- **Index direction** is encoded via trailing `*` in `itags`:
  - Index **without** trailing `*` → **ingoing** (ket)
  - Index **with** trailing `*` → **outgoing** (bra/conjugate)
- **Tensor conjugate** (`conj(QSpace)`) operation:
  1. Reverses all arrows (toggles `*` flags on all indices)
  2. Keeps the same qlabels (quantum number labels)
  3. Complex conjugates all data if applicable

This design allows QSpace to naturally represent operators (with mixed in/out indices) and state vectors (with consistent direction).

## IndexLike Trait

The `IndexLike` trait abstracts index types and provides a unified interface for index operations:

```rust
pub trait IndexLike: Clone + Eq + Hash + Debug + Send + Sync + 'static {
    type Id: Clone + Eq + Hash + Debug + Send + Sync;
    
    fn id(&self) -> &Self::Id;
    fn dim(&self) -> usize;
    fn conj_state(&self) -> ConjState;
    fn conj(&self) -> Self;
    fn is_contractable(&self, other: &Self) -> bool;
    // ... other methods
}
```

### Equality vs contractability (important)

This design intentionally separates two concepts:

- **Equality (`Eq`/`==`)**: “Are these two index *objects* the same?”
  - This is used by Rust collections like `HashSet`/`HashMap`.
  - `Hash` must be consistent with `Eq` (if `a == b`, they must hash the same).
- **Contractability (`is_contractable`)**: “Can these two indices be contracted?”
  - This is the tensor-network rule.
  - It may depend on direction (Ket/Bra) and always checks `id` and `dim`.

In other words: **do not use `==` to decide contraction**; use `is_contractable`.

### ConjState Enum

The `ConjState` enum represents the direction (conjugate state) of an index:

```rust
pub enum ConjState {
    Undirected,  // Directionless index (ITensors.jl-like default)
    Ket,         // Ingoing index (QSpace: no trailing `*`)
    Bra,         // Outgoing index (QSpace: trailing `*`)
}
```

**QSpace mapping**:
- `Ket` = ingoing index (QSpace: itag **without** trailing `*`)
- `Bra` = outgoing index (QSpace: itag **with** trailing `*`)

### A note on naming: “conjugate”, “bra/ket”, “ingoing/outgoing”

This document uses **Ket/Bra** to describe *index direction* (dual-ness).
This is distinct from **complex conjugation of numeric data**.

- **Index direction (Ket/Bra)**: metadata on each leg
- **Tensor conjugation (`TensorLike::conj`)**: conjugates numeric data *and* maps indices via `IndexLike::conj()`

This mirrors QSpace’s definition: conjugation reverses arrows and conjugates data (if complex).

### Contractability Rules

Two indices are contractable if:
1. They have the same `id()` and `dim()`
2. Their conjugate states are compatible:
   - `(Ket, Bra)` or `(Bra, Ket)` → **contractable** (opposite directions)
   - `(Undirected, Undirected)` → **contractable** (both directionless)
   - Mixed `(Undirected, Ket/Bra)` → **not contractable** (mixing forbidden)

The default implementation of `is_contractable()` enforces these rules.

#### Why mixing is forbidden

If one index is directed (Ket/Bra) but the other is undirected, contraction semantics become ambiguous.
To prevent “tests passing with incorrect direction handling” (especially in directed tensor-network code),
we disallow mixing by default.

### Equality and Hashing

`IndexLike` requires `Eq + Hash`. The equality semantics are:
- **Object equality**: Two indices are equal if they represent the same object
  - This includes ID, dimension, and conjugate state (if the concrete type stores it)
- **Hash consistency**: `Hash` must be consistent with `Eq` (if `a == b`, then `hash(a) == hash(b)`)

This allows indices to be used in `HashSet` and `HashMap` for efficient lookups.

## TensorLike Trait

The `TensorLike` trait provides a unified interface for tensor operations:

```rust
pub trait TensorLike: Sized + Clone + Debug + Send + Sync {
    type Index: IndexLike;
    
    fn external_indices(&self) -> Vec<Self::Index>;
    fn conj(&self) -> Self;
    // ... other methods
}
```

### Tensor Conjugate

The `conj()` method on `TensorLike`:
- Conjugates scalar data (complex conjugate for `Complex64` tensors)
- Maps indices via `IndexLike::conj()` (future-proof for QSpace compatibility)

For default undirected indices, `IndexLike::conj()` is a no-op, so this preserves ITensors.jl-like behavior while being ready for QSpace-compatible directed indices.

**Note**: `TensorLike` does **not** have `is_conjugate()` method. The conjugate state is managed at the index level, not the tensor level.

## Default Implementation

The default index type `DynIndex` (in `tensor4all-core`) implements:
- `conj_state() -> ConjState::Undirected` (directionless by default)
- `conj() -> Self` as identity (no-op)

This preserves ITensors.jl-like convenience where indices are directionless by default, making tensor operations simpler for general use cases.

### Future: Directed Index Types

When implementing QSpace-compatible tensor networks (e.g., in TreeTNN), you can introduce a directed index type that:
- Returns `ConjState::Ket` or `ConjState::Bra` from `conj_state()`
- Toggles between `Ket` and `Bra` in `conj()`
- Enforces direction-aware contractability via `is_contractable()`

This allows the same `IndexLike`/`TensorLike` API to work with both undirected (ITensors.jl-like) and directed (QSpace-like) indices.

## Minimal example: implementing a directed IndexLike

This is a conceptual example (not necessarily the exact structs used in the repo):

```rust
use std::hash::{Hash, Hasher};
use tensor4all_core::{ConjState, IndexLike};

#[derive(Clone, Debug)]
struct MyIndex {
    id: u64,
    dim: usize,
    state: ConjState,
}

impl PartialEq for MyIndex {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.dim == other.dim && self.state == other.state
    }
}
impl Eq for MyIndex {}

impl Hash for MyIndex {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.id.hash(h);
        self.dim.hash(h);
        self.state.hash(h);
    }
}

impl IndexLike for MyIndex {
    type Id = u64;

    fn id(&self) -> &Self::Id { &self.id }
    fn dim(&self) -> usize { self.dim }
    fn conj_state(&self) -> ConjState { self.state }

    fn conj(&self) -> Self {
        let state = match self.state {
            ConjState::Ket => ConjState::Bra,
            ConjState::Bra => ConjState::Ket,
            ConjState::Undirected => ConjState::Undirected,
        };
        Self { state, ..self.clone() }
    }
}
```

Key point: `Eq`/`Hash` must remain consistent, and `is_contractable` is provided by the default trait method.

## Design Rationale

### Why Undirected by Default?

ITensors.jl uses directionless indices by default, which is very convenient for general tensor operations. By making defaults undirected, tensor4all-rs-index-like maintains this convenience while still supporting QSpace-style directed indices when needed.

### Why Forbid Mixed Directed/Undirected?

Mixing directed and undirected indices in contractions would be ambiguous:
- Should an `Undirected` index contract with a `Ket` index?
- What does it mean to have a tensor with mixed directed/undirected indices?

By forbidding mixing, we ensure clear semantics:
- Undirected indices work with other undirected indices (ITensors.jl-like)
- Directed indices work with opposite-directed indices (QSpace-like)
- No ambiguity about which indices can contract

### Why Object Equality (not ID-only)?

Using object equality (including conjugate state) allows:
- `HashSet` to distinguish between `i` and `conj(i)` (same ID, different direction)
- Clear separation between "same index" (`==`) and "contractable" (`is_contractable()`)
- Future QSpace compatibility where `i` and `i*` are different objects

This is consistent with Rust's `Eq`/`Hash` contract and enables efficient lookups while maintaining clear semantics.

## References

- [QSpace v4](https://github.com/weichselbaum/QSpace) — A. Weichselbaum, Annals of Physics **327**, 2972 (2012)
- [ITensors.jl](https://github.com/ITensor/ITensors.jl) — M. Fishman, S. R. White, E. M. Stoudenmire, arXiv:2007.14822 (2020)
