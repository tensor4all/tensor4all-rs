# Index System Design

This document describes the index system design in tensor4all-rs, including compatibility with QSpace and ITensors.jl.

## Table of Contents

1. [Overview](#overview)
2. [Comparison: ITensors.jl, QSpace, tensor4all-rs](#comparison-itensorsjl-qspace-tensor4all-rs)
3. [How tensor4all-rs Supports Both ITensors.jl and QSpace](#how-tensor4all-rs-supports-both-itensorsjl-and-qspace)
4. [Rust Primer: What is a Trait?](#rust-primer-what-is-a-trait)
5. [Trait Requirements at a Glance](#trait-requirements-at-a-glance)
6. [QSpace Details](#qspace-details)
7. [IndexLike Trait](#indexlike-trait)
8. [TensorLike Trait](#tensorlike-trait)
9. [Default Implementation (DynIndex)](#default-implementation-dynindex)
10. [Design Rationale](#design-rationale)
11. [Index IDs in TreeTN](#index-ids-in-treetn)
12. [Migrating from QSpace](#migrating-from-qspace)
13. [References](#references)

---

## Overview

The index system in tensor4all-rs is designed to support both:
- **ITensors.jl-like behavior**: Directionless indices for convenient tensor operations
- **QSpace compatibility**: Directed indices (bra/ket) for quantum tensor networks

---

## Comparison: ITensors.jl, QSpace, tensor4all-rs

### Summary Table

| Aspect | ITensors.jl | QSpace | tensor4all-rs |
|--------|-------------|--------|---------------|
| **Central entity** | Index | Tensor | Configurable (Index-centric by default) |
| **Index identity** | UUID (auto-generated at creation) | itag name (string, assigned later) | `Id` associated type (u64 UUID by default) |
| **Connection mechanism** | Share same Index object | Same itag name + opposite direction | Same `Id` value + compatible `ConjState` |
| **Direction** | Undirected (default) | Directed (Ket/Bra via `*` suffix) | Both supported via `ConjState` |
| **Symmetry / quantum numbers** | Usually implicit (dense tensors) | Explicit symmetry sectors / block structure | Not in defaults; modeled by symmetry-aware concrete types |
| **When ID is assigned** | Index creation | After tensor creation (`setitags`) | Index creation |
| **ID mutability** | Immutable | Reassignable | Immutable |
| **Granularity** | One Index per tensor leg | One itag per tensor leg | One Index per tensor leg |

### ITensors.jl: Index-Centric Design

In ITensors.jl, **Index is the primary entity**. The workflow is:

```julia
# 1. Create Index independently (UUID auto-generated)
i = Index(2, "site")
j = Index(3, "bond")
k = Index(3, "bond")  # Different UUID from j!

# 2. Build tensors using those Indices
A = ITensor(i, j)
B = ITensor(j, k)  # j is shared → A and B are connected via j

# 3. Contract: shared Index j is automatically contracted
C = A * B  # Result has indices i, k
```

Key characteristics:
- **UUID at creation**: Each `Index(dim, tags)` call generates a unique UUID
- **Identity by object**: Two tensors share an index iff they hold the same Index object
- **Immutable identity**: Once created, an Index's UUID cannot be changed
- **Tags are metadata**: Tags like `"site"` are for human readability, not for matching

### QSpace: Tensor-Centric Design

In QSpace, **Tensor is the primary entity**. The workflow is:

```matlab
% 1. Create tensor with quantum number structure
A = getIdentity(...);  % rank-3 tensor with symmetry sectors

% 2. Assign itag names AFTER tensor creation
A = setitags(A, {'K01', 'K02*', 's02'});
%                  ↑      ↑       ↑
%                leg 0  leg 1   leg 2
%                (Ket)  (Bra)   (Ket)

% 3. Another tensor with matching itag
B = setitags(B, {'K02', 'K03*', 's03'});
%                  ↑
%               matches A's 'K02*' (name match + opposite direction)

% 4. Contract: matching itags are automatically found
C = contract(A, B);  % K02* and K02 contract
```

Key characteristics:
- **Name-based matching**: itag string name (e.g., `"K02"`) determines pairing
- **Direction via suffix**: trailing `*` indicates Bra (outgoing), no `*` indicates Ket (ingoing)
- **Late binding**: itags can be assigned/changed after tensor creation
- **Tensor owns structure**: Quantum numbers and block structure belong to tensor, not index

Symmetry note (important):
- In symmetry-aware libraries like QSpace, "index structure" is not just `(id, dim, direction)`.
  It also involves symmetry labels, sector dimensions, block metadata, and often constraints that
  touch both **index** and **tensor storage**. This makes symmetry handling inherently more complex
  than a single-field attribute on `Index`.

### tensor4all-rs: Flexible Design

tensor4all-rs supports both paradigms by separating:
- **Concrete types** (how indices/tensors store identity, direction, symmetry metadata, blocks, etc.)
- **Algorithms** (TreeTN, contraction, factorization) that only rely on a small, explicit interface

At a high level, ITensors.jl-like and QSpace-like systems share one key idea:
**some pairing key exists to decide which legs should connect** (UUID vs itag-name).

The differences are in *how much extra structure is attached to each leg*:
- ITensors.jl defaults to **undirected**, dense-friendly indices.
- QSpace attaches **direction + symmetry sector structure** and uses tensor-centric conventions.

Because symmetry handling involves both Index and Tensor internals, it is not realistic to cover
both ITensors-like and QSpace-like behavior with a single "one-size-fits-all" `Index` concrete type.
Instead, tensor4all-rs defines **traits** that capture the *minimum* functionality algorithms need,
and different concrete types implement those traits.

---

## How tensor4all-rs Supports Both ITensors.jl and QSpace

### Design goals

- Provide ITensors.jl-like ergonomics by default (undirected indices, simple dense tensors).
- Allow QSpace-like directed/symmetry-aware implementations without forcing those concepts into the default types.
- Keep algorithm crates (TreeTN, contraction, factorization) generic and reusable.

### Key design points

- **Shared concept: pairing key**  
  Both ecosystems rely on a pairing key: UUID (ITensors.jl) or itag-name (QSpace).
  In tensor4all-rs this is `IndexLike::Id` + `IndexLike::id()`.

- **Direction is per-index, not per-tensor**  
  QSpace-like direction (bra/ket) is modeled by `ConjState` on `IndexLike`.
  `TensorLike::conj()` is expected to conjugate data *and* map indices via `IndexLike::conj()`.

- **Symmetry is not in defaults**  
  The default `DynIndex` does not store symmetry. A QSpace-compatible non-Abelian index should be a
  separate concrete type implementing `IndexLike`, with its own symmetry metadata and storage model.

- **Algorithms depend only on trait methods**  
  Higher-level algorithms (e.g. TreeTN) operate only through the trait surface (e.g. `external_indices`,
  `tensordot`, `factorize`, `conj`, and index identity/direction accessors). This avoids leaking internal
  symmetry/block details into generic code.

---

## Rust Primer: What is a Trait?

In Rust, a **trait** is an interface that defines a set of methods a type must provide.
It is similar to an "interface" in other languages.

- A trait can have **required methods** (must be implemented by each concrete type).
- A trait can have **default methods** (a reusable implementation that can be overridden).
- A trait can have **associated types** (placeholders for types chosen by the implementer).

In tensor4all-rs, traits are the boundary that lets us:
- Keep algorithms generic (TreeTN, contraction, factorization)
- Plug in different concrete types (ITensors-like dense vs QSpace-like symmetry-aware)

### Minimal interface, implementation freedom

`IndexLike` and `TensorLike` are intentionally designed as **minimal interfaces**:
they define *what* operations an algorithm needs, but do **not** prescribe *how* they are implemented.

This makes it possible to support very different internals, for example:
- **Quantum numbers / symmetry sectors** (Abelian/non-Abelian)
- **Block-sparse storage layouts** and metadata
- **Different contraction backends/algorithms** (dense, block-sparse, optimized order, GPU, etc.)

---

## Trait Requirements at a Glance

This section lists the **practical** requirements that concrete types must satisfy so that
generic algorithms can use them.

### `IndexLike` requirements

- Provide a **pairing key** via `type Id` and `fn id(&self) -> &Id`.
- Provide a **dimension** via `fn dim(&self) -> usize`.
- Provide a **direction / conjugation state** via `fn conj_state(&self) -> ConjState`.
- Provide a **conjugation mapping** via `fn conj(&self) -> Self`.
- Provide a **fresh-but-similar index** via `fn sim(&self) -> Self`.
- Use `fn is_contractable(&self, other: &Self) -> bool` (default provided) for contraction checks.

### `TensorLike` requirements (conceptual)

Concrete tensor types should provide operations needed by the algorithms (e.g. TreeTN):
- Enumerate external indices: `external_indices() -> Vec<Index>`
- Conjugate tensor: `conj() -> Self` (should also map indices via `IndexLike::conj()`)
- Contract tensors: `tensordot(&self, other: &Self, pairs: &[(Index, Index)]) -> Result<Self>`
- Factorize / split: `factorize(...) -> Result<FactorizeResult<Self>>` (or equivalent)

### Example (illustrative pseudo-code)

```rust
// Default: ITensors.jl-like (UUID-based, undirected)
let i = DynIndex::new_dyn(2);
let j = DynIndex::new_dyn(3);
let j_clone = j.clone(); // same ID as j

// Two tensors that share the same pairing key (ID) can connect along that leg.
// (Exact tensor constructors differ by concrete tensor type; shown conceptually.)
let a = /* TensorLike */ /* ... */ ;
let b = /* TensorLike */ /* ... */ ;

// Contraction must be decided by indices (pairing key + compatibility), not tensor-global flags.
let c = a.tensordot(&b, &[(j, j_clone)]);
```

For QSpace-like behavior, implement a custom `IndexLike` with `type Id = String` (itag-name),
and a symmetry-aware tensor concrete type that understands sector/block metadata.

---

## QSpace Details

QSpace (extern/qspace-v4-pub) is a MATLAB/Octave library for tensor networks with quantum number symmetries.

### itag Internal Structure

The `itag_` class stores index tags in a compact format:

```cpp
class itag_ {
    IDT t;  // IDT = unsigned long (8 bytes)
};
```

**Storage format**:
- 8-byte packed ASCII string (maximum 8 characters)
- First byte's bit 7 (sign bit) stores the **conjugate flag**
- Remaining 7 bits of each byte store ASCII characters

**Example**:
```
itag "K01"  → bytes: ['K', '0', '1', 0, 0, 0, 0, 0]
itag "K01*" → bytes: ['K'|0x80, '0', '1', 0, 0, 0, 0, 0]
                      ↑ bit 7 set (conjugate flag)
```

### Conjugate Flag Operations

```cpp
// Check if conjugated (Bra)
bool isConj() const {
    return (((char&)t) < 0);  // First byte negative = bit 7 set
}

// Toggle conjugate flag
itag_& Conj() {
    ((char&)t) ^= char(128);  // XOR bit 7
    return *this;
}
```

### Matching Logic: `sameAs()`

The `sameAs()` method determines if two itags match:

```cpp
char itag_::sameAs(const itag_ &B, char lflag) const {
    IDT c128 = 128, a = t, b = B.t;
    bool cflag = (a & c128) != (b & c128);  // Different conjugate flags?

    a &= ~c128;  // Mask out conjugate flag
    b &= ~c128;

    if (a == b) { q = 3; }  // Names match

    if (cflag) { q = -q; }  // Opposite directions → negative return

    return q;
    // +3: same name, same direction (not contractable)
    // -3: same name, opposite direction (CONTRACTABLE!)
    //  0: different names
}
```

**Contractability rule**:
- Same itag name AND opposite conjugate flags → contractable
- `"K01"` (Ket) + `"K01*"` (Bra) → contractable (returns -3)
- `"K01"` + `"K01"` → NOT contractable (returns +3)
- `"K01"` + `"K02"` → NOT contractable (returns 0)

### Tensor Conjugate Operation

When conjugating a QSpace tensor:
1. **Toggle all itag conjugate flags** (Ket ↔ Bra for all legs)
2. **Keep quantum number labels unchanged**
3. **Complex conjugate all data** (if applicable)

This mirrors the mathematical definition of tensor conjugation in quantum mechanics.

### itag Assignment: `setitags()`

itags are assigned AFTER tensor creation:

```matlab
% Usage patterns:
A = setitags(A, {'K01', 'K02', 's02'});           % Explicit names
A = setitags(A, '-A', k);                          % Auto-generate for site k
A = setitags(A, '-op:X', 'op');                    % Operator format

% Auto-generation for MPS A-tensor at site k:
% '-A', k  →  {'K<k-1>', 'K<k>', 's<k>'}
% Example: '-A', 2  →  {'K01', 'K02', 's02'}
```

This late-binding design allows the same tensor structure to be reused with different connectivity.

---

## IndexLike Trait

The `IndexLike` trait abstracts index types and provides a unified interface for index operations.

### Trait Definition

```rust
pub trait IndexLike: Clone + Eq + Hash + Debug + Send + Sync + 'static {
    /// Lightweight identifier type (conjugate-independent).
    type Id: Clone + Eq + Hash + Debug + Send + Sync;

    /// Get the identifier of this index.
    fn id(&self) -> &Self::Id;

    /// Get the total dimension (state-space dimension) of the index.
    fn dim(&self) -> usize;

    /// Get the conjugate state (direction) of this index.
    fn conj_state(&self) -> ConjState;

    /// Create the conjugate of this index.
    fn conj(&self) -> Self;

    /// Check if this index can be contracted with another index.
    fn is_contractable(&self, other: &Self) -> bool { /* default impl */ }

    /// Check if this index has the same ID as another.
    fn same_id(&self, other: &Self) -> bool { /* default impl */ }

    /// Check if this index has the given ID.
    fn has_id(&self, id: &Self::Id) -> bool { /* default impl */ }

    /// Create a similar index with a new identity but the same structure.
    fn sim(&self) -> Self;
}
```

### ConjState Enum

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

### Contractability Rules

Two indices are contractable if:
1. They have the same `id()` and `dim()`
2. Their conjugate states are compatible:
   - `(Ket, Bra)` or `(Bra, Ket)` → **contractable** (opposite directions)
   - `(Undirected, Undirected)` → **contractable** (both directionless)
   - Mixed `(Undirected, Ket/Bra)` → **not contractable** (mixing forbidden)

Default implementation:

```rust
fn is_contractable(&self, other: &Self) -> bool {
    if self.id() != other.id() || self.dim() != other.dim() {
        return false;
    }
    match (self.conj_state(), other.conj_state()) {
        (ConjState::Ket, ConjState::Bra) | (ConjState::Bra, ConjState::Ket) => true,
        (ConjState::Undirected, ConjState::Undirected) => true,
        _ => false,  // Mixed directed/undirected is forbidden
    }
}
```

### Why mixing is forbidden

If one index is directed (Ket/Bra) but the other is undirected, contraction semantics become ambiguous.
To prevent "tests passing with incorrect direction handling" (especially in directed tensor-network code),
we disallow mixing by default.

### Equality vs Contractability (Important!)

This design intentionally separates two concepts:

| Concept | Method | Purpose |
|---------|--------|---------|
| **Equality** | `Eq` / `==` | "Are these two index *objects* the same?" |
| **Contractability** | `is_contractable()` | "Can these two indices be contracted?" |

**Critical**: Do NOT use `==` to decide contraction; use `is_contractable()`.

**Note on `Eq`/`Hash` semantics**:
- `IndexLike` requires `Eq + Hash` so indices can be used as keys (e.g. `HashSet`, `HashMap`), but it does **not** force a single universal notion of equality.
- The **default `DynIndex`** uses **ID + tags** equality/hashing (ITensors.jl-compatible semantics with plev=0).
- For **directed** index implementations (QSpace-like), it can be useful for `Eq`/`Hash` to also include direction (`ConjState`) to avoid accidental mixing in hash-based collections.
- Regardless of how `Eq`/`Hash` is defined, **contraction must use `is_contractable()`**.

For the default `DynIndex` implementation:
- `==` compares **ID and tags** (not dimension, not ConjState)
- `is_contractable()` checks ID, dimension, AND ConjState compatibility

This distinction matters for directed indices:
```rust
let i_ket = DirectedIndex::new(id, dim, ConjState::Ket);
let i_bra = DirectedIndex::new(id, dim, ConjState::Bra);

// For a properly implemented directed IndexLike:
i_ket == i_bra           // May be true or false depending on Eq impl
i_ket.is_contractable(&i_bra)  // true (opposite directions)
i_ket.is_contractable(&i_ket)  // false (same direction)
```

### A note on naming: "conjugate", "bra/ket", "ingoing/outgoing"

This document uses **Ket/Bra** to describe *index direction* (dual-ness).
This is distinct from **complex conjugation of numeric data**.

- **Index direction (Ket/Bra)**: metadata on each leg
- **Tensor conjugation (`TensorLike::conj`)**: conjugates numeric data *and* maps indices via `IndexLike::conj()`

This mirrors QSpace's definition: conjugation reverses arrows and conjugates data (if complex).

---

## TensorLike Trait

The `TensorLike` trait provides a unified interface for tensor operations:

```rust
pub trait TensorLike: Sized + Clone + Debug + Send + Sync {
    type Index: IndexLike;

    fn external_indices(&self) -> Vec<Self::Index>;
    fn conj(&self) -> Self;
    // ... other methods (contraction and factorization are also required by algorithms)
    //
    // In practice, algorithm crates (e.g. TreeTN, TT/MPS) rely on tensor operations such as:
    // - contraction: `tensordot`, `outer_product`, (and higher-level `contract_*` helpers)
    // - factorization: `factorize` (SVD/QR-based split) returning factors + a new bond index
}
```

### Tensor Conjugate

The `conj()` method on `TensorLike`:
- Conjugates scalar data (complex conjugate for `Complex64` tensors)
- Maps indices via `IndexLike::conj()` (toggles Ket↔Bra for directed indices)

For default undirected indices, `IndexLike::conj()` is a no-op, so this preserves ITensors.jl-like behavior while being ready for QSpace-compatible directed indices.

**Note**: `TensorLike` does **not** have `is_conjugate()` method. The conjugate state is managed at the index level, not the tensor level.

---

## Default Implementation (DynIndex)

The default index type `DynIndex` implements `IndexLike` with ITensors.jl-like semantics.

### Type Definition

```rust
pub type DynIndex = Index<DynId, TagSet>;

pub struct Index<Id, Tags = TagSet> {
    pub id: Id,
    pub dim: usize,
    pub tags: Tags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynId(pub u64);  // 64-bit UUID (compatible with ITensors.jl IDType)
```

### ID Generation

IDs are generated using a thread-local random number generator:

```rust
thread_local! {
    static ID_RNG: RefCell<rand::rngs::ThreadRng> = RefCell::new(rand::thread_rng());
}

pub(crate) fn generate_id() -> u64 {
    ID_RNG.with(|rng| rng.borrow_mut().gen())
}
```

This provides:
- **Low collision probability** with 64-bit random IDs (compatible with ITensors.jl's `IDType = UInt64`)
- **Thread-safe generation** without global locks
- **Identical semantics to ITensors.jl** task-local RNG

### Equality and Hashing

**Important**: `DynIndex` equality and hashing are based on **ID + tags** (compatible with ITensors.jl where equality = id + plev + tags, with plev fixed at 0):

```rust
impl<Id: PartialEq, Tags: PartialEq> PartialEq for Index<Id, Tags> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.tags == other.tags  // ID + tags
    }
}

impl<Id: Hash, Tags: Hash> Hash for Index<Id, Tags> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.tags.hash(state);  // ID + tags
    }
}
```

This means:
- Two indices with the same ID **and** same tags are equal
- Two indices with the same ID but different tags are **not** equal
- This enables efficient `HashMap<Index, ...>` and `HashSet<Index>` lookups
- Dimension mismatches are caught by `is_contractable()`, not by `==`
- Use `same_id()` for ID-only comparison when needed

### IndexLike Implementation

```rust
impl IndexLike for DynIndex {
    type Id = DynId;

    fn id(&self) -> &Self::Id { &self.id }
    fn dim(&self) -> usize { self.dim }

    fn conj_state(&self) -> ConjState {
        ConjState::Undirected  // Always undirected
    }

    fn conj(&self) -> Self {
        self.clone()  // No-op for undirected
    }

    fn sim(&self) -> Self {
        Index {
            id: DynId(generate_id()),  // New ID
            dim: self.dim,              // Same dimension
            tags: self.tags.clone(),    // Same tags
        }
    }
}
```

### Usage Example

```rust
use tensor4all_core::index::{DynIndex, Index};

// Create indices
let i = Index::new_dyn(2);                           // dim=2, no tags
let j = Index::new_dyn_with_tag(3, "bond").unwrap(); // dim=3, tag="bond"

// Clone shares the same ID
let j_clone = j.clone();
assert_eq!(j, j_clone);           // Same ID → equal
assert!(j.is_contractable(&j_clone));  // Same ID, same dim, both Undirected → contractable

// New index has different ID
let k = Index::new_dyn(3);
assert_ne!(j, k);                 // Different ID → not equal
assert!(!j.is_contractable(&k));  // Different ID → not contractable

// sim() creates new ID with same structure
let j_sim = j.sim();
assert_ne!(j, j_sim);             // Different ID
assert_eq!(j.dim(), j_sim.dim()); // Same dimension
```

---

## Design Rationale

### Why Index IDs are Needed

**Rule**: Contractable indices must have the same ID.

This rule is explicitly enforced in the `is_contractable()` method of `IndexLike`.

#### The Problem: Large Networks Need Efficient Pairing

In large and complex tensor networks (e.g., TreeTN, MPS/MPO with many sites), you often need to:
1. Build a **graph structure** to represent connections between tensors
2. Efficiently look up "which leg of tensor A connects to which leg of tensor B"
3. Propagate indices through network transformations (splitting, merging, canonicalization)

A **lightweight ID** serves as the "pairing key" that answers: *which specific legs are intended to contract?*

#### ID vs. Symmetry/Dimension Checks

| Aspect | Purpose |
|--------|---------|
| **ID** | Identifies *intent to pair* — which legs should contract |
| **dim** | Checks *dimensional compatibility* — sizes must match |
| **ConjState** | Checks *directional compatibility* — bra↔ket or undirected↔undirected |
| **Symmetry sectors** (QSpace-specific) | Checks *mathematical compatibility* — quantum numbers must match |

The `is_contractable()` method checks:
1. Same `id()`
2. Same `dim()`
3. Compatible `ConjState`

This separation allows:
- **Graph algorithms** to use IDs for fast edge lookups (O(1) with HashSet/HashMap)
- **Contraction algorithms** to verify mathematical compatibility before actual computation

### Why Undirected by Default?

ITensors.jl uses directionless indices by default, which is very convenient for general tensor operations. By making defaults undirected, tensor4all-rs maintains this convenience while still supporting QSpace-style directed indices when needed.

### Why Forbid Mixed Directed/Undirected?

Mixing directed and undirected indices in contractions would be ambiguous:
- Should an `Undirected` index contract with a `Ket` index?
- What does it mean to have a tensor with mixed directed/undirected indices?

By forbidding mixing, we ensure clear semantics:
- Undirected indices work with other undirected indices (ITensors.jl-like)
- Directed indices work with opposite-directed indices (QSpace-like)
- No ambiguity about which indices can contract

### Why ID + Tags Equality for DynIndex?

Using ID + tags equality (not including dimension) provides:

1. **ITensors.jl compatibility**: In ITensors.jl, Index equality is determined by `id + plev + tags`. Since tensor4all-rs does not use `plev` (conceptually fixed at 0), equality is `id + tags`.
2. **HDF5 format compatibility**: The same equality semantics enable round-trip compatibility with ITensors.jl HDF5 files.
3. **Efficient lookups**: `HashMap<Index, Value>` and `HashSet<Index>` work correctly with this equality.
4. **Separation of concerns**: Equality answers "same logical index?", `is_contractable()` answers "can contract?"
5. **Explicit ID-only comparison**: Use `same_id()` when you need to compare only by ID (e.g., for contraction pairing).

The tradeoff is that dimension mismatches are caught at contraction time, not at equality check time. This is acceptable because:
- Dimension mismatches are programming errors (should not happen in correct code)
- `is_contractable()` catches them before any computation occurs

---

## Index IDs in TreeTN

### Current Design

In the current `TreeTN` implementation:
- Graph edges store `T::Index` (bond index)
- Bond indices have IDs used for pairing
- After SVD/QR, new bond indices are generated and edges are updated

```rust
// Current: edges store bond index
pub struct TreeTN<V, T: TensorLike> {
    graph: NamedGraph<V, T, T::Index>,  // Edge weight = bond index
    // ...
}
```

### Alternative Design (Under Consideration)

An alternative approach being considered:
- Edges store topology only (no Index data)
- Use `is_contractable()` on-the-fly to find matching legs
- **Forbid multiple bonds between nodes** (single contractable pair per edge)

```rust
// Alternative: edges store no data
pub struct TreeTN<V, T: TensorLike> {
    graph: NamedGraph<V, T, ()>,  // Edge weight = unit
    // ...
}

// Find contractable pair on-the-fly
fn find_bond_pair<T: TensorLike>(a: &T, b: &T) -> Option<(T::Index, T::Index)> {
    for idx_a in a.external_indices() {
        for idx_b in b.external_indices() {
            if idx_a.is_contractable(&idx_b) {
                return Some((idx_a, idx_b));
            }
        }
    }
    None
}
```

**Benefits**:
- No synchronization needed after SVD/QR (tensor is the source of truth)
- Simpler edge structure
- No index information duplication

**Tradeoffs**:
- O(n*m) search per contraction (usually small n, m)
- Must forbid multiple bonds between nodes

**Status**: This is a design note only. It is **not implemented** in the current codebase.

---

## Migrating from QSpace

### Key Differences

| QSpace | tensor4all-rs |
|--------|---------------|
| itag name = pairing key | ID value = pairing key |
| Direction in itag suffix (`*`) | Direction in `ConjState` |
| Late binding (assign after creation) | Early binding (assign at creation) |
| Tensor-centric | Index-centric (by default) |

### Migration Options

#### Option 1: Use itag Base Name as ID (recommended)

Create a custom `IndexLike` implementation where `type Id` is a *conjugate-independent pairing key*.
For QSpace, this is naturally the **itag base name** (the name without the trailing `*`).

Using the full index object as `Id` (e.g. `type Id = QSpaceIndex`) is usually **not correct** because:
- `Id` must match between Ket/Bra legs (same pairing intent), but a full index often includes `ConjState`
- `Id` should be lightweight; direction (`ConjState`) and dimension (`dim`) belong on the index itself

One straightforward pattern is to introduce a small `QSpaceTag` type and use it as `Id`:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct QSpaceTag(String); // e.g. "K01" (no trailing '*')

struct QSpaceIndex {
    id: QSpaceTag,       // pairing key (conjugate-independent)
    dim: usize,
    state: ConjState,    // Ket or Bra
}

impl IndexLike for QSpaceIndex {
    type Id = QSpaceTag;

    fn id(&self) -> &QSpaceTag { &self.id }
    fn dim(&self) -> usize { self.dim }
    fn conj_state(&self) -> ConjState { self.state }

    fn conj(&self) -> Self {
        Self {
            id: self.id.clone(),
            dim: self.dim,
            state: match self.state {
                ConjState::Ket => ConjState::Bra,
                ConjState::Bra => ConjState::Ket,
                ConjState::Undirected => ConjState::Undirected,
            },
        }
    }

    fn sim(&self) -> Self {
        Self {
            id: QSpaceTag(generate_unique_name()), // New pairing key
            dim: self.dim,
            state: self.state,
        }
    }
}
```

#### Option 2: Convert itag Names to Numeric IDs

Hash itag names to create numeric IDs during import:

```rust
fn itag_to_id(itag: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    itag.hash(&mut hasher);
    hasher.finish()
}

fn import_qspace_tensor(qspace: QSpaceTensor) -> Tensor<DynIndex> {
    let indices: Vec<DynIndex> = qspace.itags.iter().map(|itag| {
        let name = itag.trim_end_matches('*');
        let is_bra = itag.ends_with('*');

        // For directed indices, you'd use a DirectedIndex type
        // For simplicity, using DynIndex (undirected):
        Index {
            id: DynId(itag_to_id(name)),
            dim: /* get from qspace */,
            tags: TagSet::from_str(name).unwrap(),
        }
    }).collect();

    Tensor::new(indices, qspace.data)
}
```

#### Option 3: Assign Random IDs at Network Construction

When building a tensor network from QSpace tensors:

```rust
fn build_network_from_qspace(tensors: Vec<QSpaceTensor>) -> TreeTN {
    // 1. Group tensors by itag name
    let mut itag_to_id: HashMap<String, DynId> = HashMap::new();

    // 2. For each unique itag name, generate a random ID
    for tensor in &tensors {
        for itag in &tensor.itags {
            let name = itag.trim_end_matches('*');
            itag_to_id.entry(name.to_string())
                .or_insert_with(|| DynId(generate_id()));
        }
    }

    // 3. Create tensor4all indices using the assigned IDs
    // ... (tensors with same itag name get same ID)
}
```

### Preserving Direction Information

If using directed indices is important (for quantum number conservation, etc.):

1. Create a `DirectedIndex` type that stores `ConjState::Ket` or `ConjState::Bra`
2. Parse the trailing `*` from QSpace itags to determine direction
3. Implement `conj()` to toggle between Ket and Bra

```rust
fn parse_qspace_itag(itag: &str) -> (String, ConjState) {
    if itag.ends_with('*') {
        (itag.trim_end_matches('*').to_string(), ConjState::Bra)
    } else {
        (itag.to_string(), ConjState::Ket)
    }
}
```

---

## References

- [QSpace v4](https://github.com/weichselbaum/QSpace) — A. Weichselbaum, Annals of Physics **327**, 2972 (2012)
- [ITensors.jl](https://github.com/ITensor/ITensors.jl) — M. Fishman, S. R. White, E. M. Stoudenmire, arXiv:2007.14822 (2020)
- tensor4all-rs source code: `crates/tensor4all-core/src/index_like.rs`, `crates/tensor4all-core/src/defaults/index.rs`

---

## Rust Primer: What is a Trait?

In Rust, a **trait** is an interface that defines a set of methods a type must provide.
It is similar to an "interface" in other languages.

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

### Minimal interface, implementation freedom

`IndexLike` and `TensorLike` are intentionally designed as **minimal interfaces**:
they define *what* operations an algorithm needs (e.g. `id`, `dim`, `conj`, `tensordot`),
but they do **not** prescribe *how* those operations must be implemented internally.

This makes it possible to plug in different concrete implementations with very different internals, for example:
- **Quantum numbers / symmetry sectors** (Abelian/non-Abelian)
- **Block-sparse storage layouts** and metadata
- **Different contraction backends/algorithms** (dense, block-sparse, optimized order, GPU, etc.)

As long as a type satisfies the trait contract, higher-level algorithms can operate on it without knowing these details.
