# Plan: `TensorLike` trait to unify `TensorDynLen` and `TreeTN` (indices + tensordot)

This document proposes a third-party-extensible trait that allows treating `tensor4all-core-tensor`'s tensor type (currently `TensorDynLen`) and `tensor4all-treetn`'s `TreeTN` through a common interface.

We start with **external (site/physical) indices** and their count, plus a single required operation: **explicit contraction (`tensordot`)**.

We also include an `as_any()` hook to allow optional runtime type inspection/downcasting
(similar in spirit to C++'s `dynamic_cast`, but explicit).

## Goals

- Provide a **single trait** that can be implemented by:
  - `tensor4all::TensorDynLen<Id, Symm>`
  - `tensor4all_treetn::TreeTN<Id, Symm, V>`
  - third-party tensor-network containers or tensor-like objects
- Expose:
  - **`num_external_indices()`**: number of external indices
  - **`external_indices()`**: *flattened* external indices (no node-order concept)
- Enable:
  - **`tensordot()`**: *explicit* binary contraction (the only required operation)
- Keep the trait **small and stable**; add additional capabilities via separate traits later.

## Non-goals

- Do not standardize contraction/SVD/QR/etc. in this trait (those should be separate capability traits).
- Do not require a node ordering for `TreeTN` (node order is a `TreeTN`-specific concept).
- Do not optimize allocations in the first iteration (a `Vec` return is acceptable initially).
  - We accept that `external_indices()` may allocate for `TreeTN` and/or for trait objects.

## Why "external indices"?

- For `TensorDynLen`, external indices naturally correspond to the tensor's index list.
- For `TreeTN`, external indices correspond to the network's site/physical indices stored in `SiteIndexNetwork`.

## Proposed API (A plan)

### Trait name

- Use `TensorLike` as the umbrella name, but document clearly that it is initially an *index-view* trait.

### Trait definition (sketch)

The initial design returns owned indices (`Vec<_>`) to remain simple and object-safe.

We also make trait objects cloneable via `dyn-clone` (chosen for developer ergonomics).
Additionally, we provide `as_any()` for optional downcasting.

```rust
pub trait TensorLike {
    type Id: Clone + std::hash::Hash + Eq;
    type Symm: Clone + tensor4all::index::Symmetry;
    type Tags: Clone;

    /// Return flattened external indices for this object.
    ///
    /// - `TensorDynLen`: the tensor's indices (or a filtered subset if we later define a rule)
    /// - `TreeTN`: union of all site/physical indices across nodes
    ///
    /// Ordering MUST be stable (recommended: sort by `id`).
    fn external_indices(&self) -> Vec<tensor4all::index::Index<Self::Id, Self::Symm, Self::Tags>>;

    /// Number of external indices.
    ///
    /// Implementations SHOULD override this for efficiency when possible.
    fn num_external_indices(&self) -> usize {
        self.external_indices().len()
    }

    /// Return `self` as `Any` for optional downcasting / runtime type inspection.
    ///
    /// Implementers typically write:
    /// `fn as_any(&self) -> &dyn Any { self }`
    ///
    /// This requires the concrete type to be `'static` (the usual `Any` constraint).
    fn as_any(&self) -> &dyn std::any::Any;

    /// Explicit contraction between two objects (the only required operation).
    ///
    /// You explicitly specify which indices are connected/contracted across the two operands.
    ///
    /// For a `TensorDynLen`, this corresponds to `contract_pairs`.
    fn tensordot(
        &self,
        other: &dyn TensorLike<Id = Self::Id, Symm = Self::Symm, Tags = Self::Tags>,
        pairs: &[(tensor4all::index::Index<Self::Id, Self::Symm, Self::Tags>,
                 tensor4all::index::Index<Self::Id, Self::Symm, Self::Tags>)],
    ) -> anyhow::Result<tensor4all::TensorDynLen<Self::Id, Self::Symm>>;
}
```

To make `Box<dyn TensorLike<...>>` cloneable, we will use `dyn-clone`:

- `TensorLike: dyn_clone::DynClone`
- `dyn_clone::clone_trait_object!(TensorLike);`

### Optional: helper methods on the trait object

Rust allows adding inherent methods to the trait object type:

```rust
impl dyn TensorLike<Id = DynId, Symm = NoSymmSpace, Tags = DefaultTagSet> {
    pub fn is<T: 'static>(&self) -> bool {
        self.as_any().is::<T>()
    }

    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
}
```

Notes:

- This requires choosing a concrete `Id/Symm/Tags` specialization for the `impl dyn ...` block.
- Downcasting should be used sparingly; prefer expressing required behavior as trait methods.

## Semantics and ordering

### Flattening

`external_indices()` is flattened by design.

- For `TreeTN`, there is no stable "node order" that should leak into the trait.
- The trait returns a single list representing the network's external interface.

### Stable ordering requirement

`SiteIndexNetwork` currently stores per-node site indices as `HashSet<Index<...>>`, which is inherently orderless.

To make `external_indices()` deterministic and usable for hashing/serialization/comparison, define:

- **Ordering rule (TreeTN canonicalization)**:
  - iterate nodes in a deterministic order (recommended: sort by node name `V`)
  - within each node, sort indices deterministically (recommended: sort by index `id`)
  - flatten the per-node lists into a single `Vec`
- This may require `V: Ord` and `Id: Ord` at the implementation site.

If we do not want `Id: Ord` in the trait bound, implementations can:

- sort by a stable hash of `id` (requires a fixed hasher), or
- keep insertion-ordered storage (bigger refactor), or
- return in arbitrary order (NOT recommended).

## Implementation notes (later work)

### `TensorDynLen`

- Implementation is straightforward:
  - `external_indices()` returns `self.indices.clone()`
  - `num_external_indices()` returns `self.indices.len()`

### `TreeTN`

- Use `site_index_network` as the source of truth for physical indices.
- For deterministic output, canonicalize the order:
  - sort nodes by name `V`
  - within each node, sort indices by `id`
  - flatten

## Third-party extensibility

This design supports external crates implementing `TensorLike` for their own types because:

- The trait is small and does not depend on internal `TreeTN` graph details.
- The associated types (`Id`, `Symm`, `Tags`) match the existing `Index<Id, Symm, Tags>` design.
- The initial API avoids lifetime/GAT complexity.
- Implementers can override `tensordot` / `tensordot_pairs` to avoid materializing via `contract_to_tensor` if they have a more efficient representation.

## Possible follow-ups (separate traits)

Keep `TensorLike` minimal. Add other traits separately when needed, e.g.:

- `ContractToTensor`: reduce an object (e.g. `TreeTN`) to a single concrete tensor
- `HasAllIndices`: returns all indices (including internal/bond)
- `HasTopology`: exposes graph/topology properties for networks
- `CachedExternalIndices`: allows borrowing external indices via `&[Index]` for performance

## Performance optimization options

Returning `Vec` is simple but allocates for `TreeTN` (and clones indices).

If this becomes an issue, consider:

1. **Cache** inside `TreeTN`:
   - store `external_indices_cache: Vec<Index<...>>`
   - invalidate cache on any mutation affecting site indices
   - then `external_indices_ref(&self) -> &[Index]` becomes possible

2. **Add a separate borrow-based trait** (advanced):
   - use GATs to return an iterator over borrowed indices
   - this reduces allocation but complicates object-safety and third-party implementations

The recommendation is to start with the `Vec` API and add caching only when profiling shows it matters.

## Storing arbitrary `TensorLike` objects inside `TreeTN` (trait objects)

If we want `TreeTN` to hold "any object implementing `TensorLike`" (heterogeneous nodes), the natural approach is to store trait objects as node payloads:

- `Box<dyn TensorLike<Id = ..., Symm = ..., Tags = ...> + Send + Sync>`
- or `Arc<dyn TensorLike<...> + Send + Sync>` if we want shared ownership / long-lived handles outside the `TreeTN`

Key constraint: within a given `TreeTN`, the associated types must be fixed (same `Id`, `Symm`, `Tags`), because:

- `Connection<Id, Symm, Tags>` stores concrete `Index<Id, Symm, Tags>` values
- contraction logic needs compatible index representations across nodes

For a first implementation, we can choose a canonical set for trait objects:

- `Id = DynId`, `Symm = NoSymmSpace`, `Tags = DefaultTagSet`

and later generalize if needed.


