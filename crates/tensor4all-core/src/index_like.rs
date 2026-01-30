//! IndexLike trait for abstracting index types.
//!
//! This trait allows algorithms to be generic over different index types
//! without needing to know about the internal ID representation.

use std::fmt::Debug;
use std::hash::Hash;

/// Conjugate state (direction) of an index.
///
/// This enum represents whether an index has a direction (bra/ket) or is directionless.
/// The direction is used to determine contractability between indices.
///
/// # QSpace Compatibility
///
/// In QSpace (extern/qspace-v4-pub), index direction is encoded via trailing `*` in `itags`:
/// - **Ket** = ingoing index (QSpace: itag **without** trailing `*`)
/// - **Bra** = outgoing index (QSpace: itag **with** trailing `*`)
///
/// # ITensors.jl Compatibility
///
/// ITensors.jl uses directionless indices by default (convenient for general tensor operations).
/// The `Undirected` variant provides this behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConjState {
    /// Directionless index (ITensors.jl-like default).
    ///
    /// Undirected indices can contract with other undirected indices
    /// if they have the same ID and dimension.
    Undirected,
    /// Ket (ingoing) index.
    ///
    /// In QSpace terminology, this corresponds to an index without a trailing `*` in its itag.
    /// Ket indices can only contract with Bra indices (and vice versa).
    Ket,
    /// Bra (outgoing) index.
    ///
    /// In QSpace terminology, this corresponds to an index with a trailing `*` in its itag.
    /// Bra indices can only contract with Ket indices (and vice versa).
    Bra,
}

/// Trait for index-like types that can be used in tensor operations.
///
/// This trait abstracts away the identity mechanism of indices, allowing algorithms
/// to work with any index type that provides equality, hashing, and dimension access.
///
/// # Design Principles
///
/// - **`Id` as associated type**: Lightweight identifier (conjugate-independent)
/// - **`Eq` by object equality**: Two indices are equal iff they represent the same object
///   (including ID, dimension, and conjugate state if applicable)
/// - **`dim()`**: Returns the dimension of the index
/// - **`conj_state()`**: Returns the conjugate state (direction) of the index
///
/// # Key Properties
///
/// - **`Eq`**: Defines object equality (includes ID, dimension, and conjugate state)
/// - **`Hash`**: Enables efficient lookup in `HashMap<I, ...>` / `HashSet<I>`
/// - **`Clone`**: Indices are small value types, freely copyable
/// - **`is_contractable()`**: Determines if two indices can be contracted
///
/// # Conjugate State and Contractability
///
/// The `conj_state()` method returns the direction of an index:
/// - `Undirected`: Directionless index (ITensors.jl-like default)
/// - `Ket`: Ingoing index (QSpace: no trailing `*` in itag)
/// - `Bra`: Outgoing index (QSpace: trailing `*` in itag)
///
/// Two indices are contractable if:
/// - They have the same `id()` and `dim()`
/// - Their conjugate states are compatible:
///   - `(Ket, Bra)` or `(Bra, Ket)` → contractable
///   - `(Undirected, Undirected)` → contractable
///   - Mixed `(Undirected, Ket/Bra)` → **not contractable** (mixing forbidden)
///
/// # Example
///
/// ```ignore
/// fn contract_common<I: IndexLike>(a: &Tensor<I>, b: &Tensor<I>) -> Tensor<I> {
///     // Algorithm doesn't need to know about Id, Symm, Tags
///     // It only needs indices to be comparable and have dimensions
///     let common: Vec<_> = a.indices().iter()
///         .filter(|idx| b.indices().contains(idx))
///         .cloned()
///         .collect();
///     // ...
/// }
/// ```
pub trait IndexLike: Clone + Eq + Hash + Debug + Send + Sync + 'static {
    /// Lightweight identifier type (conjugate-independent).
    ///
    /// **Rule**: Contractable indices must have the same ID.
    ///
    /// The ID serves as a "pairing key" to identify which legs are intended to contract.
    /// In large tensor networks, IDs enable efficient graph-based lookups (O(1) with HashSet/HashMap)
    /// to find matching legs across many tensors.
    ///
    /// This is separate from dimension/direction checks:
    /// - **ID**: "intent to pair" (which specific legs should connect)
    /// - **dim/ConjState**: "mathematical compatibility" (can they actually contract)
    type Id: Clone + Eq + Hash + Debug + Send + Sync;

    /// Get the identifier of this index.
    ///
    /// The ID is used as the pairing key during contraction.
    /// **Contractable indices must have the same ID** — this is enforced by `is_contractable()`.
    ///
    /// Two indices with the same ID represent the same logical leg (though they may differ
    /// in conjugate state for directed indices).
    fn id(&self) -> &Self::Id;

    /// Get the total dimension (state-space dimension) of the index.
    fn dim(&self) -> usize;

    /// Get the conjugate state (direction) of this index.
    ///
    /// Returns `ConjState::Undirected` for directionless indices (ITensors.jl-like default),
    /// or `ConjState::Ket`/`ConjState::Bra` for directed indices (QSpace-compatible).
    fn conj_state(&self) -> ConjState;

    /// Create the conjugate of this index.
    ///
    /// For directed indices, this toggles between `Ket` and `Bra`.
    /// For `Undirected` indices, this returns `self` unchanged (no-op).
    ///
    /// # Returns
    /// A new index with the conjugate state toggled (if directed) or unchanged (if undirected).
    fn conj(&self) -> Self;

    /// Check if this index can be contracted with another index.
    ///
    /// Two indices are contractable if:
    /// - They have the same `id()` and `dim()`
    /// - Their conjugate states are compatible:
    ///   - `(Ket, Bra)` or `(Bra, Ket)` → contractable
    ///   - `(Undirected, Undirected)` → contractable
    ///   - Mixed `(Undirected, Ket/Bra)` → **not contractable** (mixing forbidden)
    ///
    /// # Default Implementation
    ///
    /// The default implementation checks:
    /// 1. Same ID: `self.id() == other.id()`
    /// 2. Same dimension: `self.dim() == other.dim()`
    /// 3. Compatible conjugate states (see rules above)
    fn is_contractable(&self, other: &Self) -> bool {
        if self.id() != other.id() || self.dim() != other.dim() {
            return false;
        }
        match (self.conj_state(), other.conj_state()) {
            (ConjState::Ket, ConjState::Bra) | (ConjState::Bra, ConjState::Ket) => true,
            (ConjState::Undirected, ConjState::Undirected) => true,
            _ => false, // Mixed directed/undirected is forbidden
        }
    }

    /// Check if this index has the same ID as another.
    ///
    /// Default implementation compares IDs directly.
    /// This is a convenience method for pure ID comparison (does not check contractability).
    fn same_id(&self, other: &Self) -> bool {
        self.id() == other.id()
    }

    /// Check if this index has the given ID.
    ///
    /// Default implementation compares with the given ID.
    fn has_id(&self, id: &Self::Id) -> bool {
        self.id() == id
    }

    /// Create a similar index with a new identity but the same structure (dimension, tags, etc.).
    ///
    /// This is used to create "equivalent" indices that have the same properties
    /// but different identities, commonly needed in index replacement operations.
    ///
    /// # Returns
    /// A new index with a fresh identity and the same structure as `self`.
    fn sim(&self) -> Self
    where
        Self: Sized;

    /// Create a similar index with a new identity using a provided RNG.
    ///
    /// This is the deterministic version of [`sim`](Self::sim) that uses the provided
    /// random number generator instead of the thread-local RNG. Use this when you need
    /// reproducible results with the same seed.
    ///
    /// # Arguments
    /// * `rng` - A random number generator to use for ID generation
    ///
    /// # Returns
    /// A new index with a fresh identity and the same structure as `self`.
    fn sim_with_rng<R: rand::Rng>(&self, rng: &mut R) -> Self
    where
        Self: Sized;

    /// Create a pair of contractable dummy indices with dimension 1.
    ///
    /// These are used for structural connections that don't carry quantum numbers,
    /// such as connecting components in a tree tensor network.
    ///
    /// Both indices will be `Undirected` and have the same ID, making them contractable.
    ///
    /// # Returns
    /// A pair `(idx1, idx2)` where `idx1.is_contractable(&idx2)` is true.
    fn create_dummy_link_pair() -> (Self, Self)
    where
        Self: Sized;
}
