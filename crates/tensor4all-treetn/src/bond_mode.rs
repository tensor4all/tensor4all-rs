//! Bond mode marker types for TreeTN and DynTreeTN.
//!
//! This module provides marker types that control how nodes are connected in a tree tensor network:
//!
//! - [`Einsum`]: Nodes are automatically connected by matching index IDs (einsum convention)
//! - [`Explicit`]: Nodes must be connected manually via `connect()` method
//!
//! # Example
//!
//! ```ignore
//! use tensor4all_treetn::{TreeTN, bond_mode::{Einsum, Explicit}};
//!
//! // Einsum mode (default): auto-connects by shared index IDs
//! let tn = TreeTN::<DynId>::new(tensors, node_names)?;
//!
//! // Explicit mode: manual connection required
//! let mut tn = TreeTN::<DynId, _, _, Explicit>::new();
//! let n1 = tn.add_tensor(tensor1);
//! let n2 = tn.add_tensor(tensor2);
//! tn.connect(n1, &idx, n2, &idx)?;
//! ```

use std::fmt::Debug;

/// Marker trait for bond modes.
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait BondMode: Clone + Copy + Send + Sync + Debug + Default + private::Sealed + 'static {
    /// Whether this mode uses einsum convention (auto-connect by shared index IDs)
    const IS_EINSUM: bool;
}

mod private {
    pub trait Sealed {}
}

/// Einsum bond mode: nodes are connected by matching index IDs.
///
/// When using this mode:
/// - `TreeTN::new(tensors, node_names)` creates a network with auto-connected nodes
/// - Nodes sharing the same index ID are automatically connected
/// - Each index ID can appear in at most 2 tensors (tree constraint)
/// - The `connect()` method is NOT available
///
/// This is the default mode for `TreeTN` and `DynTreeTN`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Einsum;

impl private::Sealed for Einsum {}
impl BondMode for Einsum {
    const IS_EINSUM: bool = true;
}

/// Explicit bond mode: nodes must be connected manually.
///
/// When using this mode:
/// - `TreeTN::new()` creates an empty network
/// - `add_tensor()` adds tensors without auto-connecting
/// - `connect(node_a, index_a, node_b, index_b)` establishes bonds manually
/// - Index IDs can be reused freely across tensors
///
/// Use this mode when you need fine-grained control over the network topology.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Explicit;

impl private::Sealed for Explicit {}
impl BondMode for Explicit {
    const IS_EINSUM: bool = false;
}
