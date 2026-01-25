//! Operator composition for Tree Tensor Networks.
//!
//! This module provides traits and functions for working with operators on TreeTN states,
//! including composing multiple non-overlapping operators into a single operator.
//!
//! # Key Types
//!
//! - [`Operator`]: Trait for objects that can act as operators on tensor network states
//! - [`LinearOperator`]: MPO wrapper with index mapping for automatic transformations
//! - [`IndexMapping`]: Mapping between true site indices and internal MPO indices
//! - [`compose_exclusive_linear_operators`]: Compose non-overlapping LinearOperators into a single operator
//!
//! # Example
//!
//! ```ignore
//! // Build local operators acting on different regions
//! let op1 = build_local_operator(&[site_a, site_b]);  // acts on {a, b}
//! let op2 = build_local_operator(&[site_c, site_d]);  // acts on {c, d}
//!
//! // State has sites {a, b, x, c, d}
//! let target = state.site_index_network();
//!
//! // Compose into single operator on full space (identity at gap x)
//! let composed = compose_exclusive_linear_operators(target, &[&op1, &op2], &gap_indices)?;
//!
//! // Apply composed operator
//! let result = state.contract_zipup(&composed, center, rtol, max_rank)?;
//! ```

mod apply;
mod compose;
mod identity;
mod index_mapping;
mod linear_operator;

pub use apply::{apply_linear_operator, ApplyOptions, ArcLinearOperator};
pub use compose::{are_exclusive_operators, compose_exclusive_linear_operators};
pub use identity::{build_identity_operator_tensor, build_identity_operator_tensor_c64};
pub use index_mapping::IndexMapping;
pub use linear_operator::LinearOperator;

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::TensorLike;

use crate::SiteIndexNetwork;

/// Trait for operators that can act on tensor network states.
///
/// An operator has:
/// - A set of site indices it acts on
/// - A site index network describing its structure
///
/// This trait is used for composing multiple operators into a single operator
/// and for validating operator compatibility.
///
/// # Type Parameters
///
/// - `T`: Tensor type implementing `TensorLike`
/// - `V`: Node name type
pub trait Operator<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Get all site indices this operator acts on.
    ///
    /// Returns the union of site indices across all nodes.
    fn site_indices(&self) -> HashSet<T::Index>;

    /// Get the site index network describing this operator's structure.
    ///
    /// The site index network contains:
    /// - Topology: which nodes connect to which
    /// - Site space: which site indices belong to each node
    fn site_index_network(&self) -> &SiteIndexNetwork<V, T::Index>;

    /// Get the set of node names this operator covers.
    ///
    /// Default implementation extracts node names from the site index network.
    fn node_names(&self) -> HashSet<V> {
        self.site_index_network()
            .node_names()
            .into_iter()
            .cloned()
            .collect()
    }
}
