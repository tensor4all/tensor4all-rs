//! Tree Tensor Network (TreeTN) library for tensor4all.
//!
//! This crate provides data structures and algorithms for working with tree tensor networks,
//! a generalization of matrix product states (MPS) to tree-shaped graphs. Tree tensor networks
//! are useful for representing quantum states and operators on systems with tree-like connectivity.
//!
//! # Key Types
//!
//! - [`TreeTN`]: The main tree tensor network type, parameterized by tensor and node name types.
//! - [`DefaultTreeTN`]: A convenient alias for `TreeTN<TensorDynLen, NodeIndex>`.
//! - [`NamedGraph`]: A graph wrapper that maps node names to internal graph indices.
//!
//! # Features
//!
//! - **Canonicalization**: Transform networks into canonical forms (unitary, LU, CI).
//! - **Truncation**: Compress networks using SVD-based truncation.
//! - **Contraction**: Contract two networks using zip-up or fit algorithms.
//! - **Linear operators**: Apply and compose linear operators on tree tensor networks.
//! - **Linear solvers**: Solve linear systems involving tree tensor network operators.

#![warn(missing_docs)]
pub mod algorithm;
// dyn_treetn.rs has been removed.
// TreeTN uses the `T: TensorLike` pattern, making a separate dyn wrapper unnecessary.
pub mod link_index_network;
pub mod linsolve;
pub mod named_graph;
pub mod node_name_network;
pub mod operator;
pub mod options;
pub mod random;
pub mod site_index_network;
pub mod treetn;

pub use algorithm::{CanonicalForm, CompressionAlgorithm, ContractionAlgorithm};

// dyn_treetn exports removed - use TreeTN<TensorDynLen, V> directly
pub use link_index_network::LinkIndexNetwork;
pub use named_graph::NamedGraph;
pub use node_name_network::{CanonicalizeEdges, NodeNameNetwork};
pub use operator::{
    apply_linear_operator, are_exclusive_operators, build_identity_operator_tensor,
    build_identity_operator_tensor_c64, compose_exclusive_linear_operators, ApplyOptions,
    ArcLinearOperator, IndexMapping, LinearOperator, Operator,
};
pub use options::{CanonicalizationOptions, SplitOptions, TruncationOptions};
pub use random::{random_treetn_c64, random_treetn_f64, LinkSpace};
pub use site_index_network::SiteIndexNetwork;
pub use treetn::{
    // Local update
    apply_local_update_sweep,
    // Decomposition
    factorize_tensor_to_treetn,
    factorize_tensor_to_treetn_with,
    get_boundary_edges,
    BoundaryEdge,
    LocalUpdateStep,
    LocalUpdateSweepPlan,
    LocalUpdater,
    // Swap
    SwapOptions,
    SwapPlan,
    SwapStep,
    // Core type
    TreeTN,
    TreeTopology,
    TruncateUpdater,
};

// Re-export linsolve types from new location
pub use linsolve::{
    square_linsolve, EnvironmentCache, LinsolveOptions, LinsolveVerifyReport, NetworkTopology,
    NodeVerifyDetail, ProjectedOperator, ProjectedState, SquareLinsolveResult,
    SquareLinsolveUpdater,
};

use petgraph::graph::NodeIndex;
use tensor4all_core::TensorDynLen;

/// Default TreeTN type using TensorDynLen as the tensor type.
///
/// This is the most common configuration for TreeTN, equivalent to:
/// ```ignore
/// TreeTN<TensorDynLen, NodeIndex>
/// ```
///
/// Use this when you don't need custom tensor types.
pub type DefaultTreeTN<V = NodeIndex> = TreeTN<TensorDynLen, V>;
