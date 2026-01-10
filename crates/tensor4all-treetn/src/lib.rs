pub mod algorithm;
// dyn_treetn.rs has been removed per TENSOR_LIKE_DESIGN.md
// TreeTN now uses T: TensorLike pattern, making dyn_treetn unnecessary
pub mod named_graph;
pub mod node_name_network;
// TODO: Re-enable operator module after fixing for TensorLike refactoring
// pub mod operator;
pub mod options;
// TODO: Re-enable random module after fixing for TensorLike refactoring
// pub mod random;
pub mod site_index_network;
pub mod treetn;

pub use algorithm::{CanonicalForm, CompressionAlgorithm, ContractionAlgorithm};

// dyn_treetn exports removed - use TreeTN<TensorDynLen, V> directly
pub use named_graph::NamedGraph;
pub use node_name_network::{CanonicalizeEdges, NodeNameNetwork};
// pub use operator::{
//     are_exclusive_operators, build_identity_operator_tensor,
//     compose_exclusive_linear_operators, compose_exclusive_operators, Operator,
// };
pub use options::{CanonicalizationOptions, SplitOptions, TruncationOptions};
// pub use random::{random_treetn_c64, random_treetn_f64, LinkSpace};
pub use site_index_network::SiteIndexNetwork;
// TODO: Re-enable these exports after fixing for TensorLike refactoring
pub use treetn::{
    // apply_local_update_sweep,
    // contract,
    // contract_fit,
    // factorize_tensor_to_treetn,
    // factorize_tensor_to_treetn_with,
    // linsolve,
    // ContractionMethod,
    // ContractionOptions,
    // // Linsolve exports
    // EnvironmentCache,
    // FitContractionOptions,
    // FitEnvironment,
    // FitUpdater,
    // IndexMapping,
    // LinearOperator,
    // LinsolveOptions,
    // LinsolveResult,
    // LinsolveUpdater,
    // LinsolveVerifyReport,
    // LocalUpdateStep,
    // LocalUpdateSweepPlan,
    // LocalUpdater,
    // NetworkTopology,
    // NodeVerifyDetail,
    // ProjectedOperator,
    // ProjectedState,
    TreeTN,
    // TreeTopology,
    // TruncateUpdater,
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
