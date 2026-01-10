pub mod dyn_treetn;
pub mod named_graph;
pub mod node_name_network;
pub mod operator;
pub mod options;
pub mod random;
pub mod site_index_network;
pub mod treetn;

pub use dyn_treetn::{BoxedTensorLike, DynIndex, DynTreeTN};
pub use named_graph::NamedGraph;
pub use node_name_network::{CanonicalizeEdges, NodeNameNetwork};
pub use operator::{
    are_exclusive_operators, build_identity_operator_tensor,
    compose_exclusive_linear_operators, compose_exclusive_operators, Operator,
};
pub use options::{CanonicalizationOptions, SplitOptions, TruncationOptions};
pub use random::{random_treetn_c64, random_treetn_f64, LinkSpace};
pub use site_index_network::SiteIndexNetwork;
pub use treetn::{
    apply_local_update_sweep,
    contract,
    contract_fit,
    factorize_tensor_to_treetn,
    factorize_tensor_to_treetn_with,
    linsolve,
    ContractionMethod,
    ContractionOptions,
    // Linsolve exports
    EnvironmentCache,
    FitContractionOptions,
    FitEnvironment,
    FitUpdater,
    IndexMapping,
    LinearOperator,
    LinsolveOptions,
    LinsolveResult,
    LinsolveUpdater,
    LinsolveVerifyReport,
    LocalUpdateStep,
    LocalUpdateSweepPlan,
    LocalUpdater,
    NetworkTopology,
    NodeVerifyDetail,
    ProjectedOperator,
    ProjectedState,
    TreeTN,
    TreeTopology,
    TruncateUpdater,
};

use petgraph::graph::NodeIndex;
use tensor4all_core::index::{DynId, NoSymmSpace};

/// Default TreeTN type using DynId for index identity and NoSymmSpace for symmetry.
///
/// This is the most common configuration for TreeTN, equivalent to:
/// ```ignore
/// TreeTN<DynId, NoSymmSpace, NodeIndex>
/// ```
///
/// Use this when you don't need custom index types or symmetry.
pub type DefaultTreeTN<V = NodeIndex> = TreeTN<DynId, NoSymmSpace, V>;
