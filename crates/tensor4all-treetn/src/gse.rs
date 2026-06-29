//! Global subspace expansion for TreeTN TDVP.
//!
//! This module implements the v1 TreeTN form of the MPS global subspace
//! expansion documented in `docs/design/gse-chain-mps-algorithm.md`.
//! The implementation is intentionally narrow: it supports `TensorDynLen`
//! states with exactly one state site index per node and mapped TreeTN
//! operators with one input and one output index per node.
//!
//! During expansion, old bonds are reconstructed from full-rank local SVDs. This
//! preserves the represented state, but can remove exactly redundant bond
//! directions while adding missing reference directions. Local projected-density
//! contractions and basis reconstruction stay in `TensorDynLen` form; only the
//! eigenvalue cutoff itself reads detached primal eigenvalues for rank
//! selection.

use std::fmt::Debug;
use std::hash::Hash;

use anyhow::Result;
use tensor4all_core::{
    contract_pair_with_operand_options, AnyScalar, Canonical, DynIndex, FactorizeAlg, IndexLike,
    LinearizationOrder, PairwiseContractionOptions, SvdTruncationPolicy, TensorContractionLike,
    TensorDynLen, TensorFactorizationLike, TensorIndex,
};
use thiserror::Error;

use crate::local_update_support::{single_site_square_mappings, SquareSiteMappingError};
use crate::operator::{apply_linear_operator, ApplyOptions, LinearOperator};
use crate::{CanonicalizationOptions, TdvpError, TdvpOptions, TdvpResult, TreeTN};

/// Options controlling TreeTN global subspace expansion.
#[derive(Debug, Clone)]
pub struct GseOptions {
    /// Number of Krylov reference states to build as `H psi`, `H^2 psi`, ...
    ///
    /// This field is used by [`global_subspace_expand`] and [`gse_tdvp`]. It is
    /// ignored by [`global_subspace_expand_with_references`], where references
    /// are supplied directly.
    pub krylov_dim: usize,
    /// Options used when applying the operator to build each reference state.
    pub reference_apply: ApplyOptions,
    /// Optional maximum bond dimension for generated reference states.
    ///
    /// When this is `None`, generated references use `maxlinkdim(state) + 1`,
    /// matching the low-rank probe policy used by chain GSE implementations.
    pub reference_max_rank: Option<usize>,
    /// Optional SVD policy for generated reference states.
    pub reference_svd_policy: Option<SvdTruncationPolicy>,
    /// Density eigenvalue cutoff for retaining missing reference directions.
    pub density_weight_cutoff: f64,
    /// Hermitian tolerance for local projected-density eigendecompositions.
    pub hermitian_tol: f64,
    /// Normalize each generated Krylov reference after operator application.
    pub normalize_references: bool,
    /// Whether [`gse_tdvp`] expands before the first TDVP sweep.
    pub expand_before_first_sweep: bool,
}

impl Default for GseOptions {
    fn default() -> Self {
        Self {
            krylov_dim: 0,
            reference_apply: ApplyOptions::zipup(),
            reference_max_rank: None,
            reference_svd_policy: None,
            density_weight_cutoff: 1.0e-12,
            hermitian_tol: 1.0e-12,
            normalize_references: true,
            expand_before_first_sweep: true,
        }
    }
}

impl GseOptions {
    /// Set the number of generated Krylov reference states.
    pub fn with_krylov_dim(mut self, krylov_dim: usize) -> Self {
        self.krylov_dim = krylov_dim;
        self
    }

    /// Set the density eigenvalue cutoff.
    pub fn with_density_weight_cutoff(mut self, density_weight_cutoff: f64) -> Self {
        self.density_weight_cutoff = density_weight_cutoff;
        self
    }

    /// Set the Hermitian tolerance for local density eigendecompositions.
    pub fn with_hermitian_tol(mut self, hermitian_tol: f64) -> Self {
        self.hermitian_tol = hermitian_tol;
        self
    }

    /// Set the reference-state maximum rank used during operator application.
    pub fn with_reference_max_rank(mut self, reference_max_rank: usize) -> Self {
        self.reference_max_rank = Some(reference_max_rank);
        self
    }

    /// Set the reference-state SVD policy used during operator application.
    pub fn with_reference_svd_policy(mut self, reference_svd_policy: SvdTruncationPolicy) -> Self {
        self.reference_svd_policy = Some(reference_svd_policy);
        self
    }

    /// Set whether generated Krylov references are normalized.
    pub fn with_normalize_references(mut self, normalize_references: bool) -> Self {
        self.normalize_references = normalize_references;
        self
    }

    /// Set whether GSE-TDVP expands before the first TDVP sweep.
    pub fn with_expand_before_first_sweep(mut self, expand_before_first_sweep: bool) -> Self {
        self.expand_before_first_sweep = expand_before_first_sweep;
        self
    }
}

/// Result returned by a TreeTN global subspace expansion.
#[derive(Debug, Clone)]
pub struct GseResult<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Expanded TreeTN state.
    pub state: TreeTN<TensorDynLen, V>,
    /// Number of reference states supplied or generated.
    pub references_built: usize,
    /// Number of directed edges visited by the expansion sweep.
    pub edges_processed: usize,
    /// Number of edges where at least one new basis vector was appended.
    pub bonds_expanded: usize,
    /// Largest number of basis vectors appended to one edge.
    pub max_added_basis: usize,
}

/// Options for [`gse_tdvp`].
#[derive(Debug, Clone, Default)]
pub struct GseTdvpOptions {
    /// Global subspace expansion options.
    pub gse: GseOptions,
    /// Existing TreeTN TDVP options.
    pub tdvp: TdvpOptions,
}

/// Result returned by [`gse_tdvp`].
#[derive(Debug, Clone)]
pub struct GseTdvpResult<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Final evolved state.
    pub state: TreeTN<TensorDynLen, V>,
    /// Number of one-sweep TDVP calls completed.
    pub sweeps_completed: usize,
    /// Total number of projected local TDVP updates.
    pub local_updates: usize,
    /// Number of GSE expansions run before TDVP sweeps.
    pub gse_expansions: usize,
    /// Largest Krylov residual estimate reported by TDVP.
    pub max_error_estimate: f64,
    /// Largest number of Krylov iterations used by TDVP.
    pub max_krylov_iterations: usize,
}

/// Errors returned by TreeTN GSE and GSE-TDVP.
#[derive(Debug, Error)]
pub enum GseError {
    /// A numeric or algorithm option is invalid.
    #[error("invalid GSE option {option}: {reason}")]
    InvalidOption {
        /// Option name.
        option: &'static str,
        /// Why the value is invalid.
        reason: String,
    },
    /// The requested root does not exist.
    #[error("GSE center {center} is not a state node")]
    MissingCenter {
        /// Debug representation of the missing center.
        center: String,
    },
    /// State, reference, or operator topology is incompatible.
    #[error(
        "GSE requires target, references, and operator support to have the same tree topology"
    )]
    TopologyMismatch,
    /// The v1 implementation supports one state site index per node.
    #[error("GSE v1 requires exactly one state site index at node {node}, found {count}")]
    UnsupportedStateSiteCount {
        /// Debug representation of the node.
        node: String,
        /// Number of state site indices found.
        count: usize,
    },
    /// The v1 implementation supports one input/output mapping per node.
    #[error("GSE v1 requires exactly one {role} mapping at node {node}, found {count}")]
    UnsupportedMultipleSiteMappings {
        /// Debug representation of the node.
        node: String,
        /// Mapping role.
        role: &'static str,
        /// Number of mappings found.
        count: usize,
    },
    /// A required operator mapping is missing.
    #[error("GSE missing {role} mapping at node {node}")]
    MissingMapping {
        /// Debug representation of the node.
        node: String,
        /// Mapping role.
        role: &'static str,
    },
    /// An operator mapping is invalid.
    #[error("invalid GSE mapping at node {node}: {reason}")]
    InvalidMapping {
        /// Debug representation of the node.
        node: String,
        /// Why the mapping is invalid.
        reason: String,
    },
    /// The v1 implementation supports uniform f64 or Complex64 storage.
    #[error("GSE v1 unsupported scalar storage at node {node}: {reason}")]
    UnsupportedScalarStorage {
        /// Debug representation of the node.
        node: String,
        /// Why scalar storage is unsupported.
        reason: String,
    },
    /// Target and reference states must use the same scalar storage.
    #[error(
        "GSE v1 requires target and reference tensors to have the same scalar storage; target is {target}, reference is {reference}"
    )]
    ScalarStorageMismatch {
        /// Target scalar storage label.
        target: &'static str,
        /// Reference scalar storage label.
        reference: &'static str,
    },
    /// Existing TDVP failed after GSE.
    #[error("GSE-TDVP TDVP step failed: {source}")]
    Tdvp {
        /// TDVP source error.
        #[source]
        source: TdvpError,
    },
    /// A lower-level tensor-network or linalg operation failed.
    #[error("{context}: {source}")]
    Algorithm {
        /// Context for the failing operation.
        context: &'static str,
        /// Source error.
        #[source]
        source: anyhow::Error,
    },
}

impl From<SquareSiteMappingError> for GseError {
    fn from(error: SquareSiteMappingError) -> Self {
        match error {
            SquareSiteMappingError::UnsupportedStateSiteCount { node, count } => {
                Self::UnsupportedStateSiteCount { node, count }
            }
            SquareSiteMappingError::UnsupportedMultipleSiteMappings { node, role, count } => {
                Self::UnsupportedMultipleSiteMappings { node, role, count }
            }
            SquareSiteMappingError::MissingMapping { node, role } => {
                Self::MissingMapping { node, role }
            }
            SquareSiteMappingError::InvalidMapping { node, reason } => {
                Self::InvalidMapping { node, reason }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct EdgeExpansionStats {
    edges_processed: usize,
    bonds_expanded: usize,
    max_added_basis: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarKind {
    Real,
    Complex,
}

impl ScalarKind {
    fn label(self) -> &'static str {
        match self {
            Self::Real => "f64",
            Self::Complex => "Complex64",
        }
    }
}

/// Build Krylov references from `operator`, expand the TreeTN state, and return
/// the expanded state.
pub fn global_subspace_expand<V>(
    operator: &LinearOperator<TensorDynLen, V>,
    init: TreeTN<TensorDynLen, V>,
    center: &V,
    options: GseOptions,
) -> Result<GseResult<V>, GseError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug + 'static,
{
    validate_options(&options)?;
    if init.node_index(center).is_none() {
        return Err(GseError::MissingCenter {
            center: format!("{center:?}"),
        });
    }
    if !init.same_topology(operator.mpo()) {
        return Err(GseError::TopologyMismatch);
    }
    single_site_square_mappings(operator, &init).map_err(GseError::from)?;

    let references = build_references(operator, &init, center, &options)?;
    global_subspace_expand_with_references(init, references, center, options)
}

/// Expand a TreeTN state using caller-supplied reference work buffers.
///
/// The supplied references are cloned into internal work buffers. Their link
/// indices are relabeled before the sweep, so callers should not rely on the
/// references being returned or mutated.
pub fn global_subspace_expand_with_references<V>(
    init: TreeTN<TensorDynLen, V>,
    references: Vec<TreeTN<TensorDynLen, V>>,
    center: &V,
    options: GseOptions,
) -> Result<GseResult<V>, GseError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug + 'static,
{
    validate_options(&options)?;
    if init.node_index(center).is_none() {
        return Err(GseError::MissingCenter {
            center: format!("{center:?}"),
        });
    }
    validate_single_site_state(&init)?;
    let target_scalar = tree_scalar_kind(&init)?;

    let references_built = references.len();
    let mut state = init
        .canonicalize([center.clone()], CanonicalizationOptions::forced())
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to canonicalize target state",
            source,
        })?;

    let mut reference_buffers = Vec::with_capacity(references.len());
    for reference in references {
        validate_reference(&state, &reference)?;
        let reference_scalar = tree_scalar_kind(&reference)?;
        if reference_scalar != target_scalar {
            return Err(GseError::ScalarStorageMismatch {
                target: target_scalar.label(),
                reference: reference_scalar.label(),
            });
        }
        let mut reference = reference
            .canonicalize([center.clone()], CanonicalizationOptions::forced())
            .map_err(|source| GseError::Algorithm {
                context: "GSE failed to canonicalize reference state",
                source,
            })?;
        reference
            .sim_linkinds_mut()
            .map_err(|source| GseError::Algorithm {
                context: "GSE failed to relabel reference link indices",
                source,
            })?;
        reference_buffers.push(reference);
    }

    let stats = if reference_buffers.is_empty() {
        EdgeExpansionStats {
            edges_processed: 0,
            bonds_expanded: 0,
            max_added_basis: 0,
        }
    } else {
        expand_edges(&mut state, &mut reference_buffers, center, &options)?
    };

    Ok(GseResult {
        state,
        references_built,
        edges_processed: stats.edges_processed,
        bonds_expanded: stats.bonds_expanded,
        max_added_basis: stats.max_added_basis,
    })
}

/// Run GSE before selected TDVP sweeps and delegate each sweep to existing TDVP.
pub fn gse_tdvp<V>(
    operator: &LinearOperator<TensorDynLen, V>,
    init: TreeTN<TensorDynLen, V>,
    center: &V,
    options: GseTdvpOptions,
) -> Result<GseTdvpResult<V>, GseError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug + 'static,
{
    validate_options(&options.gse)?;
    let mut state = init;
    let sweeps = options.tdvp.nsweeps;
    let mut sweeps_completed = 0usize;
    let mut local_updates = 0usize;
    let mut gse_expansions = 0usize;
    let mut max_error_estimate = 0.0_f64;
    let mut max_krylov_iterations = 0usize;

    for sweep in 0..sweeps {
        if options.gse.krylov_dim > 0 && (sweep > 0 || options.gse.expand_before_first_sweep) {
            let expansion = global_subspace_expand(operator, state, center, options.gse.clone())?;
            state = expansion.state;
            gse_expansions += 1;
        }

        let mut tdvp_options = options.tdvp.clone();
        tdvp_options.nsweeps = 1;
        let TdvpResult {
            state: evolved,
            sweeps_completed: completed,
            local_updates: updates,
            max_error_estimate: error,
            max_krylov_iterations: iterations,
        } = crate::tdvp(operator, state, center, tdvp_options)
            .map_err(|source| GseError::Tdvp { source })?;
        state = evolved;
        sweeps_completed += completed;
        local_updates += updates;
        max_error_estimate = max_error_estimate.max(error);
        max_krylov_iterations = max_krylov_iterations.max(iterations);
    }

    Ok(GseTdvpResult {
        state,
        sweeps_completed,
        local_updates,
        gse_expansions,
        max_error_estimate,
        max_krylov_iterations,
    })
}

fn validate_options(options: &GseOptions) -> Result<(), GseError> {
    if !options.density_weight_cutoff.is_finite() || options.density_weight_cutoff < 0.0 {
        return Err(GseError::InvalidOption {
            option: "density_weight_cutoff",
            reason: "must be finite and non-negative".to_string(),
        });
    }
    if !options.hermitian_tol.is_finite() || options.hermitian_tol < 0.0 {
        return Err(GseError::InvalidOption {
            option: "hermitian_tol",
            reason: "must be finite and non-negative".to_string(),
        });
    }
    if let Some(max_rank) = options.reference_max_rank {
        if max_rank == 0 {
            return Err(GseError::InvalidOption {
                option: "reference_max_rank",
                reason: "must be greater than zero when set".to_string(),
            });
        }
    }
    Ok(())
}

fn build_references<V>(
    operator: &LinearOperator<TensorDynLen, V>,
    init: &TreeTN<TensorDynLen, V>,
    center: &V,
    options: &GseOptions,
) -> Result<Vec<TreeTN<TensorDynLen, V>>, GseError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug + 'static,
{
    let mut references = Vec::with_capacity(options.krylov_dim);
    let mut current = init.clone();
    let generated_reference_max_rank = options
        .reference_max_rank
        .unwrap_or_else(|| max_link_dim(init).saturating_add(1));
    for _ in 0..options.krylov_dim {
        let mut apply_options = options
            .reference_apply
            .clone()
            .with_max_rank(generated_reference_max_rank);
        if let Some(policy) = options.reference_svd_policy {
            apply_options = apply_options.with_svd_policy(policy);
        }
        let mut next =
            apply_linear_operator(operator, &current, apply_options).map_err(|source| {
                GseError::Algorithm {
                    context: "GSE failed to apply operator while building references",
                    source,
                }
            })?;
        if !next.same_topology(init) {
            return Err(GseError::TopologyMismatch);
        }
        validate_reference(init, &next)?;
        if options.normalize_references {
            let norm = next.norm().map_err(|source| GseError::Algorithm {
                context: "GSE failed to compute reference norm",
                source,
            })?;
            if norm > 0.0 {
                next.scale(AnyScalar::new_real(norm.recip()))
                    .map_err(|source| GseError::Algorithm {
                        context: "GSE failed to normalize reference",
                        source,
                    })?;
            }
        }
        next = next
            .canonicalize([center.clone()], CanonicalizationOptions::forced())
            .map_err(|source| GseError::Algorithm {
                context: "GSE failed to canonicalize generated reference",
                source,
            })?;
        current = next.clone();
        references.push(next);
    }
    Ok(references)
}

fn validate_single_site_state<V>(state: &TreeTN<TensorDynLen, V>) -> Result<(), GseError>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    for node in state.node_names() {
        let count = state.site_space(&node).map_or(0, |space| space.len());
        if count != 1 {
            return Err(GseError::UnsupportedStateSiteCount {
                node: format!("{node:?}"),
                count,
            });
        }
    }
    Ok(())
}

fn validate_reference<V>(
    target: &TreeTN<TensorDynLen, V>,
    reference: &TreeTN<TensorDynLen, V>,
) -> Result<(), GseError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    validate_single_site_state(reference)?;
    if !target.same_topology(reference) {
        return Err(GseError::TopologyMismatch);
    }
    for node in target.node_names() {
        let target_site = single_site_index(target, &node)?;
        let reference_site = single_site_index(reference, &node)?;
        if target_site.dim() != reference_site.dim() {
            return Err(GseError::InvalidMapping {
                node: format!("{node:?}"),
                reason: "reference site dimension does not match target".to_string(),
            });
        }
    }
    let target_scalar = tree_scalar_kind(target)?;
    let reference_scalar = tree_scalar_kind(reference)?;
    if target_scalar != reference_scalar {
        return Err(GseError::ScalarStorageMismatch {
            target: target_scalar.label(),
            reference: reference_scalar.label(),
        });
    }
    Ok(())
}

fn tree_scalar_kind<V>(state: &TreeTN<TensorDynLen, V>) -> Result<ScalarKind, GseError>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    let mut kind: Option<ScalarKind> = None;
    for node in state.node_names() {
        let tensor = state
            .node_index(&node)
            .and_then(|idx| state.tensor(idx))
            .ok_or_else(|| GseError::UnsupportedScalarStorage {
                node: format!("{node:?}"),
                reason: "missing tensor".to_string(),
            })?;
        let node_kind = if tensor.is_f64() {
            ScalarKind::Real
        } else if tensor.is_complex() {
            ScalarKind::Complex
        } else {
            return Err(GseError::UnsupportedScalarStorage {
                node: format!("{node:?}"),
                reason: "expected f64 or Complex64 tensor storage".to_string(),
            });
        };
        if let Some(previous) = kind {
            if previous != node_kind {
                return Err(GseError::UnsupportedScalarStorage {
                    node: format!("{node:?}"),
                    reason: format!(
                        "mixed scalar storage in one TreeTN: expected {}, found {}",
                        previous.label(),
                        node_kind.label()
                    ),
                });
            }
        } else {
            kind = Some(node_kind);
        }
    }
    kind.ok_or_else(|| GseError::UnsupportedScalarStorage {
        node: "<empty>".to_string(),
        reason: "empty TreeTN has no scalar storage".to_string(),
    })
}

fn expand_edges<V>(
    state: &mut TreeTN<TensorDynLen, V>,
    references: &mut [TreeTN<TensorDynLen, V>],
    center: &V,
    options: &GseOptions,
) -> Result<EdgeExpansionStats, GseError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug + 'static,
{
    let edge_order = state
        .site_index_network()
        .topology()
        .edges_to_canonicalize_by_names(center)
        .ok_or_else(|| GseError::MissingCenter {
            center: format!("{center:?}"),
        })?;

    let mut stats = EdgeExpansionStats {
        edges_processed: 0,
        bonds_expanded: 0,
        max_added_basis: 0,
    };
    for (child, parent) in edge_order {
        move_center_to_region_full_rank(state, std::slice::from_ref(&child)).map_err(|source| {
            GseError::Algorithm {
                context: "GSE failed to move target center to edge child",
                source,
            }
        })?;
        for reference in references.iter_mut() {
            move_center_to_region_full_rank(reference, std::slice::from_ref(&child)).map_err(
                |source| GseError::Algorithm {
                    context: "GSE failed to move reference center to edge child",
                    source,
                },
            )?;
        }
        let added = expand_one_edge(state, references, &parent, &child, options)?;
        stats.edges_processed += 1;
        if added > 0 {
            stats.bonds_expanded += 1;
            stats.max_added_basis = stats.max_added_basis.max(added);
        }
    }
    state
        .set_canonical_region([center.clone()])
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to set final canonical center",
            source,
        })?;
    Ok(stats)
}

fn expand_one_edge<V>(
    state: &mut TreeTN<TensorDynLen, V>,
    references: &mut [TreeTN<TensorDynLen, V>],
    parent: &V,
    child: &V,
    options: &GseOptions,
) -> Result<usize, GseError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug + 'static,
{
    let edge = state
        .edge_between(parent, child)
        .ok_or_else(|| GseError::Algorithm {
            context: "GSE target edge is missing",
            source: anyhow::anyhow!("missing edge between {:?} and {:?}", parent, child),
        })?;
    let old_bond = state
        .bond_index(edge)
        .cloned()
        .ok_or_else(|| GseError::Algorithm {
            context: "GSE target edge has no bond index",
            source: anyhow::anyhow!("missing bond between {:?} and {:?}", parent, child),
        })?;
    let child_idx = state
        .node_index(child)
        .ok_or_else(|| GseError::MissingCenter {
            center: format!("{child:?}"),
        })?;
    let parent_idx = state
        .node_index(parent)
        .ok_or_else(|| GseError::MissingCenter {
            center: format!("{parent:?}"),
        })?;
    let child_tensor = state
        .tensor(child_idx)
        .cloned()
        .ok_or_else(|| GseError::Algorithm {
            context: "GSE child tensor is missing",
            source: anyhow::anyhow!("missing tensor at {:?}", child),
        })?;
    let parent_tensor = state
        .tensor(parent_idx)
        .cloned()
        .ok_or_else(|| GseError::Algorithm {
            context: "GSE parent tensor is missing",
            source: anyhow::anyhow!("missing tensor at {:?}", parent),
        })?;
    let q_indices = child_tensor
        .external_indices()
        .into_iter()
        .filter(|idx| idx != &old_bond)
        .collect::<Vec<_>>();
    let q_dim = product_dim(&q_indices);

    let factorized = child_tensor
        .factorize_full_rank(
            std::slice::from_ref(&old_bond),
            FactorizeAlg::SVD,
            Canonical::Right,
        )
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to extract old target row basis",
            source: source.into(),
        })?;
    let basis_bond = factorized.bond_index;
    let basis_rank = basis_bond.dim();
    let basis_order = std::iter::once(basis_bond.clone())
        .chain(q_indices.iter().cloned())
        .collect::<Vec<_>>();
    let basis_tensor = factorized
        .right
        .permuteinds(&basis_order)
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to align old target row basis",
            source,
        })?;

    let q_left = fresh_indices_like(&q_indices);
    let q_right = fresh_indices_like(&q_indices);
    let density = build_reference_density(
        state, references, parent, child, &q_indices, &q_left, &q_right,
    )?;
    let trace_identity = identity_on_index_pairs(&q_left, &q_right)?;
    let trace_tensor = TensorDynLen::contract(&[&density, &trace_identity]).map_err(|source| {
        GseError::Algorithm {
            context: "GSE failed to compute local reference-density trace",
            source,
        }
    })?;
    let trace = trace_tensor.sum().map_err(|source| GseError::Algorithm {
        context: "GSE failed to read local reference-density trace",
        source,
    })?;

    let mut basis_rows = old_basis_rows(&basis_tensor, &basis_bond, basis_rank)?;
    if trace.real() > 0.0 {
        let normalized_density =
            density
                .scale(AnyScalar::new_real(1.0) / trace)
                .map_err(|source| GseError::Algorithm {
                    context: "GSE failed to normalize local reference density",
                    source,
                })?;
        let missing = projected_missing_density_tensor(
            &normalized_density,
            &basis_tensor,
            &basis_bond,
            &q_indices,
            &q_left,
            &q_right,
        )?;
        let missing = hermitianize_by_index_groups(&missing, &q_left, &q_right)?;
        let flat_left = DynIndex::new_dyn(q_dim);
        let flat_right = DynIndex::new_dyn(q_dim);
        let missing_matrix = missing
            .fuse_indices(&q_left, flat_left.clone(), LinearizationOrder::ColumnMajor)
            .and_then(|tensor| {
                tensor.fuse_indices(
                    &q_right,
                    flat_right.clone(),
                    LinearizationOrder::ColumnMajor,
                )
            })
            .and_then(|tensor| tensor.permuteinds(&[flat_left.clone(), flat_right]))
            .map_err(|source| GseError::Algorithm {
                context: "GSE failed to reshape projected reference density for eigensolve",
                source,
            })?;
        let decomp = missing_matrix
            .hermitian_eigendecomposition(options.hermitian_tol)
            .map_err(|source| GseError::Algorithm {
                context: "GSE failed to diagonalize projected reference density",
                source,
            })?;
        for (col, &lambda) in decomp.eigenvalues.iter().enumerate() {
            if lambda > options.density_weight_cutoff {
                basis_rows.push(eigenvector_basis_row(
                    &decomp.eigenvectors,
                    &flat_left,
                    &decomp.eigenvector_index,
                    col,
                    &q_left,
                    &q_indices,
                )?);
            }
        }
    }

    let added = basis_rows.len() - basis_rank;
    let new_dim = basis_rows.len();
    let new_bond = DynIndex::new_bond(new_dim).map_err(|source| GseError::Algorithm {
        context: "GSE failed to create expanded target bond index",
        source: anyhow::anyhow!("{source:?}"),
    })?;
    let target_child = stack_basis_rows(&basis_rows, new_bond.clone())?;
    let coeff_tensor = coefficient_tensor(&child_tensor, &target_child)?;
    let target_parent =
        TensorDynLen::contract(&[&parent_tensor, &coeff_tensor]).map_err(|source| {
            GseError::Algorithm {
                context: "GSE failed to absorb expanded coefficients into target parent",
                source,
            }
        })?;

    state
        .replace_edge_bond(edge, new_bond.clone())
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to replace target edge bond",
            source,
        })?;
    state
        .replace_tensor(child_idx, target_child.clone())
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to replace target child tensor",
            source,
        })?;
    state
        .replace_tensor(parent_idx, target_parent)
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to replace target parent tensor",
            source,
        })?;
    state
        .set_edge_ortho_towards(edge, Some(parent.clone()))
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to set target edge orthogonality direction",
            source,
        })?;
    state
        .set_canonical_region([parent.clone()])
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to move target canonical metadata after expansion",
            source,
        })?;

    for reference in references.iter_mut() {
        update_reference_edge(
            reference,
            state,
            parent,
            child,
            &q_indices,
            &target_child,
            &new_bond,
        )?;
    }

    Ok(added)
}

#[allow(clippy::too_many_arguments)]
fn update_reference_edge<V>(
    reference: &mut TreeTN<TensorDynLen, V>,
    target: &TreeTN<TensorDynLen, V>,
    parent: &V,
    child: &V,
    target_q_indices: &[DynIndex],
    expanded_basis: &TensorDynLen,
    target_new_bond: &DynIndex,
) -> Result<(), GseError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let edge = reference
        .edge_between(parent, child)
        .ok_or_else(|| GseError::Algorithm {
            context: "GSE reference edge is missing during update",
            source: anyhow::anyhow!(
                "missing reference edge between {:?} and {:?}",
                parent,
                child
            ),
        })?;
    let _old_bond = reference
        .bond_index(edge)
        .cloned()
        .ok_or_else(|| GseError::Algorithm {
            context: "GSE reference edge has no bond index during update",
            source: anyhow::anyhow!(
                "missing reference bond between {:?} and {:?}",
                parent,
                child
            ),
        })?;
    let child_idx = reference
        .node_index(child)
        .ok_or_else(|| GseError::MissingCenter {
            center: format!("{child:?}"),
        })?;
    let parent_idx = reference
        .node_index(parent)
        .ok_or_else(|| GseError::MissingCenter {
            center: format!("{parent:?}"),
        })?;
    let child_tensor = reference
        .tensor(child_idx)
        .cloned()
        .ok_or_else(|| GseError::Algorithm {
            context: "GSE reference child tensor is missing during update",
            source: anyhow::anyhow!("missing reference tensor at {:?}", child),
        })?;
    let parent_tensor =
        reference
            .tensor(parent_idx)
            .cloned()
            .ok_or_else(|| GseError::Algorithm {
                context: "GSE reference parent tensor is missing during update",
                source: anyhow::anyhow!("missing reference tensor at {:?}", parent),
            })?;
    let ref_q_indices = map_q_indices(target, reference, child, parent, target_q_indices)?;
    let new_bond =
        DynIndex::new_bond(target_new_bond.dim()).map_err(|source| GseError::Algorithm {
            context: "GSE failed to create expanded reference bond index",
            source: anyhow::anyhow!("{source:?}"),
        })?;
    let child_replacement = expanded_basis
        .replaceind(target_new_bond, &new_bond)
        .and_then(|tensor| tensor.replaceinds(target_q_indices, &ref_q_indices))
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to relabel expanded basis for reference child",
            source,
        })?;
    let coeff_tensor =
        coefficient_tensor(&child_tensor, &child_replacement).map_err(|error| match error {
            GseError::Algorithm { source, .. } => GseError::Algorithm {
                context: "GSE failed to build reference coefficient tensor",
                source,
            },
            other => other,
        })?;
    let parent_replacement =
        TensorDynLen::contract(&[&parent_tensor, &coeff_tensor]).map_err(|source| {
            GseError::Algorithm {
                context: "GSE failed to absorb expanded coefficients into reference parent",
                source,
            }
        })?;

    reference
        .replace_edge_bond(edge, new_bond)
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to replace reference edge bond",
            source,
        })?;
    reference
        .replace_tensor(child_idx, child_replacement)
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to replace reference child tensor",
            source,
        })?;
    reference
        .replace_tensor(parent_idx, parent_replacement)
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to replace reference parent tensor",
            source,
        })?;
    reference
        .set_edge_ortho_towards(edge, Some(parent.clone()))
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to set reference edge orthogonality direction",
            source,
        })?;
    reference
        .set_canonical_region([parent.clone()])
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to move reference canonical metadata after expansion",
            source,
        })?;
    Ok(())
}

fn build_reference_density<V>(
    target: &TreeTN<TensorDynLen, V>,
    references: &[TreeTN<TensorDynLen, V>],
    parent: &V,
    child: &V,
    target_q_indices: &[DynIndex],
    q_left: &[DynIndex],
    q_right: &[DynIndex],
) -> Result<TensorDynLen, GseError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut density: Option<TensorDynLen> = None;
    for reference in references {
        let ref_edge =
            reference
                .edge_between(parent, child)
                .ok_or_else(|| GseError::Algorithm {
                    context: "GSE reference edge is missing",
                    source: anyhow::anyhow!(
                        "missing reference edge between {:?} and {:?}",
                        parent,
                        child
                    ),
                })?;
        let ref_bond =
            reference
                .bond_index(ref_edge)
                .cloned()
                .ok_or_else(|| GseError::Algorithm {
                    context: "GSE reference edge has no bond index",
                    source: anyhow::anyhow!(
                        "missing reference bond between {:?} and {:?}",
                        parent,
                        child
                    ),
                })?;
        let ref_child_idx = reference
            .node_index(child)
            .ok_or_else(|| GseError::MissingCenter {
                center: format!("{child:?}"),
            })?;
        let ref_child_tensor =
            reference
                .tensor(ref_child_idx)
                .ok_or_else(|| GseError::Algorithm {
                    context: "GSE reference child tensor is missing",
                    source: anyhow::anyhow!("missing reference tensor at {:?}", child),
                })?;
        let ref_q_indices = map_q_indices(target, reference, child, parent, target_q_indices)?;
        let bra = ref_child_tensor
            .replaceinds(&ref_q_indices, q_left)
            .map_err(|source| GseError::Algorithm {
                context: "GSE failed to relabel reference density bra indices",
                source,
            })?;
        let ket = ref_child_tensor
            .replaceinds(&ref_q_indices, q_right)
            .map_err(|source| GseError::Algorithm {
                context: "GSE failed to relabel reference density ket indices",
                source,
            })?;
        let contribution = contract_pair_with_operand_options(
            &bra,
            &ket,
            PairwiseContractionOptions::new().with_lhs_conj(true),
        )
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to contract reference density contribution",
            source,
        })?;
        debug_assert!(!contribution.external_indices().contains(&ref_bond));
        density = Some(match density {
            Some(accumulated) => {
                accumulated
                    .add(&contribution)
                    .map_err(|source| GseError::Algorithm {
                        context: "GSE failed to accumulate reference density",
                        source,
                    })?
            }
            None => contribution,
        });
    }
    density.ok_or_else(|| GseError::Algorithm {
        context: "GSE failed to build reference density",
        source: anyhow::anyhow!("no references were supplied"),
    })
}

fn old_basis_rows(
    basis: &TensorDynLen,
    basis_bond: &DynIndex,
    basis_rank: usize,
) -> Result<Vec<TensorDynLen>, GseError> {
    (0..basis_rank)
        .map(|row| {
            basis
                .select_indices(std::slice::from_ref(basis_bond), &[row])
                .map_err(|source| GseError::Algorithm {
                    context: "GSE failed to slice old target basis row",
                    source,
                })
        })
        .collect()
}

fn eigenvector_basis_row(
    eigenvectors: &TensorDynLen,
    flat_left: &DynIndex,
    eigenvector_index: &DynIndex,
    col: usize,
    q_left: &[DynIndex],
    q_indices: &[DynIndex],
) -> Result<TensorDynLen, GseError> {
    let vector = eigenvectors
        .select_indices(std::slice::from_ref(eigenvector_index), &[col])
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to select projected-density eigenvector",
            source,
        })?;
    let row = vector
        .unfuse_index(flat_left, q_left, LinearizationOrder::ColumnMajor)
        .and_then(|tensor| tensor.conj().replaceinds(q_left, q_indices))
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to reshape projected-density eigenvector into a basis row",
            source,
        })?;
    Ok(row)
}

fn stack_basis_rows(rows: &[TensorDynLen], new_bond: DynIndex) -> Result<TensorDynLen, GseError> {
    let refs = rows.iter().collect::<Vec<_>>();
    TensorDynLen::stack_along_new_index(&refs, new_bond, 0).map_err(|source| GseError::Algorithm {
        context: "GSE failed to stack expanded basis rows",
        source,
    })
}

fn coefficient_tensor(
    child_tensor: &TensorDynLen,
    expanded_basis: &TensorDynLen,
) -> Result<TensorDynLen, GseError> {
    contract_pair_with_operand_options(
        child_tensor,
        expanded_basis,
        PairwiseContractionOptions::new().with_rhs_conj(true),
    )
    .map_err(|source| GseError::Algorithm {
        context: "GSE failed to build expansion coefficient tensor",
        source,
    })
}

fn projected_missing_density_tensor(
    density: &TensorDynLen,
    basis: &TensorDynLen,
    basis_bond: &DynIndex,
    q_indices: &[DynIndex],
    q_left: &[DynIndex],
    q_right: &[DynIndex],
) -> Result<TensorDynLen, GseError> {
    let identity = identity_on_index_pairs(q_left, q_right)?;
    let basis_left =
        basis
            .replaceinds(q_indices, q_left)
            .map_err(|source| GseError::Algorithm {
                context: "GSE failed to relabel represented-basis bra indices",
                source,
            })?;
    let basis_right =
        basis
            .replaceinds(q_indices, q_right)
            .map_err(|source| GseError::Algorithm {
                context: "GSE failed to relabel represented-basis ket indices",
                source,
            })?;
    let represented = contract_pair_with_operand_options(
        &basis_left,
        &basis_right,
        PairwiseContractionOptions::new().with_lhs_conj(true),
    )
    .map_err(|source| GseError::Algorithm {
        context: "GSE failed to contract represented basis projector",
        source,
    })?;
    debug_assert!(!represented.external_indices().contains(basis_bond));
    let projector = identity
        .axpby(
            AnyScalar::new_real(1.0),
            &represented,
            AnyScalar::new_real(-1.0),
        )
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to subtract represented basis projector",
            source,
        })?;

    let q_mid_left = fresh_indices_like(q_left);
    let q_mid_right = fresh_indices_like(q_right);
    let p_left_mid = projector
        .replaceinds(q_right, &q_mid_left)
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to relabel left projected-density projector",
            source,
        })?;
    let density_mid = density
        .replaceinds(q_left, &q_mid_left)
        .and_then(|tensor| tensor.replaceinds(q_right, &q_mid_right))
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to relabel middle projected-density tensor",
            source,
        })?;
    let p_mid_right = projector
        .replaceinds(q_left, &q_mid_right)
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to relabel right projected-density projector",
            source,
        })?;
    TensorDynLen::contract(&[&p_left_mid, &density_mid, &p_mid_right]).map_err(|source| {
        GseError::Algorithm {
            context: "GSE failed to contract projected missing reference density",
            source,
        }
    })
}

fn identity_on_index_pairs(
    left: &[DynIndex],
    right: &[DynIndex],
) -> Result<TensorDynLen, GseError> {
    if left.len() != right.len() {
        return Err(GseError::Algorithm {
            context: "GSE failed to build identity on q-space",
            source: anyhow::anyhow!(
                "left/right q-index count mismatch: {} vs {}",
                left.len(),
                right.len()
            ),
        });
    }
    let mut identity: Option<TensorDynLen> = None;
    for (left_index, right_index) in left.iter().zip(right.iter()) {
        let pair = TensorDynLen::copy_tensor(
            vec![left_index.clone(), right_index.clone()],
            AnyScalar::new_real(1.0),
        )
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to build one-index-pair identity",
            source,
        })?;
        identity = Some(match identity {
            Some(accumulated) => {
                accumulated
                    .outer_product(&pair)
                    .map_err(|source| GseError::Algorithm {
                        context: "GSE failed to combine q-space identity factors",
                        source,
                    })?
            }
            None => pair,
        });
    }
    match identity {
        Some(tensor) => Ok(tensor),
        None => TensorDynLen::scalar(1.0_f64).map_err(|source| GseError::Algorithm {
            context: "GSE failed to build scalar q-space identity",
            source,
        }),
    }
}

fn hermitianize_by_index_groups(
    tensor: &TensorDynLen,
    left: &[DynIndex],
    right: &[DynIndex],
) -> Result<TensorDynLen, GseError> {
    let adjoint = adjoint_by_index_groups(tensor, left, right)?;
    tensor
        .axpby(AnyScalar::new_real(0.5), &adjoint, AnyScalar::new_real(0.5))
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to Hermitianize projected reference density",
            source,
        })
}

fn adjoint_by_index_groups(
    tensor: &TensorDynLen,
    left: &[DynIndex],
    right: &[DynIndex],
) -> Result<TensorDynLen, GseError> {
    if left.len() != right.len() {
        return Err(GseError::Algorithm {
            context: "GSE failed to adjoint grouped tensor",
            source: anyhow::anyhow!(
                "left/right q-index count mismatch: {} vs {}",
                left.len(),
                right.len()
            ),
        });
    }
    let temporary = fresh_indices_like(left);
    tensor
        .conj()
        .replaceinds(left, &temporary)
        .and_then(|tensor| tensor.replaceinds(right, left))
        .and_then(|tensor| tensor.replaceinds(&temporary, right))
        .map_err(|source| GseError::Algorithm {
            context: "GSE failed to swap grouped tensor indices for adjoint",
            source,
        })
}

fn fresh_indices_like(indices: &[DynIndex]) -> Vec<DynIndex> {
    indices
        .iter()
        .map(|index| DynIndex::new_dyn(index.dim()))
        .collect()
}

fn map_q_indices<V>(
    target: &TreeTN<TensorDynLen, V>,
    reference: &TreeTN<TensorDynLen, V>,
    child: &V,
    parent: &V,
    target_q_indices: &[DynIndex],
) -> Result<Vec<DynIndex>, GseError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let target_site = single_site_index(target, child)?;
    let reference_site = single_site_index(reference, child)?;
    let mut mapped = Vec::with_capacity(target_q_indices.len());
    for target_index in target_q_indices {
        if target_index == &target_site {
            mapped.push(reference_site.clone());
            continue;
        }
        let neighbor =
            neighbor_for_bond(target, child, target_index)?.ok_or_else(|| GseError::Algorithm {
                context: "GSE failed to map target q bond to reference",
                source: anyhow::anyhow!(
                    "index {:?} at child {:?} is neither site nor bond",
                    target_index,
                    child
                ),
            })?;
        if &neighbor == parent {
            return Err(GseError::Algorithm {
                context: "GSE parent bond cannot appear in q index map",
                source: anyhow::anyhow!("parent bond was included in q for child {:?}", child),
            });
        }
        let ref_edge =
            reference
                .edge_between(child, &neighbor)
                .ok_or_else(|| GseError::Algorithm {
                    context: "GSE reference topology is missing q edge",
                    source: anyhow::anyhow!(
                        "missing reference edge between {:?} and {:?}",
                        child,
                        neighbor
                    ),
                })?;
        let ref_bond =
            reference
                .bond_index(ref_edge)
                .cloned()
                .ok_or_else(|| GseError::Algorithm {
                    context: "GSE reference q edge has no bond",
                    source: anyhow::anyhow!(
                        "missing reference bond between {:?} and {:?}",
                        child,
                        neighbor
                    ),
                })?;
        if ref_bond.dim() != target_index.dim() {
            return Err(GseError::InvalidMapping {
                node: format!("{child:?}"),
                reason: "reference child-side bond dimension does not match target".to_string(),
            });
        }
        mapped.push(ref_bond);
    }
    Ok(mapped)
}

fn neighbor_for_bond<V>(
    state: &TreeTN<TensorDynLen, V>,
    node: &V,
    bond: &DynIndex,
) -> Result<Option<V>, GseError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    for neighbor in state.site_index_network().neighbors(node) {
        let Some(edge) = state.edge_between(node, &neighbor) else {
            continue;
        };
        if state.bond_index(edge) == Some(bond) {
            return Ok(Some(neighbor));
        }
    }
    Ok(None)
}

fn single_site_index<V>(state: &TreeTN<TensorDynLen, V>, node: &V) -> Result<DynIndex, GseError>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    let site_space = state
        .site_space(node)
        .ok_or_else(|| GseError::UnsupportedStateSiteCount {
            node: format!("{node:?}"),
            count: 0,
        })?;
    if site_space.len() != 1 {
        return Err(GseError::UnsupportedStateSiteCount {
            node: format!("{node:?}"),
            count: site_space.len(),
        });
    }
    site_space
        .iter()
        .next()
        .cloned()
        .ok_or_else(|| GseError::UnsupportedStateSiteCount {
            node: format!("{node:?}"),
            count: 0,
        })
}

fn product_dim(indices: &[DynIndex]) -> usize {
    indices.iter().map(DynIndex::dim).product()
}

fn max_link_dim<V>(state: &TreeTN<TensorDynLen, V>) -> usize
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    state.link_dims().into_iter().max().unwrap_or(1)
}

fn move_center_to_region_full_rank<V>(
    state: &mut TreeTN<TensorDynLen, V>,
    region: &[V],
) -> Result<()>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    if region.is_empty() {
        return Err(anyhow::anyhow!("GSE cannot move center to an empty region"));
    }
    let current = state.canonical_region();
    if current.is_empty() {
        return Err(anyhow::anyhow!("GSE state is not canonicalized"));
    }
    if region.iter().any(|node| current.contains(node)) {
        return Ok(());
    }

    let (path, target) = {
        let topology = state.site_index_network().topology();
        let mut best_path = None;
        for src in current {
            let Some(src_idx) = topology.node_index(src) else {
                continue;
            };
            for dst in region {
                let Some(dst_idx) = topology.node_index(dst) else {
                    continue;
                };
                let Some(path) = topology.path_between(src_idx, dst_idx) else {
                    continue;
                };
                if best_path
                    .as_ref()
                    .is_none_or(|best: &Vec<_>| path.len() < best.len())
                {
                    best_path = Some(path);
                }
            }
        }
        let path = best_path.ok_or_else(|| {
            anyhow::anyhow!(
                "GSE could not find a path from canonical region {:?} to {:?}",
                current,
                region
            )
        })?;
        let target = path
            .last()
            .and_then(|idx| topology.node_name(*idx))
            .ok_or_else(|| anyhow::anyhow!("GSE center path ended at an unknown node"))?
            .clone();
        (path, target)
    };
    for pair in path.windows(2) {
        let src = pair[0];
        let dst = pair[1];
        state.sweep_edge_full_rank(
            src,
            dst,
            FactorizeAlg::QR,
            Canonical::Left,
            "GSE path-only center move",
        )?;
    }
    state.set_canonical_region([target])?;
    Ok(())
}
