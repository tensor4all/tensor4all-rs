//! SquareLinsolveUpdater: Local update implementation for square linsolve.
//!
//! Uses GMRES (via tensor4all_core::krylov) to solve the local linear problem at each sweep step.
//! This is the V_in = V_out specialized version.

use crate::error::TreeTNOperationError;
use std::cell::Cell;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Instant;

use anyhow::{Context, Result};

use tensor4all_core::krylov::{gmres_affine, gmres_affine_with_absolute_tolerance, GmresOptions};
use tensor4all_core::{
    AnyScalar, Canonical, FactorizeAlg, FactorizeOptions, IndexLike, SvdTruncationPolicy,
    TensorLike,
};

use super::local_linop::LocalLinOp;
use super::projected_state::ProjectedState;
use crate::linsolve::common::{
    GmresToleranceMode, LinsolveOptions, ProjectedOperator, ResidualEnrichment,
};
use crate::local_update_support;
use crate::operator::IndexMapping;
use crate::{factorize_tensor_to_treetn_with, LocalUpdateStep, LocalUpdater, TreeTN, TreeTopology};

static LOCAL_SOLVE_TRACE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Report from SquareLinsolveUpdater::verify().
#[derive(Debug, Clone)]
pub struct LinsolveVerifyReport<V> {
    /// Whether the configuration is valid
    pub is_valid: bool,
    /// Errors that would prevent linsolve from working
    pub errors: Vec<String>,
    /// Warnings that might indicate issues
    pub warnings: Vec<String>,
    /// Per-node details
    pub node_details: Vec<NodeVerifyDetail<V>>,
}

impl<V> Default for LinsolveVerifyReport<V> {
    fn default() -> Self {
        Self {
            is_valid: false,
            errors: Vec::new(),
            warnings: Vec::new(),
            node_details: Vec::new(),
        }
    }
}

/// Per-node verification details.
#[derive(Debug, Clone)]
pub struct NodeVerifyDetail<V> {
    /// Node name
    pub node: V,
    /// State's site space index IDs
    pub state_site_indices: Vec<String>,
    /// Operator's site space index IDs
    pub op_site_indices: Vec<String>,
    /// State tensor's all index IDs with dimensions
    pub state_tensor_indices: Vec<String>,
    /// Operator tensor's all index IDs with dimensions
    pub op_tensor_indices: Vec<String>,
    /// Number of common indices between state and operator
    pub common_index_count: usize,
}

impl<V: std::fmt::Debug> std::fmt::Display for LinsolveVerifyReport<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LinsolveVerifyReport:")?;
        writeln!(f, "  Valid: {}", self.is_valid)?;

        if !self.errors.is_empty() {
            writeln!(f, "  Errors:")?;
            for err in &self.errors {
                writeln!(f, "    - {}", err)?;
            }
        }

        if !self.warnings.is_empty() {
            writeln!(f, "  Warnings:")?;
            for warn in &self.warnings {
                writeln!(f, "    - {}", warn)?;
            }
        }

        if !self.node_details.is_empty() {
            writeln!(f, "  Node Details:")?;
            for detail in &self.node_details {
                writeln!(f, "    {:?}:", detail.node)?;
                writeln!(
                    f,
                    "      State site indices: {:?}",
                    detail.state_site_indices
                )?;
                writeln!(f, "      Op site indices: {:?}", detail.op_site_indices)?;
                writeln!(
                    f,
                    "      State tensor indices: {:?}",
                    detail.state_tensor_indices
                )?;
                writeln!(f, "      Op tensor indices: {:?}", detail.op_tensor_indices)?;
                writeln!(f, "      Common index count: {}", detail.common_index_count)?;
            }
        }

        Ok(())
    }
}

/// SquareLinsolveUpdater: Implements LocalUpdater for the square linsolve algorithm.
///
/// At each sweep step:
/// 1. Compute local operator (from ProjectedOperator environments)
/// 2. Compute local RHS (from ProjectedState environments)
/// 3. Solve local linear system using GMRES
/// 4. Transport the center by QR (one-site ALS) or split by SVD (two-site MALS).
///
/// Prefer [`super::square_linsolve`] to construct the sweep. For direct one-site
/// use, supply oriented two-node steps `[source, destination]`, with
/// `new_center = destination`, so both QR-modified tensors are synchronized by
/// `after_step`. A single-node problem uses `[node]` with `new_center = node`.
///
/// This is the V_in = V_out specialized version. The current solution x is used
/// with a separate reference state (with different bond indices) for stable
/// environment computation.
pub struct SquareLinsolveUpdater<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Projected operator (3-chain), wrapped in `Arc<RwLock>` for GMRES.
    pub projected_operator: Arc<RwLock<ProjectedOperator<T, V>>>,
    /// Projected state for RHS (2-chain)
    pub projected_state: ProjectedState<T, V>,
    /// Solver options
    pub options: LinsolveOptions,
    /// Reference state (separate from ket to avoid unintended contractions).
    /// Link indices are different from ket_state to prevent bra↔ket link contractions.
    /// Boundary bonds (region ↔ outside) maintain stable IDs for cache consistency.
    reference_state: TreeTN<T, V>,
    /// Mapping from boundary edge (node_in_region, neighbor_outside) to reference-side bond index.
    /// This ensures boundary bonds keep stable IDs across updates for environment cache reuse.
    boundary_bond_map: HashMap<(V, V), T::Index>,
    /// Run the bra/ket convention precheck only once.
    did_ref_bra_ket_precheck: bool,
    /// Run MPO structure validation only once.
    did_mpo_validation: bool,
}

impl<T, V> SquareLinsolveUpdater<T, V>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    /// Create a new SquareLinsolveUpdater.
    ///
    /// The reference_state will be initialized lazily on the first `before_step` call.
    pub fn new(operator: TreeTN<T, V>, rhs: TreeTN<T, V>, options: LinsolveOptions) -> Self {
        Self {
            projected_operator: Arc::new(RwLock::new(ProjectedOperator::new(operator))),
            projected_state: ProjectedState::new(rhs),
            options,
            reference_state: TreeTN::new(),
            boundary_bond_map: HashMap::new(),
            did_ref_bra_ket_precheck: false,
            did_mpo_validation: false,
        }
    }

    /// Create a new SquareLinsolveUpdater with index mappings for correct index handling.
    ///
    /// Use this when the MPO uses internal indices (s_in_tmp, s_out_tmp) that differ
    /// from the state's site indices. The mappings define how to translate between them.
    ///
    /// The reference_state will be initialized lazily on the first `before_step` call.
    ///
    /// # Arguments
    /// * `operator` - The MPO with internal index IDs
    /// * `input_mapping` - Mapping from true input indices to MPO's internal indices
    /// * `output_mapping` - Mapping from true output indices to MPO's internal indices
    /// * `rhs` - The RHS b
    /// * `options` - Solver options
    pub fn with_index_mappings(
        operator: TreeTN<T, V>,
        input_mapping: HashMap<V, IndexMapping<T::Index>>,
        output_mapping: HashMap<V, IndexMapping<T::Index>>,
        rhs: TreeTN<T, V>,
        options: LinsolveOptions,
    ) -> Self {
        let projected_operator =
            ProjectedOperator::with_index_mappings(operator, input_mapping, output_mapping);
        Self {
            projected_operator: Arc::new(RwLock::new(projected_operator)),
            projected_state: ProjectedState::new(rhs),
            options,
            reference_state: TreeTN::new(),
            boundary_bond_map: HashMap::new(),
            did_ref_bra_ket_precheck: false,
            did_mpo_validation: false,
        }
    }

    /// Initialize reference_state from ket_state if not already initialized.
    ///
    /// Creates reference_state by relabeling link indices (using sim() for internal bonds,
    /// preserving boundary bonds for cache consistency).
    ///
    /// This is called lazily on the first `before_step` to ensure we have the initial ket_state.
    fn ensure_reference_state_initialized(&mut self, ket_state: &TreeTN<T, V>) -> Result<()> {
        let initialized = self.reference_state.node_names().is_empty();
        local_update_support::initialize_reference_state_if_empty(
            &mut self.reference_state,
            ket_state,
        )?;
        if initialized {
            self.boundary_bond_map.clear();
        }
        Ok(())
    }

    /// Verify internal data consistency between operator, RHS, and state.
    ///
    /// This function checks that:
    /// 1. The operator's site space structure is compatible with the state
    /// 2. The operator's input indices can match the state's site indices
    /// 3. Environment computation requirements are satisfiable
    ///
    /// Returns a detailed report of any inconsistencies found.
    /// # Errors
    ///
    /// Returns an error only when the `projected_operator` read lock is poisoned.
    /// Structural inconsistencies are collected in the returned report instead.
    ///
    pub fn verify(
        &self,
        state: &TreeTN<T, V>,
    ) -> std::result::Result<LinsolveVerifyReport<V>, TreeTNOperationError> {
        let mut report = LinsolveVerifyReport::default();

        let proj_op = self
            .projected_operator
            .read()
            .map_err(|e| {
                anyhow::anyhow!("Failed to acquire read lock on projected_operator: {}", e)
            })
            .context("verify: lock poisoned")?;
        let operator = proj_op.operator();
        let rhs = &self.projected_state.rhs;

        // Check node consistency
        let state_nodes: std::collections::BTreeSet<_> = state
            .site_index_network()
            .node_names()
            .into_iter()
            .collect();
        let op_nodes: std::collections::BTreeSet<_> = operator
            .site_index_network()
            .node_names()
            .into_iter()
            .collect();
        let rhs_nodes: std::collections::BTreeSet<_> =
            rhs.site_index_network().node_names().into_iter().collect();

        if state_nodes != op_nodes {
            report.errors.push(format!(
                "State and operator have different node sets. State: {:?}, Operator: {:?}",
                state_nodes, op_nodes
            ));
        }

        if state_nodes != rhs_nodes {
            report.errors.push(format!(
                "State and RHS have different node sets. State: {:?}, RHS: {:?}",
                state_nodes, rhs_nodes
            ));
        }

        // Check site index compatibility per node
        for node in &state_nodes {
            let state_site = state.site_space(node);
            let op_site = operator.site_space(node);

            // Get state tensor indices
            if let Some(state_idx) = state.node_index(node) {
                if let Some(state_tensor) = state.tensor(state_idx) {
                    let state_indices_vec = state_tensor.external_indices();

                    // Get operator tensor indices
                    if let Some(op_idx) = operator.node_index(node) {
                        if let Some(op_tensor) = operator.tensor(op_idx) {
                            let op_indices_vec = op_tensor.external_indices();

                            // Check for common full indices (should have at least bond indices)
                            let common_count = state_indices_vec
                                .iter()
                                .filter(|state_idx| {
                                    op_indices_vec.iter().any(|op_idx| op_idx == *state_idx)
                                })
                                .count();

                            report.node_details.push(NodeVerifyDetail {
                                node: (*node).clone(),
                                state_site_indices: state_site
                                    .map(|s| s.iter().map(|i| format!("{:?}", i.id())).collect())
                                    .unwrap_or_default(),
                                op_site_indices: op_site
                                    .map(|s| s.iter().map(|i| format!("{:?}", i.id())).collect())
                                    .unwrap_or_default(),
                                state_tensor_indices: state_indices_vec
                                    .iter()
                                    .map(|idx| format!("{:?}(dim={})", idx, idx.dim()))
                                    .collect(),
                                op_tensor_indices: op_indices_vec
                                    .iter()
                                    .map(|idx| format!("{:?}(dim={})", idx, idx.dim()))
                                    .collect(),
                                common_index_count: common_count,
                            });

                            // Warn if no site indices in common (expected for MPO)
                            // In proper MPO structure, operator should have input indices
                            // that match state's site indices
                            if common_count == 0 {
                                report.warnings.push(format!(
                                        "Node {:?}: No common indices between state and operator tensors. \
                                     State has {:?}, operator has {:?}",
                                        node, state_indices_vec, op_indices_vec
                                    ));
                            }
                        }
                    }
                }
            }
        }

        // Final verdict
        report.is_valid = report.errors.is_empty();

        Ok(report)
    }

    /// Contract all tensors in the region into a single local tensor.
    fn contract_region(&self, subtree: &TreeTN<T, V>, region: &[V]) -> Result<T> {
        local_update_support::contract_region(subtree, region)
    }

    /// Build TreeTopology for the subtree region from the solved tensor.
    ///
    /// Maps each node to the indices belonging to it in the solved tensor.
    fn build_subtree_topology(
        &self,
        solved_tensor: &T,
        region: &[V],
        full_treetn: &TreeTN<T, V>,
    ) -> Result<TreeTopology<V, T::Index>> {
        local_update_support::build_subtree_topology(solved_tensor, region, full_treetn)
    }

    /// Copy decomposed tensors back to subtree, preserving original bond IDs.
    fn copy_decomposed_to_subtree(
        &self,
        subtree: &mut TreeTN<T, V>,
        decomposed: &TreeTN<T, V>,
        region: &[V],
        full_treetn: &TreeTN<T, V>,
    ) -> Result<()> {
        local_update_support::copy_decomposed_to_subtree(subtree, decomposed, region, full_treetn)
    }

    /// Solve the local linear problem using GMRES.
    ///
    /// Solves: (a₀ + a₁ * H_local) |x_local⟩ = |b_local⟩
    fn solve_local(&mut self, region: &[V], init: &T, state: &TreeTN<T, V>) -> Result<T> {
        let solve_index = LOCAL_SOLVE_TRACE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let trace_limit = std::env::var("T4A_LINSOLVE_TRACE_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let trace = trace_limit.is_some_and(|limit| solve_index < limit);
        let abort_after = std::env::var("T4A_LINSOLVE_ABORT_AFTER")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let solve_started = Instant::now();

        let init_indices = init.external_indices();
        let rhs_local = self.local_rhs(region, &init_indices, state)?;

        // Create local linear operator with separate reference_state
        // This prevents unintended bra↔ket link contractions in environment computation
        let linop = LocalLinOp::with_expected_input_indices(
            Arc::clone(&self.projected_operator),
            region.to_vec(),
            state,
            &self.reference_state,
            init_indices,
        );

        // KrylovKit builds Arnoldi from the unshifted local operator H.
        // The affine coefficients `(a0 I + a1 H)` are applied in GMRES'
        // projected Hessenberg problem, not inside LocalLinOp.
        let apply_calls = Cell::new(0usize);
        let apply_elapsed_micros = Cell::new(0u128);
        let apply_a = |x: &T| {
            let apply_started = Instant::now();
            let result = linop.apply_projected(x);
            apply_calls.set(apply_calls.get() + 1);
            apply_elapsed_micros
                .set(apply_elapsed_micros.get() + apply_started.elapsed().as_micros());
            result.map_err(anyhow::Error::from)
        };

        let mut gmres_options = local_gmres_options(&self.options)?;
        if trace && std::env::var_os("T4A_LINSOLVE_VERBOSE_GMRES").is_some() {
            gmres_options.verbose = true;
        }

        // Solve using GMRES (works directly with IdxTensor)
        let result = match self.options.gmres_tolerance_mode {
            GmresToleranceMode::Relative => gmres_affine(
                apply_a,
                &rhs_local,
                init,
                self.options.a0.clone(),
                self.options.a1.clone(),
                &gmres_options,
            )?,
            GmresToleranceMode::Absolute => gmres_affine_with_absolute_tolerance(
                apply_a,
                &rhs_local,
                init,
                self.options.a0.clone(),
                self.options.a1.clone(),
                &gmres_options,
                self.options.gmres_tol,
            )?,
        };

        if trace {
            eprintln!(
                "T4A local_solve #{solve_index}: region={region:?} mode={:?} rhs_norm={:.6e} init_norm={:.6e} iterations={} residual={:.6e} converged={} apply_calls={} apply_ms={:.3} total_ms={:.3}",
                self.options.gmres_tolerance_mode,
                rhs_local.norm()?,
                init.norm()?,
                result.iterations,
                result.residual_norm,
                result.converged,
                apply_calls.get(),
                apply_elapsed_micros.get() as f64 / 1000.0,
                solve_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        if abort_after.is_some_and(|limit| solve_index + 1 >= limit) {
            anyhow::bail!("T4A_LINSOLVE_ABORT_AFTER reached after local solve #{solve_index}");
        }

        Ok(result.solution)
    }

    /// Projected right-hand side `<ref|b>` on `region`, in the index order of
    /// the local state tensor `init_indices`.
    fn local_rhs(
        &mut self,
        region: &[V],
        init_indices: &[T::Index],
        state: &TreeTN<T, V>,
    ) -> Result<T> {
        let topology = state.site_index_network();
        let rhs_local_raw =
            self.projected_state
                .local_constant_term(region, &self.reference_state, topology)?;
        let rhs_local_raw =
            self.replace_reference_boundary_bonds_with_state(rhs_local_raw, region, state)?;

        // The RHS carries indices from the `rhs` TreeTN, while the local state
        // carries indices from `state`. Krylov operations such as b - A*x need
        // the same index order, so permute when only the order differs.
        let rhs_indices = rhs_local_raw.external_indices();
        if !self.index_sets_match(init_indices, &rhs_indices) {
            return Err(anyhow::anyhow!(
                "{}",
                self.index_structure_mismatch_message(
                    init_indices,
                    &rhs_indices,
                    "Index structure mismatch between init and RHS (local tensors)",
                    "This suggests:\n  - ProjectedState environment construction may have contracted/left open unexpected indices\n  - External indices may not be properly aligned between x and b\n  - full contraction may have over-contracted external indices in the environment\n\nSee the rustdoc on [`ProjectedState`](crate::linsolve::square::ProjectedState) and `docs/design/treetn-linsolve-mpo.md` for analysis of external index handling.",
                )
            ));
        }
        let in_order = init_indices
            .iter()
            .zip(rhs_indices.iter())
            .all(|(ii, ri)| ii == ri && ii.dim() == ri.dim());
        if in_order {
            Ok(rhs_local_raw)
        } else {
            Ok(rhs_local_raw.permuteinds(init_indices)?)
        }
    }

    /// Append residual directions to the `src`–`dst` bond of a two-node subtree.
    ///
    /// Requires `src` to be left-orthonormal towards `dst` (the state after
    /// QR center transport) and the subtree's region outside `{src, dst}` to
    /// match `state`, whose environments define the two-site projection.
    /// The represented two-site tensor is unchanged; on return `src` is again
    /// left-orthonormal with a bond of dimension at most
    /// `old + enrichment.max_added`, and `dst` is the new center.
    fn enrich_bond(
        &mut self,
        subtree: &mut TreeTN<T, V>,
        src_name: &V,
        dst_name: &V,
        enrichment: &ResidualEnrichment,
        state: &TreeTN<T, V>,
    ) -> Result<()> {
        let src = subtree
            .node_index(src_name)
            .ok_or_else(|| anyhow::anyhow!("enrichment source node missing"))?;
        let dst = subtree
            .node_index(dst_name)
            .ok_or_else(|| anyhow::anyhow!("enrichment destination node missing"))?;
        let edge = subtree
            .edge_between(src_name, dst_name)
            .ok_or_else(|| anyhow::anyhow!("enrichment edge missing"))?;
        let bond = subtree
            .bond_index(edge)
            .ok_or_else(|| anyhow::anyhow!("enrichment bond missing"))?
            .clone();
        let basis = subtree
            .tensor(src)
            .ok_or_else(|| anyhow::anyhow!("enrichment source tensor missing"))?
            .clone();
        let center = subtree
            .tensor(dst)
            .ok_or_else(|| anyhow::anyhow!("enrichment destination tensor missing"))?
            .clone();

        let src_side: Vec<T::Index> = basis
            .external_indices()
            .into_iter()
            .filter(|index| index != &bond)
            .collect();
        // New directions must be orthogonal to the current basis, so the
        // src-side space dimension bounds the bond.
        let src_side_dim: usize = src_side.iter().map(|index| index.dim()).product();
        let mut budget = enrichment
            .max_added
            .min(src_side_dim.saturating_sub(bond.dim()));
        if let Some(cap) = enrichment.max_bond_dim {
            budget = budget.min(cap.saturating_sub(bond.dim()));
        }
        if budget == 0 {
            return Ok(());
        }

        let region = [src_name.clone(), dst_name.clone()];
        let two_site = T::contract(&[&basis, &center])?;
        let two_site_indices = two_site.external_indices();
        let rhs = self.local_rhs(&region, &two_site_indices, state)?;
        // `||b_local||` as `norm_scaled * norm_unit`. The plain norm can
        // overflow although every entry is finite; then it is recomputed in
        // units of the largest entry of `b_local / 2`, so `cutoff * ||b_local||`
        // stays finite whenever it is representable. Halving first keeps that
        // largest modulus finite also for a complex entry whose parts are
        // finite but whose modulus exceeds `f64::MAX`.
        let (norm_scaled, norm_unit) = match rhs.norm()? {
            norm if norm.is_finite() => (norm, 1.0),
            _ => {
                let half = rhs.scale(AnyScalar::new_real(0.5))?;
                let unit = half.maxabs()?;
                if !unit.is_finite() {
                    return Ok(());
                }
                let norm = half.scale(AnyScalar::new_real(1.0 / unit))?.norm()?;
                (2.0 * norm, unit)
            }
        };
        if norm_scaled == 0.0 {
            return Ok(());
        }
        // residual = b - a0 x - a1 A x. A zero coefficient drops its term
        // instead of scaling it by zero, which would turn an overflowing
        // `A x` (or `x`) into NaN.
        let one = AnyScalar::new_real(1.0);
        let mut residual = rhs;
        if !self.options.a0.is_zero() {
            residual = residual.axpby(one.clone(), &two_site, -self.options.a0.clone())?;
        }
        if !self.options.a1.is_zero() {
            let linop = LocalLinOp::with_expected_input_indices(
                Arc::clone(&self.projected_operator),
                region.to_vec(),
                state,
                &self.reference_state,
                two_site_indices,
            );
            let applied = linop.apply_projected(&two_site)?;
            residual = residual.axpby(one, &applied, -self.options.a1.clone())?;
        }
        // Remove the component already representable in the current src basis.
        let overlap = T::contract(&[&basis.conj(), &residual])?;
        let residual = residual.sub(&T::contract(&[&basis, &overlap])?)?;

        // The enrichment cutoff is the only selection rule: pass it to the SVD
        // as an explicit absolute per-value policy, so neither the global
        // default policy nor a second factorization is involved.
        // A zero cutoff keeps every nonzero direction whatever `||b_local||`
        // is, so it does not depend on that norm.
        let threshold = if enrichment.cutoff == 0.0 {
            0.0
        } else {
            enrichment.cutoff * norm_scaled * norm_unit
        };
        // The threshold overflows only when `cutoff * ||b_local||` exceeds
        // `f64::MAX`; no finite singular value exceeds it, so nothing is added.
        if !threshold.is_finite() {
            return Ok(());
        }
        let svd = residual.factorize(
            &src_side,
            &FactorizeOptions::svd()
                .with_canonical(Canonical::Left)
                .with_svd_policy(SvdTruncationPolicy::new(threshold).with_absolute())
                .with_max_bond_dim(budget),
        )?;
        // The SVD keeps at least one direction; drop it when even the leading
        // singular value is at most the cutoff.
        let leading = svd
            .singular_values
            .as_ref()
            .and_then(|values| values.first().copied())
            .ok_or_else(|| anyhow::anyhow!("SVD did not report singular values"))?;
        if leading <= threshold {
            return Ok(());
        }

        // The residual directions are complex when `b` or `A` is complex
        // although the basis may be real (a real initial guess); the direct
        // sum needs one scalar type, so promote the real operand.
        let complex_one = AnyScalar::new_complex(1.0, 0.0);
        let (basis, left) = match (
            basis.inner_product(&basis)?.is_complex(),
            svd.left.inner_product(&svd.left)?.is_complex(),
        ) {
            (false, true) => (basis.scale(complex_one)?, svd.left),
            (true, false) => (basis, svd.left.scale(complex_one)?),
            _ => (basis, svd.left),
        };
        let stacked = basis.direct_sum(&left, &[(bond, svd.bond_index)])?.tensor;
        // Re-orthonormalize: the new directions are orthogonal to the old
        // basis only up to rounding.
        let enlarged = stacked.factorize_full_rank(&src_side, FactorizeAlg::QR, Canonical::Left)?;
        let new_center = T::contract(&[&enlarged.left.conj(), &two_site])?;

        subtree.replace_edge_bond(edge, enlarged.bond_index)?;
        subtree.replace_tensor(src, enlarged.left)?;
        subtree.replace_tensor(dst, new_center)?;
        subtree.set_edge_ortho_towards(edge, Some(dst_name.clone()))?;
        Ok(())
    }

    /// Synchronize reference_state region with ket_state, preserving boundary bond IDs.
    ///
    /// This ensures reference_state stays in sync with ket_state updates while maintaining
    /// stable boundary bond IDs for environment cache reuse.
    fn sync_reference_state_region(
        &mut self,
        step: &LocalUpdateStep<V>,
        ket_state: &TreeTN<T, V>,
    ) -> Result<()> {
        local_update_support::sync_reference_state_region(
            &mut self.reference_state,
            Some(&mut self.boundary_bond_map),
            step,
            ket_state,
        )
    }
}

impl<T, V> LocalUpdater<T, V> for SquareLinsolveUpdater<T, V>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    fn before_step(
        &mut self,
        step: &LocalUpdateStep<V>,
        full_treetn_before: &TreeTN<T, V>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        self.options.validate_update_mode()?;
        // Initialize reference_state lazily on first call
        self.ensure_reference_state_initialized(full_treetn_before)?;

        // (1) Precheck: ensure local RHS indices align with local init indices
        // (bra/ket convention sanity for `<ref|H|x>` vs `<ref|b>`).
        if !self.did_ref_bra_ket_precheck {
            self.precheck_ref_bra_ket_convention(step, full_treetn_before)?;
            self.did_ref_bra_ket_precheck = true;
        }

        // (3) MPO structure validation (fail fast) – run once.
        if !self.did_mpo_validation {
            self.validate_mpo_external_indices(full_treetn_before)?;
            self.did_mpo_validation = true;
        }
        Ok(())
    }

    fn update(
        &mut self,
        mut subtree: TreeTN<T, V>,
        step: &LocalUpdateStep<V>,
        full_treetn: &TreeTN<T, V>,
    ) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError> {
        if self.options.nsite == 1 {
            if !(matches!(step.nodes.as_slice(), [node] if node == &step.new_center)
                || matches!(step.nodes.as_slice(), [src, dst] if src != dst && dst == &step.new_center))
            {
                return Err(anyhow::anyhow!(
                    "one-site ALS requires a single node or an oriented transport edge"
                )
                .into());
            }
            let node = step
                .nodes
                .first()
                .ok_or_else(|| anyhow::anyhow!("empty ALS update region"))?;
            // The local problem is a valid projection only when the solved
            // tensor is the orthogonality center.
            let center = full_treetn.canonical_region();
            if center.len() != 1 || !center.contains(node) {
                return Err(anyhow::anyhow!(
                    "one-site ALS must solve at the orthogonality center: \
                     source {node:?}, canonical region {center:?}"
                )
                .into());
            }
            let src = subtree
                .node_index(node)
                .ok_or_else(|| anyhow::anyhow!("ALS source node missing"))?;
            let init = subtree
                .tensor(src)
                .ok_or_else(|| anyhow::anyhow!("ALS source tensor missing"))?;
            let solved = self.solve_local(std::slice::from_ref(node), init, full_treetn)?;
            subtree.replace_tensor(src, solved)?;
            if step.nodes.len() == 2 {
                let dst = subtree
                    .node_index(&step.new_center)
                    .ok_or_else(|| anyhow::anyhow!("ALS destination node missing"))?;
                subtree.sweep_edge_full_rank(
                    src,
                    dst,
                    FactorizeAlg::QR,
                    Canonical::Left,
                    "ALS center transport",
                )?;
                if let Some(enrichment) = self.options.enrichment {
                    self.enrich_bond(
                        &mut subtree,
                        node,
                        &step.new_center,
                        &enrichment,
                        full_treetn,
                    )?;
                }
            }
            subtree.set_canonical_region([step.new_center.clone()])?;
            return Ok(subtree);
        }
        // Contract tensors in the region into a single local tensor
        let init_local = self.contract_region(&subtree, &step.nodes)?;
        // Solve local linear problem using GMRES
        let solved_local = self.solve_local(&step.nodes, &init_local, full_treetn)?;

        // Build TreeTopology for the subtree region
        let topology = self.build_subtree_topology(&solved_local, &step.nodes, full_treetn)?;

        // Decompose solved tensor back into TreeTN using factorize_tensor_to_treetn
        let mut factorize_options = FactorizeOptions::svd();
        if let Some(max_bond_dim) = self.options.truncation.max_bond_dim() {
            factorize_options = factorize_options.with_max_bond_dim(max_bond_dim);
        }
        if let Some(policy) = self.options.truncation.svd_policy() {
            factorize_options = factorize_options.with_svd_policy(policy);
        }
        // Force decomposition root to be consistent with the sweep plan's new center.
        // This keeps the norm-carrying tensor on the declared canonical center.
        let decomposed = factorize_tensor_to_treetn_with(
            &solved_local,
            &topology,
            factorize_options,
            &step.new_center,
        )?;

        // Copy decomposed tensors back to subtree, preserving original bond IDs
        self.copy_decomposed_to_subtree(&mut subtree, &decomposed, &step.nodes, full_treetn)?;

        // Force-move the canonical region metadata AND the bond ortho directions to the new center.
        // (apply_local_update_sweep also updates canonical_region on the full TreeTN, but we keep
        // the subtree self-consistent here.)
        subtree.set_canonical_region([step.new_center.clone()])?;
        if let Some(edges) = subtree.edges_to_canonicalize_by_names(&step.new_center) {
            for (from, to) in edges {
                if let Some(edge) = subtree.edge_between(&from, &to) {
                    subtree.set_edge_ortho_towards(edge, Some(to))?;
                }
            }
        }

        Ok(subtree)
    }

    fn after_step(
        &mut self,
        step: &LocalUpdateStep<V>,
        full_treetn_after: &TreeTN<T, V>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        // Use state's SiteIndexNetwork directly (implements NetworkTopology)
        let topology = full_treetn_after.site_index_network();

        // Synchronize reference_state with ket_state (full_treetn_after) for the updated region
        // while preserving boundary bond IDs for cache consistency
        self.sync_reference_state_region(step, full_treetn_after)?;

        // Invalidate all caches affected by the updated region
        {
            let mut proj_op = self
                .projected_operator
                .write()
                .map_err(|e| anyhow::anyhow!("Failed to acquire write lock: {}", e))
                .context("after_step: lock poisoned")?;
            proj_op.invalidate(&step.nodes, topology);
        }
        self.projected_state.invalidate(&step.nodes, topology);

        Ok(())
    }
}

impl<T, V> SquareLinsolveUpdater<T, V>
where
    T: TensorLike + 'static,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    fn index_sets_match(&self, init_indices: &[T::Index], rhs_indices: &[T::Index]) -> bool {
        if init_indices.len() != rhs_indices.len() {
            return false;
        }
        let init_keys: std::collections::HashSet<_> =
            init_indices.iter().map(|i| (i.clone(), i.dim())).collect();
        let rhs_keys: std::collections::HashSet<_> =
            rhs_indices.iter().map(|i| (i.clone(), i.dim())).collect();
        init_keys == rhs_keys
    }

    fn index_structure_mismatch_message(
        &self,
        init_indices: &[T::Index],
        rhs_indices: &[T::Index],
        header: &str,
        footer: &str,
    ) -> String {
        let init_keys: std::collections::HashSet<_> =
            init_indices.iter().map(|i| (i.clone(), i.dim())).collect();
        let rhs_keys: std::collections::HashSet<_> =
            rhs_indices.iter().map(|i| (i.clone(), i.dim())).collect();
        let extra_in_rhs: Vec<_> = rhs_keys
            .difference(&init_keys)
            .map(|(idx, dim)| format!("{idx:?}:{dim}"))
            .collect();
        let missing_in_rhs: Vec<_> = init_keys
            .difference(&rhs_keys)
            .map(|(idx, dim)| format!("{idx:?}:{dim}"))
            .collect();

        format!(
            "{header}:\n  init has {} indices: {:?}\n  rhs has {} indices: {:?}\n  extra in rhs (not in init): {:?}\n  missing in rhs (in init but not in rhs): {:?}\n\n{footer}",
            init_indices.len(),
            init_indices
                .iter()
                .map(|i| format!("{:?}:{}", i.id(), i.dim()))
                .collect::<Vec<_>>(),
            rhs_indices.len(),
            rhs_indices
                .iter()
                .map(|i| format!("{:?}:{}", i.id(), i.dim()))
                .collect::<Vec<_>>(),
            extra_in_rhs,
            missing_in_rhs,
        )
    }

    fn replace_reference_boundary_bonds_with_state(
        &self,
        tensor: T,
        region: &[V],
        state: &TreeTN<T, V>,
    ) -> Result<T> {
        let mut result = tensor;
        for node in region {
            for neighbor in state.site_index_network().neighbors(node) {
                if region.contains(&neighbor) {
                    continue;
                }

                let Some(state_edge) = state.edge_between(node, &neighbor) else {
                    continue;
                };
                let Some(reference_edge) = self.reference_state.edge_between(node, &neighbor)
                else {
                    continue;
                };
                let Some(state_bond) = state.bond_index(state_edge) else {
                    continue;
                };
                let Some(reference_bond) = self.reference_state.bond_index(reference_edge) else {
                    continue;
                };

                if reference_bond == state_bond {
                    continue;
                }
                if result
                    .external_indices()
                    .iter()
                    .any(|index| index == reference_bond)
                {
                    result = result.replaceind(reference_bond, state_bond)?;
                }
            }
        }
        Ok(result)
    }

    fn precheck_ref_bra_ket_convention(
        &mut self,
        step: &LocalUpdateStep<V>,
        full_treetn_before: &TreeTN<T, V>,
    ) -> Result<()> {
        let subtree = full_treetn_before.extract_subtree(&step.nodes)?;
        let init_local = self.contract_region(&subtree, &step.nodes)?;

        let topology = full_treetn_before.site_index_network();
        let rhs_local_raw = self.projected_state.local_constant_term(
            &step.nodes,
            &self.reference_state,
            topology,
        )?;
        let rhs_local_raw = self.replace_reference_boundary_bonds_with_state(
            rhs_local_raw,
            &step.nodes,
            full_treetn_before,
        )?;

        let init_indices = init_local.external_indices();
        let rhs_indices = rhs_local_raw.external_indices();

        if !self.index_sets_match(&init_indices, &rhs_indices) {
            return Err(anyhow::anyhow!(
                "{}",
                self.index_structure_mismatch_message(
                    &init_indices,
                    &rhs_indices,
                    "linsolve precheck failed (local index structure mismatch)",
                    "This suggests `<ref|H|x>` vs `<ref|b>` conventions (or external-index contraction rules) are inconsistent for the current region. See `docs/design/treetn-linsolve-mpo.md` and the [`ProjectedState`](crate::linsolve::square::ProjectedState) rustdoc for analysis.",
                )
            ));
        }

        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn validate_mpo_external_indices(&mut self, state: &TreeTN<T, V>) -> Result<()> {
        // Only validate when operator mappings exist (MPO-with-mappings path).
        let (input_mapping, output_mapping): (
            HashMap<V, IndexMapping<T::Index>>,
            HashMap<V, IndexMapping<T::Index>>,
        ) = {
            let proj_op = self.projected_operator.read().map_err(|e| {
                anyhow::anyhow!("validate_mpo_external_indices: lock poisoned: {e}")
            })?;
            let Some(input) = proj_op.input_mapping() else {
                return Ok(());
            };
            let Some(output) = proj_op.output_mapping() else {
                return Ok(());
            };
            (input.clone(), output.clone())
        };

        for node in state.node_names() {
            let Some(x_sites) = state.site_space(&node) else {
                continue;
            };
            let Some(b_sites) = self.projected_state.rhs.site_space(&node) else {
                continue;
            };

            // Only apply strict MPO validation when both look like MPO (2 site indices).
            if x_sites.len() != 2 || b_sites.len() != 2 {
                continue;
            }

            let x_contracted = input_mapping
                .get(&node)
                .ok_or_else(|| {
                    anyhow::anyhow!("MPO validation: missing input_mapping for node {:?}", node)
                })?
                .true_index
                .clone();
            let b_contracted = output_mapping
                .get(&node)
                .ok_or_else(|| {
                    anyhow::anyhow!("MPO validation: missing output_mapping for node {:?}", node)
                })?
                .true_index
                .clone();

            let x_external: Vec<_> = x_sites
                .iter()
                .filter(|idx| *idx != &x_contracted)
                .cloned()
                .collect();
            let b_external: Vec<_> = b_sites
                .iter()
                .filter(|idx| *idx != &b_contracted)
                .cloned()
                .collect();

            if x_external.len() != 1 || b_external.len() != 1 {
                return Err(anyhow::anyhow!(
                    "MPO validation: expected exactly 1 external site index after removing contracted index. node={:?}, x_site_len={}, b_site_len={}, x_external={:?}, b_external={:?}",
                    node,
                    x_sites.len(),
                    b_sites.len(),
                    x_external.iter().map(|i| format!("{:?}:{}", i.id(), i.dim())).collect::<Vec<_>>(),
                    b_external.iter().map(|i| format!("{:?}:{}", i.id(), i.dim())).collect::<Vec<_>>(),
                ));
            }

            let x_ext = &x_external[0];
            let b_ext = &b_external[0];
            if x_ext != b_ext {
                return Err(anyhow::anyhow!(
                    "MPO validation: external index mismatch at node {:?}: x has {:?}:{}, b has {:?}:{}",
                    node,
                    x_ext,
                    x_ext.dim(),
                    b_ext,
                    b_ext.dim(),
                ));
            }
        }

        Ok(())
    }
}

fn local_gmres_options(options: &LinsolveOptions) -> Result<GmresOptions> {
    if options.gmres_restart_dim == 0 {
        anyhow::bail!("LinsolveOptions::gmres_restart_dim must be greater than zero");
    }
    if options.gmres_max_restarts == 0 {
        anyhow::bail!("LinsolveOptions::gmres_max_restarts must be greater than zero");
    }

    Ok(GmresOptions {
        max_iter: options.gmres_restart_dim,
        rtol: options.gmres_tol,
        max_restarts: options.gmres_max_restarts,
        verbose: false,
        check_true_residual: true,
    })
}

#[cfg(test)]
mod tests;
