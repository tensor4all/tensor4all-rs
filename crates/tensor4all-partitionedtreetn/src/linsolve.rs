//! Patched linear solver for `(a0 + a1 A) x = b`.
//!
//! Routes automatically from the operator's projectors before any numerics:
//! - [`LinsolveRoute::PatchLocal`] when no operator block connects two distinct
//!   state patches (solves each patch independently with
//!   [`tensor4all_treetn::square_linsolve`]).
//! - [`LinsolveRoute::GlobalKrylov`] otherwise (runs
//!   [`tensor4all_core::krylov::gmres_with_truncation`] over the whole
//!   [`PartitionedTreeTN`]).

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::krylov::{gmres_with_truncation, GmresOptions};
use tensor4all_core::{AnyScalar, DynIndex, TensorIndex, TensorVectorSpace};
use tensor4all_treetn::{
    contraction::ContractionOptions, square_linsolve, IndexMapping, LinsolveOptions,
};

use crate::error::{PartitionedTreeTNError, Result};
use crate::partitioned_tree_tn::PartitionedTreeTN;
use crate::projector::Projector;
use crate::subdomain_tree_tn::SubDomainTreeTN;

/// The route [`partitioned_linsolve`] took, chosen from the operator's
/// projectors before any numerical work.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::LinsolveRoute;
///
/// assert_ne!(LinsolveRoute::PatchLocal, LinsolveRoute::GlobalKrylov);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinsolveRoute {
    /// No operator block connects two distinct state patches; each supported
    /// patch is solved independently with `square_linsolve`.
    PatchLocal,
    /// At least one operator block couples two distinct state patches; the
    /// whole partitioned system is solved with `gmres_with_truncation`.
    ///
    /// This route carries two costs: it forms one truncated sum per Krylov
    /// vector, and convergence is not certified.
    GlobalKrylov,
}

/// Result of [`partitioned_linsolve`].
///
/// `residual` is always the relative residual of the whole coupled system, not
/// the maximum of per-patch residuals. `patch_residuals` and `sweeps` are
/// populated on [`LinsolveRoute::PatchLocal`]; `krylov_iterations` is populated
/// on [`LinsolveRoute::GlobalKrylov`].
#[derive(Clone, Debug)]
pub struct PartitionedLinsolveResult<V = usize>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Computed solution partition. Patches with neither an operator block nor
    /// a right-hand-side block are omitted.
    pub solution: PartitionedTreeTN<V>,
    /// Whether the solver reported convergence within the requested tolerance.
    pub converged: bool,
    /// Relative residual `‖(a0 + a1 A) x - b‖ / ‖b‖` of the whole coupled
    /// system (or `0.0` when `‖b‖ == 0`).
    pub residual: f64,
    /// Per-patch relative residuals on [`LinsolveRoute::PatchLocal`] (empty on
    /// [`LinsolveRoute::GlobalKrylov`]).
    pub patch_residuals: Vec<(Projector, f64)>,
    /// Per-patch sweep counts on [`LinsolveRoute::PatchLocal`] (empty on
    /// [`LinsolveRoute::GlobalKrylov`]).
    pub sweeps: Vec<(Projector, usize)>,
    /// Total Krylov iterations on [`LinsolveRoute::GlobalKrylov`] (`0` on
    /// [`LinsolveRoute::PatchLocal`]).
    pub krylov_iterations: usize,
    /// Route selected from the operator projectors.
    pub route: LinsolveRoute,
}

/// Extract `(input_state_projector, output_state_projector)` for an operator
/// patch from its projector and the optional index mappings.
fn operator_block_endpoints<V>(
    op_proj: &Projector,
    input_mapping: Option<&HashMap<V, IndexMapping<DynIndex>>>,
    output_mapping: Option<&HashMap<V, IndexMapping<DynIndex>>>,
) -> Result<(Projector, Projector)> {
    match (input_mapping, output_mapping) {
        (Some(in_map), Some(out_map)) => {
            let mut p_in = Projector::new();
            let mut p_out = Projector::new();
            for m in in_map.values() {
                if let Some(val) = op_proj.get(&m.internal_index) {
                    p_in.insert(m.true_index.clone(), val)?;
                }
            }
            for m in out_map.values() {
                if let Some(val) = op_proj.get(&m.internal_index) {
                    p_out.insert(m.true_index.clone(), val)?;
                }
            }
            Ok((p_in, p_out))
        }
        (None, None) => Ok((op_proj.clone(), op_proj.clone())),
        _ => Err(PartitionedTreeTNError::InvalidOptions {
            operation: "partitioned_linsolve",
            reason: "input_mapping and output_mapping must both be Some or both be None",
        }),
    }
}

fn reindex_partition<V>(
    part: &PartitionedTreeTN<V>,
    mapping: &HashMap<V, IndexMapping<DynIndex>>,
    to_internal: bool,
) -> Result<PartitionedTreeTN<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    let mut current = part.clone();
    for m in mapping.values() {
        let (from, to) = if to_internal {
            (&m.true_index, &m.internal_index)
        } else {
            (&m.internal_index, &m.true_index)
        };
        if from != to {
            current =
                current
                    .replaceind(from, to)
                    .map_err(|err| PartitionedTreeTNError::TreeTN {
                        source: anyhow::Error::new(err).into(),
                    })?;
        }
    }
    Ok(current)
}

fn contraction_options_from_linsolve(options: &LinsolveOptions) -> ContractionOptions {
    let mut c = ContractionOptions::default();
    if let Some(max_bond_dim) = options.truncation.max_bond_dim() {
        c = c.with_max_bond_dim(max_bond_dim);
    }
    if let Some(policy) = options.truncation.svd_policy() {
        c = c.with_svd_policy(policy);
    }
    c
}

/// Solve `(a0 + a1 A) x = b` for patched `operator` `A`, right-hand side `rhs`
/// `b`, and initial guess `init`.
///
/// The route is chosen from the operator's projectors before any numerics:
/// - If every operator block has identical input and output state projectors,
///   [`LinsolveRoute::PatchLocal`] solves each supported patch independently
///   with [`square_linsolve`], never forming or visiting absent operator blocks.
/// - Otherwise, [`LinsolveRoute::GlobalKrylov`] runs [`gmres_with_truncation`]
///   on the whole partitioned network using `options.gmres_tol`,
///   `options.gmres_restart_dim`, `options.gmres_max_restarts`, and
///   `options.truncation`. On the global route, `nfullsweeps`, `nsite`,
///   `enrichment`, and `local_solver` are unused. The global route carries two
///   costs: one truncated sum per Krylov vector, and convergence is not
///   certified.
///
/// On both routes, [`PartitionedLinsolveResult::residual`] is the residual of
/// the whole coupled system, not the maximum of per-patch residuals. No
/// solution patch is created where there is neither an operator block nor a
/// right-hand-side block.
///
/// # Errors
///
/// Returns [`PartitionedTreeTNError::Empty`] if `rhs` is empty,
/// [`PartitionedTreeTNError::InvalidOptions`] when mappings are half-specified
/// or `a0 == 0` on a patch with no operator block, or
/// [`PartitionedTreeTNError::TreeTN`] (preserving the underlying
/// `TreeTNOperationError` as source) when a per-patch solve or Krylov operation
/// fails.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use tensor4all_core::{AnyScalar, DynIndex, IdxTensor};
/// use tensor4all_partitionedtreetn::{
///     partitioned_linsolve, LinsolveRoute, PartitionedTreeTN, Projector, SubDomainTreeTN,
/// };
/// use tensor4all_treetn::{IndexMapping, LinsolveOptions, TreeTN};
///
/// let s0 = DynIndex::new_dyn(2);
/// let s1 = DynIndex::new_dyn(2);
/// let s0_in = DynIndex::new_dyn(2);
/// let s0_out = DynIndex::new_dyn(2);
/// let s1_in = DynIndex::new_dyn(2);
/// let s1_out = DynIndex::new_dyn(2);
/// let bond = DynIndex::new_dyn(1);
///
/// let mut rhs_tree = TreeTN::<IdxTensor, usize>::new();
/// let n0 = rhs_tree.add_tensor(0, IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0_f64, 0.0])?)?;
/// let n1 = rhs_tree.add_tensor(1, IdxTensor::from_dense(vec![bond.clone(), s1.clone()], vec![2.0_f64, 4.0])?)?;
/// rhs_tree.connect(n0, &bond, n1, &bond)?;
/// let p0 = Projector::from_pairs([(s0.clone(), 0)])?;
/// let rhs = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::new(rhs_tree.clone(), p0.clone())?)?;
///
/// let op_bond = DynIndex::new_dyn(1);
/// let mut op_tree = TreeTN::<IdxTensor, usize>::new();
/// let m0 = op_tree.add_tensor(0, IdxTensor::from_dense(vec![s0_out.clone(), s0_in.clone(), op_bond.clone()], vec![1.0_f64, 0.0, 0.0, 0.0])?)?;
/// let m1 = op_tree.add_tensor(1, IdxTensor::from_dense(vec![op_bond.clone(), s1_out.clone(), s1_in.clone()], vec![1.0_f64, 0.0, 0.0, 1.0])?)?;
/// op_tree.connect(m0, &op_bond, m1, &op_bond)?;
/// let op = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::new(
///     op_tree,
///     Projector::from_pairs([(s0_out.clone(), 0), (s0_in.clone(), 0)])?,
/// )?)?;
///
/// let mut in_map = HashMap::new();
/// in_map.insert(0, IndexMapping { true_index: s0.clone(), internal_index: s0_in });
/// in_map.insert(1, IndexMapping { true_index: s1.clone(), internal_index: s1_in });
/// let mut out_map = HashMap::new();
/// out_map.insert(0, IndexMapping { true_index: s0, internal_index: s0_out });
/// out_map.insert(1, IndexMapping { true_index: s1, internal_index: s1_out });
///
/// let init = rhs.scale(AnyScalar::new_real(0.0))?;
/// let opts = LinsolveOptions::default()
///     .with_coefficients(AnyScalar::new_real(1.0), AnyScalar::new_real(1.0))
///     .with_nfullsweeps(3);
/// let res = partitioned_linsolve(&op, &rhs, init, &0usize, opts, Some(in_map), Some(out_map))?;
/// assert_eq!(res.route, LinsolveRoute::PatchLocal);
/// assert!(res.residual < 1.0e-8);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn partitioned_linsolve<V>(
    operator: &PartitionedTreeTN<V>,
    rhs: &PartitionedTreeTN<V>,
    init: PartitionedTreeTN<V>,
    center: &V,
    options: LinsolveOptions,
    input_mapping: Option<HashMap<V, IndexMapping<DynIndex>>>,
    output_mapping: Option<HashMap<V, IndexMapping<DynIndex>>>,
) -> Result<PartitionedLinsolveResult<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug + 'static,
{
    if input_mapping.is_some() != output_mapping.is_some() {
        return Err(PartitionedTreeTNError::InvalidOptions {
            operation: "partitioned_linsolve",
            reason: "input_mapping and output_mapping must both be Some or both be None",
        });
    }
    if rhs.is_empty() {
        return Err(PartitionedTreeTNError::Empty);
    }

    // Inspect operator incidence from projectors alone, before any numerics.
    let mut is_coupled = false;
    let mut diag_blocks: HashMap<Projector, &SubDomainTreeTN<V>> = HashMap::new();
    let mut op_output_projectors: Vec<Projector> = Vec::new();
    let rhs_projectors: HashSet<Projector> = rhs.iter().map(|(p, _)| p.clone()).collect();

    for (op_proj, op_patch) in operator {
        let (p_in, p_out) =
            operator_block_endpoints(op_proj, input_mapping.as_ref(), output_mapping.as_ref())?;
        op_output_projectors.push(p_out.clone());
        if p_in != p_out {
            is_coupled = true;
        } else {
            // If an operator block overlaps a rhs patch without matching its
            // exact projector key (e.g., an unprojected or coarser operator
            // block), route to GlobalKrylov so patched contraction handles it.
            let overlaps_nonexact = rhs_projectors
                .iter()
                .any(|rp| rp != &p_out && p_out.intersection(rp).is_some());
            if overlaps_nonexact {
                is_coupled = true;
            } else {
                diag_blocks.insert(p_out, op_patch);
            }
        }
    }

    if !is_coupled {
        solve_patch_local(
            &diag_blocks,
            rhs,
            init,
            center,
            options,
            input_mapping,
            output_mapping,
        )
    } else {
        solve_global_krylov(
            operator,
            rhs,
            init,
            &rhs_projectors,
            &op_output_projectors,
            center,
            options,
            input_mapping,
            output_mapping,
        )
    }
}

fn solve_patch_local<V>(
    diag_blocks: &HashMap<Projector, &SubDomainTreeTN<V>>,
    rhs: &PartitionedTreeTN<V>,
    init: PartitionedTreeTN<V>,
    center: &V,
    options: LinsolveOptions,
    input_mapping: Option<HashMap<V, IndexMapping<DynIndex>>>,
    output_mapping: Option<HashMap<V, IndexMapping<DynIndex>>>,
) -> Result<PartitionedLinsolveResult<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug + 'static,
{
    let b_norm = rhs.norm()?;
    let mut solution_patches = Vec::new();
    let mut patch_residuals = Vec::new();
    let mut sweeps = Vec::new();
    let mut all_converged = true;
    let mut residual_abs = 0.0_f64;

    for (proj, rhs_patch) in rhs {
        let bp_norm = rhs_patch.norm()?;
        if let Some(op_patch) = diag_blocks.get(proj) {
            let init_tree = match init.get(proj) {
                Some(p) => p.data().clone(),
                None => rhs_patch
                    .data()
                    .scale(AnyScalar::new_real(0.0))
                    .map_err(PartitionedTreeTNError::from)?,
            };
            let mut patch_options = options.clone();
            patch_options.check_residual = true;
            let res = square_linsolve(
                op_patch.data(),
                rhs_patch.data(),
                init_tree,
                center,
                patch_options,
                input_mapping.clone(),
                output_mapping.clone(),
            )
            .map_err(PartitionedTreeTNError::from)?;
            let patch_res = res.residual.unwrap_or(f64::NAN);
            residual_abs = residual_abs.hypot(patch_res * bp_norm);
            all_converged &= res.converged;
            patch_residuals.push((proj.clone(), patch_res));
            sweeps.push((proj.clone(), res.sweeps));
            solution_patches.push(SubDomainTreeTN::new(res.solution, proj.clone())?);
        } else {
            // Absent operator block is zero: equation reduces to a0 * x_p = b_p.
            if options.a0.is_zero() {
                return Err(PartitionedTreeTNError::InvalidOptions {
                    operation: "partitioned_linsolve",
                    reason: "a0 is zero on a patch with no operator block; system is singular",
                });
            }
            let inv_a0 = AnyScalar::new_real(1.0) / options.a0.clone();
            let sol_tree = rhs_patch
                .data()
                .scale(inv_a0)
                .map_err(PartitionedTreeTNError::from)?;
            patch_residuals.push((proj.clone(), 0.0));
            sweeps.push((proj.clone(), 0));
            solution_patches.push(SubDomainTreeTN::new(sol_tree, proj.clone())?);
        }
    }

    let solution = PartitionedTreeTN::from_subdomains(solution_patches)?;
    let residual = if b_norm == 0.0 {
        0.0
    } else {
        residual_abs / b_norm
    };

    Ok(PartitionedLinsolveResult {
        solution,
        converged: all_converged,
        residual,
        patch_residuals,
        sweeps,
        krylov_iterations: 0,
        route: LinsolveRoute::PatchLocal,
    })
}

#[allow(clippy::too_many_arguments)]
fn solve_global_krylov<V>(
    operator: &PartitionedTreeTN<V>,
    rhs: &PartitionedTreeTN<V>,
    init: PartitionedTreeTN<V>,
    rhs_projectors: &HashSet<Projector>,
    op_output_projectors: &[Projector],
    center: &V,
    options: LinsolveOptions,
    input_mapping: Option<HashMap<V, IndexMapping<DynIndex>>>,
    output_mapping: Option<HashMap<V, IndexMapping<DynIndex>>>,
) -> Result<PartitionedLinsolveResult<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug + 'static,
{
    // Restrict initial guess to supported output patches so unsupported patches
    // never enter the Krylov subspace or the returned solution.
    let mut init_patches: Vec<SubDomainTreeTN<V>> = init
        .into_iter()
        .filter(|(proj, _)| {
            rhs_projectors.contains(proj)
                || op_output_projectors
                    .iter()
                    .any(|p_out| p_out.intersection(proj).is_some())
        })
        .map(|(_, patch)| patch)
        .collect();
    if init_patches.is_empty() {
        for (proj, rhs_patch) in rhs {
            let zero_tree = rhs_patch
                .data()
                .scale(AnyScalar::new_real(0.0))
                .map_err(PartitionedTreeTNError::from)?;
            init_patches.push(SubDomainTreeTN::new(zero_tree, proj.clone())?);
        }
    }
    let init_supported = PartitionedTreeTN::from_subdomains(init_patches)?;

    let c_opts = contraction_options_from_linsolve(&options);
    let trunc_opts = options.truncation;
    let a0 = options.a0.clone();
    let a1 = options.a1.clone();

    let apply_a = |x: &PartitionedTreeTN<V>| -> anyhow::Result<PartitionedTreeTN<V>> {
        if a1.is_zero() {
            return Ok(x.scale(a0.clone())?);
        }
        let x_in = match input_mapping.as_ref() {
            Some(map) => reindex_partition(x, map, true)?,
            None => x.clone(),
        };
        let ax_out = operator.contract(&x_in, center, c_opts.clone())?;
        let ax = match output_mapping.as_ref() {
            Some(map) => reindex_partition(&ax_out, map, false)?,
            None => ax_out,
        };
        if a0.is_zero() {
            Ok(ax.scale(a1.clone())?)
        } else {
            Ok(x.axpby_truncated(a0.clone(), &ax, a1.clone(), center, trunc_opts)?)
        }
    };

    let truncate_fn = |x: &mut PartitionedTreeTN<V>| -> anyhow::Result<()> {
        let mut truncated_patches = Vec::with_capacity(x.len());
        for (_, patch) in x.iter() {
            let mut p = patch.clone();
            p.truncate(center, trunc_opts)?;
            truncated_patches.push(p);
        }
        *x = PartitionedTreeTN::from_subdomains(truncated_patches)?;
        Ok(())
    };

    let gmres_opts = GmresOptions {
        max_iter: options.gmres_restart_dim,
        rtol: options.gmres_tol,
        max_restarts: options.gmres_max_restarts,
        verbose: false,
        check_true_residual: true,
    };

    let gmres_res = gmres_with_truncation(
        |x| apply_a(x),
        rhs,
        &init_supported,
        &gmres_opts,
        truncate_fn,
    )
    .map_err(|err| PartitionedTreeTNError::TreeTN {
        source: anyhow::Error::new(err).into(),
    })?;

    // Compute true residual of the whole coupled system.
    let b_norm = rhs.norm()?;
    let residual = if b_norm == 0.0 {
        0.0
    } else {
        let ax = apply_a(&gmres_res.solution).map_err(|err| {
            match err.downcast::<PartitionedTreeTNError>() {
                Ok(e) => e,
                Err(other) => PartitionedTreeTNError::TreeTN {
                    source: other.into(),
                },
            }
        })?;
        let diff = ax
            .axpby(AnyScalar::new_real(1.0), rhs, AnyScalar::new_real(-1.0))
            .map_err(|err| PartitionedTreeTNError::TreeTN {
                source: anyhow::Error::new(err).into(),
            })?;
        diff.norm()? / b_norm
    };

    Ok(PartitionedLinsolveResult {
        solution: gmres_res.solution,
        converged: gmres_res.converged,
        residual,
        patch_residuals: Vec::new(),
        sweeps: Vec::new(),
        krylov_iterations: gmres_res.iterations,
        route: LinsolveRoute::GlobalKrylov,
    })
}
