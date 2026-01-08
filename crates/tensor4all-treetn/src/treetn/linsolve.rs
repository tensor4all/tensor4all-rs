//! Linear equation solver for Tree Tensor Networks.
//!
//! This module provides the `linsolve` function for solving linear systems
//! of the form `(a₀ + a₁ * A) * x = b` where A is a TTN operator and x, b are TTN states.
//!
//! # Algorithm
//!
//! The algorithm uses alternating updates (sweeping) similar to DMRG:
//! 1. Position environments to expose a local region
//! 2. Solve the local linear problem using GMRES (via kryst)
//! 3. Factorize the result and move the orthogonality center
//! 4. Update environment caches
//!
//! # Inspired By
//!
//! This implementation is inspired by:
//! - [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) - Core algorithm structure
//! - [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl) - Krylov solver integration pattern
//! - [kryst](https://github.com/tmathis720/kryst) - Rust GMRES implementation
//!
//! # References
//!
//! - Phys. Rev. B 72, 180403 (2005) - Noise term technique (not implemented in initial version)

mod environment;
mod linear_operator;
mod local_linop;
mod options;
mod projected_operator;
mod projected_state;
mod updater;

pub use environment::{EnvironmentCache, NetworkTopology};
pub use linear_operator::{IndexMapping, LinearOperator};
pub use options::LinsolveOptions;
pub use projected_operator::ProjectedOperator;
pub use projected_state::ProjectedState;
pub use updater::{LinsolveUpdater, LinsolveVerifyReport, NodeVerifyDetail};

use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::index::{DynId, NoSymmSpace, Symmetry};

use super::localupdate::{apply_local_update_sweep, LocalUpdateSweepPlan};
use super::TreeTN;
use crate::CanonicalizationOptions;

/// Result of linsolve operation.
#[derive(Debug, Clone)]
pub struct LinsolveResult<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    Symm: Clone + Symmetry + std::fmt::Debug,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// The solution TreeTN
    pub solution: TreeTN<Id, Symm, V>,
    /// Number of sweeps performed
    pub sweeps: usize,
    /// Final residual norm (if computed)
    pub residual: Option<f64>,
    /// Converged flag
    pub converged: bool,
}

/// Validate that operator, rhs, and init have compatible structures for linsolve.
///
/// Checks:
/// 1. Operator can act on init (same topology)
/// 2. Result of operator action has compatible site dimensions with rhs
fn validate_linsolve_inputs<Id, Symm, V>(
    operator: &TreeTN<Id, Symm, V>,
    rhs: &TreeTN<Id, Symm, V>,
    init: &TreeTN<Id, Symm, V>,
) -> Result<()>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug,
    Symm: Clone + Symmetry + PartialEq + std::fmt::Debug,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let init_network = init.site_index_network();
    let op_network = operator.site_index_network();
    let rhs_network = rhs.site_index_network();

    // Check 1: Operator can act on init
    let result_network = init_network
        .apply_operator_topology(op_network)
        .map_err(|e| anyhow::anyhow!("Operator cannot act on init: {}", e))?;

    // Check 2: Result has compatible dimensions with rhs
    if !result_network.compatible_site_dimensions(rhs_network) {
        return Err(anyhow::anyhow!(
            "Result of operator action is not compatible with RHS"
        ));
    }

    Ok(())
}

/// Solve the linear system `(a₀ + a₁ * H) |x⟩ = |b⟩` for TreeTN.
///
/// # Arguments
///
/// * `operator` - The operator H as a TreeTN (must have compatible structure with `rhs`)
/// * `rhs` - The right-hand side |b⟩ as a TreeTN
/// * `init` - Initial guess for |x⟩
/// * `center` - Node to use as sweep center
/// * `options` - Solver options
///
/// # Returns
///
/// The solution TreeTN, or an error if solving fails.
///
/// # Example
///
/// ```ignore
/// let solution = linsolve(&h_mpo, &b_mps, x0_mps, "center", LinsolveOptions::default())?;
/// ```
pub fn linsolve<Id, Symm, V>(
    operator: &TreeTN<Id, Symm, V>,
    rhs: &TreeTN<Id, Symm, V>,
    init: TreeTN<Id, Symm, V>,
    center: &V,
    options: LinsolveOptions,
) -> Result<LinsolveResult<Id, Symm, V>>
where
    Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + From<DynId> + Send + Sync + 'static,
    Symm:
        Clone + Symmetry + From<NoSymmSpace> + PartialEq + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    // Validate inputs before proceeding
    validate_linsolve_inputs(operator, rhs, &init)?;

    // Canonicalize initial guess towards center
    let mut x = init.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    // Create LinsolveUpdater
    let mut updater = LinsolveUpdater::new(operator.clone(), rhs.clone(), options.clone());

    // Create sweep plan (nsite=2 for 2-site updates)
    let plan = LocalUpdateSweepPlan::from_treetn(&x, center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create sweep plan"))?;

    let converged = false;
    let mut final_sweeps = 0;

    // Perform sweeps
    for sweep in 0..options.nsweeps {
        final_sweeps = sweep + 1;

        // TODO: Compute residual for convergence check
        // For now, just run all sweeps

        apply_local_update_sweep(&mut x, &plan, &mut updater)?;

        // Early termination check would go here
        if let Some(_tol) = options.convergence_tol {
            // TODO: Compute ||Hx - b|| / ||b|| and check convergence
        }
    }

    Ok(LinsolveResult {
        solution: x,
        sweeps: final_sweeps,
        residual: None,
        converged,
    })
}
