//! Linear equation solver for Tree Tensor Networks (V_in = V_out case).
//!
//! This module provides the `square_linsolve` function for solving linear systems
//! of the form `(a₀ + a₁ * A) * x = b` where A is a TTN operator and x, b are TTN states,
//! and the input and output spaces are the same (V_in = V_out).
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

mod local_linop;
mod projected_state;
mod updater;

pub use projected_state::ProjectedState;
pub use updater::{LinsolveVerifyReport, NodeVerifyDetail, SquareLinsolveUpdater};

use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::TensorLike;

use crate::linsolve::common::LinsolveOptions;
use crate::{apply_local_update_sweep, CanonicalizationOptions, LocalUpdateSweepPlan, TreeTN};

/// Result of square_linsolve operation.
#[derive(Debug, Clone)]
pub struct SquareLinsolveResult<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// The solution TreeTN
    pub solution: TreeTN<T, V>,
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
fn validate_linsolve_inputs<T, V>(
    operator: &TreeTN<T, V>,
    rhs: &TreeTN<T, V>,
    init: &TreeTN<T, V>,
) -> Result<()>
where
    T: TensorLike,
    <T::Index as tensor4all_core::IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
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
/// This solver is for the square case where V_in = V_out (input and output spaces are the same).
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
/// let solution = square_linsolve(&h_mpo, &b_mps, x0_mps, "center", LinsolveOptions::default())?;
/// ```
pub fn square_linsolve<T, V>(
    operator: &TreeTN<T, V>,
    rhs: &TreeTN<T, V>,
    init: TreeTN<T, V>,
    center: &V,
    options: LinsolveOptions,
) -> Result<SquareLinsolveResult<T, V>>
where
    T: TensorLike + 'static,
    <T::Index as tensor4all_core::IndexLike>::Id:
        Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync + 'static,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug + 'static,
{
    // Validate inputs before proceeding
    validate_linsolve_inputs(operator, rhs, &init)?;

    // Canonicalize initial guess towards center
    let mut x = init.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    // Create SquareLinsolveUpdater
    let mut updater = SquareLinsolveUpdater::new(operator.clone(), rhs.clone(), options.clone());

    // Create sweep plan (nsite=2 for 2-site updates)
    let plan = LocalUpdateSweepPlan::from_treetn(&x, center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create sweep plan"))?;

    let mut final_sweeps = 0;

    // Perform sweeps
    for sweep in 0..options.nfullsweeps {
        final_sweeps = sweep + 1;
        apply_local_update_sweep(&mut x, &plan, &mut updater)?;
    }

    // Note: Residual computation (||Hx - b|| / ||b||) and convergence checking
    // are not yet implemented. Currently, all requested sweeps are performed.
    Ok(SquareLinsolveResult {
        solution: x,
        sweeps: final_sweeps,
        residual: None,
        converged: false,
    })
}
