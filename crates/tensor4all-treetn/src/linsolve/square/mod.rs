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

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::{DynIndex, TensorDynLen};

    fn create_simple_2site_mps() -> TreeTN<TensorDynLen, String> {
        let mut mps = TreeTN::<TensorDynLen, String>::new();
        let s0 = DynIndex::new_dyn(2);
        let s1 = DynIndex::new_dyn(2);
        let b01 = DynIndex::new_dyn(2);

        let t0 = TensorDynLen::from_dense_f64(vec![s0.clone(), b01.clone()], vec![1.0; 4]);
        let t1 = TensorDynLen::from_dense_f64(vec![b01.clone(), s1.clone()], vec![1.0; 4]);

        let n0 = mps.add_tensor("site0".to_string(), t0).unwrap();
        let n1 = mps.add_tensor("site1".to_string(), t1).unwrap();
        mps.connect(n0, &b01, n1, &b01).unwrap();

        mps
    }

    fn create_simple_2site_mpo() -> TreeTN<TensorDynLen, String> {
        let mut mpo = TreeTN::<TensorDynLen, String>::new();
        let s0_in = DynIndex::new_dyn(2);
        let s0_out = DynIndex::new_dyn(2);
        let s1_in = DynIndex::new_dyn(2);
        let s1_out = DynIndex::new_dyn(2);
        let b_mpo = DynIndex::new_dyn(1);

        let id_data = vec![1.0, 0.0, 0.0, 1.0]; // Identity matrix
        let t0_mpo = TensorDynLen::from_dense_f64(
            vec![s0_out.clone(), s0_in.clone(), b_mpo.clone()],
            id_data.clone(),
        );
        let t1_mpo = TensorDynLen::from_dense_f64(
            vec![b_mpo.clone(), s1_out.clone(), s1_in.clone()],
            id_data,
        );

        let n0_mpo = mpo.add_tensor("site0".to_string(), t0_mpo).unwrap();
        let n1_mpo = mpo.add_tensor("site1".to_string(), t1_mpo).unwrap();
        mpo.connect(n0_mpo, &b_mpo, n1_mpo, &b_mpo).unwrap();

        mpo
    }

    #[test]
    fn test_validate_linsolve_inputs_success() {
        let operator = create_simple_2site_mpo();
        let rhs = create_simple_2site_mps();
        let init = create_simple_2site_mps();

        // Should succeed for compatible inputs
        let result = validate_linsolve_inputs(&operator, &rhs, &init);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_linsolve_inputs_incompatible_dimensions() {
        let operator = create_simple_2site_mpo();
        let rhs = create_simple_2site_mps();

        // Create init with different dimensions
        let mut init = TreeTN::<TensorDynLen, String>::new();
        let s0 = DynIndex::new_dyn(3); // Different dimension
        let s1 = DynIndex::new_dyn(3);
        let b01 = DynIndex::new_dyn(2);

        let t0 = TensorDynLen::from_dense_f64(vec![s0.clone(), b01.clone()], vec![1.0; 6]);
        let t1 = TensorDynLen::from_dense_f64(vec![b01.clone(), s1.clone()], vec![1.0; 6]);

        let n0 = init.add_tensor("site0".to_string(), t0).unwrap();
        let n1 = init.add_tensor("site1".to_string(), t1).unwrap();
        init.connect(n0, &b01, n1, &b01).unwrap();

        // Should fail due to incompatible dimensions
        let result = validate_linsolve_inputs(&operator, &rhs, &init);
        assert!(result.is_err());
    }

    #[test]
    fn test_square_linsolve_basic() {
        let operator = create_simple_2site_mpo();
        let rhs = create_simple_2site_mps();
        let init = create_simple_2site_mps();

        let options = LinsolveOptions::default().with_nfullsweeps(1);
        let result = square_linsolve(&operator, &rhs, init, &"site0".to_string(), options);

        assert!(result.is_ok());
        let solution = result.unwrap();
        assert_eq!(solution.solution.node_count(), 2);
        assert_eq!(solution.sweeps, 1);
        assert!(!solution.converged);
    }

    #[test]
    fn test_square_linsolve_result_fields() {
        let operator = create_simple_2site_mpo();
        let rhs = create_simple_2site_mps();
        let init = create_simple_2site_mps();

        let options = LinsolveOptions::default().with_nfullsweeps(2);
        let result = square_linsolve(&operator, &rhs, init, &"site0".to_string(), options).unwrap();

        assert_eq!(result.sweeps, 2);
        assert_eq!(result.residual, None);
        assert!(!result.converged);
        assert_eq!(result.solution.node_count(), 2);
    }
}
