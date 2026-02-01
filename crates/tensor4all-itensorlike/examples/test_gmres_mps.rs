//! Test: GMRES solver with MPS/MPO format
//!
//! A = Identity MPO
//! x = MPS with all elements = 1
//! b = A * x = x (since A = I)
//!
//! Run:
//!   cargo run -p tensor4all-itensorlike --example test_gmres_mps --release

use tensor4all_core::krylov::{gmres, GmresOptions};
use tensor4all_core::{AnyScalar, DynIndex, IndexLike, TensorDynLen, TensorIndex};
use tensor4all_itensorlike::{ContractOptions, TensorTrain};

/// Shared indices for all MPS/MPO operations.
/// This ensures all tensors can be added and have inner products computed.
struct SharedIndices {
    /// Physical (site) indices
    sites: Vec<DynIndex>,
    /// Bond indices between sites
    bonds: Vec<DynIndex>,
    /// MPO output indices (separate from input = site indices)
    mpo_outputs: Vec<DynIndex>,
}

impl SharedIndices {
    fn new(n: usize, phys_dim: usize) -> Self {
        let sites: Vec<DynIndex> = (0..n).map(|_| DynIndex::new_dyn(phys_dim)).collect();
        let bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
            .map(|_| DynIndex::new_dyn(1))
            .collect();
        let mpo_outputs: Vec<DynIndex> = (0..n).map(|_| DynIndex::new_dyn(phys_dim)).collect();

        SharedIndices {
            sites,
            bonds,
            mpo_outputs,
        }
    }
}

fn main() -> anyhow::Result<()> {
    // Use 2 sites to have proper bond structure
    let n = 2;
    let phys_dim = 2;

    println!("=== Test: GMRES with MPS ===");
    println!("N = {}, phys_dim = {}", n, phys_dim);

    // Create shared indices for all operations
    let indices = SharedIndices::new(n, phys_dim);

    // Create Identity MPO (with separate input/output physical indices)
    let mpo = create_identity_mpo(&indices)?;
    println!("Identity MPO created with {} sites", mpo.len());

    // Create x_true (all ones MPS)
    let x_true = create_ones_mps(&indices)?;
    println!("x_true MPS created with {} sites, norm: {:.6}", x_true.len(), x_true.norm());

    // Compute b = A * x_true
    let b = apply_mpo(&mpo, &x_true, &indices)?;
    println!("b = A * x_true computed, norm: {:.6}", b.norm());

    // Use b as initial guess (x0 = b)
    let x0 = b.clone();
    println!("x0 (initial guess = b) created, norm: {:.6}", x0.norm());

    // Define apply_a closure
    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> { apply_mpo(&mpo, x, &indices) };

    // Compute initial residual: |Ax0 - b| / |b|
    let b_norm = b.norm();
    let ax0 = apply_a(&x0)?;
    let r0 = ax0.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let initial_residual = r0.norm() / b_norm;
    println!("\n=== Initial Residual ===");
    println!("|Ax0 - b| / |b| = {:.6e}", initial_residual);

    // Solve with GMRES
    let options = GmresOptions {
        max_iter: 5,
        rtol: 1e-12,
        max_restarts: 1,
        verbose: true,
    };

    println!("\n=== Running GMRES (max_iter={}) ===", options.max_iter);
    let result = gmres(&apply_a, &b, &x0, &options)?;

    // Compute final residual: |Ax_sol - b| / |b|
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let final_residual = r_final.norm() / b_norm;

    // Results
    println!("\n=== Results ===");
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations);
    println!("GMRES reported residual: {:.6e}", result.residual_norm);
    println!("Final |Ax - b| / |b|:    {:.6e}", final_residual);

    // Compute error ||x_sol - x_true||
    let diff = result
        .solution
        .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    let error = diff.norm();
    println!("Error ||x_sol - x_true||: {:.6e}", error);

    println!("\n=== Done ===");
    Ok(())
}

/// Create an identity MPO using shared indices.
fn create_identity_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.sites.len();
    let mut tensors = Vec::with_capacity(n);

    // MPO bond indices (separate from MPS bonds)
    let mpo_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    for i in 0..n {
        let site_dim = indices.sites[i].dim();
        let s_in = indices.sites[i].clone();
        let s_out = indices.mpo_outputs[i].clone();

        // Identity tensor: delta(s_in, s_out)
        if i == 0 && n == 1 {
            // Single site: [s_in, s_out]
            let mut data = vec![0.0; site_dim * site_dim];
            for j in 0..site_dim {
                data[j * site_dim + j] = 1.0;
            }
            let tensor = TensorDynLen::from_dense_f64(vec![s_in, s_out], data);
            tensors.push(tensor);
        } else if i == 0 {
            // First site: [s_in, s_out, right]
            let right_bond = mpo_bonds[i].clone();
            let mut data = vec![0.0; site_dim * site_dim * 1];
            for j in 0..site_dim {
                data[j * site_dim + j] = 1.0;
            }
            let tensor = TensorDynLen::from_dense_f64(vec![s_in, s_out, right_bond], data);
            tensors.push(tensor);
        } else if i == n - 1 {
            // Last site: [left, s_in, s_out]
            let left_bond = mpo_bonds[i - 1].clone();
            let mut data = vec![0.0; 1 * site_dim * site_dim];
            for j in 0..site_dim {
                data[j * site_dim + j] = 1.0;
            }
            let tensor = TensorDynLen::from_dense_f64(vec![left_bond, s_in, s_out], data);
            tensors.push(tensor);
        } else {
            // Middle site: [left, s_in, s_out, right]
            let left_bond = mpo_bonds[i - 1].clone();
            let right_bond = mpo_bonds[i].clone();
            let mut data = vec![0.0; 1 * site_dim * site_dim * 1];
            for j in 0..site_dim {
                data[j * site_dim + j] = 1.0;
            }
            let tensor =
                TensorDynLen::from_dense_f64(vec![left_bond, s_in, s_out, right_bond], data);
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create an MPS with all elements = 1, using shared indices.
fn create_ones_mps(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.sites.len();
    let mut tensors = Vec::with_capacity(n);

    for i in 0..n {
        let site_dim = indices.sites[i].dim();
        let site_idx = indices.sites[i].clone();

        if i == 0 && n == 1 {
            let data = vec![1.0; site_dim];
            let tensor = TensorDynLen::from_dense_f64(vec![site_idx], data);
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = indices.bonds[i].clone();
            let data = vec![1.0; site_dim * 1];
            let tensor = TensorDynLen::from_dense_f64(vec![site_idx, right_bond], data);
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = indices.bonds[i - 1].clone();
            let data = vec![1.0; 1 * site_dim];
            let tensor = TensorDynLen::from_dense_f64(vec![left_bond, site_idx], data);
            tensors.push(tensor);
        } else {
            let left_bond = indices.bonds[i - 1].clone();
            let right_bond = indices.bonds[i].clone();
            let data = vec![1.0; 1 * site_dim * 1];
            let tensor = TensorDynLen::from_dense_f64(vec![left_bond, site_idx, right_bond], data);
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Apply MPO to MPS and return result with same external indices as input MPS.
///
/// After contraction, replaces MPO output indices with site indices,
/// ensuring the result is in the same vector space as the input.
fn apply_mpo(
    mpo: &TensorTrain,
    mps: &TensorTrain,
    indices: &SharedIndices,
) -> anyhow::Result<TensorTrain> {
    // Contract MPO with MPS using fit method (more stable)
    let options = ContractOptions::fit().with_nhalfsweeps(4);
    let result = mpo
        .contract(mps, &options)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Replace MPO output indices with site indices
    let result = result.replaceinds(&indices.mpo_outputs, &indices.sites)?;

    Ok(result)
}
