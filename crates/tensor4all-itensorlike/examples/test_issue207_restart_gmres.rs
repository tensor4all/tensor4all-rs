//! Test: Issue #207 - Can restart_gmres_with_truncation fix the orthogonality loss problem?
//!
//! Issue #207 identified that gmres_with_truncation reports false convergence for N=3 Pauli-X:
//! - GMRES reported residual: ~1e-16
//! - Actual residual: ~0.4 (40% error)
//!
//! This test compares gmres_with_truncation vs restart_gmres_with_truncation
//! to verify that the restart variant fixes this issue by checking true residuals.
//!
//! Run:
//!   cargo run -p tensor4all-itensorlike --example test_issue207_restart_gmres --release

use tensor4all_core::krylov::{
    gmres_with_truncation, restart_gmres_with_truncation, GmresOptions, RestartGmresOptions,
};
use tensor4all_core::{AnyScalar, DynIndex, IndexLike, TensorDynLen, TensorIndex};
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};

/// Shared indices for MPO operations.
struct SharedIndices {
    inputs: Vec<DynIndex>,
    outputs: Vec<DynIndex>,
    bonds: Vec<DynIndex>,
    operator_outputs: Vec<DynIndex>,
}

impl SharedIndices {
    fn new(n: usize, phys_dim: usize) -> Self {
        let inputs: Vec<DynIndex> = (0..n).map(|_| DynIndex::new_dyn(phys_dim)).collect();
        let outputs: Vec<DynIndex> = (0..n).map(|_| DynIndex::new_dyn(phys_dim)).collect();
        let bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
            .map(|_| DynIndex::new_dyn(1))
            .collect();
        let operator_outputs: Vec<DynIndex> = (0..n).map(|_| DynIndex::new_dyn(phys_dim)).collect();
        Self {
            inputs,
            outputs,
            bonds,
            operator_outputs,
        }
    }
}

fn main() -> anyhow::Result<()> {
    println!("========================================");
    println!("  Issue #207: restart_gmres_with_truncation Test");
    println!("========================================\n");
    println!("Issue: gmres_with_truncation reports false convergence for N=3 Pauli-X");
    println!("       (reported residual ~1e-16, actual residual ~0.4)\n");

    let n = 3;

    // Run multiple times within the same process to check reproducibility
    println!("Running 3 times within the same process...\n");
    for run in 1..=3 {
        println!("############## RUN {} ##############\n", run);

        println!("========================================");
        println!("  Test 1: gmres_with_truncation (original)");
        println!("========================================\n");

        let original_result = test_gmres_original(n)?;

        println!("\n========================================");
        println!("  Test 2: restart_gmres_with_truncation (fix)");
        println!("========================================\n");

        let restart_result = test_gmres_restart(n)?;

        print_summary(original_result, restart_result);
        println!();
    }

    Ok(())
}

fn print_summary(
    original_result: (f64, f64, bool, usize),
    restart_result: (f64, f64, bool, usize),
) {
    let (orig_reported, orig_true, orig_conv, _) = original_result;
    let (restart_reported, restart_true, restart_conv, restart_outer) = restart_result;

    println!("\n========================================");
    println!("  Summary Comparison (N=3 Pauli-X)");
    println!("========================================\n");

    println!("gmres_with_truncation (original):");
    println!("  Reported residual: {:.6e}", orig_reported);
    println!("  True residual:     {:.6e}", orig_true);
    println!("  Converged:         {}", orig_conv);

    println!("\nrestart_gmres_with_truncation (fix):");
    println!("  Reported residual: {:.6e}", restart_reported);
    println!("  True residual:     {:.6e}", restart_true);
    println!("  Converged:         {}", restart_conv);
    println!("  Outer iterations:  {}", restart_outer);

    println!("\n========================================");
    println!("  Issue #207 Verification");
    println!("========================================\n");

    if orig_true > 0.1 && restart_true < 1e-6 && restart_conv {
        println!("✓ Issue #207 is FIXED by restart_gmres_with_truncation!");
    } else if restart_true < orig_true * 0.01 {
        println!("✓ restart_gmres_with_truncation significantly improves the result.");
    } else if orig_true < 1e-6 {
        println!("Note: Original gmres_with_truncation already works for N=3.");
        println!("      The issue may have been fixed by iterative reorthogonalization.");
    } else {
        println!("✗ Issue #207 is NOT fully fixed.");
    }
}

/// Test gmres_with_truncation (original, affected by issue #207)
/// Returns (reported_residual, true_residual, converged, iterations)
fn test_gmres_original(n: usize) -> anyhow::Result<(f64, f64, bool, usize)> {
    let phys_dim = 2;
    println!("--- gmres_with_truncation: N={} Pauli-X ---", n);

    let indices = SharedIndices::new(n, phys_dim);

    // Create Pauli-X operator MPO
    let pauli_x_op = create_pauli_x_operator_mpo(&indices)?;

    // Create x_true = identity MPO
    let x_true = create_identity_mpo(&indices)?;

    // b = A(x_true) = σ_x * I = σ_x
    let b = apply_operator_to_mpo(&pauli_x_op, &x_true, &indices)?;
    let b_norm = b.norm();

    // Initial guess: 0.5 * b
    let x0 = b.scale(AnyScalar::new_real(0.5))?;

    // Define apply_a closure
    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> {
        apply_operator_to_mpo(&pauli_x_op, x, &indices)
    };

    // Truncation options (same as issue #207)
    let truncate_opts = TruncateOptions::svd().with_rtol(1e-8).with_max_rank(20);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    // GMRES options
    let options = GmresOptions {
        max_iter: 10,
        rtol: 1e-8,
        max_restarts: 1,
        verbose: true,
    };

    let result = gmres_with_truncation(&apply_a, &b, &x0, &options, truncate_fn)?;

    // Compute true residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let true_residual = r_final.norm() / b_norm;

    println!("  GMRES reported residual: {:.6e}", result.residual_norm);
    println!("  True residual:           {:.6e}", true_residual);
    println!("  Converged:               {}", result.converged);
    println!("  Iterations:              {}", result.iterations);

    if result.converged && true_residual > 0.01 {
        println!("  *** FALSE CONVERGENCE DETECTED ***");
    }

    Ok((
        result.residual_norm,
        true_residual,
        result.converged,
        result.iterations,
    ))
}

/// Test restart_gmres_with_truncation (fix for issue #207)
/// Returns (reported_residual, true_residual, converged, outer_iterations)
fn test_gmres_restart(n: usize) -> anyhow::Result<(f64, f64, bool, usize)> {
    let phys_dim = 2;
    println!("--- restart_gmres_with_truncation: N={} Pauli-X ---", n);

    let indices = SharedIndices::new(n, phys_dim);

    // Create Pauli-X operator MPO
    let pauli_x_op = create_pauli_x_operator_mpo(&indices)?;

    // Create x_true = identity MPO
    let x_true = create_identity_mpo(&indices)?;

    // b = A(x_true) = σ_x * I = σ_x
    let b = apply_operator_to_mpo(&pauli_x_op, &x_true, &indices)?;
    let b_norm = b.norm();

    // Initial guess: 0.5 * b
    let x0 = b.scale(AnyScalar::new_real(0.5))?;

    // Define apply_a closure
    let apply_a = |x: &TensorTrain| -> anyhow::Result<TensorTrain> {
        apply_operator_to_mpo(&pauli_x_op, x, &indices)
    };

    // Truncation options (same as issue #207)
    let truncate_opts = TruncateOptions::svd().with_rtol(1e-8).with_max_rank(20);
    let truncate_fn = |x: &mut TensorTrain| -> anyhow::Result<()> {
        x.truncate(&truncate_opts)?;
        Ok(())
    };

    // Restart GMRES options
    let options = RestartGmresOptions {
        max_outer_iters: 30,
        rtol: 1e-8,
        inner_max_iter: 10,
        inner_max_restarts: 0,
        min_reduction: None,
        inner_rtol: Some(0.1),
        verbose: true,
    };

    let result = restart_gmres_with_truncation(&apply_a, &b, Some(&x0), &options, &truncate_fn)?;

    // Compute true residual
    let ax_sol = apply_a(&result.solution)?;
    let r_final = ax_sol.axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))?;
    let true_residual = r_final.norm() / b_norm;

    println!("  Reported residual:       {:.6e}", result.residual_norm);
    println!("  True residual:           {:.6e}", true_residual);
    println!("  Converged:               {}", result.converged);
    println!("  Outer iterations:        {}", result.outer_iterations);
    println!("  Total inner iterations:  {}", result.iterations);

    // Compute error ||x_sol - x_true||
    let diff =
        result
            .solution
            .axpby(AnyScalar::new_real(1.0), &x_true, AnyScalar::new_real(-1.0))?;
    println!("  Error ||x - x_true||:    {:.6e}", diff.norm());

    if result.converged && true_residual > 0.01 {
        println!("  *** FALSE CONVERGENCE DETECTED ***");
    }

    Ok((
        result.residual_norm,
        true_residual,
        result.converged,
        result.outer_iterations,
    ))
}

// ============================================================================
// Helper functions (copied from test_gmres_mpo.rs)
// ============================================================================

/// Create an identity MPO: I[in, out] = δ(in, out)
fn create_identity_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.inputs.len();
    let mut tensors = Vec::with_capacity(n);

    for i in 0..n {
        let in_dim = indices.inputs[i].dim();
        let out_dim = indices.outputs[i].dim();
        let in_idx = indices.inputs[i].clone();
        let out_idx = indices.outputs[i].clone();

        let mut data = vec![0.0_f64; in_dim * out_dim];
        for j in 0..in_dim.min(out_dim) {
            data[j * out_dim + j] = 1.0;
        }

        if i == 0 && n == 1 {
            let tensor = TensorDynLen::from_dense_f64(vec![in_idx, out_idx], data);
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = indices.bonds[i].clone();
            let tensor = TensorDynLen::from_dense_f64(vec![in_idx, out_idx, right_bond], data);
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = indices.bonds[i - 1].clone();
            let tensor = TensorDynLen::from_dense_f64(vec![left_bond, in_idx, out_idx], data);
            tensors.push(tensor);
        } else {
            let left_bond = indices.bonds[i - 1].clone();
            let right_bond = indices.bonds[i].clone();
            let tensor =
                TensorDynLen::from_dense_f64(vec![left_bond, in_idx, out_idx, right_bond], data);
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Create a Pauli-X operator MPO.
fn create_pauli_x_operator_mpo(indices: &SharedIndices) -> anyhow::Result<TensorTrain> {
    let n = indices.inputs.len();
    let mut tensors = Vec::with_capacity(n);

    let op_bonds: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(1))
        .collect();

    // Pauli-X: [[0, 1], [1, 0]]
    let pauli_x = [0.0, 1.0, 1.0, 0.0];

    for i in 0..n {
        let in_idx = indices.inputs[i].clone();
        let op_out_idx = indices.operator_outputs[i].clone();

        if i == 0 && n == 1 {
            let tensor = TensorDynLen::from_dense_f64(vec![in_idx, op_out_idx], pauli_x.to_vec());
            tensors.push(tensor);
        } else if i == 0 {
            let right_bond = op_bonds[i].clone();
            let tensor = TensorDynLen::from_dense_f64(
                vec![in_idx, op_out_idx, right_bond],
                pauli_x.to_vec(),
            );
            tensors.push(tensor);
        } else if i == n - 1 {
            let left_bond = op_bonds[i - 1].clone();
            let tensor =
                TensorDynLen::from_dense_f64(vec![left_bond, in_idx, op_out_idx], pauli_x.to_vec());
            tensors.push(tensor);
        } else {
            let left_bond = op_bonds[i - 1].clone();
            let right_bond = op_bonds[i].clone();
            let tensor = TensorDynLen::from_dense_f64(
                vec![left_bond, in_idx, op_out_idx, right_bond],
                pauli_x.to_vec(),
            );
            tensors.push(tensor);
        }
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Apply an operator MPO to an MPO state: result = O * X
fn apply_operator_to_mpo(
    op: &TensorTrain,
    mpo: &TensorTrain,
    indices: &SharedIndices,
) -> anyhow::Result<TensorTrain> {
    let options = ContractOptions::zipup().with_rtol(1e-10).with_max_rank(50);

    let result = op
        .contract(mpo, &options)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let result = result.replaceinds(&indices.operator_outputs, &indices.inputs)?;

    Ok(result)
}
