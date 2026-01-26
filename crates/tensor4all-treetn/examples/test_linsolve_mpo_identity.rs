//! Test: linsolve with MPO for x and b (all identity operators).
//!
//! This test focuses on the case where x and b are MPOs (Matrix Product Operators)
//! instead of MPS (Matrix Product States).
//!
//! Test setup:
//! - A = I (identity operator)
//! - x = I (identity operator, as MPO)
//! - b = I (identity operator, as MPO)
//! - Equation: A * x = b => I * I = I
//!
//! This is the simplest case where all operators are identity.
//!
//! NOTE: This test currently encounters an issue because SquareLinsolveUpdater
//! expects x and b to have the same site index structure as MPS (one index per site),
//! but MPOs have two indices per site (input and output). This test demonstrates
//! the attempt to use MPOs for x and b, but may require modifications to
//! SquareLinsolveUpdater to fully support MPO cases.
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example test_linsolve_mpo_identity --release

use std::collections::{HashMap, HashSet};

use tensor4all_core::{index::DynId, DynIndex, IndexLike, TensorDynLen};
use tensor4all_treetn::{
    apply_local_update_sweep, CanonicalizationOptions, IndexMapping, LinsolveOptions,
    LocalUpdateSweepPlan, SquareLinsolveUpdater, TreeTN,
};

fn make_node_name(i: usize) -> String {
    format!("site{i}")
}

fn unique_dyn_index(used: &mut HashSet<DynId>, dim: usize) -> DynIndex {
    loop {
        let idx = DynIndex::new_dyn(dim);
        if used.insert(*idx.id()) {
            return idx;
        }
    }
}

/// Create an N-site identity MPO with internal indices (bond dim = 1).
/// Returns (mpo, input_mapping, output_mapping).
fn create_identity_mpo_with_mappings(
    n: usize,
    phys_dim: usize,
    true_site_indices: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
) -> anyhow::Result<(
    TreeTN<TensorDynLen, String>,
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
)> {
    anyhow::ensure!(true_site_indices.len() == n, "site index count mismatch");

    let mut mpo = TreeTN::<TensorDynLen, String>::new();

    // MPO bonds: dim 1
    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, 1))
        .collect();

    // Internal indices (MPO-only)
    let s_in_tmp: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim))
        .collect();
    let s_out_tmp: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim))
        .collect();

    let mut input_mapping: HashMap<String, IndexMapping<DynIndex>> = HashMap::new();
    let mut output_mapping: HashMap<String, IndexMapping<DynIndex>> = HashMap::new();

    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let node_name = make_node_name(i);

        // Index ordering matches tests/linsolve.rs conventions:
        // - first:  [s_out_tmp, s_in_tmp, bond_right]
        // - middle: [bond_left, s_out_tmp, s_in_tmp, bond_right]
        // - last:   [bond_left, s_out_tmp, s_in_tmp]
        let indices = if n == 1 {
            vec![s_out_tmp[i].clone(), s_in_tmp[i].clone()]
        } else if i == 0 {
            vec![s_out_tmp[i].clone(), s_in_tmp[i].clone(), bonds[i].clone()]
        } else if i + 1 == n {
            vec![
                bonds[i - 1].clone(),
                s_out_tmp[i].clone(),
                s_in_tmp[i].clone(),
            ]
        } else {
            vec![
                bonds[i - 1].clone(),
                s_out_tmp[i].clone(),
                s_in_tmp[i].clone(),
                bonds[i].clone(),
            ]
        };

        // Base identity matrix on (s_out_tmp, s_in_tmp).
        let mut data = vec![0.0_f64; phys_dim * phys_dim];
        for k in 0..phys_dim {
            data[k * phys_dim + k] = 1.0;
        }
        let base =
            TensorDynLen::from_dense_f64(vec![s_out_tmp[i].clone(), s_in_tmp[i].clone()], data);
        let t = if indices.len() == 2 {
            base
        } else {
            let bond_indices: Vec<_> = indices
                .iter()
                .filter(|idx| idx.dim() == 1)
                .cloned()
                .collect();
            let ones = TensorDynLen::from_dense_f64(bond_indices, vec![1.0_f64; 1]);
            TensorDynLen::outer_product(&base, &ones)?
        };

        let node = mpo.add_tensor(node_name.clone(), t).unwrap();
        nodes.push(node);

        input_mapping.insert(
            node_name.clone(),
            IndexMapping {
                true_index: true_site_indices[i].clone(),
                internal_index: s_in_tmp[i].clone(),
            },
        );
        output_mapping.insert(
            node_name,
            IndexMapping {
                true_index: true_site_indices[i].clone(),
                internal_index: s_out_tmp[i].clone(),
            },
        );
    }

    for i in 0..n.saturating_sub(1) {
        mpo.connect(nodes[i], &bonds[i], nodes[i + 1], &bonds[i])
            .unwrap();
    }

    Ok((mpo, input_mapping, output_mapping))
}

fn print_bond_dims(treetn: &TreeTN<TensorDynLen, String>, label: &str) {
    let edges: Vec<_> = treetn.site_index_network().edges().collect();
    if edges.is_empty() {
        println!("{label}: no bonds");
        return;
    }
    let mut dims = Vec::new();
    for (node_a, node_b) in edges {
        if let Some(edge) = treetn.edge_between(&node_a, &node_b) {
            if let Some(bond) = treetn.bond_index(edge) {
                dims.push(bond.dim);
            }
        }
    }
    println!("{label}: bond_dims = {:?}", dims);
}

fn main() -> anyhow::Result<()> {
    let n = 3usize;
    let phys_dim = 2usize;

    println!("=== Test: linsolve with MPO for x and b (all identity operators) ===");
    println!("N = {n}, phys_dim = {phys_dim}");
    println!("A = I (identity operator)");
    println!("x = I (identity operator, as MPO)");
    println!("b = I (identity operator, as MPO)");
    println!("Equation: A * x = b => I * I = I");
    println!();

    let mut used_ids = HashSet::<DynId>::new();

    // Create site indices for x and b (they are MPOs, so they need input and output indices)
    // For MPO, we need separate indices for input and output
    // Note: We use the same site indices for x and b to ensure they match
    // (SquareLinsolveUpdater expects x and b to have the same site indices)
    let b_site_indices: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(&mut used_ids, phys_dim))
        .collect();

    // Create operator A = I (identity)
    // Note: We'll create A with b's site indices to ensure consistency
    // (A will be recreated later with matching indices)

    println!("Creating operator A (identity MPO)...");

    // Create x = I (identity MPO)
    // Note: We'll create x with b's site indices to ensure consistency
    // (x will be recreated later with matching indices)

    println!("Creating x (identity MPO)...");

    // Create b = I (identity MPO)
    let (b_mpo, _b_input_mapping, _b_output_mapping) = create_identity_mpo_with_mappings(
        n,
        phys_dim,
        &b_site_indices,
        &mut used_ids,
    )?;

    println!("Created b (identity MPO)");
    print_bond_dims(&b_mpo, "b bond dimensions");
    println!();

    // For linsolve, we need to set up the equation A * x = b
    // The operator A's output indices should match x's input indices
    // x's output indices should match b's input indices
    // b's output indices are the final output
    //
    // Note: For MPO case, x and b need to have matching index structure.
    // Since both are identity MPOs, they should have the same structure.
    // However, SquareLinsolveUpdater expects x and b to have the same site indices.
    // We need to ensure that x and b use the same site indices.

    // Create initial guess (use identity MPO as initial guess)
    // Use b's site indices for x to ensure they match
    let (x_mpo_matched, _x_input_mapping_matched, _x_output_mapping_matched) =
        create_identity_mpo_with_mappings(n, phys_dim, &b_site_indices, &mut used_ids)?;
    let init = x_mpo_matched.clone();

    // Canonicalize towards center
    let center = make_node_name(n / 2);
    let mut x = init.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    println!("Initial guess (x_init) created");
    print_bond_dims(&x, "x_init bond dimensions");
    println!();

    // Setup linsolve options
    let options = LinsolveOptions::default()
        .with_nfullsweeps(5)
        .with_max_rank(4)
        .with_krylov_tol(1e-8)
        .with_krylov_maxiter(20)
        .with_krylov_dim(30)
        .with_coefficients(1.0, 0.0); // a0=1, a1=0 => I * x = b

    println!("Linsolve options: nfullsweeps=5, max_rank=4, krylov_tol=1e-8");
    println!();

    // Create updater
    // Note: For MPO case, we need to ensure index mappings are correct
    // The operator A's output should map to x's input, and x's output should map to b's input
    // However, SquareLinsolveUpdater expects x and b to have the same site indices.
    // We use b's site indices for both x and the operator A.
    let (operator_a_matched, a_input_mapping_matched, a_output_mapping_matched) =
        create_identity_mpo_with_mappings(n, phys_dim, &b_site_indices, &mut used_ids)?;
    let mut updater = SquareLinsolveUpdater::with_index_mappings(
        operator_a_matched,
        a_input_mapping_matched,
        a_output_mapping_matched,
        b_mpo,
        options,
    );

    // Create sweep plan
    let plan = LocalUpdateSweepPlan::from_treetn(&x, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    println!("Starting sweeps...");
    println!();

    // Run sweeps
    for sweep in 1..=5 {
        println!("Sweep {sweep}/5...");
        apply_local_update_sweep(&mut x, &plan, &mut updater)?;
        print_bond_dims(&x, &format!("x bond dimensions (after sweep {sweep})"));
        println!();
    }

    println!("=== Test completed successfully ===");
    println!("Final solution x:");
    print_bond_dims(&x, "x bond dimensions (final)");

    // For identity case, solution should remain identity
    // We can verify by checking that x is still identity-like
    println!();
    println!("Note: For I * x = I, the solution x = I should be recovered.");

    Ok(())
}
