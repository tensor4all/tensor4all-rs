//! Test: linsolve identity residual (N=3).
//!
//! Fixed conditions:
//! - N = 3
//! - A = identity (diagonal MPO with all diag values = 1.0) with internal indices + index mappings
//! - init = rhs.sim_linkinds() (same physical indices, independent bond/link index IDs)
//! - coefficients: a0 = 0, a1 = 1, so the equation is A x = b
//!
//! This example runs 5 times 2-site sweep and prints the relative residual:
//!   ||r||_2 / ||b||_2, where r = A x - b.
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example test_linsolve_identity_residual_n3 --release

use std::collections::HashMap;

use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_treetn::{
    apply_linear_operator, apply_local_update_sweep, ApplyOptions, CanonicalizationOptions,
    IndexMapping, LinearOperator, LinsolveOptions, LinsolveUpdater, LocalUpdateSweepPlan, TreeTN,
};

/// Create an N-site MPS chain with identity-like structure.
/// Returns (mps, site_indices).
fn create_n_site_mps(
    n_sites: usize,
    phys_dim: usize,
    bond_dim: usize,
) -> (TreeTN<TensorDynLen, String>, Vec<DynIndex>) {
    assert!(n_sites >= 2, "Need at least 2 sites");

    let mut mps = TreeTN::<TensorDynLen, String>::new();

    // Physical indices
    let site_indices: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();

    // Bond indices (n_sites - 1 bonds)
    let bond_indices: Vec<DynIndex> = (0..n_sites - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();

    // Create tensors for each site (index order matches tests/linsolve.rs)
    for i in 0..n_sites {
        let name = format!("site{i}");
        let tensor = if i == 0 {
            // First site: [s0, b01] - identity-like
            let mut data = vec![0.0; phys_dim * bond_dim];
            for j in 0..phys_dim.min(bond_dim) {
                data[j * bond_dim + j] = 1.0;
            }
            TensorDynLen::from_dense_f64(
                vec![site_indices[i].clone(), bond_indices[i].clone()],
                data,
            )
        } else if i == n_sites - 1 {
            // Last site: [b_{n-2,n-1}, s_{n-1}] - identity-like
            let mut data = vec![0.0; bond_dim * phys_dim];
            for j in 0..phys_dim.min(bond_dim) {
                data[j * phys_dim + j] = 1.0;
            }
            TensorDynLen::from_dense_f64(
                vec![bond_indices[i - 1].clone(), site_indices[i].clone()],
                data,
            )
        } else {
            // Middle sites: [b_{i-1,i}, s_i, b_{i,i+1}] - identity-like
            let mut data = vec![0.0; bond_dim * phys_dim * bond_dim];
            for j in 0..phys_dim.min(bond_dim) {
                let idx = j * phys_dim * bond_dim + j * bond_dim + j;
                data[idx] = 1.0;
            }
            TensorDynLen::from_dense_f64(
                vec![
                    bond_indices[i - 1].clone(),
                    site_indices[i].clone(),
                    bond_indices[i].clone(),
                ],
                data,
            )
        };
        mps.add_tensor(name, tensor).unwrap();
    }

    // Connect adjacent sites
    for (i, bond) in bond_indices.iter().enumerate() {
        let name_i = format!("site{i}");
        let name_j = format!("site{}", i + 1);
        let ni = mps.node_index(&name_i).unwrap();
        let nj = mps.node_index(&name_j).unwrap();
        mps.connect(ni, bond, nj, bond).unwrap();
    }

    (mps, site_indices)
}

/// Create an N-site diagonal (identity) MPO with internal indices.
/// Returns (mpo, s_in_tmp, s_out_tmp).
fn create_n_site_mpo_with_internal_indices(
    diag_values: &[f64],
    phys_dim: usize,
) -> (TreeTN<TensorDynLen, String>, Vec<DynIndex>, Vec<DynIndex>) {
    let n_sites = diag_values.len();
    assert!(n_sites >= 2, "Need at least 2 sites");

    let mut mpo = TreeTN::<TensorDynLen, String>::new();

    // Internal indices (independent IDs)
    let s_in_tmp: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let s_out_tmp: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();

    // Bond indices (dim 1 for diagonal operator)
    let bond_indices: Vec<DynIndex> = (0..n_sites - 1).map(|_| DynIndex::new_dyn(1)).collect();

    for i in 0..n_sites {
        let name = format!("site{i}");
        let mut data = vec![0.0; phys_dim * phys_dim];
        for j in 0..phys_dim {
            data[j * phys_dim + j] = diag_values[i];
        }

        let tensor = if i == 0 {
            TensorDynLen::from_dense_f64(
                vec![
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                    bond_indices[i].clone(),
                ],
                data,
            )
        } else if i == n_sites - 1 {
            TensorDynLen::from_dense_f64(
                vec![
                    bond_indices[i - 1].clone(),
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                ],
                data,
            )
        } else {
            TensorDynLen::from_dense_f64(
                vec![
                    bond_indices[i - 1].clone(),
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                    bond_indices[i].clone(),
                ],
                data,
            )
        };
        mpo.add_tensor(name, tensor).unwrap();
    }

    // Connect adjacent sites
    for (i, bond) in bond_indices.iter().enumerate() {
        let name_i = format!("site{i}");
        let name_j = format!("site{}", i + 1);
        let ni = mpo.node_index(&name_i).unwrap();
        let nj = mpo.node_index(&name_j).unwrap();
        mpo.connect(ni, bond, nj, bond).unwrap();
    }

    (mpo, s_in_tmp, s_out_tmp)
}

/// Create N-site index mappings from MPO and state site indices.
/// Returns (input_mapping, output_mapping).
fn create_n_site_index_mappings(
    state_site_indices: &[DynIndex],
    s_in_tmp: &[DynIndex],
    s_out_tmp: &[DynIndex],
) -> (
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
) {
    let n_sites = state_site_indices.len();
    assert_eq!(s_in_tmp.len(), n_sites);
    assert_eq!(s_out_tmp.len(), n_sites);

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for i in 0..n_sites {
        let site = format!("site{i}");
        input_mapping.insert(
            site.clone(),
            IndexMapping {
                true_index: state_site_indices[i].clone(),
                internal_index: s_in_tmp[i].clone(),
            },
        );
        output_mapping.insert(
            site,
            IndexMapping {
                true_index: state_site_indices[i].clone(),
                internal_index: s_out_tmp[i].clone(),
            },
        );
    }

    (input_mapping, output_mapping)
}

fn main() -> anyhow::Result<()> {
    let n_sites = 3usize;
    let phys_dim = 2usize;
    let bond_dim = 2usize;

    // RHS
    let (rhs, site_indices) = create_n_site_mps(n_sites, phys_dim, bond_dim);

    // A = I (diagonal MPO with ones)
    let diag_values: Vec<f64> = vec![1.0; n_sites];
    let (mpo, s_in_tmp, s_out_tmp) =
        create_n_site_mpo_with_internal_indices(&diag_values, phys_dim);
    let (input_mapping, output_mapping) =
        create_n_site_index_mappings(&site_indices, &s_in_tmp, &s_out_tmp);

    // init fixed: rhs with independent link indices
    let init = rhs.sim_linkinds()?;
    let mut x = init.canonicalize(["site0".to_string()], CanonicalizationOptions::default())?;

    // linsolve: fixed coefficients a0=0, a1=1 (A x = b)
    let options = LinsolveOptions::default()
        .with_nfullsweeps(1)
        .with_krylov_tol(1e-10)
        .with_max_rank(4)
        .with_coefficients(0.0, 1.0);

    let mut updater = LinsolveUpdater::with_index_mappings(
        mpo.clone(),
        input_mapping.clone(),
        output_mapping.clone(),
        rhs.clone(),
        options,
    );

    // 5 times 2-site sweep
    let plan = LocalUpdateSweepPlan::from_treetn(&x, &"site0".to_string(), 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;
    for _ in 0..5 {
        apply_local_update_sweep(&mut x, &plan, &mut updater)?;
    }

    // Residual r = A x - b (measured in full contracted space)
    let linop = LinearOperator::new(mpo, input_mapping, output_mapping);
    let ax = apply_linear_operator(&linop, &x, ApplyOptions::default())?;

    let ax_full = ax.contract_to_tensor()?;
    let b_full = rhs.contract_to_tensor()?;
    let ax_vec = ax_full.to_vec_f64()?;
    let b_vec = b_full.to_vec_f64()?;
    anyhow::ensure!(ax_vec.len() == b_vec.len(), "vector length mismatch");

    let mut r2 = 0.0_f64;
    let mut b2 = 0.0_f64;
    for (ax_i, b_i) in ax_vec.iter().zip(b_vec.iter()) {
        let r_i = ax_i - b_i;
        r2 += r_i * r_i;
        b2 += b_i * b_i;
    }
    let rel_r2 = if b2 > 0.0 {
        (r2 / b2).sqrt()
    } else {
        r2.sqrt()
    };
    println!("N={n_sites}, a0=0, a1=1");
    println!("||r||_2 / ||b||_2 = {:.3e}", rel_r2);

    Ok(())
}
