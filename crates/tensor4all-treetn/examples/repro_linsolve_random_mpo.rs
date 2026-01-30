//! Minimal reproduction test for linsolve with random MPO operator.
//!
//! Two test cases:
//! 1. A = Identity operator, x_true = all-ones MPO, b = A * x_true
//!    Expected: converges (this works in other tests)
//! 2. A = Random MPO operator, x_true = all-ones MPO, b = A * x_true
//!    Expected: may not converge (reproduces the issue)
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example repro_linsolve_random_mpo --release

use std::collections::{HashMap, HashSet};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor4all_core::{
    index::{DynId, Index},
    DynIndex, IndexLike, TensorDynLen, TensorLike,
};
use tensor4all_treetn::{
    apply_linear_operator, apply_local_update_sweep, ApplyOptions, CanonicalizationOptions,
    IndexMapping, LinearOperator, LinsolveOptions, LocalUpdateSweepPlan, SquareLinsolveUpdater,
    TreeTN,
};

fn make_node_name(i: usize) -> String {
    format!("site{i}")
}

/// Create a DynIndex with a deterministic ID from the seeded RNG.
fn unique_dyn_index(used: &mut HashSet<DynId>, dim: usize, rng: &mut StdRng) -> DynIndex {
    loop {
        let id = DynId(rng.gen());
        if used.insert(id) {
            return Index::new(id, dim);
        }
    }
}

fn mpo_node_indices(
    n: usize,
    i: usize,
    bonds: &[DynIndex],
    s_out: &[DynIndex],
    s_in: &[DynIndex],
) -> Vec<DynIndex> {
    if n == 1 {
        vec![s_out[i].clone(), s_in[i].clone()]
    } else if i == 0 {
        vec![s_out[i].clone(), s_in[i].clone(), bonds[i].clone()]
    } else if i + 1 == n {
        vec![bonds[i - 1].clone(), s_out[i].clone(), s_in[i].clone()]
    } else {
        vec![
            bonds[i - 1].clone(),
            s_out[i].clone(),
            s_in[i].clone(),
            bonds[i].clone(),
        ]
    }
}

fn bond_indices(indices: &[DynIndex]) -> Vec<DynIndex> {
    indices
        .iter()
        .filter(|idx| idx.dim() == 1)
        .cloned()
        .collect()
}

/// Create an N-site identity MPO for the operator A.
/// Tensors have [s_out, s_in] (+ bonds) with no separate "external" index.
#[allow(clippy::type_complexity)]
fn create_identity_operator(
    n: usize,
    phys_dim: usize,
    true_site_indices: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
    rng: &mut StdRng,
) -> anyhow::Result<(
    TreeTN<TensorDynLen, String>,
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
)> {
    anyhow::ensure!(true_site_indices.len() == n, "site index count mismatch");

    let mut mpo = TreeTN::<TensorDynLen, String>::new();

    // MPO bonds: dim 1 for identity
    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, 1, rng))
        .collect();

    let s_in_tmp: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim, rng))
        .collect();
    let s_out_tmp: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim, rng))
        .collect();

    let mut input_mapping: HashMap<String, IndexMapping<DynIndex>> = HashMap::new();
    let mut output_mapping: HashMap<String, IndexMapping<DynIndex>> = HashMap::new();
    let mut nodes = Vec::with_capacity(n);

    for i in 0..n {
        let node_name = make_node_name(i);
        let indices = mpo_node_indices(n, i, &bonds, &s_out_tmp, &s_in_tmp);

        // Identity: δ(s_out, s_in)
        let mut base_data = vec![0.0_f64; phys_dim * phys_dim];
        for k in 0..phys_dim {
            base_data[k * phys_dim + k] = 1.0;
        }
        let base = TensorDynLen::from_dense_f64(
            vec![s_out_tmp[i].clone(), s_in_tmp[i].clone()],
            base_data,
        );

        let t = if indices.len() == 2 {
            base
        } else {
            let bond_inds = bond_indices(&indices);
            let ones = TensorDynLen::from_dense_f64(bond_inds, vec![1.0_f64; 1]);
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

/// Create an N-site random MPO for the operator A.
/// Tensors have [s_out, s_in] (+ bonds) with no separate "external" index.
#[allow(clippy::type_complexity)]
fn create_random_operator(
    n: usize,
    phys_dim: usize,
    bond_dim: usize,
    true_site_indices: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
    rng: &mut StdRng,
) -> anyhow::Result<(
    TreeTN<TensorDynLen, String>,
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
)> {
    anyhow::ensure!(true_site_indices.len() == n, "site index count mismatch");

    let mut mpo = TreeTN::<TensorDynLen, String>::new();

    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, bond_dim, rng))
        .collect();

    let s_in_tmp: Vec<DynIndex> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim, rng))
        .collect();
    let s_out_tmp: Vec<DynIndex> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim, rng))
        .collect();

    let mut input_mapping: HashMap<String, IndexMapping<DynIndex>> = HashMap::new();
    let mut output_mapping: HashMap<String, IndexMapping<DynIndex>> = HashMap::new();

    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let node_name = make_node_name(i);
        let indices = mpo_node_indices(n, i, &bonds, &s_out_tmp, &s_in_tmp);
        let t = TensorDynLen::random_f64(rng, indices);

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

/// Create an N-site all-ones MPO for x_true.
/// Returns (mpo, input_mapping, output_mapping).
#[allow(clippy::type_complexity)]
fn create_ones_mpo(
    n: usize,
    phys_dim: usize,
    bond_dim: usize,
    true_site_indices: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
    rng: &mut StdRng,
) -> anyhow::Result<(
    TreeTN<TensorDynLen, String>,
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
)> {
    anyhow::ensure!(true_site_indices.len() == n, "site index count mismatch");

    let mut mpo = TreeTN::<TensorDynLen, String>::new();

    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, bond_dim, rng))
        .collect();

    let s_in_tmp: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim, rng))
        .collect();
    let s_out_tmp: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim, rng))
        .collect();

    let mut input_mapping: HashMap<String, IndexMapping<DynIndex>> = HashMap::new();
    let mut output_mapping: HashMap<String, IndexMapping<DynIndex>> = HashMap::new();

    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let node_name = make_node_name(i);
        let indices = mpo_node_indices(n, i, &bonds, &s_out_tmp, &s_in_tmp);

        // Include true_site_index as external index
        let mut all_indices = vec![true_site_indices[i].clone()];
        all_indices.extend(indices.iter().cloned());

        let t = TensorDynLen::ones(&all_indices)?;

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

fn compute_rel_residual(
    op: &TreeTN<TensorDynLen, String>,
    im: &HashMap<String, IndexMapping<DynIndex>>,
    om: &HashMap<String, IndexMapping<DynIndex>>,
    a0: f64,
    a1: f64,
    x: &TreeTN<TensorDynLen, String>,
    rhs: &TreeTN<TensorDynLen, String>,
) -> anyhow::Result<f64> {
    let linop = LinearOperator::new(op.clone(), im.clone(), om.clone());
    let ax = apply_linear_operator(&linop, x, ApplyOptions::default())?;

    let ax_full = ax.contract_to_tensor()?;
    let x_full = x.contract_to_tensor()?;
    let b_full = rhs.contract_to_tensor()?;

    let ax_vec = ax_full.to_vec_f64()?;
    let x_vec = x_full.to_vec_f64()?;
    let b_vec = b_full.to_vec_f64()?;

    anyhow::ensure!(ax_vec.len() == b_vec.len(), "vector length mismatch");
    anyhow::ensure!(x_vec.len() == b_vec.len(), "vector length mismatch");

    let mut r2 = 0.0_f64;
    let mut b2 = 0.0_f64;
    for ((ax_i, x_i), b_i) in ax_vec.iter().zip(x_vec.iter()).zip(b_vec.iter()) {
        let opx_i = a0 * x_i + a1 * ax_i;
        let r_i = opx_i - b_i;
        r2 += r_i * r_i;
        b2 += b_i * b_i;
    }
    Ok(if b2 > 0.0 {
        (r2 / b2).sqrt()
    } else {
        r2.sqrt()
    })
}

fn run_linsolve_test(
    test_name: &str,
    operator: &TreeTN<TensorDynLen, String>,
    input_mapping: &HashMap<String, IndexMapping<DynIndex>>,
    output_mapping: &HashMap<String, IndexMapping<DynIndex>>,
    rhs: &TreeTN<TensorDynLen, String>,
    a0: f64,
    a1: f64,
    n_sweeps: usize,
    max_rank: usize,
) -> anyhow::Result<()> {
    println!("--- {test_name} ---");

    let n = operator.node_count();
    let center = make_node_name(n / 2);

    let options = LinsolveOptions::default()
        .with_nfullsweeps(n_sweeps)
        .with_max_rank(max_rank)
        .with_krylov_tol(1e-8)
        .with_krylov_maxiter(50)
        .with_krylov_dim(30)
        .with_coefficients(a0, a1);

    // Initial guess: x0 = rhs
    let init = rhs.clone();
    let mut x = init.canonicalize([center.clone()], CanonicalizationOptions::default())?;

    let plan = LocalUpdateSweepPlan::from_treetn(&x, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    let mut updater = SquareLinsolveUpdater::with_index_mappings(
        operator.clone(),
        input_mapping.clone(),
        output_mapping.clone(),
        rhs.clone(),
        options,
    );

    let r0 = compute_rel_residual(operator, input_mapping, output_mapping, a0, a1, &x, rhs)?;
    println!("  Initial residual: {:.6e}", r0);

    for sweep in 1..=n_sweeps {
        apply_local_update_sweep(&mut x, &plan, &mut updater)?;
        let r = compute_rel_residual(operator, input_mapping, output_mapping, a0, a1, &x, rhs)?;
        println!("  After sweep {sweep}: residual = {:.6e}", r);
    }

    let r_final = compute_rel_residual(operator, input_mapping, output_mapping, a0, a1, &x, rhs)?;
    println!("  Final residual: {:.6e}", r_final);

    if r_final < 1e-4 {
        println!("  [PASS] Converged");
    } else {
        println!("  [FAIL] Did not converge (residual > 1e-4)");
    }
    println!();

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let n_sites = 3usize;
    let phys_dim = 2usize;
    let bond_dim = 4usize; // Small bond dimension for testing
    let n_sweeps = 5usize;
    let seed = 1234_u64;

    println!("=== Minimal Reproduction Test for linsolve ===");
    println!("N = {n_sites}, phys_dim = {phys_dim}, bond_dim = {bond_dim}");
    println!("n_sweeps = {n_sweeps}, seed = {seed}");
    println!();

    let mut used_ids = HashSet::<DynId>::new();
    let mut rng = StdRng::seed_from_u64(seed);

    // Create true site indices (shared)
    let true_site_indices: Vec<_> = (0..n_sites)
        .map(|_| unique_dyn_index(&mut used_ids, phys_dim, &mut rng))
        .collect();

    // ========================================
    // Test 1: A = Identity operator
    // ========================================
    println!("=== Test 1: A = Identity operator ===");

    let (identity_op, id_input_mapping, id_output_mapping) = create_identity_operator(
        n_sites,
        phys_dim,
        &true_site_indices,
        &mut used_ids,
        &mut rng,
    )?;

    // x_true = all-ones MPO (bond_dim=1 for simplicity with identity)
    let (x_true_id, _, _) = create_ones_mpo(
        n_sites,
        phys_dim,
        1, // bond_dim = 1 for identity test
        &true_site_indices,
        &mut used_ids,
        &mut rng,
    )?;

    // b = A * x_true = I * x_true = x_true
    let linop_id = LinearOperator::new(
        identity_op.clone(),
        id_input_mapping.clone(),
        id_output_mapping.clone(),
    );
    let rhs_id = apply_linear_operator(&linop_id, &x_true_id, ApplyOptions::default())?;

    // Equation: I * x = b (a0=1, a1=0 means (1*I + 0*A)*x = b => I*x = b)
    // But for A*x = b, we use a0=0, a1=1
    run_linsolve_test(
        "Identity operator: A*x = b where A=I, b=A*x_true",
        &identity_op,
        &id_input_mapping,
        &id_output_mapping,
        &rhs_id,
        0.0, // a0
        1.0, // a1: (0*I + 1*A)*x = b => A*x = b
        n_sweeps,
        4,
    )?;

    // ========================================
    // Test 2: A = Random MPO operator
    // ========================================
    println!("=== Test 2: A = Random MPO operator ===");

    let (random_op, rand_input_mapping, rand_output_mapping) = create_random_operator(
        n_sites,
        phys_dim,
        bond_dim,
        &true_site_indices,
        &mut used_ids,
        &mut rng,
    )?;

    // x_true = all-ones MPO
    let (x_true_rand, _, _) = create_ones_mpo(
        n_sites,
        phys_dim,
        bond_dim,
        &true_site_indices,
        &mut used_ids,
        &mut rng,
    )?;

    // b = A * x_true
    let linop_rand = LinearOperator::new(
        random_op.clone(),
        rand_input_mapping.clone(),
        rand_output_mapping.clone(),
    );
    let rhs_rand = apply_linear_operator(&linop_rand, &x_true_rand, ApplyOptions::default())?;

    run_linsolve_test(
        "Random operator: A*x = b where A=random MPO, b=A*x_true",
        &random_op,
        &rand_input_mapping,
        &rand_output_mapping,
        &rhs_rand,
        0.0, // a0
        1.0, // a1
        n_sweeps,
        bond_dim * 2, // Allow higher rank for random case
    )?;

    // ========================================
    // Test 2b: Random operator with only 1 sweep (to verify sweep 1 works)
    // ========================================
    println!("=== Test 2b: Random operator with ONLY 1 sweep ===");

    // Recreate with fresh RNG state for reproducibility
    let mut rng2 = StdRng::seed_from_u64(seed + 100);
    let mut used_ids2 = HashSet::<DynId>::new();

    let true_site_indices2: Vec<_> = (0..n_sites)
        .map(|_| unique_dyn_index(&mut used_ids2, phys_dim, &mut rng2))
        .collect();

    let (random_op2, rand_input_mapping2, rand_output_mapping2) = create_random_operator(
        n_sites,
        phys_dim,
        bond_dim,
        &true_site_indices2,
        &mut used_ids2,
        &mut rng2,
    )?;

    let (x_true_rand2, _, _) = create_ones_mpo(
        n_sites,
        phys_dim,
        bond_dim,
        &true_site_indices2,
        &mut used_ids2,
        &mut rng2,
    )?;

    let linop_rand2 = LinearOperator::new(
        random_op2.clone(),
        rand_input_mapping2.clone(),
        rand_output_mapping2.clone(),
    );
    let rhs_rand2 = apply_linear_operator(&linop_rand2, &x_true_rand2, ApplyOptions::default())?;

    run_linsolve_test(
        "Random operator (1 sweep only)",
        &random_op2,
        &rand_input_mapping2,
        &rand_output_mapping2,
        &rhs_rand2,
        0.0,
        1.0,
        1, // Only 1 sweep
        bond_dim * 2,
    )?;

    println!("=== Test completed ===");
    Ok(())
}
