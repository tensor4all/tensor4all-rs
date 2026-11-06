//! Regression tests for issue #695: the `max_bond_dim` contract of zip-up
//! contraction.
//!
//! For six binary sites the exact MPO x MPS product has bond dimensions at
//! most 8, but zip-up factorizes local blocks that still carry the open input
//! bonds, whose rank reaches 16 here. A cap of 8 therefore truncates
//! non-negligible weight, as documented on
//! `TreeTN::contract_zipup`; a cap covering the intermediate ranks is exact,
//! and SRC is exact at the final rank.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::contraction::{contract, ContractionOptions};
use tensor4all_treetn::TreeTN;

type Network = TreeTN<IdxTensor, String>;
const ALPHA: f64 = -0.5;

fn random_tensor(indices: Vec<DynIndex>, rng: &mut ChaCha8Rng) -> IdxTensor {
    let elements: usize = indices.iter().map(IndexLike::dim).product();
    let mut data: Vec<f64> = (0..elements)
        .map(|_| ALPHA + (1.0 - ALPHA) * rng.random_range(0.0_f64..1.0_f64))
        .collect();
    let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
    for value in &mut data {
        *value /= norm;
    }
    IdxTensor::from_dense(indices, data).expect("valid tensor")
}

fn make_mpo_mps(
    n_sites: usize,
    physical_dim: usize,
    bond_dim: usize,
    seed: u64,
) -> (Network, Network) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let inputs = (0..n_sites)
        .map(|_| DynIndex::new_dyn(physical_dim))
        .collect::<Vec<_>>();
    let outputs = (0..n_sites)
        .map(|_| DynIndex::new_dyn(physical_dim))
        .collect::<Vec<_>>();
    let operator_bonds = (0..n_sites - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();
    let state_bonds = (0..n_sites - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();
    let mut operator_tensors = Vec::with_capacity(n_sites);
    let mut state_tensors = Vec::with_capacity(n_sites);
    for site in 0..n_sites {
        let mut oi = Vec::with_capacity(4);
        if site > 0 {
            oi.push(operator_bonds[site - 1].clone());
        }
        oi.push(inputs[site].clone());
        oi.push(outputs[site].clone());
        if site + 1 < n_sites {
            oi.push(operator_bonds[site].clone());
        }
        operator_tensors.push(random_tensor(oi, &mut rng));

        let mut si = Vec::with_capacity(3);
        if site > 0 {
            si.push(state_bonds[site - 1].clone());
        }
        si.push(inputs[site].clone());
        if site + 1 < n_sites {
            si.push(state_bonds[site].clone());
        }
        state_tensors.push(random_tensor(si, &mut rng));
    }
    let names = (0..n_sites).map(|s| format!("S{s}")).collect::<Vec<_>>();
    (
        Network::from_tensors(operator_tensors, names.clone()).unwrap(),
        Network::from_tensors(state_tensors, names).unwrap(),
    )
}

fn relative_error(result: &Network, exact: &IdxTensor) -> f64 {
    let dense = result.to_dense().unwrap();
    dense.sub(exact).unwrap().norm().unwrap() / exact.norm().unwrap()
}

/// Zip-up is exact once `max_bond_dim` covers the intermediate local ranks
/// (16 for this fixture), not only the final rank (8).
#[test]
fn zipup_is_exact_when_cap_covers_intermediate_ranks() {
    let (operator, state) = make_mpo_mps(6, 2, 8, 7);
    let exact = operator.contract_naive(&state).unwrap();
    let center = "S0".to_string();
    for max_rank in [16usize, 64] {
        let result = contract(
            &operator,
            &state,
            &center,
            ContractionOptions::zipup().with_max_bond_dim(max_rank),
        )
        .unwrap();
        let err = relative_error(&result, &exact);
        assert!(err < 1e-10, "max_rank={max_rank}: relative error {err:e}");
        assert!(result.link_dims().iter().all(|&d| d <= 8));
    }
}

/// SRC reaches the exact product with the cap at the final rank.
#[test]
fn src_is_exact_with_cap_at_final_rank() {
    let (operator, state) = make_mpo_mps(6, 2, 8, 7);
    let exact = operator.contract_naive(&state).unwrap();
    let result = contract(
        &operator,
        &state,
        &"S0".to_string(),
        ContractionOptions::src().with_max_bond_dim(8),
    )
    .unwrap();
    let err = relative_error(&result, &exact);
    assert!(err < 1e-9, "relative error {err:e}");
}
