//! Benchmark and correctness harness for issue #1186 part 1: a partial
//! `apply_linear_operator` must not contract and factorize the nodes its
//! operator does not touch.
//!
//! Run it with
//!
//! ```text
//! cargo run --release -p tensor4all-treetn --example benchmark_partial_apply_untouched
//! ```
//!
//! The reported wall time is only meaningful on an otherwise idle machine; the
//! work-section numbers (nodes skipped, agreement with the exact local apply)
//! are deterministic. The same file compiles against a checkout without the
//! pass-through, so running it before and after that change measures the change
//! itself.
//!
//! The state is a comb: a degree-three scalar hub with three chains of `length`
//! binary sites, bond dimensions `min(2^(i+1), max_bond)` and an amplitude
//! weight `exp(-decay * bond dimension)` per leg, so the spectra decay
//! geometrically. The operator is a random MPO of bond dimension `mpo_bond` on
//! one branch.

use std::collections::HashMap;
use std::time::Instant;

use tensor4all_core::{DynIndex, IdxTensor, IndexLike, SvdTruncationPolicy};
use tensor4all_treetn::{
    apply_linear_operator, ApplyOptions, IndexMapping, LinearOperator, TreeTN,
};

/// Small deterministic value stream (an LCG), so runs are comparable.
struct Values(u64);

impl Values {
    fn take(&mut self, count: usize) -> Vec<f64> {
        (0..count)
            .map(|_| {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((self.0 >> 33) as f64 / (1u64 << 31) as f64) - 0.5
            })
            .collect()
    }
}

fn index(dim: usize) -> DynIndex {
    DynIndex::new_dyn(dim)
}

fn build_comb(
    branches: usize,
    length: usize,
    max_bond: usize,
    decay: f64,
) -> (TreeTN<IdxTensor, String>, Vec<(String, DynIndex)>) {
    let mut values = Values(0x5eed_1186);
    let mut tensors: Vec<IdxTensor> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut sites: Vec<(String, DynIndex)> = Vec::new();

    let hub_site = index(1);
    let hub_bonds: Vec<DynIndex> = (0..branches).map(|_| index(max_bond.min(3))).collect();
    let mut hub_legs = vec![hub_site.clone()];
    hub_legs.extend(hub_bonds.iter().cloned());
    let hub_len: usize = hub_legs.iter().map(|leg| leg.dim()).product();
    tensors.push(IdxTensor::from_dense(hub_legs, values.take(hub_len)).unwrap());
    names.push("hub".to_string());
    sites.push(("hub".to_string(), hub_site));

    for (branch, hub_bond) in hub_bonds.iter().enumerate() {
        // The hub owns the first bond; the rest are fresh.
        let mut bonds: Vec<DynIndex> = vec![hub_bond.clone()];
        bonds.extend((0..length).map(|i| index((1usize << (i + 1)).min(max_bond))));

        for site_index in 0..length {
            let site = index(2);
            let mut legs = vec![bonds[site_index].clone(), site.clone()];
            if site_index + 1 < length {
                legs.push(bonds[site_index + 1].clone());
            }
            let weights: Vec<f64> = legs
                .iter()
                .map(|leg| (-decay * leg.dim() as f64).exp())
                .collect();
            let len: usize = legs.iter().map(|leg| leg.dim()).product();
            let raw = values.take(len);
            let data = raw
                .iter()
                .enumerate()
                .map(|(flat, value)| {
                    // Column-major position `flat` over `legs`; only the weight
                    // of the leg each digit selects matters here.
                    let mut rest = flat;
                    let mut weight = 1.0;
                    for (leg, leg_weight) in legs.iter().zip(weights.iter()) {
                        let digit = rest % leg.dim();
                        rest /= leg.dim();
                        weight *= leg_weight.powi(digit as i32);
                    }
                    value * weight
                })
                .collect();
            tensors.push(IdxTensor::from_dense(legs, data).unwrap());
            names.push(format!("{branch}_{site_index}"));
            sites.push((format!("{branch}_{site_index}"), site));
        }
    }

    (TreeTN::from_tensors(tensors, names).unwrap(), sites)
}

/// A random MPO over `count` consecutive sites of branch `branch`, starting at
/// `first`.
fn build_branch_operator(
    sites: &[(String, DynIndex)],
    branch: usize,
    first: usize,
    count: usize,
    mpo_bond: usize,
) -> LinearOperator<IdxTensor, String> {
    let mut values = Values(0xfeed_1186);
    let inputs: Vec<DynIndex> = (0..count).map(|_| index(2)).collect();
    let outputs: Vec<DynIndex> = (0..count).map(|_| index(2)).collect();
    let bonds: Vec<DynIndex> = (0..count.saturating_sub(1))
        .map(|_| index(mpo_bond))
        .collect();

    let mut tensors = Vec::with_capacity(count);
    let mut names = Vec::with_capacity(count);
    for site in 0..count {
        let mut legs = Vec::new();
        if site > 0 {
            legs.push(bonds[site - 1].clone());
        }
        legs.push(inputs[site].clone());
        legs.push(outputs[site].clone());
        if site + 1 < count {
            legs.push(bonds[site].clone());
        }
        let len: usize = legs.iter().map(|leg| leg.dim()).product();
        tensors.push(IdxTensor::from_dense(legs, values.take(len)).unwrap());
        names.push(format!("{branch}_{}", first + site));
    }
    let mpo = TreeTN::from_tensors(tensors, names.clone()).unwrap();

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    for (site, name) in names.iter().enumerate() {
        let true_index = sites
            .iter()
            .find(|(node, _)| node == name)
            .expect("branch node")
            .1
            .clone();
        input_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index: true_index.clone(),
                internal_index: inputs[site].clone(),
            },
        );
        output_mapping.insert(
            name.clone(),
            IndexMapping {
                true_index,
                internal_index: outputs[site].clone(),
            },
        );
    }

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

/// Largest absolute difference between two dense results of the same apply.
///
/// The apply result keeps the identity extension's dimension-one legs and the
/// exact local reference does not, so the leg counts differ and `sub`/`maxabs`
/// cannot be used; the data sequences line up because every node carries one
/// site index.
fn dense_deviation(left: &IdxTensor, right: &IdxTensor) -> f64 {
    let left = left.to_vec::<f64>().unwrap();
    let right = right.to_vec::<f64>().unwrap();
    assert_eq!(left.len(), right.len(), "amplitude counts");
    for (new, reference) in left.iter().zip(right.iter()) {
        assert!((new - reference).is_finite(), "non-finite difference");
    }
    left.iter()
        .zip(right.iter())
        .map(|(new, reference)| (new - reference).abs())
        .fold(0.0, f64::max)
}

fn main() {
    // Agreement with the exact local apply on a comb small enough to compare
    // densely (13 sites, 8192 amplitudes).
    let (state, sites) = build_comb(3, 4, 16, 0.1);
    let state = state
        .canonicalize(["hub".to_string()], Default::default())
        .unwrap();
    let operator = build_branch_operator(&sites, 0, 0, 4, 4);
    let options = ApplyOptions {
        center: Some("hub".to_string()),
        ..ApplyOptions::zipup()
    };
    let result = apply_linear_operator(&operator, &state, options).unwrap();
    let exact = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
    println!(
        "13-site comb (4 sites per branch, bonds <= 16): max |zipup - exact local| = {:e}",
        dense_deviation(&result.to_dense().unwrap(), &exact.to_dense().unwrap())
    );

    // The shape the issue reports: three 20-site branches, bonds up to
    // `MAX_BOND`, with the operator covering a whole branch first and then only
    // three of its sites, so the untouched share of the nodes grows.
    const LENGTH: usize = 20;
    const MAX_BOND: usize = 160;
    const MPO_BOND: usize = 14;
    const REPEATS: usize = 3;

    let (state, sites) = build_comb(3, LENGTH, MAX_BOND, 0.1);
    let state = state
        .canonicalize(["hub".to_string()], Default::default())
        .unwrap();
    for covered in [LENGTH, 3] {
        // Every node outside the operator's support is an identity, except the
        // hub, which receives the operator branch's remainder and is factorized.
        let skipped = 3 * LENGTH - covered;
        let operator = build_branch_operator(&sites, 0, 0, covered, MPO_BOND);
        let options = ApplyOptions {
            center: Some("hub".to_string()),
            svd_policy: Some(SvdTruncationPolicy::new(1.0e-8)),
            ..ApplyOptions::zipup()
        };
        let started = Instant::now();
        for _ in 0..REPEATS {
            let result = apply_linear_operator(&operator, &state, options.clone()).unwrap();
            std::hint::black_box(&result);
        }
        let elapsed = started.elapsed().as_secs_f64();
        println!(
            "{}-site comb (3 branches x {LENGTH} sites, bonds <= {MAX_BOND}, bond-{MPO_BOND} MPO \
             on {covered} sites of one branch, {skipped} nodes skipped): {:.3} s for \
             {REPEATS} applies ({:.3} s each)",
            3 * LENGTH + 1,
            elapsed,
            elapsed / REPEATS as f64
        );
    }
}
