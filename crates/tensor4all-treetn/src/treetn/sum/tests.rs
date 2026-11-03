//! `TreeTN::exact_sum`: the engines build the same network, `NaryWithin`
//! switches at its limit, and the size estimate is the built sum's.

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};

use super::*;

/// A four-node star `1 - 0 - 2, 0 - 3` over `sites`, with bond dimensions
/// `bonds` to the leaves and values derived from `seed`.
fn star_term(sites: &[DynIndex], bonds: [usize; 3], seed: usize) -> TreeTN<IdxTensor, usize> {
    let values = |len: usize, seed: usize| -> Vec<f64> {
        (0..len)
            .map(|k| ((seed * 31 + k) as f64 * 0.618).sin())
            .collect()
    };
    let links: Vec<DynIndex> = bonds.iter().map(|&dim| DynIndex::new_dyn(dim)).collect();
    let mut tensors = vec![IdxTensor::from_dense(
        vec![
            sites[0].clone(),
            links[0].clone(),
            links[1].clone(),
            links[2].clone(),
        ],
        values(2 * bonds.iter().product::<usize>(), seed),
    )
    .unwrap()];
    for (leaf, link) in links.iter().enumerate() {
        tensors.push(
            IdxTensor::from_dense(
                vec![link.clone(), sites[leaf + 1].clone()],
                values(2 * link.dim(), seed + leaf + 1),
            )
            .unwrap(),
        );
    }
    TreeTN::from_tensors(tensors, vec![0, 1, 2, 3]).unwrap()
}

fn star_sites() -> Vec<DynIndex> {
    (0..4).map(|_| DynIndex::new_dyn(2)).collect()
}

const BONDS: [[usize; 3]; 3] = [[1, 2, 3], [2, 1, 1], [3, 2, 2]];

/// Summed bonds 6, 5, 6: center 2 * 6 * 5 * 6 elements, leaves 6 * 2, 5 * 2
/// and 6 * 2, at 16 bytes each.
const STAR_ESTIMATE: usize = (2 * 6 * 5 * 6 + 6 * 2 + 5 * 2 + 6 * 2) * 16;

fn star_terms(sites: &[DynIndex]) -> Vec<TreeTN<IdxTensor, usize>> {
    BONDS
        .iter()
        .enumerate()
        .map(|(seed, &bonds)| star_term(sites, bonds, 10 * seed))
        .collect()
}

fn assert_same_tensors(left: &TreeTN<IdxTensor, usize>, right: &TreeTN<IdxTensor, usize>) {
    for node in left.node_names() {
        let a = left.tensor(left.node_index(&node).unwrap()).unwrap();
        let b = right.tensor(right.node_index(&node).unwrap()).unwrap();
        assert_eq!(a.dims(), b.dims(), "node {node}");
        assert_eq!(
            a.to_vec::<f64>().unwrap(),
            b.to_vec::<f64>().unwrap(),
            "node {node}"
        );
    }
}

#[test]
fn exact_sum_engines_build_the_same_network() {
    let sites = star_sites();
    let terms = star_terms(&sites);
    let refs: Vec<_> = terms.iter().collect();

    let binary = TreeTN::exact_sum(&refs, SumEngine::Binary).unwrap();
    let nary = TreeTN::exact_sum(&refs, SumEngine::Nary).unwrap();

    assert!(!binary.used_nary);
    assert!(nary.used_nary);
    assert_eq!(binary.estimated_nary_bytes, STAR_ESTIMATE);
    assert_eq!(nary.estimated_nary_bytes, STAR_ESTIMATE);
    assert_same_tensors(&binary.network, &nary.network);
}

#[test]
fn exact_sum_default_engine_is_binary() {
    let sites = star_sites();
    let terms = star_terms(&sites);
    let refs: Vec<_> = terms.iter().collect();

    let sum = TreeTN::exact_sum(&refs, SumEngine::default()).unwrap();

    assert!(!sum.used_nary);
}

#[test]
fn nary_within_switches_exactly_at_its_limit() {
    let sites = star_sites();
    let terms = star_terms(&sites);
    let refs: Vec<_> = terms.iter().collect();

    let at_limit = TreeTN::exact_sum(
        &refs,
        SumEngine::NaryWithin {
            limit_bytes: STAR_ESTIMATE,
        },
    )
    .unwrap();
    let below_limit = TreeTN::exact_sum(
        &refs,
        SumEngine::NaryWithin {
            limit_bytes: STAR_ESTIMATE - 1,
        },
    )
    .unwrap();

    assert!(at_limit.used_nary);
    assert!(!below_limit.used_nary);
    assert_same_tensors(&at_limit.network, &below_limit.network);
}

#[test]
fn exact_sum_of_one_term_returns_it() {
    let sites = star_sites();
    let term = star_term(&sites, [1, 2, 3], 0);

    for engine in [SumEngine::Binary, SumEngine::Nary] {
        let sum = TreeTN::exact_sum(&[&term], engine).unwrap();
        assert_same_tensors(&sum.network, &term);
        assert_eq!(sum.network.link_dims(), term.link_dims());
    }
}

#[test]
fn exact_sum_checks_the_terms_before_estimating() {
    let sites = star_sites();
    let term = star_term(&sites, [1, 1, 1], 0);
    let single = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![IdxTensor::from_dense(vec![sites[0].clone()], vec![1.0, 2.0]).unwrap()],
        vec![0],
    )
    .unwrap();

    let no_terms = TreeTN::<IdxTensor, usize>::exact_sum(&[], SumEngine::Nary).unwrap_err();
    assert!(
        format!("{no_terms:#}").contains("at least one term"),
        "{no_terms:#}"
    );
    let mismatch = TreeTN::exact_sum(&[&term, &single], SumEngine::Nary).unwrap_err();
    assert!(
        format!("{mismatch:#}").contains("different topologies"),
        "{mismatch:#}"
    );
}
