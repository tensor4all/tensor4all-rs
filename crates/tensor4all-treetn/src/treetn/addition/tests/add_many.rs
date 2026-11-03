//! N-ary exact sum of TreeTNs.

use tensor4all_core::{AnyScalar, DynIndex, IdxTensor, IndexLike};

use crate::treetn::TreeTN;

/// Site indices of a four-node tree whose node 0 has degree 3.
fn star_sites() -> Vec<DynIndex> {
    (0..4).map(|_| DynIndex::new_dyn(2)).collect()
}

fn values(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|k| ((seed * 31 + k) as f64 * 0.618).sin())
        .collect()
}

/// One term on the star `1 - 0 - 2, 0 - 3`, with bond dimensions `bonds`
/// (to leaves 1, 2, 3) and values derived from `seed`.
fn star_term(sites: &[DynIndex], bonds: [usize; 3], seed: usize) -> TreeTN<IdxTensor, usize> {
    let links: Vec<DynIndex> = bonds.iter().map(|&dim| DynIndex::new_dyn(dim)).collect();
    let center_len = 2 * bonds.iter().product::<usize>();
    let mut tensors = vec![IdxTensor::from_dense(
        vec![
            sites[0].clone(),
            links[0].clone(),
            links[1].clone(),
            links[2].clone(),
        ],
        values(center_len, seed),
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

fn dense_sum(terms: &[&TreeTN<IdxTensor, usize>]) -> IdxTensor {
    let mut sum = terms[0].to_dense().unwrap();
    for term in &terms[1..] {
        sum = sum
            .axpby(
                AnyScalar::new_real(1.0),
                &term.to_dense().unwrap(),
                AnyScalar::new_real(1.0),
            )
            .unwrap();
    }
    sum
}

#[test]
fn add_many_on_a_branching_tree_equals_the_dense_sum() {
    let sites = star_sites();
    let a = star_term(&sites, [1, 2, 3], 0);
    let b = star_term(&sites, [2, 1, 1], 10);
    let c = star_term(&sites, [3, 2, 2], 20);
    let d = star_term(&sites, [1, 1, 2], 30);
    let terms = [&a, &b, &c, &d];

    let sum = TreeTN::add_many(&terms).unwrap();

    let mut link_dims = sum.link_dims();
    link_dims.sort();
    assert_eq!(link_dims, vec![6, 7, 8]);
    let expected = dense_sum(&terms);
    let error = sum.to_dense().unwrap().distance(&expected).unwrap();
    assert!(error < 1e-12, "dense error {error}");
}

#[test]
fn add_many_has_the_same_tensors_as_an_add_chain() {
    let sites = star_sites();
    let a = star_term(&sites, [1, 2, 3], 0);
    let b = star_term(&sites, [2, 1, 1], 10);
    let c = star_term(&sites, [3, 2, 2], 20);

    let many = TreeTN::add_many(&[&a, &b, &c]).unwrap();
    let chain = a.add(&b).unwrap().add(&c).unwrap();

    // Same leg order and same blocks in the same positions: the payloads
    // agree bit for bit, only the bond index IDs differ.
    for node in many.node_names() {
        let many_tensor = many.tensor(many.node_index(&node).unwrap()).unwrap();
        let chain_tensor = chain.tensor(chain.node_index(&node).unwrap()).unwrap();
        assert_eq!(many_tensor.dims(), chain_tensor.dims(), "node {node}");
        assert_eq!(
            many_tensor.to_vec::<f64>().unwrap(),
            chain_tensor.to_vec::<f64>().unwrap(),
            "node {node}"
        );
    }
}

#[test]
fn add_many_of_one_term_returns_it_unchanged() {
    let sites = star_sites();
    let a = star_term(&sites, [1, 2, 3], 0);

    let sum = TreeTN::add_many(&[&a]).unwrap();

    for leaf in 1..4 {
        let bond = |tn: &TreeTN<IdxTensor, usize>| {
            tn.bond_index(tn.edge_between(&0, &leaf).unwrap())
                .unwrap()
                .clone()
        };
        assert_eq!(bond(&sum), bond(&a), "leaf {leaf}");
    }
    let error = sum
        .to_dense()
        .unwrap()
        .distance(&a.to_dense().unwrap())
        .unwrap();
    assert_eq!(error, 0.0);
}

#[test]
fn add_many_of_single_node_networks_adds_elementwise() {
    let s = DynIndex::new_dyn(3);
    let make = |values: Vec<f64>| {
        let tensor = IdxTensor::from_dense(vec![s.clone()], values).unwrap();
        TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0]).unwrap()
    };
    let a = make(vec![1.0, 2.0, 3.0]);
    let b = make(vec![10.0, 20.0, 30.0]);
    let c = make(vec![100.0, 200.0, 300.0]);

    let sum = TreeTN::add_many(&[&a, &b, &c]).unwrap();

    let expected = IdxTensor::from_dense(vec![s.clone()], vec![111.0, 222.0, 333.0]).unwrap();
    assert!(sum.to_dense().unwrap().distance(&expected).unwrap() < 1e-12);
}

#[test]
fn add_many_rejects_no_terms() {
    let err = TreeTN::<IdxTensor, usize>::add_many(&[]).unwrap_err();
    assert!(format!("{err:#}").contains("at least one term"), "{err:#}");
}

#[test]
fn add_many_rejects_a_later_term_with_another_topology() {
    let sites = star_sites();
    let a = star_term(&sites, [1, 1, 1], 0);
    let b = star_term(&sites, [1, 1, 1], 10);
    let single = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![IdxTensor::from_dense(vec![sites[0].clone()], vec![1.0, 2.0]).unwrap()],
        vec![0],
    )
    .unwrap();

    let err = TreeTN::add_many(&[&a, &b, &single]).unwrap_err();
    assert!(
        format!("{err:#}").contains("different topologies"),
        "{err:#}"
    );
}

#[test]
fn add_many_rejects_a_later_term_with_other_site_indices() {
    let sites = star_sites();
    let a = star_term(&sites, [1, 1, 1], 0);
    let b = star_term(&sites, [1, 1, 1], 10);
    let other = star_term(&star_sites(), [1, 1, 1], 20);

    let err = TreeTN::add_many(&[&a, &b, &other]).unwrap_err();
    assert!(
        format!("{err:#}").contains("different site indices"),
        "{err:#}"
    );
}

#[test]
fn add_many_rejects_mixed_scalar_types() {
    let sites = star_sites();
    let a = star_term(&sites, [1, 1, 1], 0);
    let b = star_term(&sites, [1, 1, 1], 10);
    let complex = b.scale(AnyScalar::new_complex(0.0, 1.0)).unwrap();

    let err = TreeTN::add_many(&[&a, &b, &complex]).unwrap_err();
    assert!(
        format!("{err:#}").contains("same dense scalar type"),
        "{err:#}"
    );
}
