use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

use super::prepare_problem;
use crate::{TreeAciError, TreeAciOptions};

fn make_tree(
    edges: &[(usize, usize)],
    physical: &[Vec<DynIndex>],
    bond_dim: usize,
) -> TreeTN<IdxTensor, usize> {
    let bonds = edges
        .iter()
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();
    let tensors = (0..physical.len())
        .map(|node| {
            let mut indices = physical[node].clone();
            for (edge, &(left, right)) in edges.iter().enumerate() {
                if left == node || right == node {
                    indices.push(bonds[edge].clone());
                }
            }
            let len = indices.iter().map(IndexLike::dim).product();
            IdxTensor::from_dense(indices, vec![1.0; len]).unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..physical.len()).collect()).unwrap()
}

fn one_site_indices(node_count: usize) -> Vec<Vec<DynIndex>> {
    (0..node_count)
        .map(|_| vec![DynIndex::new_dyn(2)])
        .collect()
}

#[test]
fn accepts_path_y_and_degree_four_topologies() {
    for edges in [
        vec![(0, 1), (1, 2), (2, 3)],
        vec![(0, 1), (0, 2), (0, 3)],
        vec![(0, 1), (0, 2), (0, 3), (0, 4)],
    ] {
        let node_count = edges.len() + 1;
        let tree = make_tree(&edges, &one_site_indices(node_count), 2);
        let prepared = prepare_problem::<f64, _>(&[tree], &TreeAciOptions::default()).unwrap();
        assert_eq!(prepared.node_order.len(), node_count);
        assert_eq!(prepared.directed_edges.len(), 2 * edges.len());
    }
}

#[test]
fn directed_edges_record_reverse_and_incoming_branches() {
    let tree = make_tree(&[(0, 1), (0, 2), (0, 3)], &one_site_indices(4), 2);
    let prepared = prepare_problem::<f64, _>(&[tree], &TreeAciOptions::default()).unwrap();

    for arc in &prepared.directed_edges {
        let reverse = &prepared.directed_edges[arc.reverse];
        assert_eq!(reverse.from, arc.to);
        assert_eq!(reverse.to, arc.from);
        assert_eq!(reverse.reverse, arc.id);
        assert!(arc
            .incoming_to_from
            .iter()
            .all(|incoming| prepared.directed_edges[*incoming].to == arc.from));
        assert!(arc
            .incoming_to_from
            .iter()
            .all(|incoming| prepared.directed_edges[*incoming].from != arc.to));
    }
    let away_from_center = prepared
        .directed_edges
        .iter()
        .filter(|arc| arc.from == 0)
        .collect::<Vec<_>>();
    assert!(away_from_center
        .iter()
        .all(|arc| arc.incoming_to_from.len() == 2));
}

#[test]
fn physical_layout_follows_reference_tensor_order_and_allows_zero_legs() {
    let scalar_tree = make_tree(&[], &[vec![]], 1);
    let scalar = prepare_problem::<f64, _>(&[scalar_tree], &TreeAciOptions::default()).unwrap();
    assert_eq!(scalar.physical[0].local_dim, 1);
    assert!(scalar.physical[0].strides.is_empty());

    let first = DynIndex::new_dyn(2);
    let second = DynIndex::new_dyn(3);
    let tree = make_tree(&[], &[vec![second.clone(), first.clone()]], 1);
    let prepared = prepare_problem::<f64, _>(&[tree], &TreeAciOptions::default()).unwrap();
    assert_eq!(prepared.physical[0].indices, vec![second, first]);
    assert_eq!(prepared.physical[0].dims, vec![3, 2]);
    assert_eq!(prepared.physical[0].strides, vec![1, 3]);
    assert_eq!(prepared.physical[0].local_dim, 6);
}

#[test]
fn full_physical_index_equality_is_required() {
    let site = DynIndex::new_dyn(2);
    let reference = make_tree(&[], &[vec![site.clone()]], 1);
    let primed = make_tree(&[], &[vec![site.prime()]], 1);

    assert!(matches!(
        prepare_problem::<f64, _>(&[reference, primed], &TreeAciOptions::default()),
        Err(TreeAciError::PhysicalIndexMismatch { input: 1, .. })
    ));
}

#[test]
fn input_bond_dimensions_may_differ() {
    let physical = one_site_indices(2);
    let rank_two = make_tree(&[(0, 1)], &physical, 2);
    let rank_five = make_tree(&[(0, 1)], &physical, 5);
    assert!(prepare_problem::<f64, _>(&[rank_two, rank_five], &TreeAciOptions::default()).is_ok());
}

#[test]
fn rejects_missing_empty_mismatched_and_oversized_inputs() {
    assert!(matches!(
        prepare_problem::<f64, usize>(&[], &TreeAciOptions::default()),
        Err(TreeAciError::NoInputs)
    ));
    assert!(matches!(
        prepare_problem::<f64, usize>(&[TreeTN::new()], &TreeAciOptions::default()),
        Err(TreeAciError::EmptyTree { input: 0 })
    ));

    let sites = one_site_indices(3);
    let path = make_tree(&[(0, 1), (1, 2)], &sites, 2);
    let star = make_tree(&[(0, 1), (0, 2)], &sites, 2);
    assert!(matches!(
        prepare_problem::<f64, _>(&[path, star], &TreeAciOptions::default()),
        Err(TreeAciError::TopologyMismatch { input: 1 })
    ));

    let tree = make_tree(&[], &[vec![DynIndex::new_dyn(8)]], 1);
    let options = TreeAciOptions {
        max_core_elements: Some(7),
        ..TreeAciOptions::default()
    };
    assert!(matches!(
        prepare_problem::<f64, _>(&[tree], &options),
        Err(TreeAciError::ResourceLimit {
            resource: "core elements",
            requested: 8,
            limit: 7
        })
    ));
}

#[test]
fn rejects_cycles_zero_dimensions_and_unknown_roots() {
    let b01 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(2);
    let b20 = DynIndex::new_dyn(2);
    let tensors = vec![
        IdxTensor::from_dense(vec![b01.clone(), b20.clone()], vec![1.0; 4]).unwrap(),
        IdxTensor::from_dense(vec![b01.clone(), b12.clone()], vec![1.0; 4]).unwrap(),
        IdxTensor::from_dense(vec![b12.clone(), b20.clone()], vec![1.0; 4]).unwrap(),
    ];
    let mut cycle = TreeTN::new();
    let nodes = tensors
        .into_iter()
        .enumerate()
        .map(|(name, tensor)| cycle.add_tensor(name, tensor).unwrap())
        .collect::<Vec<_>>();
    cycle.connect(nodes[0], &b01, nodes[1], &b01).unwrap();
    cycle.connect(nodes[1], &b12, nodes[2], &b12).unwrap();
    cycle.connect(nodes[2], &b20, nodes[0], &b20).unwrap();
    assert!(matches!(
        prepare_problem::<f64, _>(&[cycle], &TreeAciOptions::default()),
        Err(TreeAciError::TreeTN(_))
    ));

    let zero = DynIndex::new_dyn(0);
    assert!(IdxTensor::from_dense(vec![zero], Vec::<f64>::new()).is_err());

    let tree = make_tree(&[(0, 1)], &one_site_indices(2), 2);
    let options = TreeAciOptions {
        root: Some(9),
        ..TreeAciOptions::default()
    };
    assert!(matches!(
        prepare_problem::<f64, _>(&[tree], &options),
        Err(TreeAciError::InvalidOption { option: "root", .. })
    ));
}
