mod dispatch;

use super::*;
use tensor4all_core::DynIndex;

fn varying_bonds() -> TreeTN<IdxTensor, usize> {
    let bonds: Vec<_> = (1..=3).map(DynIndex::new_dyn).collect();
    let mut hub = vec![DynIndex::new_dyn(2)];
    hub.extend(bonds.iter().cloned());
    let mut tensors = vec![IdxTensor::from_dense(hub, vec![1.; 12]).unwrap()];
    tensors.extend(bonds.into_iter().enumerate().map(|(node, bond)| {
        IdxTensor::from_dense(vec![bond, DynIndex::new_dyn(2)], vec![1.; 2 * (node + 1)]).unwrap()
    }));
    TreeTN::from_tensors(tensors, vec![0, 1, 2, 3]).unwrap()
}

fn expected(tree: &TreeTN<IdxTensor, usize>, root: usize) -> Vec<EdgeRank<usize>> {
    tree.edges_to_canonicalize_by_names(&root)
        .unwrap()
        .into_iter()
        .map(|(child, parent)| EdgeRank {
            rank: child.max(parent),
            child,
            parent,
        })
        .collect()
}

#[test]
fn stored_bonds_and_first_saturated_edge_follow_report_order() {
    let tree = varying_bonds();
    let expected = expected(&tree, 0);
    for threshold in [
        None,
        NonZeroUsize::new(1),
        NonZeroUsize::new(2),
        NonZeroUsize::new(3),
        NonZeroUsize::new(4),
    ] {
        match finish(tree.clone(), &0, threshold).unwrap() {
            ContractionOutcome::Completed { report, .. } => {
                assert!(threshold.is_none() || threshold == NonZeroUsize::new(4));
                assert_eq!(report.edge_ranks, expected);
            }
            ContractionOutcome::Saturated { edge } => {
                let threshold = threshold.unwrap().get();
                assert_eq!(
                    &edge,
                    expected.iter().find(|e| e.rank >= threshold).unwrap()
                );
            }
        }
    }
}

#[test]
fn requested_root_then_unique_canonical_root_then_smallest_node() {
    let mut tree = varying_bonds();
    for (canonical, requested, root) in [
        (vec![3], 0, 0),
        (vec![3], 99, 3),
        (vec![], 99, 0),
        (vec![0, 1], 99, 0),
    ] {
        tree.set_canonical_region(canonical).unwrap();
        let ContractionOutcome::Completed { report, .. } =
            finish(tree.clone(), &requested, None).unwrap()
        else {
            panic!("no threshold must complete");
        };
        assert_eq!(report.edge_ranks, expected(&tree, root));
    }
}

#[test]
fn invalid_output_metadata_returns_diagnostics() {
    let mut tree = varying_bonds();
    tree.canonical_region = [99].into_iter().collect();
    let error = finish(tree, &88, None).unwrap_err();
    assert!(error.to_string().contains("output root is absent"));

    let mut tree = varying_bonds();
    let removed = tree.edge_between(&0, &2).unwrap();
    tree.graph.graph_mut().remove_edge(removed);
    let error = finish(tree, &0, None).unwrap_err();
    assert!(error.to_string().contains("missing output bond"));
}
