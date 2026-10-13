//! Directed components, component keys, the component cost index, and center
//! selection.

use super::*;

/// [AI Supplied] A directed component is a graph property, not a property
/// of the center chosen for one evaluation. Moving the center across a
/// chain must therefore reuse the same `2E` physical component layouts.
#[test]
fn directed_component_layouts_are_shared_across_centers() {
    let (tree, indices) = typed_three_node_chain::<f64>();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let values = [0usize, 0, 0, 1, 1, 1];
    let points = ColMajorArrayRef::new(&values, &[3usize, 2usize]).unwrap();

    for center in 0..3 {
        evaluator
            .evaluate_batched_with_hint(points, EvaluationHint::around(center))
            .unwrap();
    }

    assert_eq!(evaluator.directed_component_layouts().len(), 4);
    assert_eq!(
        evaluator
            .directed_component_layouts()
            .values()
            .map(|layout| layout.layout.input_positions.len())
            .sum::<usize>(),
        6
    );
}

/// [AI Supplied] The composed key is checked against the direct encoder
/// for every unique assignment in a nested directed component. This keeps
/// the cache-key representation an exact equality-preserving encoding,
/// rather than an opaque tuple of temporary assignment IDs.
#[test]
fn directed_component_composition_matches_direct_encoding() {
    let (tree, indices) = typed_three_node_chain::<f64>();
    let evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let values = [0usize, 0, 0, 1, 1, 1, 2, 0, 1];
    let points = ColMajorArrayRef::new(&values, &[3usize, 3usize]).unwrap();
    let plan = RootedMessagePlan::new(&tree, &0).unwrap();
    let assignment_batches = evaluator
        .build_message_assignment_batches(&plan, points)
        .unwrap();

    for node in &plan.postorder {
        let parent = plan.parent.get(node).and_then(Clone::clone).unwrap();
        let component = evaluator
            .directed_component_layouts()
            .get(&(*node, parent))
            .unwrap();
        let batch = assignment_batches.get(node).unwrap();
        for (assignment_id, &point) in batch.first_points.iter().enumerate() {
            let raw = component
                .layout
                .input_positions
                .iter()
                .map(|&row| value_at(points, row, point, "test").unwrap())
                .collect::<Vec<_>>();
            let direct = component.layout.indexer.encode(&raw).unwrap();
            assert_eq!(batch.keys[assignment_id], direct, "node={node:?}");
        }
    }
}

/// [AI Supplied] KeyBuilder coverage includes zero-width, nested, wide,
/// duplicate/reordered assignments, invalid values, and checked overflow.
#[test]
fn key_builder_component_edge_cases_are_checked() {
    assert!(FlatIndexer::try_new(&[0]).is_err());
    let empty = FlatIndexer::try_new(&[]).unwrap().encode(&[]).unwrap();
    let mut empty_builder = KeyBuilder::with_capacity_bits(0).unwrap();
    empty_builder.push(&empty).unwrap();
    assert_eq!(empty_builder.finish(), empty);

    let left_indexer = FlatIndexer::try_new(&[2, 3]).unwrap();
    let right_indexer = FlatIndexer::try_new(&[5]).unwrap();
    let whole_indexer = FlatIndexer::try_new(&[2, 3, 5]).unwrap();
    let left = left_indexer.encode(&[1, 2]).unwrap();
    let right = right_indexer.encode(&[4]).unwrap();
    let mut nested = KeyBuilder::with_capacity_bits(whole_indexer.width_bits()).unwrap();
    nested.push(&left).unwrap();
    nested.push(&right).unwrap();
    assert_eq!(nested.finish(), whole_indexer.encode(&[1, 2, 4]).unwrap());

    let duplicate = whole_indexer.encode(&[1, 2, 4]).unwrap();
    let reordered = whole_indexer.encode(&[1, 2, 4]).unwrap();
    assert_eq!(duplicate, reordered);
    assert!(whole_indexer.encode(&[1, 2, 4]).is_ok());
    assert!(whole_indexer.encode(&[1, 3, 4]).is_err());

    let wide_indexer = FlatIndexer::try_new(&[2; 130]).unwrap();
    let wide_values = vec![1usize; 130];
    let wide = wide_indexer.encode(&wide_values).unwrap();
    let mut wide_builder = KeyBuilder::with_capacity_bits(wide.width_bits()).unwrap();
    wide_builder.push(&wide).unwrap();
    assert_eq!(wide_builder.finish(), wide);

    let mut overflow = KeyBuilder::with_capacity_bits(left.width_bits()).unwrap();
    overflow.push(&left).unwrap();
    assert!(overflow.push(&right).is_err());
}

fn topology_tree(edges: &[(usize, usize)]) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let n_nodes = edges
        .iter()
        .flat_map(|&(left, right)| [left, right])
        .max()
        .map_or(1, |node| node + 1);
    let physical = (0..n_nodes)
        .map(|_| DynIndex::new_dyn(2))
        .collect::<Vec<_>>();
    let bonds = edges
        .iter()
        .enumerate()
        .map(|(edge, _)| DynIndex::new_dyn(2 + edge))
        .collect::<Vec<_>>();
    let mut tensors = Vec::with_capacity(n_nodes);
    for (node, physical_index) in physical.iter().enumerate() {
        let mut tensor_indices = vec![physical_index.clone()];
        for (edge, &(left, right)) in edges.iter().enumerate() {
            if node == left || node == right {
                tensor_indices.push(bonds[edge].clone());
            }
        }
        let size = tensor_indices.iter().map(IndexLike::dim).product();
        tensors.push(IdxTensor::from_dense(tensor_indices, vec![1.0_f64; size]).unwrap());
    }
    (
        TreeTN::from_tensors(tensors, (0..n_nodes).collect()).unwrap(),
        physical,
    )
}

fn assert_shared_metadata_for_topology(edges: &[(usize, usize)], expected_refs: usize) {
    let (tree, indices) = topology_tree(edges);
    let n_nodes = tree.node_count();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::<usize>::default())
            .unwrap();
    let values = vec![0usize; n_nodes];
    let shape = [n_nodes, 1];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    for center in tree.node_names() {
        evaluator
            .evaluate_batched_with_hint(points, EvaluationHint::around(center))
            .unwrap();
    }

    assert_eq!(
        evaluator.directed_component_layouts().len(),
        edges.len() * 2
    );
    assert_eq!(
        evaluator
            .directed_component_layouts()
            .values()
            .map(|layout| layout.layout.input_positions.len())
            .sum::<usize>(),
        expected_refs
    );
}

/// [AI Supplied] Path, Y, comb, and unequal-bond topologies retain one
/// component layout per directed edge. For one physical index per node,
/// the position census is `N * E`, independent of how many centers were
/// visited.
#[test]
fn directed_component_metadata_scales_with_edges_not_centers() {
    for n_sites in [4usize, 8, 16] {
        let path_edges = (0..n_sites - 1)
            .map(|node| (node, node + 1))
            .collect::<Vec<_>>();
        assert_shared_metadata_for_topology(&path_edges, n_sites * (n_sites - 1));
    }

    let y_edges = [(0, 1), (0, 2), (0, 3)];
    assert_shared_metadata_for_topology(&y_edges, 4 * 3);

    let comb_edges = [(0, 1), (1, 2), (2, 3), (1, 4), (2, 5), (3, 6)];
    assert_shared_metadata_for_topology(&comb_edges, 7 * 6);

    let (unequal_tree, unequal_indices) = typed_unequal_y_tree::<f64>();
    let mut unequal_evaluator = TreeTNCachedEvaluator::new(
        &unequal_tree,
        &unequal_indices,
        CachedEvaluatorOptions::<usize>::default(),
    )
    .unwrap();
    let unequal_values = [0usize; 4];
    let unequal_points = ColMajorArrayRef::new(&unequal_values, &[4usize, 1usize]).unwrap();
    for center in unequal_tree.node_names() {
        unequal_evaluator
            .evaluate_batched_with_hint(unequal_points, EvaluationHint::around(center))
            .unwrap();
    }
    assert_eq!(unequal_evaluator.directed_component_layouts().len(), 6);
    assert_eq!(
        unequal_evaluator
            .directed_component_layouts()
            .values()
            .map(|layout| layout.layout.input_positions.len())
            .sum::<usize>(),
        12
    );
}

#[test]
fn fixed_center_and_scan_hint_evaluation_match() {
    let (tree, indices) = complex_three_node_chain();
    let shape = [3usize, 2usize];
    let values = [0usize, 0, 0, 0, 0, 1];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    let mut hinted =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::<usize>::default())
            .unwrap();
    let mut fixed_center =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::<usize>::default())
            .unwrap();

    let hinted_values = hinted
        .evaluate_batched_with_hint(points, EvaluationHint::around(1))
        .unwrap();
    let fixed_center_values = fixed_center.evaluate_batched(points).unwrap();

    assert_scalars_close(&hinted_values, &fixed_center_values);
}

#[test]
fn changing_center_hint_reuses_unchanged_directed_messages() {
    let (tree, indices) = complex_three_node_chain();
    let shape = [3usize, 1usize];
    let values = [0usize, 0, 0];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::<usize>::default())
            .unwrap();

    let first = evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(0))
        .unwrap();
    let second = evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(1))
        .unwrap();

    assert_scalars_close(&first, &expected);
    assert_scalars_close(&second, &expected);
    let stats = evaluator.stats_for_test();
    assert_eq!(
        stats.message_cache_hits, 2,
        "both selected cut directions should be reused after changing centers: {stats:?}"
    );
    assert_eq!(
        stats.message_cache_misses, 0,
        "changing the center should not invalidate either cut direction: {stats:?}"
    );
}

fn three_node_chain() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let s0 = DynIndex::new_dyn(2);
    let b01 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), b01.clone()], vec![1.0_f64; 4]).unwrap();
    let t1 = IdxTensor::from_dense(vec![b01, s1.clone(), b12.clone()], vec![1.0_f64; 8]).unwrap();
    let t2 = IdxTensor::from_dense(vec![b12, s2.clone()], vec![1.0_f64; 4]).unwrap();
    let tree = TreeTN::<_, usize>::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap();
    (tree, vec![s0, s1, s2])
}

#[test]
fn component_cost_index_counts_unique_directed_components() {
    let (tree, indices) = three_node_chain();
    let values = vec![0, 0, 0, 0, 1, 1, 1, 1, 1];
    let shape = [3, 3];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    let cost_index = ComponentCostIndex::new(&tree, &indices, points).unwrap();

    assert_eq!(cost_index.component_count(&(0, 1)).unwrap(), 2);
    assert_eq!(cost_index.component_count(&(1, 0)).unwrap(), 2);
    assert_eq!(cost_index.component_count(&(1, 2)).unwrap(), 3);
    assert_eq!(cost_index.component_count(&(2, 1)).unwrap(), 2);
    assert_eq!(cost_index.center_cost(&0).unwrap(), 2);
    assert_eq!(cost_index.center_cost(&1).unwrap(), 4);
    assert_eq!(cost_index.center_cost(&2).unwrap(), 3);
}

#[test]
fn component_cost_index_rejects_wrong_batch_row_count() {
    let (tree, indices) = three_node_chain();
    let values = vec![0, 1, 1, 0];
    let shape = [2, 2];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    let err = ComponentCostIndex::new(&tree, &indices, points)
        .err()
        .unwrap();
    assert!(err.to_string().contains("row count"));
}

#[test]
fn greedy_center_search_descends_to_lower_cost_neighbor() {
    let cost_index = ComponentCostIndex::from_parts_for_test(
        HashMap::from([(0, vec![1]), (1, vec![0, 2]), (2, vec![1, 3]), (3, vec![2])]),
        HashMap::from([(0, 40), (1, 20), (2, 10), (3, 15)]),
    );

    let result = GreedyCenterSearch::<usize>::default()
        .search(&cost_index, &[0])
        .unwrap();

    assert_eq!(result.center, 2);
    assert_eq!(result.cost, 10);
    assert_eq!(result.path, vec![0, 1, 2]);
}

#[test]
fn greedy_center_search_uses_best_of_multiple_starts() {
    let cost_index = ComponentCostIndex::from_parts_for_test(
        HashMap::from([
            ("a", vec!["b"]),
            ("b", vec!["a", "c"]),
            ("c", vec!["b", "d"]),
            ("d", vec!["c"]),
        ]),
        HashMap::from([("a", 8), ("b", 6), ("c", 5), ("d", 2)]),
    );

    let result = GreedyCenterSearch::<&str>::with_max_steps(Some(1))
        .search(&cost_index, &["a", "d"])
        .unwrap();

    assert_eq!(result.center, "d");
    assert_eq!(result.cost, 2);
}

#[test]
fn cached_evaluator_selects_greedy_center_when_center_is_not_fixed() {
    let (tree, indices) = three_node_chain();
    let values = vec![0, 0, 0, 0, 1, 1, 1, 1, 1];
    let shape = [3, 3];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            initial_centers: vec![1],
            ..Default::default()
        },
    )
    .unwrap();

    let actual = evaluator.evaluate_batched(points).unwrap();

    assert_eq!(evaluator.center(), Some(&0));
    assert_scalars_close(&actual, &expected);
}

#[test]
fn cached_evaluator_rejects_unknown_initial_center() {
    let (tree, indices) = three_node_chain();
    let err = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            initial_centers: vec![99],
            ..Default::default()
        },
    )
    .err()
    .unwrap();

    assert!(err.to_string().contains("initial center"));
}

#[test]
fn cached_evaluator_computes_one_environment_per_unique_subtree_assignment() {
    let (tree, indices) = three_node_chain();
    let values = vec![0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1];
    let shape = [3, 4];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let actual = evaluator.evaluate_batched(points).unwrap();

    assert_scalars_close(&actual, &expected);
    assert_eq!(evaluator.stats_for_test().subtree_environment_count, 4);
}

#[test]
fn cached_evaluator_reuses_directed_messages_inside_components() {
    let (tree, indices) = five_node_chain();
    let values = vec![
        0, 0, 0, 0, 0, //
        0, 1, 0, 0, 1, //
        1, 0, 1, 1, 0, //
        1, 1, 1, 1, 1,
    ];
    let shape = [5, 4];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    let actual = evaluator.evaluate_batched(points).unwrap();

    assert_scalars_close(&actual, &expected);
    assert_eq!(evaluator.stats_for_test().directed_message_count, 12);
}

#[test]
fn cached_evaluator_batches_directed_messages() {
    let (tree, indices) = five_node_chain();
    let values = vec![
        0, 0, 0, 0, 0, //
        0, 1, 0, 0, 1, //
        1, 0, 1, 1, 0, //
        1, 1, 1, 1, 1,
    ];
    let shape = [5, 4];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    let actual = evaluator.evaluate_batched(points).unwrap();
    let stats = evaluator.stats_for_test();

    assert_scalars_close(&actual, &expected);
    assert!(stats.batched_message_contract_count < stats.directed_message_count);
}
