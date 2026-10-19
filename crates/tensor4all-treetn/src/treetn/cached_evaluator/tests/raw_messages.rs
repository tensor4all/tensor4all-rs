//! Raw leaf, chain, branch, and center message kernels, checked against the
//! generic contraction.

use super::*;

#[derive(Clone, Copy, Default)]
struct WorkToken;

impl std::ops::AddAssign for WorkToken {
    fn add_assign(&mut self, _rhs: Self) {}
}

impl std::ops::Mul for WorkToken {
    type Output = Self;

    fn mul(self, _rhs: Self) -> Self::Output {
        self
    }
}

// [AI Supplied] Small exact fixture plumbing around the #671 relation.
fn measured_raw_center_core_visits(physical_dim: usize, bond_dims: &[usize]) -> usize {
    let mut dims = Vec::with_capacity(1 + bond_dims.len());
    dims.push(physical_dim);
    dims.extend_from_slice(bond_dims);
    // A zero-sized scalar executes the real loop nest at chi=256 without
    // allocating the equivalent 268 MiB f64 degree-3 core.
    let core = vec![WorkToken; dims.iter().product()];
    let physical_values = (0..physical_dim).collect::<Vec<_>>();
    let components = bond_dims
        .iter()
        .enumerate()
        .map(|(position, &dim)| RawCenterComponent {
            axis: position + 1,
            dim,
            point_to_assignment: vec![0; physical_dim],
            values: vec![WorkToken; dim],
        })
        .collect::<Vec<_>>();

    reset_raw_center_core_visits_for_test();
    let values = contract_raw_center(&core, &dims, 0, &physical_values, &components).unwrap();
    assert_eq!(values.len(), physical_dim);
    raw_center_core_visits_for_test()
}

fn assert_representative_raw_center_values(physical_dim: usize, bond_dims: &[usize]) {
    let mut dims = Vec::with_capacity(1 + bond_dims.len());
    dims.push(physical_dim);
    dims.extend_from_slice(bond_dims);
    let core = vec![1.0_f64; dims.iter().product()];
    let physical_values = (0..physical_dim).collect::<Vec<_>>();
    let components = bond_dims
        .iter()
        .enumerate()
        .map(|(position, &dim)| RawCenterComponent {
            axis: position + 1,
            dim,
            point_to_assignment: vec![0; physical_dim],
            values: vec![1.0; dim],
        })
        .collect::<Vec<_>>();

    let values = contract_raw_center(&core, &dims, 0, &physical_values, &components).unwrap();
    let expected_value = bond_dims.iter().product::<usize>() as f64;
    assert_eq!(values, vec![expected_value; physical_dim]);
}

/// Hiroshi's issue #671 complexity relation is a deterministic work-count
/// invariant, not a wall-clock threshold: a dense center contraction must
/// visit `d * product(incident bond dimensions)` core elements when every
/// physical value is evaluated once.
#[test]
fn raw_center_work_scales_with_coordination_number_and_actual_bond_dimensions() {
    const REPRESENTATIVE_BONDS: [usize; 3] = [64, 128, 256];
    let z2_visits = REPRESENTATIVE_BONDS.map(|chi| {
        let visits = measured_raw_center_core_visits(2, &[chi, chi]);
        assert_eq!(visits, 2 * chi.pow(2));
        visits
    });
    assert!(z2_visits
        .windows(2)
        .all(|pair| pair[1] / pair[0] == 2usize.pow(2)));

    let z3_visits = REPRESENTATIVE_BONDS.map(|chi| {
        let visits = measured_raw_center_core_visits(2, &[chi, chi, chi]);
        assert_eq!(visits, 2 * chi.pow(3));
        visits
    });
    assert!(z3_visits
        .windows(2)
        .all(|pair| pair[1] / pair[0] == 2usize.pow(3)));

    assert_eq!(
        measured_raw_center_core_visits(3, &[64, 128, 256]),
        3 * 64 * 128 * 256,
        "unequal bonds must use their actual product rather than max(chi)^z"
    );

    // Exercise real f64 arithmetic and storage at a representative
    // degree-3 bond dimension (about 32 MiB of dense core payload).
    assert_representative_raw_center_values(2, &[128, 128, 128]);
}

/// Root cause of the cache slowdown (see the message-cache-prototype
/// worklog) is `IdxTensor::to_vec` on a `contract_with_options` result
/// hitting an expensive non-contiguous/backend-resident fallback. This
/// tests the fix's first slice: a leaf node's message computed directly
/// from the tree tensor's raw data, with no `contract_with_options` and
/// no intermediate `IdxTensor` at all, must match the existing generic
/// path exactly.
#[test]
fn raw_leaf_message_matches_generic_contraction() {
    let (tree, indices) = varied_three_node_chain();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let shape = [3usize, 2usize];
    let values = [0usize, 0, 0, 0, 0, 1];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    // Force layout/center bookkeeping the same way evaluate_batched would.
    evaluator.evaluate_batched(points).unwrap();

    let plan = RootedMessagePlan::new(&tree, &0).unwrap();
    let assignment_batches = evaluator
        .build_message_assignment_batches(&plan, points)
        .unwrap();
    let leaf_points = assignment_batches.get(&2).unwrap().first_points.clone();

    let expected_message = evaluator
        .compute_stacked_message(
            &2,
            points,
            &leaf_points,
            &plan,
            &assignment_batches,
            &HashMap::new(),
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap();
    let expected = tensor_values_any(expected_message.tensor.as_ref().unwrap()).unwrap();

    let actual = evaluator
        .try_compute_leaf_message_raw(
            &2,
            points,
            &leaf_points,
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap()
        .expect("leaf node with one physical index and a real tensor must be eligible");

    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            (a - e.real()).abs() < 1.0e-12,
            "raw={a} generic={}",
            e.real()
        );
    }
}

#[test]
fn raw_complex_leaf_message_matches_generic_contraction() {
    let (tree, indices) = complex_three_node_chain();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let shape = [3usize, 2usize];
    let values = [0usize, 0, 0, 0, 0, 1];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    evaluator.evaluate_batched(points).unwrap();

    let plan = RootedMessagePlan::new(&tree, &0).unwrap();
    let assignment_batches = evaluator
        .build_message_assignment_batches(&plan, points)
        .unwrap();
    let leaf_points = assignment_batches.get(&2).unwrap().first_points.clone();
    let expected_message = evaluator
        .compute_stacked_message(
            &2,
            points,
            &leaf_points,
            &plan,
            &assignment_batches,
            &HashMap::new(),
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap();
    let expected = tensor_values_any(expected_message.tensor.as_ref().unwrap()).unwrap();

    let actual = evaluator
        .try_compute_leaf_message_complex_raw(
            &2,
            points,
            &leaf_points,
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap()
        .expect("complex leaf message should use the raw path");

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual.re - expected.real()).abs() < 1.0e-12);
        assert!((actual.im - expected.imag()).abs() < 1.0e-12);
    }
}

#[test]
fn raw_chain_message_matches_generic_contraction() {
    let (tree, indices) = varied_three_node_chain();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let shape = [3usize, 2usize];
    let values = [0usize, 0, 0, 0, 0, 1];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    evaluator.evaluate_batched(points).unwrap();

    let plan = RootedMessagePlan::new(&tree, &0).unwrap();
    let assignment_batches = evaluator
        .build_message_assignment_batches(&plan, points)
        .unwrap();

    // Node 2 (leaf, child of node 1) via the generic oracle path.
    let leaf_points = assignment_batches.get(&2).unwrap().first_points.clone();
    let node2_message = evaluator
        .compute_stacked_message(
            &2,
            points,
            &leaf_points,
            &plan,
            &assignment_batches,
            &HashMap::new(),
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap();
    let mut messages = HashMap::new();
    messages.insert(2usize, node2_message);

    let node1_points = assignment_batches.get(&1).unwrap().first_points.clone();
    let expected_message = evaluator
        .compute_stacked_message(
            &1,
            points,
            &node1_points,
            &plan,
            &assignment_batches,
            &messages,
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap();
    let expected = tensor_values_any(expected_message.tensor.as_ref().unwrap()).unwrap();

    let actual = evaluator
        .try_compute_chain_message_raw(
            &1,
            points,
            &node1_points,
            &plan,
            &assignment_batches,
            &messages,
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap()
        .expect("interior node with one child and one physical index must be eligible");

    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            (a - e.real()).abs() < 1.0e-10,
            "raw={a} generic={}",
            e.real()
        );
    }
}

#[test]
fn raw_complex_chain_message_matches_generic_contraction() {
    let (tree, indices) = complex_three_node_chain();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let shape = [3usize, 2usize];
    let values = [0usize, 0, 0, 0, 0, 1];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    evaluator.evaluate_batched(points).unwrap();

    let plan = RootedMessagePlan::new(&tree, &0).unwrap();
    let assignment_batches = evaluator
        .build_message_assignment_batches(&plan, points)
        .unwrap();

    let leaf_points = assignment_batches.get(&2).unwrap().first_points.clone();
    let node2_message = evaluator
        .compute_stacked_message(
            &2,
            points,
            &leaf_points,
            &plan,
            &assignment_batches,
            &HashMap::new(),
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap();
    let mut messages = HashMap::new();
    messages.insert(2usize, node2_message);

    let node1_points = assignment_batches.get(&1).unwrap().first_points.clone();
    let expected_message = evaluator
        .compute_stacked_message(
            &1,
            points,
            &node1_points,
            &plan,
            &assignment_batches,
            &messages,
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap();
    let expected = tensor_values_any(expected_message.tensor.as_ref().unwrap()).unwrap();

    let actual = evaluator
        .try_compute_chain_message_complex_raw(
            &1,
            points,
            &node1_points,
            &plan,
            &assignment_batches,
            &messages,
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap()
        .expect("complex chain message should use the raw path");

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual.re - expected.real()).abs() < 1.0e-10);
        assert!((actual.im - expected.imag()).abs() < 1.0e-10);
    }
}

#[test]
fn raw_branch_message_matches_generic_contraction() {
    let (tree, indices) = star_tree();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let shape = [4usize, 2usize];
    let values = [0usize, 0, 0, 0, 1, 0, 1, 1];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    evaluator.evaluate_batched(points).unwrap();

    let plan = RootedMessagePlan::new(&tree, &1).unwrap();
    let assignment_batches = evaluator
        .build_message_assignment_batches(&plan, points)
        .unwrap();

    // Nodes 2 and 3 (leaves, the hub's two children when rooted at leaf
    // 1) via the generic oracle path.
    let mut messages = HashMap::new();
    for leaf in [2usize, 3usize] {
        let leaf_points = assignment_batches.get(&leaf).unwrap().first_points.clone();
        let leaf_message = evaluator
            .compute_stacked_message(
                &leaf,
                points,
                &leaf_points,
                &plan,
                &assignment_batches,
                &HashMap::new(),
                &mut CachedEvaluationRoute::global(),
            )
            .unwrap();
        messages.insert(leaf, leaf_message);
    }

    let hub_points = assignment_batches.get(&0).unwrap().first_points.clone();
    let expected_message = evaluator
        .compute_stacked_message(
            &0,
            points,
            &hub_points,
            &plan,
            &assignment_batches,
            &messages,
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap();
    let expected = tensor_values_any(expected_message.tensor.as_ref().unwrap()).unwrap();

    let actual = evaluator
        .try_compute_branch_message_raw(
            &0,
            points,
            &hub_points,
            &plan,
            &assignment_batches,
            &messages,
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap()
        .expect("branch node with two children and one physical index must be eligible");

    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            (a - e.real()).abs() < 1.0e-10,
            "raw={a} generic={}",
            e.real()
        );
    }
}

#[test]
fn degree_three_hub_keeps_raw_messages_when_center_is_a_leaf() {
    let (tree, indices) = star_tree();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let shape = [4usize, 2usize];
    let values = [0usize, 0, 0, 0, 1, 0, 1, 1];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    let (_, environments) = evaluator
        .build_environment_cache(&1, points, &mut CachedEvaluationRoute::global())
        .unwrap();
    let hub_message = environments
        .get(&0)
        .expect("leaf-centered star must expose the hub environment");

    assert!(
        hub_message.raw_values.is_some(),
        "a degree-3 hub rooted at a leaf should retain its raw branch message"
    );
    assert!(
        hub_message.tensor.is_none(),
        "raw branch messages should not be materialized into an IdxTensor"
    );
}

#[cfg(feature = "diagnostics")]
#[test]
fn build_environment_cache_records_guard_diagnostics_per_node_with_correct_coordination_numbers() {
    use crate::treetn::diagnostics;

    let (tree, indices) = star_tree();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let shape = [4usize, 2usize];
    let values = [0usize, 0, 0, 0, 1, 0, 1, 1];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    diagnostics::reset();
    let _ = evaluator
        .build_environment_cache(&1, points, &mut CachedEvaluationRoute::global())
        .unwrap();

    let snapshot = diagnostics::snapshot();
    let hub_record = snapshot
        .iter()
        .find(|record| record.node == "tree:0")
        .expect("hub node (0) recorded");
    assert_eq!(hub_record.coordination_number, 3);
    assert_eq!(hub_record.bond_dims.len(), 3);
    assert!(hub_record.guard_cache_hits + hub_record.guard_cache_misses > 0);

    for leaf in ["tree:2", "tree:3"] {
        let leaf_record = snapshot
            .iter()
            .find(|record| record.node == leaf)
            .unwrap_or_else(|| panic!("leaf node ({leaf}) recorded"));
        assert_eq!(leaf_record.coordination_number, 1);
    }
}

/// A degree-4 hub used to disqualify the whole tree from the raw path,
/// because its leaf-rooted form has three children and no kernel covered
/// that. `try_compute_multi_branch_message_raw` does, so the tree keeps
/// its raw messages (tensor4all-rs #727).
#[test]
fn degree_four_hub_keeps_raw_messages_when_center_is_a_leaf() {
    let (tree, indices) = four_arm_star_tree();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let values = vec![0usize; 5];
    let shape = [5usize, 1usize];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    let (_, environments) = evaluator
        .build_environment_cache(&1, points, &mut CachedEvaluationRoute::global())
        .unwrap();
    let hub_message = environments
        .get(&0)
        .expect("leaf-centered star must expose the hub environment");

    assert!(
        hub_message.raw_values.is_some(),
        "a degree-4 hub is covered by the arbitrary-degree message kernel"
    );
    assert!(hub_message.tensor.is_none());
}

fn complex_star_tree() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let sc = DynIndex::new_dyn(2);
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let b0 = DynIndex::new_dyn(2);
    let b1 = DynIndex::new_dyn(2);
    let b2 = DynIndex::new_dyn(2);
    let center_data: Vec<Complex64> = (0..16)
        .map(|value| Complex64::new(value as f64 + 1.0, -(value as f64) * 0.25))
        .collect();
    let center = IdxTensor::from_dense(
        vec![sc.clone(), b0.clone(), b1.clone(), b2.clone()],
        center_data,
    )
    .unwrap();
    let leaf0 = IdxTensor::from_dense(
        vec![b0, s0.clone()],
        vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(0.5, -0.25),
            Complex64::new(1.5, 0.75),
            Complex64::new(2.0, -1.0),
        ],
    )
    .unwrap();
    let leaf1 = IdxTensor::from_dense(
        vec![b1, s1.clone()],
        vec![
            Complex64::new(0.25, -0.5),
            Complex64::new(1.0, 0.25),
            Complex64::new(1.25, -0.75),
            Complex64::new(2.0, 1.0),
        ],
    )
    .unwrap();
    let leaf2 = IdxTensor::from_dense(
        vec![b2, s2.clone()],
        vec![
            Complex64::new(2.0, 0.5),
            Complex64::new(1.0, -0.25),
            Complex64::new(0.75, 0.5),
            Complex64::new(1.5, -1.0),
        ],
    )
    .unwrap();
    let tree =
        TreeTN::<_, usize>::from_tensors(vec![center, leaf0, leaf1, leaf2], vec![0, 1, 2, 3])
            .unwrap();
    (tree, vec![sc, s0, s1, s2])
}

#[test]
fn raw_complex_branch_message_matches_generic_contraction() {
    let (tree, indices) = complex_star_tree();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let shape = [4usize, 2usize];
    let values = [0usize, 0, 0, 0, 1, 0, 1, 1];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    evaluator.evaluate_batched(points).unwrap();

    let plan = RootedMessagePlan::new(&tree, &1).unwrap();
    let assignment_batches = evaluator
        .build_message_assignment_batches(&plan, points)
        .unwrap();

    let mut messages = HashMap::new();
    for leaf in [2usize, 3usize] {
        let leaf_points = assignment_batches.get(&leaf).unwrap().first_points.clone();
        let leaf_message = evaluator
            .compute_stacked_message(
                &leaf,
                points,
                &leaf_points,
                &plan,
                &assignment_batches,
                &HashMap::new(),
                &mut CachedEvaluationRoute::global(),
            )
            .unwrap();
        messages.insert(leaf, leaf_message);
    }

    let hub_points = assignment_batches.get(&0).unwrap().first_points.clone();
    let expected_message = evaluator
        .compute_stacked_message(
            &0,
            points,
            &hub_points,
            &plan,
            &assignment_batches,
            &messages,
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap();
    let expected = tensor_values_any(expected_message.tensor.as_ref().unwrap()).unwrap();

    let actual = evaluator
        .try_compute_branch_message_complex_raw(
            &0,
            points,
            &hub_points,
            &plan,
            &assignment_batches,
            &messages,
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap()
        .expect("complex branch message should use the raw path");

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual.re - expected.real()).abs() < 1.0e-10);
        assert!((actual.im - expected.imag()).abs() < 1.0e-10);
    }
}

#[test]
fn raw_degree_three_center_matches_exact_real_evaluation() {
    let (tree, indices) = star_tree();
    let values = [0usize, 0, 0, 0, 1, 0, 1, 1];
    let points = ColMajorArrayRef::new(&values, &[4, 2]).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::<usize>::default())
            .unwrap();

    let actual = evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(0))
        .unwrap();

    assert_scalars_close(&actual, &expected);
}

#[test]
fn raw_degree_three_center_matches_exact_complex_evaluation() {
    let (tree, indices) = complex_star_tree();
    let values = [0usize, 0, 0, 0, 1, 0, 1, 1];
    let points = ColMajorArrayRef::new(&values, &[4, 2]).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::<usize>::default())
            .unwrap();

    let actual = evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(0))
        .unwrap();

    assert_scalars_close(&actual, &expected);
}

/// Returns the `(grouped, flat)` route counts of the kernel call under
/// test, so the caller can pin which route this configuration takes
/// without counting the warm-up evaluation's own calls.
fn assert_multi_branch_matches_generic<T>(arms: usize, point_count: usize) -> (usize, usize)
where
    T: CachedEvaluatorTestScalar
        + tensor4all_tensorbackend::StorageScalar
        + BlasMul
        + Copy
        + Default
        + std::ops::AddAssign
        + std::ops::Mul<Output = T>,
{
    let (tree, indices) = unequal_star_tree::<T>(arms);
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let values = star_points(&indices, point_count);
    let shape = [indices.len(), point_count];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    evaluator.evaluate_batched(points).unwrap();

    let plan = RootedMessagePlan::new(&tree, &1).unwrap();
    let assignment_batches = evaluator
        .build_message_assignment_batches(&plan, points)
        .unwrap();

    // Every leaf except the centre is a child of the hub; compute those
    // through the generic oracle so only the hub's own step differs.
    let mut messages = HashMap::new();
    for leaf in 2..=arms {
        let leaf_points = assignment_batches.get(&leaf).unwrap().first_points.clone();
        let leaf_message = evaluator
            .compute_stacked_message(
                &leaf,
                points,
                &leaf_points,
                &plan,
                &assignment_batches,
                &HashMap::new(),
                &mut CachedEvaluationRoute::global(),
            )
            .unwrap();
        messages.insert(leaf, leaf_message);
    }
    assert_eq!(
        plan.children.get(&0).map(Vec::len),
        Some(arms - 1),
        "the hub must have every non-centre arm as a rooted child"
    );

    let hub_points = assignment_batches.get(&0).unwrap().first_points.clone();
    let expected_message = evaluator
        .compute_stacked_message(
            &0,
            points,
            &hub_points,
            &plan,
            &assignment_batches,
            &messages,
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap();
    let expected = tensor_values_any(expected_message.tensor.as_ref().unwrap()).unwrap();

    reset_multi_branch_routes_for_test();
    let actual = evaluator
        .try_compute_multi_branch_message_raw::<T>(
            &0,
            points,
            &hub_points,
            &plan,
            &assignment_batches,
            &messages,
            if <T as CachedEvaluatorTestScalar>::IS_COMPLEX {
                ScalarKind::C64
            } else {
                ScalarKind::F64
            },
            |value| match (value, <T as CachedEvaluatorTestScalar>::IS_COMPLEX) {
                (CachedScalar::F64(value), false) => Some(T::from_parts(*value, 0.0)),
                (CachedScalar::C64(value), true) => Some(T::from_parts(value.re, value.im)),
                _ => None,
            },
            &mut CachedEvaluationRoute::global(),
        )
        .unwrap()
        .unwrap_or_else(|| panic!("a hub with {} rooted children must be eligible", arms - 1));

    assert_eq!(actual.len(), expected.len());
    let mut residual = 0.0f64;
    for (raw, generic) in actual.iter().zip(expected.iter()) {
        let raw = raw.into_any_scalar();
        residual = residual
            .max((raw.real() - generic.real()).abs())
            .max((raw.imag() - generic.imag()).abs());
    }
    assert!(
        residual <= 1.0e-10,
        "arbitrary-degree message differs from the generic contraction by {residual:.3e} \
         (arms={arms}, points={point_count})"
    );
    multi_branch_routes_for_test()
}

/// The arbitrary-degree message kernel must reproduce the generic
/// contraction at every degree it claims, for both scalar kinds, and on
/// both of its routes.
///
/// The route a case takes follows from its own arithmetic -- the grouped
/// kernel is chosen when `parent_dim * prod(child_dims) * points` clears
/// the backend threshold -- so rather than restate that rule per case,
/// the matrix asserts that each call took exactly one route and that the
/// matrix as a whole exercised both.
#[test]
fn multi_branch_raw_message_matches_generic_contraction() {
    let mut grouped_total = 0usize;
    let mut flat_total = 0usize;
    for arms in [4usize, 5, 6] {
        for point_count in [1usize, 32] {
            for complex in [false, true] {
                let (grouped, flat) = if complex {
                    assert_multi_branch_matches_generic::<Complex64>(arms, point_count)
                } else {
                    assert_multi_branch_matches_generic::<f64>(arms, point_count)
                };
                assert_eq!(
                    grouped + flat,
                    1,
                    "arms={arms} points={point_count} complex={complex} took {grouped} grouped \
                     and {flat} flat routes"
                );
                grouped_total += grouped;
                flat_total += flat;
            }
        }
    }
    assert!(
        grouped_total > 0 && flat_total > 0,
        "the matrix must exercise both routes, saw {grouped_total} grouped and {flat_total} flat"
    );
}

/// The two routes of the arbitrary-degree kernel are the same
/// contraction, so they must agree with each other as well as with the
/// generic path -- the fallback-parity half of the routing contract.
#[test]
fn multi_branch_grouped_and_flat_routes_agree() {
    // A hub of shape [parent=3, physical=2, c0=2, c1=3, c2=2] with the
    // physical axis in the middle and a non-contiguous parent axis.
    let spec = MultiBranchContractionSpec {
        strides: vec![1, 3, 6, 12, 36],
        physical_axis: 1,
        parent_axis: 0,
        child_axes: vec![2, 3, 4],
        parent_dim: 3,
        child_dims: vec![2, 3, 2],
    };
    let raw = (0..72)
        .map(|value| (value % 11) as f64 * 0.25 - 1.0)
        .collect::<Vec<f64>>();
    let physical_values = vec![0usize, 1, 1, 0];
    let child_columns = vec![
        (0..8)
            .map(|v| (v % 5) as f64 * 0.5 - 0.75)
            .collect::<Vec<f64>>(),
        (0..12)
            .map(|v| (v % 7) as f64 * 0.25 - 0.5)
            .collect::<Vec<f64>>(),
        (0..8)
            .map(|v| (v % 3) as f64 * 0.75 - 0.25)
            .collect::<Vec<f64>>(),
    ];

    let flat =
        scalar_multi_branch_message_contraction(&spec, &raw, &physical_values, &child_columns)
            .unwrap();
    let grouped = grouped_multi_branch_message_contraction(
        &mut CachedEvaluationRoute::global(),
        &spec,
        &raw,
        &physical_values,
        &child_columns,
    )
    .unwrap();

    assert_eq!(flat.len(), physical_values.len() * spec.parent_dim);
    assert_eq!(flat.len(), grouped.len());
    let residual = flat
        .iter()
        .zip(&grouped)
        .fold(0.0f64, |residual, (left, right)| {
            residual.max((left - right).abs())
        });
    assert!(
        residual <= 1.0e-12,
        "the grouped and flat routes disagree by {residual:.3e}"
    );

    // A hand-computed reference for one output element pins the two
    // routes to the intended contraction rather than to each other:
    // point 0 has physical 0 and reads child columns 0..d of each child.
    let mut reference = 0.0f64;
    for c0 in 0..spec.child_dims[0] {
        for c1 in 0..spec.child_dims[1] {
            for c2 in 0..spec.child_dims[2] {
                // Point 0 has physical value 0, so its physical
                // offset contributes nothing to this reference.
                let offset = c0 * spec.strides[spec.child_axes[0]]
                    + c1 * spec.strides[spec.child_axes[1]]
                    + c2 * spec.strides[spec.child_axes[2]]
                    + 2 * spec.strides[spec.parent_axis];
                reference += raw[offset]
                    * child_columns[0][c0]
                    * child_columns[1][c1]
                    * child_columns[2][c2];
            }
        }
    }
    assert!(
        (flat[2] - reference).abs() <= 1.0e-12,
        "flat route gives {} against the hand-computed {reference}",
        flat[2]
    );
}

/// A centre of coordination four or more is contracted by the same raw
/// centre kernel that used to stop at three components.
#[test]
fn raw_center_contraction_covers_high_coordination_centers() {
    for arms in [4usize, 5] {
        let (tree, indices) = unequal_star_tree::<f64>(arms);
        let values = star_points(&indices, 3);
        let shape = [indices.len(), 3];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        let expected = tree.evaluate(&indices, points).unwrap();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(0),
                ..Default::default()
            },
        )
        .unwrap();

        reset_raw_center_core_visits_for_test();
        let actual = evaluator.evaluate_batched(points).unwrap();
        assert_scalars_close(&actual, &expected);
        assert!(
            raw_center_core_visits_for_test() > 0,
            "a degree-{arms} centre must reach the raw centre kernel"
        );
    }
}
