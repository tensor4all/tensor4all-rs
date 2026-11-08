//! Batch chunking: chunk-limit validation, automatic chunk sizing, and branch
//! hints under chunked evaluation.

use super::*;

#[test]
fn chunk_limit_validation_scalar_batches_and_late_errors() {
    let tree = TreeTN::from_tensors(vec![IdxTensor::scalar(2.0).unwrap()], vec![0_usize]).unwrap();
    let plan = CachedEvaluatorPlan::new(&tree, &[]).unwrap();
    let invalid = CachedEvaluatorOptions {
        max_batch_points: Some(0),
        ..Default::default()
    };
    assert!(TreeTNCachedEvaluator::new(&tree, &[], invalid.clone()).is_err());
    assert!(TreeTNCachedEvaluator::with_plan(&tree, &plan, invalid).is_err());
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &[], CachedEvaluatorOptions::default()).unwrap();
    let actual = evaluator
        .evaluate_batched_typed::<f64>(
            ColMajorArrayRef::new(&[], &[0, 33]).unwrap(),
            EvaluationHint::default(),
        )
        .unwrap();
    assert_eq!(actual, vec![2.0; 33]);
    assert_eq!(evaluator.stats_for_test().batched_center_contract_count, 1);
    let (tree, indices) = typed_three_node_chain::<f64>();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    let mut values = vec![0_usize; 3 * 257];
    values[3 * 256] = indices[0].dim(); // invalid value in the last internal chunk
    let points = ColMajorArrayRef::new(&values, &[3, 257]).unwrap();
    assert!(evaluator.evaluate_batched(points).is_err());
    values[3 * 256] = 1;
    let points = ColMajorArrayRef::new(&values, &[3, 257]).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let actual = evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(0))
        .unwrap();
    assert_typed_results::<f64>(&actual, &expected);
    assert!(evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(99))
        .is_err());
}

fn assert_chunked_branch_hints<T: CachedEvaluatorTestScalar>() {
    let (tree, indices) = typed_unequal_y_tree::<T>();
    let values = (0..33)
        .flat_map(|p| [p % 2, (p / 2) % 2, (p / 4) % 2, (p / 8) % 2])
        .collect::<Vec<_>>();
    let points = ColMajorArrayRef::new(&values, &[4, 33]).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    for limit in [1_usize, 7, 16, usize::MAX] {
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                max_batch_points: Some(limit),
                message_cache_max_bytes: 0,
                branch_slice_cache_max_bytes: 0,
                ..Default::default()
            },
        )
        .unwrap();
        for hint in [
            EvaluationHint::default(),
            EvaluationHint::around(0),
            EvaluationHint::around(1),
        ] {
            let actual = evaluator.evaluate_batched(points).unwrap();
            assert_typed_results::<T>(&actual, &expected);
            let actual = evaluator.evaluate_batched_typed::<T>(points, hint).unwrap();
            let actual = actual
                .into_iter()
                .map(CachedEvaluatorTestScalar::into_any_scalar)
                .collect::<Vec<AnyScalar>>();
            assert_typed_results::<T>(&actual, &expected);
            assert_eq!(
                evaluator.stats_for_test().batched_center_contract_count,
                33_usize.div_ceil(limit)
            );
        }
    }
}

#[test]
fn chunked_branch_hints_and_disabled_caches_preserve_all_scalar_kinds() {
    assert_chunked_branch_hints::<f32>();
    assert_chunked_branch_hints::<f64>();
    assert_chunked_branch_hints::<Complex32>();
    assert_chunked_branch_hints::<Complex64>();
}

#[test]
fn automatic_chunk_limit_shrinks_for_large_local_tensors() {
    let indices = vec![DynIndex::new_dyn(128), DynIndex::new_dyn(128)];
    let tensor = IdxTensor::from_dense(indices.clone(), vec![2.0_f64; 128 * 128]).unwrap();
    let tree = TreeTN::from_tensors(vec![tensor], vec![0_usize]).unwrap();
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    let limit =
        (32 * 1024 * 1024 / (128 * 128 * std::mem::size_of::<CachedScalar>())).clamp(1, 256);
    assert_eq!(evaluator.batch_point_limit, limit);
    assert!(limit < 256);
    let values = vec![0_usize; 2 * (limit + 1)];
    let shape = [2, limit + 1];
    let actual = evaluator
        .evaluate_batched_typed::<f64>(
            ColMajorArrayRef::new(&values, &shape).unwrap(),
            EvaluationHint::default(),
        )
        .unwrap();
    assert_eq!(actual, vec![2.0; limit + 1]);
    assert_eq!(evaluator.stats_for_test().batched_center_contract_count, 2);
}

/// A degree-three centre of a large bond dimension with dimension-one physical
/// indices: the shape #1184 reports, where the automatic chunk had dropped to a
/// single point.
fn large_raw_centre_tree() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let sites: Vec<DynIndex> = (0..4).map(|_| DynIndex::new_dyn(1)).collect();
    let bonds: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(128)).collect();
    let centre_indices = std::iter::once(sites[0].clone())
        .chain(bonds.iter().cloned())
        .collect::<Vec<_>>();
    let centre = IdxTensor::from_dense(centre_indices, vec![1.0_f64; 128 * 128 * 128]).unwrap();
    let leaves = bonds
        .iter()
        .enumerate()
        .map(|(leaf, bond)| {
            IdxTensor::from_dense(
                vec![bond.clone(), sites[leaf + 1].clone()],
                vec![0.5_f64; 128],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut tensors = vec![centre];
    tensors.extend(leaves);
    let tree = TreeTN::from_tensors(tensors, vec![0_usize, 1, 2, 3]).unwrap();
    (tree, sites)
}

/// The raw centre kernels read the core in place and build one message column per
/// point, so a large internal centre must not shrink a batch to a single point
/// (tensor4all-rs #1184).
#[test]
fn automatic_chunk_limit_ignores_the_core_the_raw_centre_route_never_stacks() {
    let (tree, indices) = large_raw_centre_tree();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(evaluator.center(), Some(&0));
    // The core charged per point is 48 MiB, above the whole budget: the estimate
    // that charged it to every route clamps to one point.
    let core_bytes = 128 * 128 * 128 * std::mem::size_of::<CachedScalar>();
    assert_eq!((32 * 1024 * 1024 / core_bytes).clamp(1, 256), 1);
    // The raw route's per-point buffers are the 773 message-column scalars plus
    // the hub's 16512 grouped-message rows (128 x 128 plus the shared column),
    // 405 KiB, so 80 points fit.
    assert_eq!(evaluator.batch_point_limit, 80);

    // Two points therefore share one batch, and give the dense result.
    let values = vec![0_usize; indices.len() * 2];
    let shape = [indices.len(), 2];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let actual = evaluator.evaluate_batched(points).unwrap();
    assert_typed_results::<f64>(&actual, &expected);
    assert_eq!(evaluator.stats_for_test().batched_center_contract_count, 1);
}

/// The reported asymmetric star (#1194): a dim-1 centre with bonds 1024, 1 and
/// 2048 and 256-wide leaf sites.
fn asymmetric_star_tree() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let centre_index = DynIndex::new_dyn(1);
    let bonds: Vec<DynIndex> = [1024_usize, 1, 2048]
        .into_iter()
        .map(DynIndex::new_dyn)
        .collect();
    let centre_indices = std::iter::once(centre_index.clone())
        .chain(bonds.iter().cloned())
        .collect::<Vec<_>>();
    // Bonds 1024, 1 and 2048 around a dim-1 physical index.
    let centre = IdxTensor::from_dense(centre_indices, vec![1.0_f64; 1024 * 2048]).unwrap();
    let mut tensors = vec![centre];
    let mut indices = vec![centre_index];
    for (label, bond) in bonds.iter().enumerate() {
        let site = DynIndex::new_dyn(256);
        tensors.push(
            IdxTensor::from_dense(
                vec![bond.clone(), site.clone()],
                vec![1.0_f64; bond.dim() * 256],
            )
            .unwrap(),
        );
        indices.push(site);
        let _ = label;
    }
    let tree = TreeTN::from_tensors(tensors, vec![0_usize, 1, 2, 3]).unwrap();
    (tree, indices)
}

/// The estimate has to come from the grouped-message geometry, not from a local
/// tensor: with bonds 1024/1/2048 the hub's message asks for `1024 x 2048` rows
/// per point, twice its own larger bond, and the element-count estimate charged
/// 1024 (#1194).
#[test]
fn automatic_chunk_limit_bounds_the_grouped_message_intermediate() {
    let (tree, indices) = asymmetric_star_tree();
    let plan = CachedEvaluatorPlan::new(&tree, &indices).unwrap();
    let rows = grouped_message_intermediate_per_point(&tree, &plan.inner).unwrap();
    // The 3-bond branch kernel's rows, plus the one-wide shared child column it
    // multiplies them by.
    let gemm_rows = 1024 * 2048;
    assert_eq!(rows, gemm_rows + 1);
    // What the element-count estimate charged per point, 2048 times short.
    assert_eq!(gemm_rows / 2048, 1_024);

    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    assert_eq!(evaluator.batch_point_limit, 1);
    // The measured allocation of one GEMM in the reporter's shape: 2,097,152 rows
    // by one column of f64 is 16 MiB, so one point fits and a second would not.
    let intermediate_bytes = evaluator.batch_point_limit * gemm_rows * std::mem::size_of::<f64>();
    assert_eq!(intermediate_bytes, 16 * 1024 * 1024);
    assert!(intermediate_bytes <= 32 * 1024 * 1024);

    // Two columns of the reporter's `[0, 0, p, 0]`, unhinted and hinted at the
    // hub: both run one point per batch, and the hub really is the branch whose
    // GEMM this bounds.
    let values = [0, 0, 0, 0, 0, 0, 1, 0];
    let shape = [indices.len(), 2];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    let actual = evaluator.evaluate_batched(points).unwrap();
    assert_typed_results::<f64>(&actual, &expected);
    assert_eq!(evaluator.stats_for_test().batched_center_contract_count, 2);
    let hinted = evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(0))
        .unwrap();
    assert_typed_results::<f64>(&hinted, &expected);
    assert_eq!(evaluator.stats_for_test().batched_center_contract_count, 2);
}

/// The multi-branch kernel materialises the same way — `rows x points` with
/// `rows = parent x product(other children)` — for a node with three or more
/// children, so a four-bond hub is bounded by the same geometry, and a multi-point
/// batch really runs that grouped kernel (#1194).
#[test]
fn automatic_chunk_limit_bounds_the_multi_branch_message_intermediate() {
    let hub_site = DynIndex::new_dyn(1);
    let bonds: Vec<DynIndex> = (0..4).map(|_| DynIndex::new_dyn(32)).collect();
    let hub_indices = std::iter::once(hub_site.clone())
        .chain(bonds.iter().cloned())
        .collect::<Vec<_>>();
    let hub = IdxTensor::from_dense(hub_indices, vec![1.0_f64; 32 * 32 * 32 * 32]).unwrap();
    let mut tensors = vec![hub];
    let mut indices = vec![hub_site];
    for bond in &bonds {
        let site = DynIndex::new_dyn(2);
        tensors.push(
            IdxTensor::from_dense(
                vec![bond.clone(), site.clone()],
                vec![1.0_f64; bond.dim() * 2],
            )
            .unwrap(),
        );
        indices.push(site);
    }
    let tree = TreeTN::from_tensors(tensors, vec![0_usize, 1, 2, 3, 4]).unwrap();
    let plan = CachedEvaluatorPlan::new(&tree, &indices).unwrap();
    let rows = grouped_message_intermediate_per_point(&tree, &plan.inner).unwrap();
    // 32 x 32 x 32 rows, plus the 32-wide shared child column.
    assert_eq!(rows, 32 * 32 * 32 + 32);
    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    assert_eq!(evaluator.batch_point_limit, 42);
    let intermediate_bytes =
        evaluator.batch_point_limit * (32 * 32 * 32) * std::mem::size_of::<f64>();
    assert!(
        intermediate_bytes <= 32 * 1024 * 1024,
        "one batch asks for {intermediate_bytes} intermediate bytes"
    );

    // Eight points with distinct leaf assignments, the centre hinted at leaf 1: the
    // hub is then a branch with three children, so its message runs the grouped
    // multi-branch kernel with eight columns inside **one** batch.
    let n_points = 8;
    let mut values = vec![0_usize; indices.len() * n_points];
    for point in 0..n_points {
        for (leaf, slot) in values
            .chunks_mut(indices.len())
            .nth(point)
            .unwrap()
            .iter_mut()
            .enumerate()
            .skip(1)
        {
            *slot = (point + leaf) % 2;
        }
    }
    let shape = [indices.len(), n_points];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    let expected = tree.evaluate(&indices, points).unwrap();
    reset_multi_branch_routes_for_test();
    let actual = evaluator
        .evaluate_batched_with_hint(points, EvaluationHint::around(1))
        .unwrap();
    assert_eq!(evaluator.stats_for_test().batched_center_contract_count, 1);
    let (grouped, _flat) = multi_branch_routes_for_test();
    assert!(grouped > 0, "the grouped multi-branch kernel did not run");
    assert_typed_results::<f64>(&actual, &expected);
}

/// The tree of the #1184 report: a dim-1 centre with bonds 145, 165 and 151 and
/// three branches of 20 binary sites. The automatic chunk had dropped to a single
/// point here; it now holds as many as the grouped-message rows allow.
#[test]
fn automatic_chunk_limit_grows_for_the_reported_large_centre() {
    let (tree, indices) = reported_large_centre_tree();
    let plan = CachedEvaluatorPlan::new(&tree, &indices).unwrap();
    let rows = grouped_message_intermediate_per_point(&tree, &plan.inner).unwrap();
    // The widest neighbour of the junction as its parent: 165 x 151 rows plus the
    // 145-wide shared column.
    assert_eq!(rows, 165 * 151 + 145);
    let evaluator =
        TreeTNCachedEvaluator::new(&tree, &indices, CachedEvaluatorOptions::default()).unwrap();
    // The core-based estimate is 1 for the 145x165x151 junction, so the raw-route
    // budget decides: 33,554,432 / ((13,632 + 25,060) x 24) = 36 points.
    assert_eq!(evaluator.batch_point_limit, 36);
}

/// The centre of the #1184 report with three 20-site binary branches.
fn reported_large_centre_tree() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    // The reproducer's `synthetic_comb` (simonb282/t4a-perf-repro, ift_speed):
    // `d` branches of `r` binary sites, node (level i, branch j) named `i*d + j`
    // and the junction `r*d`, with bond `min(2^(i+1), chi_j)` above level i.
    const R: usize = 20;
    const CHIS: [usize; 3] = [145, 165, 151];
    let d = CHIS.len();
    let bond_dim = |level: usize, branch: usize| (1usize << (level + 1).min(40)).min(CHIS[branch]);
    let bonds: Vec<Vec<DynIndex>> = (0..d)
        .map(|branch| {
            (0..R)
                .map(|level| DynIndex::new_dyn(bond_dim(level, branch)))
                .collect()
        })
        .collect();
    let mut tensors = Vec::with_capacity(R * d + 1);
    let mut indices = Vec::with_capacity(R * d + 1);
    for level in 0..R {
        for branch_bonds in &bonds {
            let site = DynIndex::new_dyn(2);
            let mut node_indices = vec![site.clone()];
            if level > 0 {
                node_indices.push(branch_bonds[level - 1].clone());
            }
            node_indices.push(branch_bonds[level].clone());
            let elements = node_indices
                .iter()
                .fold(1_usize, |n, index| n * index.dim());
            tensors.push(IdxTensor::from_dense(node_indices, vec![1.0_f64; elements]).unwrap());
            indices.push(site);
        }
    }
    let junction_site = DynIndex::new_dyn(1);
    let mut junction_indices = vec![junction_site.clone()];
    junction_indices.extend(bonds.iter().map(|branch| branch[R - 1].clone()));
    let elements = junction_indices
        .iter()
        .fold(1_usize, |n, index| n * index.dim());
    tensors.push(IdxTensor::from_dense(junction_indices, vec![1.0_f64; elements]).unwrap());
    indices.push(junction_site);
    let tree = TreeTN::from_tensors(tensors, (0..=R * d).collect()).unwrap();
    (tree, indices)
}

/// A batch that the core-based estimate still fills keeps that size: only the one
/// that had collapsed to a single point moves (#1184).
#[test]
fn automatic_chunk_limit_keeps_its_core_based_value_where_it_still_allows_a_batch() {
    let expected =
        (32 * 1024 * 1024 / (32 * 32 * 32 * std::mem::size_of::<CachedScalar>())).clamp(1, 256);
    assert_eq!(expected, 42);
    // f64 keeps the plan on the raw kernels, f32 off them: neither value moves,
    // because the core-based estimate already allows more than one point.
    let f64_tree = star_tree::<f64>();
    let evaluator =
        TreeTNCachedEvaluator::new(&f64_tree.0, &f64_tree.1, CachedEvaluatorOptions::default())
            .unwrap();
    assert_eq!(evaluator.batch_point_limit, expected);
    let f32_tree = star_tree::<f32>();
    let evaluator =
        TreeTNCachedEvaluator::new(&f32_tree.0, &f32_tree.1, CachedEvaluatorOptions::default())
            .unwrap();
    assert_eq!(evaluator.batch_point_limit, expected);
}

/// A degree-three centre of bond dimension 32 with dimension-one physical
/// indices: its core-based estimate is 42 points.
fn star_tree<T: CachedEvaluatorTestScalar>() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let sites: Vec<DynIndex> = (0..4).map(|_| DynIndex::new_dyn(1)).collect();
    let bonds: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(32)).collect();
    let centre_indices = std::iter::once(sites[0].clone())
        .chain(bonds.iter().cloned())
        .collect::<Vec<_>>();
    let centre = IdxTensor::from_dense(
        centre_indices,
        (0..32 * 32 * 32)
            .map(|value| T::from_parts(value as f64 * 0.5, 0.0))
            .collect(),
    )
    .unwrap();
    let leaves = bonds
        .iter()
        .enumerate()
        .map(|(leaf, bond)| {
            IdxTensor::from_dense(
                vec![bond.clone(), sites[leaf + 1].clone()],
                (0..32)
                    .map(|value| T::from_parts(0.5 + value as f64, -0.25))
                    .collect(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut tensors = vec![centre];
    tensors.extend(leaves);
    let tree = TreeTN::from_tensors(tensors, vec![0_usize, 1, 2, 3]).unwrap();
    (tree, sites)
}

fn assert_default_batch_chunking<T: CachedEvaluatorTestScalar>() {
    let (tree, indices) = typed_three_node_chain::<T>();
    for n_points in [0_usize, 1, 255, 256, 257, 512, 513] {
        let values = (0..n_points)
            .flat_map(|point| [point % 2, (point / 2) % 2, point % 3])
            .collect::<Vec<_>>();
        let shape = [3, n_points];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        let expected = tree.evaluate(&indices, points).unwrap();
        let mut evaluator = TreeTNCachedEvaluator::new(
            &tree,
            &indices,
            CachedEvaluatorOptions {
                center: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        let actual = evaluator.evaluate_batched(points).unwrap();
        assert_typed_results::<T>(&actual, &expected);
        assert_eq!(
            evaluator.stats_for_test().batched_center_contract_count,
            n_points.div_ceil(256)
        );
    }
}

#[test]
fn default_batch_chunking_bounds_center_work_and_preserves_results() {
    assert_default_batch_chunking::<f32>();
    assert_default_batch_chunking::<f64>();
    assert_default_batch_chunking::<Complex32>();
    assert_default_batch_chunking::<Complex64>();
}
