use anyhow::Result;
use num_complex::{Complex32, Complex64};
use tensor4all_core::{CommonScalar, IdxTensor, MatrixLuciScalar, Scalar, TensorElement};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, TreeTciEdge, TreeTciGraph, TreeTciOptions,
};

fn memo_parity<T>()
where
    T: FullPivLuScalar
        + CommonScalar
        + MatrixLuciScalar
        + TensorElement
        + tensor4all_treetci::globalpivot::ScalarParts
        + Scalar
        + PartialEq
        + std::fmt::Debug,
{
    for (dims, edges) in [
        (
            vec![2; 4],
            vec![
                TreeTciEdge::new(0, 1),
                TreeTciEdge::new(1, 2),
                TreeTciEdge::new(2, 3),
            ],
        ),
        (
            vec![1, 2, 2, 2],
            vec![
                TreeTciEdge::new(0, 1),
                TreeTciEdge::new(0, 2),
                TreeTciEdge::new(0, 3),
            ],
        ),
    ] {
        let mut reference = None;
        for limit in [None, Some(0), Some(16), Some(1024)] {
            let mut evaluated = 0;
            let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<T>> {
                evaluated += batch.n_points(); // FnMut, no RefCell wrapper required.
                Ok(batch
                    .data()
                    .chunks_exact(batch.n_sites())
                    .map(|point| {
                        <T as Scalar>::from_f64(point.iter().map(|&i| (i + 1) as f64).product())
                    })
                    .collect())
            };
            let result = crossinterpolate2::<T, _, _>(
                evaluate,
                dims.clone(),
                TreeTciGraph::new(dims.len(), &edges).unwrap(),
                vec![vec![0; dims.len()]],
                TreeTciOptions {
                    evaluation_cache_bytes: limit,
                    seed: Some(4),
                    tolerance: <T as Scalar>::epsilon() * 100.0,
                    ..Default::default()
                },
                None,
                &DefaultProposer,
            )
            .unwrap();
            assert_eq!(result.evaluation.evaluated_points, evaluated);
            assert_eq!(result.evaluation.retained_byte_limit, limit);
            assert!(result.evaluation.requested_points >= evaluated);
            if let Some(limit) = limit {
                assert!(result.evaluation.retained_bytes <= limit);
                if limit == 1024 {
                    assert!(result.evaluation.cache_hits > 0);
                    assert!(evaluated < result.evaluation.requested_points);
                    assert_eq!(result.evaluation.cached_entries, evaluated);
                }
            } else {
                assert_eq!(result.evaluation.requested_points, evaluated);
                assert_eq!(result.evaluation.cached_entries, 0);
            }
            let dense = result.treetn.contract_to_tensor().unwrap();
            let values = dense.to_vec::<T>().unwrap();
            // Product fixture is invariant to physical-axis order; unit axes
            // contribute exactly one. Materialize once and compare by tensor subtraction.
            let expected = (0..values.len())
                .map(|mut linear| {
                    let mut value = 1.0;
                    for dim in dense.dims() {
                        value *= (linear % dim + 1) as f64;
                        linear /= dim;
                    }
                    <T as Scalar>::from_f64(value)
                })
                .collect::<Vec<_>>();
            let expected = IdxTensor::from_dense(dense.indices().to_vec(), expected).unwrap();
            assert!(
                dense.sub(&expected).unwrap().maxabs().unwrap() < <T as Scalar>::epsilon() * 1600.0
            );
            let fingerprint = (values, result.ranks, result.errors, result.termination);
            if let Some(reference) = &reference {
                assert_eq!(&fingerprint, reference);
            } else {
                reference = Some(fingerprint);
            }
        }
    }
}

#[test]
fn optional_memo_matches_uncached_runs_for_every_scalar_kind() {
    memo_parity::<f32>();
    memo_parity::<f64>();
    memo_parity::<Complex32>();
    memo_parity::<Complex64>();
}

#[test]
fn public_materialization_crosses_default_chunk_boundary() {
    use tensor4all_treetci::{to_treetn, TreeTCI2};
    let mut state =
        TreeTCI2::<f64>::new(vec![65_539, 1], TreeTciGraph::linear_chain(2).unwrap()).unwrap();
    state.add_global_pivots(&[vec![0, 0]]).unwrap();
    let sizes = std::cell::RefCell::new(Vec::new());
    let tn = to_treetn(
        &state,
        |batch| {
            sizes.borrow_mut().push(batch.n_points());
            Ok(batch
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| p[0] as f64 + 0.25)
                .collect())
        },
        None,
    )
    .unwrap();
    assert_eq!(*sizes.borrow(), vec![65_536, 3, 1, 1]);
    let dense = tn.contract_to_tensor().unwrap();
    let expected = IdxTensor::from_dense(
        dense.indices().to_vec(),
        (0..65_539).map(|i| i as f64 + 0.25).collect(),
    )
    .unwrap();
    assert_eq!(dense.sub(&expected).unwrap().maxabs().unwrap(), 0.0);
}

#[test]
fn public_edge_updates_cross_default_chunk_boundary_in_column_major_order() {
    use tensor4all_treetci::{optimize_default, TreeTCI2};
    let mut state =
        TreeTCI2::<f64>::new(vec![257, 257], TreeTciGraph::linear_chain(2).unwrap()).unwrap();
    state.add_global_pivots(&[vec![0, 0]]).unwrap();
    state.max_sample_value = 1.0;
    let mut calls = Vec::new();
    optimize_default(
        &mut state,
        |batch: GlobalIndexBatch<'_>| {
            // Each candidate matrix is column-major: site zero varies first.
            if batch.n_points() > 257 {
                calls.push(batch.data().to_vec());
            }
            Ok(batch
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| (p[0] + 1) as f64 * (p[1] + 1) as f64)
                .collect())
        },
        &TreeTciOptions {
            max_iter: 1,
            enable_global_pivots: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(calls.len(), 4); // Two half-sweeps, each with two chunks.
    for chunks in calls.as_chunks::<2>().0.iter() {
        assert_eq!(chunks[0].len(), 2 * 65_536);
        assert_eq!(chunks[1].len(), 2 * (257 * 257 - 65_536));
        let points = chunks.concat();
        for (position, point) in points.as_chunks::<2>().0.iter().enumerate() {
            assert_eq!(*point, [position % 257, position / 257]);
        }
    }
    assert_eq!(state.max_bond_dim(), 1);
}

#[test]
fn continued_optimization_and_caller_rng_each_own_a_fresh_cache() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use tensor4all_treetci::{optimize_with_proposer, optimize_with_proposer_with_rng, TreeTCI2};
    let mut state =
        TreeTCI2::<f64>::new(vec![2, 2], TreeTciGraph::linear_chain(2).unwrap()).unwrap();
    state.add_global_pivots(&[vec![0, 0]]).unwrap();
    state.max_sample_value = 2.0;
    let options = TreeTciOptions {
        evaluation_cache_bytes: Some(1024),
        max_iter: 1,
        ..Default::default()
    };
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    for seeded in [true, false] {
        let mut points = 0;
        let evaluate = |batch: GlobalIndexBatch<'_>| {
            points += batch.n_points();
            Ok(vec![2.0; batch.n_points()])
        };
        let result = if seeded {
            optimize_with_proposer(&mut state, evaluate, &options, &DefaultProposer)
        } else {
            optimize_with_proposer_with_rng(
                &mut state,
                evaluate,
                &options,
                &DefaultProposer,
                &mut rng,
            )
        }
        .unwrap();
        assert_eq!(points, 4);
        assert_eq!(result.evaluation.evaluated_points, points);
        assert_eq!(result.evaluation.cached_entries, 4);
        assert!(result.evaluation.cache_hits > 0);
    }
}

#[test]
fn memo_accepts_a_borrowed_mutable_tree_evaluator() {
    use tensor4all_core::ColMajorArrayRef;
    use tensor4all_treetn::TreeTNCachedEvaluator;
    let graph = TreeTciGraph::linear_chain(2).unwrap();
    let source = crossinterpolate2::<f64, _, _>(
        |batch| {
            Ok(batch
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| 2.0 * (p[0] + 1) as f64 * (p[1] + 1) as f64)
                .collect())
        },
        vec![2, 2],
        graph.clone(),
        vec![vec![0, 0]],
        TreeTciOptions {
            seed: Some(0),
            ..Default::default()
        },
        None,
        &DefaultProposer,
    )
    .unwrap()
    .treetn;
    let indices = (0..2)
        .map(|site| {
            source
                .tensor(source.node_index(&site).unwrap())
                .unwrap()
                .indices()[0]
                .clone()
        })
        .collect::<Vec<_>>();
    let mut oracle =
        TreeTNCachedEvaluator::<usize>::new(&source, &indices, Default::default()).unwrap();
    let result = crossinterpolate2::<f64, _, _>(
        |batch: GlobalIndexBatch<'_>| {
            let shape = [batch.n_sites(), batch.n_points()];
            Ok(oracle
                .evaluate_batched(ColMajorArrayRef::new(batch.data(), &shape)?)?
                .iter()
                .map(|v| v.real())
                .collect())
        },
        vec![2, 2],
        graph,
        vec![vec![0, 0]],
        TreeTciOptions {
            evaluation_cache_bytes: Some(1024),
            seed: Some(0),
            ..Default::default()
        },
        None,
        &DefaultProposer,
    )
    .unwrap();
    assert_eq!(result.evaluation.evaluated_points, 4);
    let dense = result.treetn.contract_to_tensor().unwrap();
    let expected =
        IdxTensor::from_dense(dense.indices().to_vec(), vec![2.0, 4.0, 4.0, 8.0]).unwrap();
    assert!(dense.sub(&expected).unwrap().maxabs().unwrap() < 1e-12);
}
