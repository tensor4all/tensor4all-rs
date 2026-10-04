use num_complex::Complex32;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_tensorbackend::{
    default_cpu_execution_context, CpuExecutionContext, ExecutionContext, Matrix,
};
use tensor4all_treersi::{
    hadamard_many, hadamard_many_with_rng_in, TreeRsiError, TreeRsiOptions, TreeRsiProbes,
};
use tensor4all_treetn::TreeTN;

fn single(values: Vec<f64>) -> TreeTN<IdxTensor, usize> {
    TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![DynIndex::new_dyn(values.len())], values).unwrap()],
        vec![0],
    )
    .unwrap()
}
fn options() -> TreeRsiOptions<usize> {
    TreeRsiOptions {
        max_bond_dim: Some(4),
        ..Default::default()
    }
}

#[test]
fn invalid_controls_and_nonfinite_inputs_fail_without_advancing_rng() {
    let tree = single(vec![1.0, 2.0]);
    let context = ExecutionContext::Cpu(default_cpu_execution_context());
    let bad = [
        TreeRsiOptions {
            max_bond_dim: Some(0),
            ..options()
        },
        TreeRsiOptions {
            sketch_dim: Some(0),
            ..options()
        },
        TreeRsiOptions {
            max_bond_dim: None,
            sketch_dim: None,
            ..options()
        },
        TreeRsiOptions {
            rel_tol: f64::NAN,
            ..options()
        },
        TreeRsiOptions {
            rel_tol: -1.0,
            ..options()
        },
        TreeRsiOptions {
            max_local_elements: 0,
            ..options()
        },
        TreeRsiOptions {
            root: Some(99),
            ..options()
        },
    ];
    for options in bad {
        let mut rng = ChaCha8Rng::seed_from_u64(71);
        let mut expected = rng.clone();
        let err = hadamard_many_with_rng_in::<f64, _, _>(
            std::slice::from_ref(&tree),
            &options,
            &mut rng,
            &context,
        )
        .unwrap_err();
        assert!(matches!(err, TreeRsiError::InvalidOption { .. }), "{err}");
        assert_eq!(rng.random::<u64>(), expected.random::<u64>());
    }
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut rng = ChaCha8Rng::seed_from_u64(71);
        let mut expected = rng.clone();
        assert!(matches!(
            hadamard_many_with_rng_in::<f64, _, _>(
                &[single(vec![x, 1.0])],
                &options(),
                &mut rng,
                &context
            ),
            Err(TreeRsiError::NonFiniteValue { .. })
        ));
        assert_eq!(rng.random::<u64>(), expected.random::<u64>());
    }
}

#[test]
fn empty_inputs_limits_scalar_and_physical_index_mismatches_are_errors() {
    assert!(matches!(
        hadamard_many::<f64, usize>(&[], &options()),
        Err(TreeRsiError::NoInputs)
    ));
    let a = single(vec![1.0, 2.0]);
    assert!(matches!(
        hadamard_many::<f32, _>(std::slice::from_ref(&a), &options()),
        Err(TreeRsiError::ScalarKind { .. })
    ));
    assert!(matches!(
        hadamard_many::<f64, _>(
            std::slice::from_ref(&a),
            &TreeRsiOptions {
                max_local_elements: 1,
                ..options()
            }
        ),
        Err(TreeRsiError::ResourceLimit { .. })
    ));
    assert!(matches!(
        hadamard_many::<f64, _>(&[a, single(vec![1.0, 2.0])], &options()),
        Err(TreeRsiError::PhysicalIndexMismatch { .. })
    ));
}

#[test]
fn context_mismatch_is_rejected_before_rng_advances() {
    let context = ExecutionContext::Cpu(
        CpuExecutionContext::from_backend(tenferro_cpu::CpuBackend::new()).into(),
    );
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let mut before = rng.clone();
    let error = hadamard_many_with_rng_in::<f64, _, _>(
        &[single(vec![1.0, 2.0])],
        &options(),
        &mut rng,
        &context,
    )
    .unwrap_err();
    assert!(matches!(error, TreeRsiError::InputContext(_)));
    assert_eq!(rng.random::<u64>(), before.random::<u64>());
}

#[test]
fn zero_products_and_single_node_products_have_known_values() {
    for values in [vec![0.0, 0.0], vec![2.0, -3.0]] {
        let a = single(values.clone());
        let out = hadamard_many::<f64, _>(&[a.clone(), a], &options()).unwrap();
        assert_eq!(
            out.tree.to_dense().unwrap().to_vec::<f64>().unwrap(),
            values.iter().map(|x| x * x).collect::<Vec<_>>()
        );
    }
}

#[test]
fn complex32_subnormal_products_preserve_phase() {
    let site = DynIndex::new_dyn(2);
    let x = Complex32::new(1e-20, 1e-20);
    let a = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site], vec![x, -x]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let result = hadamard_many::<Complex32, _>(&[a.clone(), a], &options()).unwrap();
    for z in result
        .tree
        .to_dense()
        .unwrap()
        .to_vec::<Complex32>()
        .unwrap()
    {
        assert_eq!(z.re, 0.0);
        assert!((z.im - 2e-40).abs() <= 2.0 * f32::from_bits(1));
    }
}

#[test]
fn unrepresentable_outputs_are_errors_instead_of_successful_zeros_or_infinities() {
    for value in [1e-300, 1e300] {
        let a = single(vec![value]);
        assert!(matches!(
            hadamard_many::<f64, _>(&[a.clone(), a], &options()),
            Err(TreeRsiError::DynamicRange { .. })
        ));
    }
}

#[test]
fn scaled_gemm_rejects_underflow_when_the_final_hadamard_product_is_well_scaled() {
    fn diagonal(values: [f64; 3]) -> Vec<f64> {
        let mut data = vec![0.0; 9];
        for (i, value) in values.into_iter().enumerate() {
            data[i + 3 * i] = value;
        }
        data
    }

    fn two_site(sites: &[DynIndex], left: [f64; 3], right: [f64; 3]) -> TreeTN<IdxTensor, usize> {
        let bond = DynIndex::new_dyn(3);
        TreeTN::from_tensors(
            vec![
                IdxTensor::from_dense(vec![sites[0].clone(), bond.clone()], diagonal(left))
                    .unwrap(),
                IdxTensor::from_dense(vec![bond, sites[1].clone()], diagonal(right)).unwrap(),
            ],
            vec![0, 1],
        )
        .unwrap()
    }

    let sites = vec![DynIndex::new_dyn(3), DynIndex::new_dyn(3)];
    let a = two_site(&sites, [1e200, 1e-100, 1e-100], [1e-100, 1e200, 1e-100]);
    let b = two_site(&sites, [1e-100, 1e-100, 1e100], [1.0, 1.0, 1e100]);
    let dense_a = a
        .to_dense()
        .unwrap()
        .permute_indices(&sites)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    let dense_b = b
        .to_dense()
        .unwrap()
        .permute_indices(&sites)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    let exact_product = dense_a
        .iter()
        .zip(&dense_b)
        .map(|(x, y)| x * y)
        .collect::<Vec<_>>();
    assert_eq!(
        exact_product,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );

    let options = TreeRsiOptions {
        max_bond_dim: Some(3),
        sketch_dim: Some(3),
        ..Default::default()
    };
    assert!(matches!(
        hadamard_many::<f64, _>(&[a, b], &options),
        Err(TreeRsiError::DynamicRange { .. })
    ));
}

/// The true product is constant 2 (rank one). A rank-one row ID of the
/// specified sketches flips a sign. Success/rank/local pivots cannot certify it.
#[test]
fn explicit_probes_reproduce_the_rank_one_product_counterexample() {
    let sites = [
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(1),
        DynIndex::new_dyn(2),
    ];
    let inputs = [[1.0, 2.0, 2.0, 1.0], [2.0, 1.0, 1.0, 2.0]]
        .into_iter()
        .map(|data| {
            let (a, b) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
            TreeTN::from_tensors(
                vec![
                    IdxTensor::from_dense(vec![a.clone(), sites[0].clone()], data.to_vec())
                        .unwrap(),
                    IdxTensor::from_dense(
                        vec![a, b.clone(), sites[1].clone()],
                        vec![1.0, 0.0, 0.0, 1.0],
                    )
                    .unwrap(),
                    IdxTensor::from_dense(vec![b, sites[2].clone()], vec![1.0, 0.0, 0.0, 1.0])
                        .unwrap(),
                ],
                vec![0usize, 1, 2],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let probes = TreeRsiProbes::new(vec![
        HashMap::from([(2, Matrix::from_col_major_vec(2, 1, vec![1.0, 1.0]))]),
        HashMap::from([(2, Matrix::from_col_major_vec(2, 1, vec![1.0, -1.0]))]),
    ]);
    let options = TreeRsiOptions {
        max_bond_dim: Some(1),
        sketch_dim: Some(1),
        probes: Some(probes),
        ..Default::default()
    };
    let result = hadamard_many::<f64, _>(&inputs, &options).unwrap();
    let actual = result
        .tree
        .to_dense()
        .unwrap()
        .permute_indices(&sites)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    assert_eq!(actual, vec![2.0, -2.0, 2.0, -2.0]);
    let relative = (actual.iter().map(|x| (x - 2.0).powi(2)).sum::<f64>() / 16.0).sqrt();
    assert!((relative - 2.0_f64.sqrt()).abs() < 1e-14);
}

#[test]
fn required_explicit_probes_are_validated_before_consuming_rng() {
    let sites = (0..4).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let bonds = (0..3).map(|_| DynIndex::new_dyn(1)).collect::<Vec<_>>();
    let tensors = (0..4)
        .map(|i| {
            let mut indices = vec![sites[i].clone()];
            if i > 0 {
                indices.push(bonds[i - 1].clone());
            }
            if i < 3 {
                indices.push(bonds[i].clone());
            }
            IdxTensor::from_dense(indices, vec![1.0_f64, 2.0]).unwrap()
        })
        .collect();
    let tree = TreeTN::from_tensors(tensors, vec![0usize, 1, 2, 3]).unwrap();
    let context = ExecutionContext::Cpu(default_cpu_execution_context());
    for maps in [
        vec![],
        vec![HashMap::new()],
        vec![HashMap::from([
            (2, Matrix::from_col_major_vec(2, 1, vec![f64::NAN, 1.0])),
            (3, Matrix::from_col_major_vec(2, 1, vec![1.0, 1.0])),
        ])],
    ] {
        let options = TreeRsiOptions {
            max_bond_dim: Some(1),
            sketch_dim: Some(1),
            probes: Some(TreeRsiProbes::new(maps)),
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut expected = rng.clone();
        assert!(matches!(
            hadamard_many_with_rng_in::<f64, _, _>(
                std::slice::from_ref(&tree),
                &options,
                &mut rng,
                &context
            ),
            Err(TreeRsiError::InvalidProbes { .. })
        ));
        assert_eq!(rng.random::<u64>(), expected.random::<u64>());
    }
}

#[test]
fn physical_groups_and_axis_permutations_preserve_semantic_indices() {
    let (s, t, b) = (
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(3),
        DynIndex::new_dyn(2),
    );
    let tensor = IdxTensor::from_dense(
        vec![s.clone(), t.clone(), b.clone()],
        (1..=12).map(|x| x as f64).collect(),
    )
    .unwrap();
    let end = IdxTensor::from_dense(vec![b.clone()], vec![2.0_f64, -1.0]).unwrap();
    let a = TreeTN::from_tensors(vec![tensor.clone(), end.clone()], vec![0usize, 1]).unwrap();
    let reordered = tensor
        .permute_indices(&[b.clone(), t.clone(), s.clone()])
        .unwrap();
    let c = TreeTN::from_tensors(vec![reordered, end], vec![0usize, 1]).unwrap();
    let exact = a
        .to_dense()
        .unwrap()
        .permute_indices(&[s.clone(), t.clone()])
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    let out = hadamard_many::<f64, _>(&[a, c], &options()).unwrap();
    let values = out
        .tree
        .to_dense()
        .unwrap()
        .permute_indices(&[s, t])
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    for (v, x) in values.into_iter().zip(exact) {
        assert!((v - x * x).abs() < 1e-12);
    }
}

#[test]
fn topology_and_sketch_size_overflow_are_rejected() {
    let tree = single(vec![1.0, 2.0]);
    let other = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![1.0_f64, 2.0]).unwrap()],
        vec![99usize],
    )
    .unwrap();
    assert!(matches!(
        hadamard_many::<f64, _>(&[tree.clone(), other], &options()),
        Err(TreeRsiError::TopologyMismatch { .. })
    ));
    for options in [
        TreeRsiOptions {
            sketch_dim: Some(usize::MAX),
            ..options()
        },
        TreeRsiOptions {
            max_bond_dim: Some(usize::MAX),
            ..options()
        },
    ] {
        assert!(matches!(
            hadamard_many::<f64, _>(std::slice::from_ref(&tree), &options),
            Err(TreeRsiError::SizeOverflow { .. })
        ));
    }
}

#[test]
fn zero_local_sketch_has_a_well_defined_zero_product_convention() {
    let (s, t, b) = (
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(1),
    );
    let a = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![s, b.clone()], vec![0.0_f64, 0.0]).unwrap(),
            IdxTensor::from_dense(vec![b, t], vec![2.0_f64, 3.0]).unwrap(),
        ],
        vec![0usize, 1],
    )
    .unwrap();
    let out = hadamard_many::<f64, _>(&[a.clone(), a], &options()).unwrap();
    assert_eq!(
        out.tree.to_dense().unwrap().to_vec::<f64>().unwrap(),
        vec![0.0; 4]
    );
    assert_eq!(out.diagnostics.edges[0].rank, 1);
    assert_eq!(out.diagnostics.edges[0].relative_pivot, 0.0);
}
