use num_complex::Complex64;
use tensor4all_core::{
    AnyScalar, DynIndex, IdxTensor, IndexLike, Scalar, SvdTruncationPolicy, TensorElement,
};

use super::*;

fn diagonal<T: TensorElement + Scalar>(
    sites: &[DynIndex],
    values: [T; 2],
) -> TreeTN<IdxTensor, usize> {
    let bond = DynIndex::new_dyn(2);
    TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![sites[0].clone(), bond.clone()],
                vec![
                    T::from_f64(1.0),
                    T::from_f64(0.0),
                    T::from_f64(0.0),
                    T::from_f64(1.0),
                ],
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![bond, sites[1].clone()],
                vec![values[0], T::from_f64(0.0), T::from_f64(0.0), values[1]],
            )
            .unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap()
}

fn dense_error_squared(actual: &TreeTN<IdxTensor, usize>, expected: &IdxTensor) -> f64 {
    let difference = actual.to_dense().unwrap().sub(expected).unwrap();
    let norm = difference.norm().unwrap();
    norm * norm
}

// Removing i != j density contributions would retain a cancelled second direction.
#[test]
fn density_matrix_sum_retains_interference_and_cancellation() {
    let sites: Vec<_> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    for phase in [AnyScalar::new_real(1.0), AnyScalar::new_complex(0.0, 1.0)] {
        let terms: Vec<_> = [[3.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
            .into_iter()
            .map(|values| {
                if phase.is_complex() {
                    diagonal(&sites, values.map(|value| Complex64::new(0.0, value)))
                } else {
                    diagonal(&sites, values)
                }
            })
            .collect();
        let references: Vec<_> = terms.iter().collect();
        let result = TreeTN::density_matrix_sum(&references, &0, 1e-12, None).unwrap();
        let expected = IdxTensor::from_dense(sites.clone(), vec![3.0, 0.0, 0.0, 0.0])
            .unwrap()
            .scale(phase.clone())
            .unwrap();
        assert_eq!(result.network.link_dims(), vec![1]);
        assert!(dense_error_squared(&result.network, &expected) < 1e-24);
        assert!(result.discarded_weight.abs() < 1e-12);

        let opposite = terms[0].scale(AnyScalar::new_real(-1.0)).unwrap();
        let zero = TreeTN::density_matrix_sum(&[&terms[0], &opposite], &1, 0.0, None).unwrap();
        assert!(zero.network.to_dense().unwrap().maxabs().unwrap() < 1e-12);
        assert_eq!(zero.network.link_dims(), vec![1]);
    }
}

// Per-eigenvalue thresholding and ignoring the cap each violate the known tail.
#[test]
fn density_matrix_sum_uses_discarded_weight_and_hard_cap() {
    let sites: Vec<_> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    let term = diagonal(&sites, [3.0, 1.0]);
    let exact = term.to_dense().unwrap();
    for center in [0, 1] {
        for (cutoff, cap, rank, tail) in [
            (0.09, None, 2, 0.0),
            (0.11, None, 1, 1.0),
            (0.0, Some(1), 1, 1.0),
            (1.0, None, 1, 1.0),
        ] {
            let result = TreeTN::density_matrix_sum(&[&term], &center, cutoff, cap).unwrap();
            assert_eq!(result.network.link_dims(), vec![rank]);
            assert!((result.discarded_weight - tail).abs() < 1e-12);
            assert!((dense_error_squared(&result.network, &exact) - tail).abs() < 1e-12);
            assert_eq!(
                result
                    .network
                    .canonical_region()
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>(),
                vec![center]
            );
            assert_eq!(
                result.network.canonical_form(),
                Some(crate::CanonicalForm::Unitary)
            );
            result.network.verify_internal_consistency().unwrap();
        }
    }
}

fn branched(sites: &[DynIndex], shift: f64, complex: bool) -> TreeTN<IdxTensor, usize> {
    let bonds: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let indices = vec![
        vec![
            sites[0].clone(),
            sites[1].clone(),
            bonds[0].clone(),
            bonds[1].clone(),
            bonds[2].clone(),
        ],
        vec![bonds[0].clone(), sites[2].clone()],
        vec![bonds[1].clone()],
        vec![bonds[2].clone(), sites[3].clone()],
    ];
    let tensors = indices
        .into_iter()
        .enumerate()
        .map(|(node, indices)| {
            let size: usize = indices.iter().map(|index| index.dim()).product();
            let values: Vec<_> = (0..size)
                .map(|i| {
                    let real = ((i * 7 + node * 3) % 11) as f64 / 11.0 - shift;
                    let imaginary = if complex {
                        ((i * 3 + node) % 7) as f64 / 9.0
                    } else {
                        0.0
                    };
                    Complex64::new(real, imaginary)
                })
                .collect();
            if complex {
                IdxTensor::from_dense(indices, values).unwrap()
            } else {
                IdxTensor::from_dense(indices, values.into_iter().map(|v| v.re).collect()).unwrap()
            }
        })
        .collect();
    TreeTN::from_tensors(tensors, vec![0, 1, 2, 3]).unwrap()
}

// Conjugation errors, chain-only traversal, or local metrics instead of exterior
// overlaps all change this dense oracle; omitting sibling projections can only
// overestimate the reported bound.
#[test]
fn density_matrix_sum_branches_complex_sites_and_error_bound() {
    let sites: Vec<_> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
    for complex in [false, true] {
        let terms = [
            branched(&sites, 0.3, complex),
            branched(&sites, 0.7, complex),
        ];
        let expected = terms[0]
            .to_dense()
            .unwrap()
            .add(&terms[1].to_dense().unwrap())
            .unwrap();
        for center in [0, 2] {
            for (cutoff, cap) in [(0.0, None), (0.03, None), (0.0, Some(1))] {
                let result =
                    TreeTN::density_matrix_sum(&[&terms[0], &terms[1]], &center, cutoff, cap)
                        .unwrap();
                let error = dense_error_squared(&result.network, &expected);
                assert!(
                    error <= result.discarded_weight + 1e-10,
                    "{error} > {}",
                    result.discarded_weight
                );
                if cutoff == 0.0 && cap.is_none() {
                    assert!(error < 1e-22, "{error}");
                }
                if let Some(cap) = cap {
                    assert!(result.network.link_dims().iter().all(|&rank| rank <= cap));
                }
                if cap.is_none() {
                    let norm = expected.norm().unwrap();
                    assert!(result.discarded_weight <= 3.0 * cutoff * norm * norm + 1e-10);
                }
                assert!(result
                    .network
                    .share_equivalent_site_index_network(&terms[0]));
                result.network.verify_internal_consistency().unwrap();
            }
        }
    }
}

// Fast paths must not accept invalid options, mixed scalar kinds, or site aliases.
#[test]
fn density_matrix_sum_validates_before_single_node_fast_path() {
    let site = DynIndex::new_dyn(2);
    let real = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], vec![1.0, 2.0]).unwrap()],
        vec![0],
    )
    .unwrap();
    let complex = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], vec![Complex64::new(1.0, 0.0); 2]).unwrap()],
        vec![0],
    )
    .unwrap();
    let narrow = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site], vec![1.0_f32, 2.0]).unwrap()],
        vec![0],
    )
    .unwrap();
    let result = TreeTN::density_matrix_sum(&[&real, &real], &0, 0.0, None).unwrap();
    assert_eq!(
        result.network.to_dense().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0, 4.0]
    );
    assert_eq!(result.discarded_weight, 0.0);
    assert!(TreeTN::<IdxTensor, usize>::density_matrix_sum(&[], &0, 0.0, None).is_err());
    assert!(
        TreeTN::density_matrix_sum(&[&TreeTN::<IdxTensor, usize>::new()], &0, 0.0, None).is_err()
    );
    for cutoff in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(TreeTN::density_matrix_sum(&[&real], &0, cutoff, None).is_err());
    }
    assert!(TreeTN::density_matrix_sum(&[&real], &0, 0.0, Some(0)).is_err());
    assert!(TreeTN::density_matrix_sum(&[&real], &1, 0.0, None).is_err());
    assert!(TreeTN::density_matrix_sum(&[&real, &complex], &0, 0.0, None).is_err());
    assert!(TreeTN::density_matrix_sum(&[&real, &narrow], &0, 0.0, None).is_err());
    let other = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![1.0, 2.0]).unwrap()],
        vec![0],
    )
    .unwrap();
    assert!(TreeTN::density_matrix_sum(&[&real, &other], &0, 0.0, None).is_err());
    let primed_site = real.node_site_indices(&0).unwrap()[0].prime();
    let primed = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![primed_site], vec![1.0, 2.0]).unwrap()],
        vec![0],
    )
    .unwrap();
    assert!(TreeTN::density_matrix_sum(&[&real, &primed], &0, 0.0, None).is_err());
    let mut resized_site = real.node_site_indices(&0).unwrap()[0].clone();
    resized_site.dim = 3;
    let resized = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![resized_site], vec![1.0, 2.0, 3.0]).unwrap()],
        vec![0],
    )
    .unwrap();
    for error in [
        TreeTN::density_matrix_sum(&[&real, &resized], &0, 0.0, None).unwrap_err(),
        TreeTN::exact_sum(&[&real, &resized], crate::SumEngine::Nary).unwrap_err(),
    ] {
        assert!(matches!(error, TreeTNOperationError::InvalidGraph { .. }));
    }
    let sites: Vec<_> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    for term in [
        diagonal(&sites, [3.0_f32, 1.0]),
        diagonal(
            &sites,
            [
                num_complex::Complex32::new(0.0, 3.0),
                num_complex::Complex32::new(1.0, 0.0),
            ],
        ),
    ] {
        let result = TreeTN::density_matrix_sum(&[&term], &0, 0.11, None).unwrap();
        let expected = term.to_dense().unwrap();
        assert!((dense_error_squared(&result.network, &expected) - 1.0).abs() < 1e-4);
        assert_eq!(result.network.link_dims(), vec![1]);
        let tensor = tree_tensor(&result.network, &0).unwrap();
        assert_eq!(tensor.is_f32(), tree_tensor(&term, &0).unwrap().is_f32());
        assert_eq!(tensor.is_c32(), tree_tensor(&term, &0).unwrap().is_c32());
    }
}

fn ghz(
    sites: &[DynIndex],
    edges: &[(usize, usize)],
    amplitudes: [f64; 2],
) -> TreeTN<IdxTensor, usize> {
    let bonds: Vec<_> = edges.iter().map(|_| DynIndex::new_dyn(2)).collect();
    let tensors =
        sites
            .iter()
            .enumerate()
            .map(|(node, site)| {
                let mut indices = vec![site.clone()];
                indices.extend(edges.iter().zip(&bonds).filter_map(|(&(a, b), bond)| {
                    (node == a || node == b).then_some(bond.clone())
                }));
                IdxTensor::from_diag(
                    indices,
                    if node == 0 {
                        amplitudes.to_vec()
                    } else {
                        vec![1.0, 1.0]
                    },
                )
                .unwrap()
            })
            .collect();
    TreeTN::from_tensors(tensors, (0..sites.len()).collect()).unwrap()
}

// The same per-edge cutoff has a known rank-one optimum in both algorithms;
// this catches silently interpreting cutoff as an amplitude threshold.
#[test]
fn density_matrix_sum_matches_exact_then_truncate_ranks() {
    let sites: Vec<_> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
    for edges in [vec![(0, 1), (1, 2), (2, 3)], vec![(0, 1), (0, 2), (0, 3)]] {
        let terms = [
            ghz(&sites, &edges, [2.0, 0.6]),
            ghz(&sites, &edges, [1.0, 0.4]),
        ];
        let references = [&terms[0], &terms[1]];
        let exact = TreeTN::exact_sum(&references, crate::SumEngine::Nary)
            .unwrap()
            .network;
        let expected = exact.to_dense().unwrap();
        let truncated = exact
            .truncate(
                [0],
                crate::TruncationOptions::default().with_svd_policy(
                    SvdTruncationPolicy::new(0.11)
                        .with_relative()
                        .with_squared_values()
                        .with_discarded_tail_sum(),
                ),
            )
            .unwrap();
        let result = TreeTN::density_matrix_sum(&references, &0, 0.11, None).unwrap();
        assert_eq!(result.network.link_dims(), vec![1; 3]);
        assert_eq!(result.network.link_dims(), truncated.link_dims());
        assert!((dense_error_squared(&result.network, &expected) - 1.0).abs() < 1e-12);
        assert!(dense_error_squared(&result.network, &expected) <= result.discarded_weight + 1e-12);
        assert!(result.discarded_weight <= 3.0 * 0.11 * 10.0 + 1e-12);
    }
}

// A full tensor on 65 binary sites overflows usize; local environments stay tiny.
#[test]
fn density_matrix_sum_long_chain_stays_local() {
    let sites: Vec<_> = (0..65).map(|_| DynIndex::new_dyn(2)).collect();
    let edges: Vec<_> = (0..64).map(|i| (i, i + 1)).collect();
    let terms = [
        ghz(&sites, &edges, [2.0, 0.6]),
        ghz(&sites, &edges, [1.0, 0.4]),
    ];
    let result = TreeTN::density_matrix_sum(&[&terms[0], &terms[1]], &0, 0.11, None).unwrap();
    assert_eq!(result.network.link_dims(), vec![1; 64]);
    assert!(
        (result
            .network
            .evaluate_point(&sites, &[0; 65])
            .unwrap()
            .real()
            - 3.0)
            .abs()
            < 1e-12
    );
    assert!(
        result
            .network
            .evaluate_point(&sites, &[1; 65])
            .unwrap()
            .abs()
            < 1e-12
    );
}
