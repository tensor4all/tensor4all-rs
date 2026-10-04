use num_complex::{Complex32, Complex64};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_tensorbackend::{default_cpu_execution_context, ExecutionContext};
use tensor4all_treersi::{hadamard_many, hadamard_many_with_rng_in, TreeRsiOptions, TreeRsiScalar};
use tensor4all_treetn::TreeTN;

type Tree = TreeTN<IdxTensor, usize>;
#[path = "../examples/support/mod.rs"]
mod support;
use support::random_tree;

fn whole_product<T: TreeRsiScalar>(sample: impl Fn(f64, f64) -> T + Copy, tolerance: f64) {
    let topologies = [
        (vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], 6),
        (vec![(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)], 0),
        (vec![(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)], 6),
        (vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)], 5),
    ];
    for (edges, root) in topologies {
        let sites = (0..7).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
        for operands in [1, 2, 3, 4] {
            let inputs = (0..operands)
                .map(|a| random_tree(&sites, &edges, 2, 100 + a, sample))
                .collect::<Vec<_>>();
            let mut expected = vec![T::one(); 128];
            for input in &inputs {
                let dense = input
                    .to_dense()
                    .unwrap()
                    .permute_indices(&sites)
                    .unwrap()
                    .to_vec::<T>()
                    .unwrap();
                for (x, y) in expected.iter_mut().zip(dense) {
                    *x = *x * y;
                }
            }
            for k in [9, 128] {
                let options = TreeRsiOptions {
                    max_bond_dim: Some(16),
                    sketch_dim: Some(k),
                    root: Some(root),
                    rel_tol: T::epsilon() * 10.0,
                    ..Default::default()
                };
                let result = hadamard_many::<T, _>(&inputs, &options).unwrap();
                let oracle = IdxTensor::from_dense(sites.clone(), expected.clone()).unwrap();
                let error = result
                    .tree
                    .to_dense()
                    .unwrap()
                    .sub(&oracle)
                    .unwrap()
                    .maxabs()
                    .unwrap();
                let scale = oracle.maxabs().unwrap();
                assert!(
                    error / scale < tolerance,
                    "operands={operands}, k={k}, root={root}, rel max={}",
                    error / scale
                );
                if k == 9 {
                    assert!(result.diagnostics.edges.iter().any(|e| !e.exact_columns));
                } else {
                    assert!(result.diagnostics.edges.iter().all(|e| e.exact_columns));
                }
            }
        }
    }
}

#[test]
fn whole_products_f64() {
    whole_product(|x, _| x, 1e-10);
}
#[test]
fn whole_products_c64() {
    whole_product(Complex64::new, 1e-10);
}
#[test]
fn whole_products_f32() {
    whole_product(|x, _| x as f32, 2e-4);
}
#[test]
fn whole_products_c32() {
    whole_product(|x, y| Complex32::new(x as f32, y as f32), 2e-4);
}

fn constant_chain(n: usize, physical: bool, gauged: bool) -> (Tree, Vec<DynIndex>) {
    let sites = (0..n).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let bonds = (1..n).map(|_| DynIndex::new_dyn(1)).collect::<Vec<_>>();
    let tensors = (0..n)
        .map(|i| {
            let mut indices = Vec::new();
            if i > 0 {
                indices.push(bonds[i - 1].clone());
            }
            if physical {
                indices.push(sites[i].clone());
            }
            if i + 1 < n {
                indices.push(bonds[i].clone());
            }
            let scale = if gauged {
                2.0_f64.powi(if i % 2 == 0 { 500 } else { -500 })
            } else {
                1.0
            };
            IdxTensor::from_dense(indices, vec![scale; if physical { 2 } else { 1 }]).unwrap()
        })
        .collect();
    (
        TreeTN::from_tensors(tensors, (0..n).collect()).unwrap(),
        sites,
    )
}

#[test]
fn long_rank_one_constant_and_extreme_gauges_remain_one() {
    for gauged in [false, true] {
        let (tree, sites) = constant_chain(2048, true, gauged);
        let options = TreeRsiOptions {
            max_bond_dim: Some(1),
            sketch_dim: Some(6),
            ..Default::default()
        };
        let result = hadamard_many::<f64, _>(&[tree.clone(), tree], &options).unwrap();
        let points = (0..3)
            .flat_map(|p| {
                (0..2048).map(move |i| match p {
                    0 => 0,
                    1 => 1,
                    _ => i % 2,
                })
            })
            .collect::<Vec<_>>();
        let values = result
            .tree
            .evaluate(&sites, ColMajorArrayRef::new(&points, &[2048, 3]).unwrap())
            .unwrap();
        for value in values {
            assert!((value.real() - 1.0).abs() < 1e-12, "{value:?}");
        }
        assert_eq!(result.diagnostics.max_rank(), 1);
    }
}

#[test]
fn long_nonphysical_chain_uses_linear_exact_message_count() {
    let (tree, _) = constant_chain(1024, false, true);
    let options = TreeRsiOptions {
        max_bond_dim: Some(1),
        ..Default::default()
    };
    let result = hadamard_many::<f64, _>(&[tree.clone(), tree], &options).unwrap();
    assert_eq!(
        result.tree.to_dense().unwrap().to_vec::<f64>().unwrap(),
        vec![1.0]
    );
    assert_eq!(result.diagnostics.exact_messages_per_input, 2 * 1023);
    assert_eq!(result.diagnostics.sketch_messages_per_input, 0);
}

#[test]
fn subnormal_f32_product_does_not_overflow_the_scaling_factor() {
    for scale in [1.0_f32, 1e-15, 1e-20] {
        let (s0, s1, b) = (
            DynIndex::new_dyn(2),
            DynIndex::new_dyn(2),
            DynIndex::new_dyn(2),
        );
        let data = vec![scale, 2.0 * scale, 3.0 * scale, 4.0 * scale];
        let tree = TreeTN::from_tensors(
            vec![
                IdxTensor::from_dense(vec![s0.clone(), b.clone()], data.clone()).unwrap(),
                IdxTensor::from_dense(vec![b, s1.clone()], vec![1.0_f32, 0.0, 0.0, 1.0]).unwrap(),
            ],
            vec![0usize, 1],
        )
        .unwrap();
        let out = hadamard_many::<f32, _>(
            &[tree.clone(), tree],
            &TreeRsiOptions {
                max_bond_dim: Some(2),
                rel_tol: 1e-6,
                ..Default::default()
            },
        )
        .unwrap();
        let actual = out
            .tree
            .to_dense()
            .unwrap()
            .permute_indices(&[s0, s1])
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        for (a, x) in actual.into_iter().zip(data) {
            let expected = x * x;
            assert!(
                (a - expected).abs() <= expected.abs() * 2e-6 + 2.0 * f32::from_bits(1),
                "{a} != {expected}"
            );
        }
    }
}

#[test]
fn caller_rng_is_consumed_and_explicit_context_is_preserved() {
    let sites = (0..7).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let edges = (0..6).map(|i| (i, i + 1)).collect::<Vec<_>>();
    let a = random_tree(&sites, &edges, 2, 13, |x, _| x);
    let options = TreeRsiOptions {
        max_bond_dim: Some(4),
        sketch_dim: Some(2),
        ..Default::default()
    };
    let context = ExecutionContext::Cpu(default_cpu_execution_context());
    let mut rng = ChaCha8Rng::seed_from_u64(77);
    let before = rng.clone().random::<u64>();
    let out = hadamard_many_with_rng_in::<f64, _, _>(&[a.clone(), a], &options, &mut rng, &context)
        .unwrap();
    out.tree.validate_context(&context).unwrap();
    assert_ne!(rng.random::<u64>(), before);
    assert!(out.diagnostics.edges.iter().any(|e| !e.exact_columns));
}

#[test]
fn rank_one_star_prefix_products_preserve_full_product_under_rerooting() {
    let sites = (0..9).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let edges = (1..9).map(|v| (0, v)).collect::<Vec<_>>();
    let inputs = [7, 13].map(|seed| random_tree(&sites, &edges, 1, seed, |x, _| x));
    let a = inputs[0]
        .to_dense()
        .unwrap()
        .permute_indices(&sites)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    let b = inputs[1]
        .to_dense()
        .unwrap()
        .permute_indices(&sites)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    let expected =
        IdxTensor::from_dense(sites, a.iter().zip(b).map(|(x, y)| x * y).collect()).unwrap();
    for root in [0, 8] {
        let result = hadamard_many::<f64, _>(
            &inputs,
            &TreeRsiOptions {
                root: Some(root),
                max_bond_dim: Some(1),
                sketch_dim: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            result
                .tree
                .to_dense()
                .unwrap()
                .sub(&expected)
                .unwrap()
                .maxabs()
                .unwrap()
                < 1e-12
        );
    }
}

#[test]
fn inputs_may_have_different_bond_dimensions() {
    let sites = (0..6).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let edges = (1..6).map(|v| (v - 1, v)).collect::<Vec<_>>();
    let inputs = [
        random_tree(&sites, &edges, 2, 11, |x, _| x),
        random_tree(&sites, &edges, 3, 17, |x, _| x),
    ];
    let a = inputs[0]
        .to_dense()
        .unwrap()
        .permute_indices(&sites)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    let b = inputs[1]
        .to_dense()
        .unwrap()
        .permute_indices(&sites)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    let expected =
        IdxTensor::from_dense(sites, a.iter().zip(b).map(|(x, y)| x * y).collect()).unwrap();
    let out = hadamard_many::<f64, _>(
        &inputs,
        &TreeRsiOptions {
            max_bond_dim: Some(6),
            sketch_dim: Some(4),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        out.tree
            .to_dense()
            .unwrap()
            .sub(&expected)
            .unwrap()
            .maxabs()
            .unwrap()
            / expected.maxabs().unwrap()
            < 1e-10
    );
}
