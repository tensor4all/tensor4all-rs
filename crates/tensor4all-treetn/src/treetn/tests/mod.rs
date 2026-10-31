use super::*;
use std::sync::Arc;

use crate::options::TruncationOptions;
use num_complex::Complex64;
use tensor4all_core::{
    DynIndex, FactorizeResult, IdxTensor, TensorConstructionLike, TensorContractionLike,
    TensorFactorizationLike, TensorIndex,
};
use tensor4all_tensorbackend::CpuBackend;
use tensor4all_tensorbackend::{CpuExecutionContext, ExecutionContext};

#[test]
fn from_tensors_empty_returns_empty_network() {
    let tn = TreeTN::<IdxTensor, usize>::from_tensors(Vec::new(), Vec::new()).unwrap();

    assert_eq!(tn.node_count(), 0);
    assert_eq!(tn.edge_count(), 0);
    assert!(tn.node_names().is_empty());
}

// ------------------------------------------------------------------------
// Empty-side factorization
// ------------------------------------------------------------------------

fn cpu_context() -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::new(),
    )))
}

/// Build a column-major tensor, complex when `phase` is given and owned by
/// `context` when one is given.
fn sample_tensor(
    indices: Vec<DynIndex>,
    data: &[f64],
    phase: Option<Complex64>,
    context: Option<&ExecutionContext>,
) -> IdxTensor {
    match (phase, context) {
        (None, None) => IdxTensor::from_dense(indices, data.to_vec()),
        (None, Some(context)) => {
            <IdxTensor as TensorConstructionLike>::from_dense_in(context, indices, data.to_vec())
        }
        (Some(phase), None) => IdxTensor::from_dense(
            indices,
            data.iter().map(|value| phase * *value).collect::<Vec<_>>(),
        ),
        (Some(phase), Some(context)) => <IdxTensor as TensorConstructionLike>::from_dense_in(
            context,
            indices,
            data.iter().map(|value| phase * *value).collect::<Vec<_>>(),
        ),
    }
    .unwrap()
}

/// Which side of a two-site split has no indices.
#[derive(Clone, Copy, Debug)]
enum EmptySide {
    Left,
    Right,
    Both,
}

/// A tensor and the left indices whose split leaves `side` empty.
fn empty_side_case(
    side: EmptySide,
    phase: Option<Complex64>,
    context: Option<&ExecutionContext>,
) -> (IdxTensor, Vec<DynIndex>) {
    let (i, j) = (DynIndex::new_dyn(2), DynIndex::new_dyn(3));
    let data = [0.5, -1.0, 2.0, 0.25, 1.5, -0.75];
    match side {
        EmptySide::Left => (sample_tensor(vec![i, j], &data, phase, context), Vec::new()),
        EmptySide::Right => (
            sample_tensor(vec![i.clone(), j.clone()], &data, phase, context),
            vec![j, i],
        ),
        EmptySide::Both => (
            sample_tensor(Vec::new(), &[-2.5], phase, context),
            Vec::new(),
        ),
    }
}

/// Copy a tensor into the default context through its host data, so
/// tensors owned by different execution contexts can be compared.
fn on_host(tensor: &IdxTensor) -> IdxTensor {
    let indices = tensor.external_indices();
    if tensor.is_complex() {
        IdxTensor::from_dense(indices, tensor.to_vec::<Complex64>().unwrap())
    } else {
        IdxTensor::from_dense(indices, tensor.to_vec::<f64>().unwrap())
    }
    .unwrap()
}

fn sorted_indices(mut indices: Vec<DynIndex>) -> Vec<DynIndex> {
    indices.sort_by_key(|index| *index.id());
    indices
}

/// Check the structural and numerical contract of an empty-side split.
fn assert_empty_side_split(
    tensor: &IdxTensor,
    left_indices: &[DynIndex],
    result: &FactorizeResult<IdxTensor>,
    alg: FactorizeAlg,
    canonical: Canonical,
    label: &str,
) {
    let bond = &result.bond_index;
    assert_eq!(bond.dim(), 1, "{label}: bond is not dimension one");
    assert_eq!(result.rank, 1, "{label}: rank");

    let mut expected_left = left_indices.to_vec();
    expected_left.push(bond.clone());
    let mut expected_right: Vec<DynIndex> = tensor
        .external_indices()
        .into_iter()
        .filter(|index| !left_indices.contains(index))
        .collect();
    expected_right.push(bond.clone());
    assert_eq!(
        sorted_indices(result.left.external_indices()),
        sorted_indices(expected_left),
        "{label}: left factor indices"
    );
    assert_eq!(
        sorted_indices(result.right.external_indices()),
        sorted_indices(expected_right),
        "{label}: right factor indices"
    );

    let reconstructed = on_host(&result.left)
        .contract_pair(&on_host(&result.right))
        .unwrap();
    let residual = reconstructed
        .sub(&on_host(tensor))
        .unwrap()
        .maxabs()
        .unwrap();
    let scale = tensor.maxabs().unwrap();
    assert!(
        residual <= 1e-12 * scale,
        "{label}: reconstruction residual {residual}"
    );

    // With a dimension-one bond, a unitary canonical factor has unit
    // norm and an LU/CI canonical factor has a unit pivot.
    let canonical_factor = match canonical {
        Canonical::Left => &result.left,
        Canonical::Right => &result.right,
    };
    let measure = match alg {
        FactorizeAlg::SVD | FactorizeAlg::QR => canonical_factor.norm().unwrap(),
        FactorizeAlg::LU | FactorizeAlg::CI => canonical_factor.maxabs().unwrap(),
    };
    assert!(
        (measure - 1.0).abs() <= 1e-12,
        "{label}: canonical factor is not normalized ({measure})"
    );
}

const EMPTY_SIDES: [EmptySide; 3] = [EmptySide::Left, EmptySide::Right, EmptySide::Both];
const PHASES: [Option<Complex64>; 2] = [None, Some(Complex64::new(0.6, -0.8))];

#[test]
fn empty_side_split_honours_algorithm_and_canonical_direction() {
    let variants = [
        (FactorizeAlg::SVD, Canonical::Left),
        (FactorizeAlg::SVD, Canonical::Right),
        (FactorizeAlg::QR, Canonical::Left),
        (FactorizeAlg::LU, Canonical::Left),
        (FactorizeAlg::LU, Canonical::Right),
        (FactorizeAlg::CI, Canonical::Left),
        (FactorizeAlg::CI, Canonical::Right),
    ];
    for (alg, canonical) in variants {
        for side in EMPTY_SIDES {
            for phase in PHASES {
                let label = format!("{alg:?}/{canonical:?}, empty {side:?}, phase {phase:?}");
                let (tensor, left_indices) = empty_side_case(side, phase, None);
                let result = factorize_allowing_empty_side(&tensor, &left_indices, |t, left| {
                    Ok(t.factorize_full_rank(left, alg, canonical)?)
                })
                .unwrap();
                assert_empty_side_split(&tensor, &left_indices, &result, alg, canonical, &label);
            }
        }
    }
}

#[test]
fn empty_side_split_with_truncating_options_keeps_unit_bond() {
    for canonical in [Canonical::Left, Canonical::Right] {
        let options = FactorizeOptions::svd()
            .with_canonical(canonical)
            .with_max_bond_dim(1);
        for side in EMPTY_SIDES {
            let label = format!("truncating SVD/{canonical:?}, empty {side:?}");
            let (tensor, left_indices) = empty_side_case(side, None, None);
            let result = factorize_allowing_empty_side(&tensor, &left_indices, |t, left| {
                Ok(t.factorize(left, &options)?)
            })
            .unwrap();
            assert_empty_side_split(
                &tensor,
                &left_indices,
                &result,
                FactorizeAlg::SVD,
                canonical,
                &label,
            );
        }
    }
}

#[test]
fn empty_side_split_keeps_factors_in_explicit_context() {
    let context = cpu_context();
    let variants = [
        (FactorizeAlg::SVD, Canonical::Left),
        (FactorizeAlg::SVD, Canonical::Right),
        (FactorizeAlg::QR, Canonical::Left),
    ];
    for (alg, canonical) in variants {
        for side in EMPTY_SIDES {
            for phase in PHASES {
                let label =
                    format!("in-context {alg:?}/{canonical:?}, empty {side:?}, phase {phase:?}");
                let (tensor, left_indices) = empty_side_case(side, phase, Some(&context));
                let result = factorize_allowing_empty_side(&tensor, &left_indices, |t, left| {
                    Ok(t.factorize_full_rank_in(left, alg, canonical, &context)?)
                })
                .unwrap_or_else(|e| panic!("{label}: {e:#}"));
                result.left.validate_context(&context).unwrap();
                result.right.validate_context(&context).unwrap();
                assert_empty_side_split(&tensor, &left_indices, &result, alg, canonical, &label);
            }
        }
    }
}

#[test]
fn non_degenerate_split_is_delegated_unchanged() {
    let (i, j) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let tensor = sample_tensor(vec![i.clone(), j], &[1.0, 0.0, 0.0, 2.0], None, None);
    let result = factorize_allowing_empty_side(&tensor, std::slice::from_ref(&i), |t, left| {
        Ok(t.factorize_full_rank(left, FactorizeAlg::SVD, Canonical::Left)?)
    })
    .unwrap();
    assert_eq!(result.bond_index.dim(), 2);
    assert_eq!(result.singular_values, Some(vec![2.0, 1.0]));
    let reconstructed = result.left.contract_pair(&result.right).unwrap();
    assert!(reconstructed.sub(&tensor).unwrap().maxabs().unwrap() < 1e-12);
}

/// A left index that the tensor does not carry is rejected before
/// `factorize` runs, for both a non-degenerate and an empty right side.
/// The primed copy differs from a tensor index only by prime level.
#[test]
fn split_rejects_left_index_outside_tensor() {
    let (i, j) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let foreign = DynIndex::new_dyn(2);
    let primed_i = i.prime();
    let tensor = sample_tensor(
        vec![i.clone(), j.clone()],
        &[1.0, 0.0, 0.0, 2.0],
        None,
        None,
    );
    for left_indices in [
        vec![foreign.clone()],
        vec![i.clone(), foreign],
        vec![i, j, primed_i],
    ] {
        let error = factorize_allowing_empty_side(&tensor, &left_indices, |_, _| {
            panic!("factorize must not run for an invalid left index")
        })
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("is not an index of the tensor to factorize"),
            "{error:#}"
        );
    }
}

// ------------------------------------------------------------------------
// Site-free nodes in sweeps
// ------------------------------------------------------------------------

/// `[0] -- bond(3) -- [1]`: node 0 is a site-free leaf, node 1 carries a
/// dimension-two site. Both tensors are multiplied by `phase`, so the
/// represented vector is `phase^2 * [14, 32]`.
fn site_free_leaf_tree(
    phase: Option<Complex64>,
    context: Option<&ExecutionContext>,
) -> TreeTN<IdxTensor, usize> {
    let bond = DynIndex::new_dyn(3);
    let site = DynIndex::new_dyn(2);
    let leaf = sample_tensor(vec![bond.clone()], &[1.0, 2.0, 3.0], phase, context);
    let parent = sample_tensor(
        vec![bond, site],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        phase,
        context,
    );
    TreeTN::from_tensors(vec![leaf, parent], vec![0, 1]).unwrap()
}

/// Check the leaf edge has dimension one and the values are unchanged.
fn assert_site_free_leaf_values(tree: &TreeTN<IdxTensor, usize>, phase: Option<Complex64>) {
    let edge = tree.edge_between(&0, &1).unwrap();
    assert_eq!(tree.bond_index(edge).unwrap().dim(), 1);
    let actual = on_host(&tree.to_dense().unwrap());
    let phase_squared = phase.map(|phase| phase * phase);
    let expected = sample_tensor(
        actual.external_indices(),
        &[14.0, 32.0],
        phase_squared,
        None,
    );
    let residual = actual.sub(&expected).unwrap().maxabs().unwrap();
    assert!(residual < 1e-12, "site-free leaf residual {residual}");
}

/// Check every tensor of `tree` belongs to `context`.
fn assert_in_context(tree: &TreeTN<IdxTensor, usize>, context: &ExecutionContext) {
    for node in tree.node_names() {
        let index = tree.node_index(&node).unwrap();
        tree.tensor(index)
            .unwrap()
            .validate_context(context)
            .unwrap();
    }
}

#[test]
fn sweep_absorbs_site_free_leaf_with_and_without_context() {
    let context = cpu_context();
    for phase in PHASES {
        for scoped in [true, false] {
            let label = format!("phase {phase:?}, scoped {scoped}");
            let mut tree = site_free_leaf_tree(phase, scoped.then_some(&context));
            let (src, dst) = (tree.node_index(&0).unwrap(), tree.node_index(&1).unwrap());
            if scoped {
                tree.sweep_edge_full_rank_in(
                    src,
                    dst,
                    FactorizeAlg::QR,
                    Canonical::Left,
                    "test",
                    &context,
                )
                .unwrap_or_else(|e| panic!("{label}: {e:#}"));
                assert_in_context(&tree, &context);
            } else {
                tree.sweep_edge_full_rank(src, dst, FactorizeAlg::QR, Canonical::Left, "test")
                    .unwrap_or_else(|e| panic!("{label}: {e:#}"));
            }
            assert_site_free_leaf_values(&tree, phase);
            // The swept leaf keeps a unit-modulus scalar.
            let leaf = tree.tensor(src).unwrap();
            assert!((leaf.norm().unwrap() - 1.0).abs() < 1e-12, "{label}");
        }
    }
}

#[test]
fn scoped_truncation_keeps_site_free_leaf_network_in_context() {
    let context = cpu_context();
    for phase in PHASES {
        for center in [0, 1] {
            let mut tree = site_free_leaf_tree(phase, Some(&context));
            tree.truncate_impl_in([center], None, Some(4), "test", &context)
                .unwrap_or_else(|e| panic!("phase {phase:?}, center {center}: {e:#}"));
            tree.verify_internal_consistency().unwrap();
            assert_in_context(&tree, &context);
            assert_site_free_leaf_values(&tree, phase);
        }
    }
}

#[test]
fn swap_on_edge_between_two_site_free_nodes_keeps_scalar() {
    // Both sides of the split are empty: the merged tensor is a scalar.
    let bond = DynIndex::new_dyn(3);
    let mut tree = TreeTN::<IdxTensor, usize>::from_tensors(
        vec![
            sample_tensor(vec![bond.clone()], &[1.0, 2.0, 3.0], None, None),
            sample_tensor(vec![bond], &[4.0, 5.0, 6.0], None, None),
        ],
        vec![0, 1],
    )
    .unwrap();
    let (a, b) = (tree.node_index(&0).unwrap(), tree.node_index(&1).unwrap());
    let no_sites = HashSet::new();
    tree.swap_on_edge(a, b, &no_sites, &no_sites, &FactorizeOptions::svd())
        .unwrap();

    tree.verify_internal_consistency().unwrap();
    assert_eq!(tree.link_dims(), vec![1]);
    assert!((tree.tensor(a).unwrap().norm().unwrap() - 1.0).abs() < 1e-12);
    assert_eq!(
        tree.to_dense().unwrap().to_vec::<f64>().unwrap(),
        vec![32.0]
    );
}

// ------------------------------------------------------------------------
// Truncation on a caller-owned context
// ------------------------------------------------------------------------

/// A 4-site chain (site dimension 2, bonds of dimension 4) with fixed
/// pseudo-random entries; the cuts after the first and before the last site
/// have rank at most 2, so a bond cap of 3 changes the outer bonds.
fn random_chain(context: Option<&ExecutionContext>) -> TreeTN<IdxTensor, usize> {
    let mut state = 12_345u64;
    let mut next = move |len: usize| -> Vec<f64> {
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    };
    let sites: Vec<DynIndex> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(4)).collect();
    let tensors = vec![
        sample_tensor(
            vec![sites[0].clone(), bonds[0].clone()],
            &next(8),
            None,
            context,
        ),
        sample_tensor(
            vec![bonds[0].clone(), sites[1].clone(), bonds[1].clone()],
            &next(32),
            None,
            context,
        ),
        sample_tensor(
            vec![bonds[1].clone(), sites[2].clone(), bonds[2].clone()],
            &next(32),
            None,
            context,
        ),
        sample_tensor(
            vec![bonds[2].clone(), sites[3].clone()],
            &next(8),
            None,
            context,
        ),
    ];
    TreeTN::from_tensors(tensors, vec![0, 1, 2, 3]).unwrap()
}

fn dense_values(tree: &TreeTN<IdxTensor, usize>) -> Vec<f64> {
    tree.to_dense().unwrap().to_vec::<f64>().unwrap()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn truncate_in_matches_truncate() {
    let context = cpu_context();
    let options = || TruncationOptions::default().with_max_bond_dim(3);
    let plain = random_chain(None).truncate([1], options()).unwrap();
    let scoped = random_chain(Some(&context))
        .truncate_in([1], options(), &context)
        .unwrap();
    scoped.verify_internal_consistency().unwrap();
    assert_in_context(&scoped, &context);
    assert_eq!(scoped.link_dims(), plain.link_dims());
    assert!(
        scoped.link_dims().iter().all(|&d| d <= 3),
        "{:?}",
        scoped.link_dims()
    );
    assert_eq!(
        scoped
            .canonical_region()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![1]
    );
    let (a, b) = (dense_values(&plain), dense_values(&scoped));
    let scale = a.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(max_abs_diff(&a, &b) <= 1e-12 * scale);
}

#[test]
fn truncate_mut_in_is_truncate_in_bitwise() {
    let context = cpu_context();
    let options = || TruncationOptions::default().with_max_bond_dim(3);
    let by_value = random_chain(Some(&context))
        .truncate_in([2], options(), &context)
        .unwrap();
    let mut in_place = random_chain(Some(&context));
    in_place.truncate_mut_in([2], options(), &context).unwrap();
    assert_in_context(&in_place, &context);
    assert_eq!(in_place.link_dims(), by_value.link_dims());
    assert_eq!(dense_values(&in_place), dense_values(&by_value));
}

/// The numerics run on the context's own two workers, not the process default.
#[test]
fn truncate_in_runs_on_the_context_it_is_given() {
    let cpu = Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(2).unwrap(),
    ));
    assert_eq!(cpu.with_backend(|backend| backend.num_threads()), 2);
    let context = ExecutionContext::Cpu(Arc::clone(&cpu));
    let plain = random_chain(None)
        .truncate([0], TruncationOptions::default().with_max_bond_dim(3))
        .unwrap();
    let scoped = random_chain(Some(&context))
        .truncate_in(
            [0],
            TruncationOptions::default().with_max_bond_dim(3),
            &context,
        )
        .unwrap();
    assert_in_context(&scoped, &context);
    assert_eq!(scoped.link_dims(), plain.link_dims());
    let (a, b) = (dense_values(&plain), dense_values(&scoped));
    let scale = a.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(max_abs_diff(&a, &b) <= 1e-12 * scale);
}

#[test]
fn truncate_in_rejects_invalid_options_and_multi_node_centers() {
    let context = cpu_context();
    let zero = TruncationOptions::default().with_max_bond_dim(0);
    assert!(random_chain(Some(&context))
        .truncate_in([1], zero, &context)
        .is_err());
    assert!(random_chain(Some(&context))
        .truncate_in(
            [1, 2],
            TruncationOptions::default().with_max_bond_dim(3),
            &context
        )
        .is_err());
}
