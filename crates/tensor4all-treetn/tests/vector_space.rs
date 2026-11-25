//! TreeTN vectors use exact algebra, preserve index identity, and remain scalable.

use std::cell::Cell;

use num_complex::{Complex32, Complex64};
use tensor4all_core::krylov::{
    gmres_with_truncation, restart_gmres_with_truncation, GmresOptions, RestartGmresOptions,
};
use tensor4all_core::{
    AnyScalar, DynIndex, IdxTensor, SvdTruncationPolicy, TensorContractionLike, TensorElement,
    TensorIndex, TensorVectorSpace,
};
use tensor4all_treetn::{
    CanonicalForm, CanonicalizationOptions, TreeTN, TreeTNOperationError, TruncationOptions,
};

type Tree = TreeTN<IdxTensor, usize>;

fn sites() -> Vec<DynIndex> {
    (0..4).map(|_| DynIndex::new_dyn(2)).collect()
}

/// A degree-three center representing |0000> + 2 |1111>.
fn star<S: TensorElement>(sites: &[DynIndex], scalar: impl Fn(f64) -> S) -> Tree {
    let bonds = (0..3).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let mut center_data = vec![scalar(0.0); 16];
    center_data[0] = scalar(1.0);
    center_data[15] = scalar(2.0);
    let mut center_indices = vec![sites[0].clone()];
    center_indices.extend(bonds.iter().cloned());
    let mut tensors = vec![IdxTensor::from_dense(center_indices, center_data).unwrap()];
    for (bond, site) in bonds.into_iter().zip(&sites[1..]) {
        tensors.push(
            IdxTensor::from_dense(
                vec![bond, site.clone()],
                vec![scalar(1.0), scalar(0.0), scalar(0.0), scalar(1.0)],
            )
            .unwrap(),
        );
    }
    Tree::from_tensors(tensors, vec![0, 1, 2, 3]).unwrap()
}

fn assert_dense_equal(actual: &Tree, expected: &Tree, tolerance: f64) {
    let actual = actual.to_dense().unwrap();
    let expected = expected.to_dense().unwrap();
    let error = actual.sub(&expected).unwrap().maxabs().unwrap();
    assert!(error < tolerance, "max absolute error: {error}");
}

fn check_scalar_kind<S: TensorElement>(scalar: impl Fn(f64) -> S, single: bool, complex: bool) {
    let tree = star(&sites(), scalar);
    assert_eq!(tree.site_index_network().neighbors(&0).count(), 3);
    assert_eq!(TensorVectorSpace::is_single_precision(&tree), single);
    assert_eq!(TensorVectorSpace::is_complex(&tree), complex);
    TensorVectorSpace::validate(&tree).unwrap();
    assert!(TensorVectorSpace::is_all_finite(&tree).unwrap());
    let tolerance = if single { 1e-5 } else { 1e-12 };
    let inner = TensorVectorSpace::inner_product(&tree, &tree).unwrap();
    assert!((inner.real() - 5.0).abs() < tolerance);
    assert!(inner.imag().abs() < tolerance);
    if single {
        // The owning IdxTensor factorization API currently supports
        // single-precision CI, but not the unitary QR needed by TreeTN norms.
        assert!(matches!(
            TensorVectorSpace::norm(&tree),
            Err(TreeTNOperationError::Operation { .. })
        ));
        assert!(matches!(
            TensorVectorSpace::norm_squared(&tree),
            Err(TreeTNOperationError::Operation { .. })
        ));
        return;
    }
    assert!((TensorVectorSpace::norm(&tree).unwrap() - 5.0_f64.sqrt()).abs() < tolerance);
    assert!((TensorVectorSpace::norm_squared(&tree).unwrap() - 5.0).abs() < tolerance);
}

#[test]
fn scalar_kinds_and_known_branched_norms() {
    check_scalar_kind(|x| x, false, false);
    check_scalar_kind(|x| x as f32, true, false);
    check_scalar_kind(|x| Complex64::new(x, 0.0), false, true);
    check_scalar_kind(|x| Complex32::new(x as f32, 0.0), true, true);
}

#[test]
fn mixed_core_storage_reports_any_single_or_complex_component() {
    let mut tree = star(&sites(), |x| x);
    let node = tree.node_index(&0).unwrap();
    let indices = tree.tensor(node).unwrap().external_indices();
    let replacement = IdxTensor::from_dense(indices, vec![Complex32::new(1.0, 0.0); 16]).unwrap();
    tree.replace_tensor(node, replacement).unwrap();
    assert!(TensorVectorSpace::is_single_precision(&tree));
    assert!(TensorVectorSpace::is_complex(&tree));
}

#[test]
fn axpby_is_exact_without_implicit_compression() {
    let tree = star(&sites(), |x| x);
    let sum = TensorVectorSpace::axpby(
        &tree,
        AnyScalar::new_real(2.0),
        &tree,
        AnyScalar::new_real(-0.5),
    )
    .unwrap();
    let expected = tree.scale(AnyScalar::new_real(1.5)).unwrap();
    assert_dense_equal(&sum, &expected, 1e-12);
    // Identical summands could be compressed back to rank two, but exact
    // axpby leaves that decision to the solver's explicit truncation callback.
    assert_eq!(sum.link_dims(), vec![4, 4, 4]);
}

#[test]
fn scaling_preserves_canonical_gauge_and_complex_inner_product() {
    let tree = star(&sites(), |x| x)
        .canonicalize([0], CanonicalizationOptions::default())
        .unwrap();
    let scaled = TensorVectorSpace::scale(&tree, AnyScalar::new_complex(0.0, 2.0)).unwrap();
    assert_eq!(scaled.canonical_region(), tree.canonical_region());
    assert_eq!(scaled.canonical_form(), tree.canonical_form());
    scaled.validate_ortho_consistency().unwrap();
    let inner = TensorVectorSpace::inner_product(&scaled, &tree).unwrap();
    assert!(inner.real().abs() < 1e-12);
    assert!((inner.imag() + 10.0).abs() < 1e-12);
    assert!((TensorVectorSpace::norm_squared(&scaled).unwrap() - 20.0).abs() < 1e-11);
    let zero = TensorVectorSpace::scale(&tree, AnyScalar::new_real(0.0)).unwrap();
    assert_eq!(TensorVectorSpace::norm(&zero).unwrap(), 0.0);
}

#[test]
fn trait_norms_do_not_mutate_the_input_or_square_before_reducing() {
    let mut tree = star(&sites(), |x| x);
    assert!(tree.canonical_region().is_empty());
    let original_bonds = tree.link_dims();
    assert!((TensorVectorSpace::norm(&tree).unwrap() - 5.0_f64.sqrt()).abs() < 1e-12);
    assert!(tree.canonical_region().is_empty());
    assert_eq!(tree.link_dims(), original_bonds);
    // The inherent mutating API remains available explicitly.
    assert!((Tree::norm(&mut tree).unwrap() - 5.0_f64.sqrt()).abs() < 1e-12);
    assert_eq!(
        tree.canonical_region(),
        &std::collections::HashSet::from([0])
    );
    for exponent in [-200, 200] {
        let scale = 10.0_f64.powi(exponent);
        let tensor =
            IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![3.0 * scale, 4.0 * scale])
                .unwrap();
        let tree = Tree::from_tensors(vec![tensor], vec![0]).unwrap();
        let relative_error = (TensorVectorSpace::norm(&tree).unwrap() / (5.0 * scale) - 1.0).abs();
        assert!(
            relative_error < 1e-12,
            "relative norm error: {relative_error}"
        );
    }
}

#[test]
fn full_site_identity_is_required_even_for_zero_coefficients() {
    let site_indices = sites();
    let tree = star(&site_indices, |x| x);
    let mut primed_sites = site_indices.clone();
    primed_sites[0] = primed_sites[0].prime();
    let primed = star(&primed_sites, |x| x);
    assert!(TensorVectorSpace::inner_product(&tree, &primed).is_err());
    assert!(TensorVectorSpace::axpby(
        &tree,
        AnyScalar::new_real(1.0),
        &primed,
        AnyScalar::new_real(0.0),
    )
    .is_err());
    let aligned = primed.reindex_site_space_like(&tree).unwrap();
    assert!(
        (TensorVectorSpace::inner_product(&tree, &aligned)
            .unwrap()
            .real()
            - 5.0)
            .abs()
            < 1e-12
    );
}

#[test]
fn unsupported_elementwise_metrics_and_invalid_vectors_return_errors() {
    let tree = star(&sites(), |x| x);
    assert!(matches!(
        TensorVectorSpace::maxabs(&tree),
        Err(TreeTNOperationError::InvalidConfiguration { .. })
    ));
    assert!(matches!(
        TensorVectorSpace::min_nonzero_abs(&tree),
        Err(TreeTNOperationError::InvalidConfiguration { .. })
    ));
    let empty = Tree::new();
    assert!(matches!(
        TensorVectorSpace::validate(&empty),
        Err(TreeTNOperationError::InvalidGraph { .. })
    ));
    let apply_called = Cell::new(false);
    let rejected = gmres_with_truncation(
        |tree: &Tree| {
            apply_called.set(true);
            Ok(tree.clone())
        },
        &empty,
        &empty,
        &GmresOptions::default(),
        |_tree| Ok(()),
    );
    assert!(rejected.is_err());
    assert!(!apply_called.get());
    assert!(TensorVectorSpace::norm(&empty).is_err());
    assert!(TensorVectorSpace::scale(&empty, AnyScalar::new_real(0.0)).is_err());
    let mut disconnected = tree.clone();
    disconnected
        .add_tensor(
            4,
            IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![1.0_f64, 0.0]).unwrap(),
        )
        .unwrap();
    assert!(matches!(
        TensorVectorSpace::validate(&disconnected),
        Err(TreeTNOperationError::InvalidGraph { .. })
    ));
    let mut bad_gauge = tree;
    bad_gauge.set_canonical_region([0]).unwrap();
    assert!(TensorVectorSpace::validate(&bad_gauge).is_err());
}

#[test]
fn finiteness_checks_stored_components_without_dense_contraction() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut tree = star(&sites(), |x| x);
        let node = tree.node_index(&0).unwrap();
        let indices = tree.tensor(node).unwrap().external_indices();
        tree.replace_tensor(
            node,
            IdxTensor::from_dense(indices, vec![value; 16]).unwrap(),
        )
        .unwrap();
        assert!(!TensorVectorSpace::is_all_finite(&tree).unwrap());
    }
    let big = 0.75 * f64::MAX;
    let tensor =
        IdxTensor::from_dense(vec![DynIndex::new_dyn(1)], vec![Complex64::new(big, big)]).unwrap();
    let tree = Tree::from_tensors(vec![tensor], vec![0]).unwrap();
    assert!(TensorVectorSpace::is_all_finite(&tree).unwrap());
}

fn diagonal_operator<S: TensorElement>(site: &DynIndex, diagonal: [S; 2], zero: S) -> IdxTensor {
    IdxTensor::from_dense(
        vec![site.prime(), site.clone()],
        vec![diagonal[0], zero, zero, diagonal[1]],
    )
    .unwrap()
}

fn apply_diagonal(tree: &Tree, operator: &IdxTensor) -> anyhow::Result<Tree> {
    let node = tree.node_index(&0).unwrap();
    let transformed = IdxTensor::contract(&[tree.tensor(node).unwrap(), operator])?;
    let mut result = tree.clone();
    result.replace_tensor(node, transformed)?;
    // Applying an operator produces fresh physical legs. Match them explicitly
    // before returning a vector to the same Krylov space.
    Ok(result.reindex_site_space_like(tree)?)
}

fn truncate_at_center(tree: &mut Tree, calls: &Cell<usize>) -> anyhow::Result<()> {
    calls.set(calls.get() + 1);
    tree.truncate_mut(
        [0],
        TruncationOptions::default()
            .with_max_bond_dim(2)
            .with_svd_policy(SvdTruncationPolicy::new(1e-12)),
    )?;
    assert_eq!(
        tree.canonical_region(),
        &std::collections::HashSet::from([0])
    );
    assert_eq!(tree.canonical_form(), Some(CanonicalForm::Unitary));
    tree.validate_ortho_consistency()?;
    Ok(())
}

fn check_gmres<S: TensorElement>(scalar: impl Fn(f64) -> S, diagonal: [S; 2], zero: S) {
    let sites = sites();
    let expected = star(&sites, scalar);
    let operator = diagonal_operator(&sites[0], diagonal, zero);
    let rhs = apply_diagonal(&expected, &operator).unwrap();
    let initial = expected.scale(AnyScalar::new_real(0.0)).unwrap();
    let calls = Cell::new(0);
    let options = GmresOptions {
        max_iter: 4,
        max_restarts: 2,
        check_true_residual: true,
        ..Default::default()
    };
    let result = gmres_with_truncation(
        |tree| apply_diagonal(tree, &operator),
        &rhs,
        &initial,
        &options,
        |tree| truncate_at_center(tree, &calls),
    )
    .unwrap();
    assert!(result.converged, "residual: {}", result.residual_norm);
    assert!(calls.get() > 0);
    assert_dense_equal(&result.solution, &expected, 1e-9);
    let residual =
        TensorVectorSpace::sub(&apply_diagonal(&result.solution, &operator).unwrap(), &rhs)
            .unwrap();
    let relative_residual =
        TensorVectorSpace::norm(&residual).unwrap() / TensorVectorSpace::norm(&rhs).unwrap();
    assert!(
        relative_residual < 1e-10,
        "relative residual: {relative_residual}"
    );
}

#[test]
fn gmres_with_truncation_solves_a_non_chain_real_tree() {
    check_gmres(|x| x, [2.0, 3.0], 0.0);
}

#[test]
fn gmres_with_truncation_solves_a_non_chain_complex_tree() {
    check_gmres(
        |x| Complex64::new(x, 0.0),
        [Complex64::new(2.0, 1.0), Complex64::new(3.0, -1.0)],
        Complex64::new(0.0, 0.0),
    );
}

#[test]
fn restarted_gmres_with_truncation_solves_a_non_chain_tree() {
    let sites = sites();
    let expected = star(&sites, |x| x);
    let operator = diagonal_operator(&sites[0], [2.0, 3.0], 0.0);
    let rhs = apply_diagonal(&expected, &operator).unwrap();
    let calls = Cell::new(0);
    let options = RestartGmresOptions {
        max_outer_iters: 32,
        inner_max_iter: 1,
        ..Default::default()
    };
    let result = restart_gmres_with_truncation(
        |tree| apply_diagonal(tree, &operator),
        &rhs,
        None,
        &options,
        |tree| truncate_at_center(tree, &calls),
    )
    .unwrap();
    assert!(result.converged, "residual: {}", result.residual_norm);
    assert!(result.outer_iterations > 1);
    assert!(calls.get() > 0);
    assert_dense_equal(&result.solution, &expected, 1e-9);
}

#[test]
fn long_vectors_do_not_require_dense_materialization() {
    let length = 64;
    let bonds = (0..length - 1)
        .map(|_| DynIndex::new_dyn(1))
        .collect::<Vec<_>>();
    let tensors = (0..length)
        .map(|node| {
            let mut indices = Vec::new();
            if node > 0 {
                indices.push(bonds[node - 1].clone());
            }
            indices.push(DynIndex::new_dyn(2));
            if node + 1 < length {
                indices.push(bonds[node].clone());
            }
            IdxTensor::from_dense(indices, vec![0.6_f64, 0.8]).unwrap()
        })
        .collect();
    let tree = Tree::from_tensors(tensors, (0..length).collect()).unwrap();
    let sum = TensorVectorSpace::axpby(
        &tree,
        AnyScalar::new_real(0.25),
        &tree,
        AnyScalar::new_real(0.75),
    )
    .unwrap();
    let residual = TensorVectorSpace::sub(&sum, &tree).unwrap();
    assert!(
        (TensorVectorSpace::inner_product(&tree, &tree)
            .unwrap()
            .real()
            - 1.0)
            .abs()
            < 1e-12
    );
    assert!(TensorVectorSpace::is_all_finite(&sum).unwrap());
    assert!(TensorVectorSpace::norm(&residual).unwrap() < 1e-12);
    assert!(TensorVectorSpace::maxabs(&tree).is_err());
}
