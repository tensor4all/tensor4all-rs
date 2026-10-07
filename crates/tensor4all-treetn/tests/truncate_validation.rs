//! Regression tests for SVD truncation/contraction option validation before
//! empty-center, single-node, zero-sweep, and method-dispatch shortcuts.
//!
//! These cover the issue #655 contract requirement that invalid thresholds and
//! `max_bond_dim == 0` are rejected on every path, including the paths that
//! previously returned `Ok` without performing any factorization.

use tensor4all_core::{DynIndex, FactorizeError, FactorizeOptions, IdxTensor, SvdTruncationPolicy};
use tensor4all_treetn::{
    contraction::ContractionOptions, factorize_tensor_to_treetn_with, CanonicalForm, TreeTN,
    TreeTNOperationError, TreeTopology, TruncationOptions,
};

fn assert_invalid_factorization_options(error: TreeTNOperationError) {
    assert!(
        matches!(
            error.source.downcast_ref::<FactorizeError>(),
            Some(FactorizeError::InvalidOptions(_))
        ),
        "expected the typed factorization error, got {error:?}"
    );
}

fn one_site(name: usize, dim: usize) -> TreeTN<IdxTensor, usize> {
    let site = DynIndex::new_dyn(dim);
    TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site], vec![1.0_f64; dim]).unwrap()],
        vec![name],
    )
    .unwrap()
}

#[test]
fn truncate_rejects_invalid_options_before_the_empty_center_shortcut() {
    // A NaN/infinite/negative policy or `max_bond_dim == 0` must be rejected
    // even when the empty center would otherwise be a no-op.
    let tree = one_site(0usize, 2);
    for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        let options = TruncationOptions::new().with_svd_policy(SvdTruncationPolicy::new(threshold));
        assert!(
            tree.clone().truncate([], options).is_err(),
            "empty-center truncate must reject threshold {threshold}"
        );
        assert!(
            tree.clone().truncate([0usize], options).is_err(),
            "ordinary single-node truncate must reject threshold {threshold}"
        );
    }

    let zero_cap = TruncationOptions::default().with_max_bond_dim(0);
    assert!(tree.clone().truncate([], zero_cap).is_err());
    assert!(tree.clone().truncate([0usize], zero_cap).is_err());
}

#[test]
fn contract_dispatch_rejects_invalid_options_before_method_shortcuts() {
    // Both Zipup and Naive dispatch must reject the invalid policy before any
    // single-node / dense shortcut runs.
    let left = one_site(0usize, 2);
    let right = one_site(1usize, 3);
    for method in [
        tensor4all_treetn::contraction::ContractionMethod::Zipup,
        tensor4all_treetn::contraction::ContractionMethod::Naive,
    ] {
        for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let options = ContractionOptions::new(method)
                .with_svd_policy(SvdTruncationPolicy::new(threshold));
            assert!(
                tensor4all_treetn::contraction::contract(&left, &right, &0, options).is_err(),
                "contract {method:?} must reject threshold {threshold}"
            );
        }
        let zero_cap = ContractionOptions::new(method).with_max_bond_dim(0);
        assert!(
            tensor4all_treetn::contraction::contract(&left, &right, &0, zero_cap).is_err(),
            "contract {method:?} must reject max_bond_dim == 0"
        );
    }
}

#[test]
fn decomposition_validates_options_before_single_node_and_empty_topology_paths() {
    let site = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_dense(vec![site.clone()], vec![2.0, 3.0]).unwrap();
    let topology = TreeTopology::new([(0, vec![site])].into(), vec![]);
    let empty = TreeTopology::<usize, DynIndex>::new(Default::default(), vec![]);
    let mut invalid = vec![FactorizeOptions::svd().with_max_bond_dim(0)];
    for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        invalid.push(FactorizeOptions::svd().with_svd_policy(SvdTruncationPolicy::new(threshold)));
    }
    invalid.push(FactorizeOptions::svd().with_qr_rtol(1e-8));
    for options in [
        FactorizeOptions::qr(),
        FactorizeOptions::lu(),
        FactorizeOptions::ci(),
    ] {
        invalid.push(options.with_svd_policy(SvdTruncationPolicy::new(1e-8)));
    }
    for options in [FactorizeOptions::lu(), FactorizeOptions::ci()] {
        invalid.push(options.with_qr_rtol(1e-8));
    }
    for options in invalid {
        for topology in [&topology, &empty] {
            assert_invalid_factorization_options(
                factorize_tensor_to_treetn_with(&tensor, topology, options.clone(), &0)
                    .unwrap_err(),
            );
        }
    }

    for options in [
        FactorizeOptions::svd(),
        FactorizeOptions::qr(),
        FactorizeOptions::lu(),
        FactorizeOptions::ci(),
    ] {
        let result =
            factorize_tensor_to_treetn_with(&tensor, &topology, options.with_max_bond_dim(1), &0)
                .unwrap();
        assert_eq!(
            result
                .contract_to_tensor()
                .unwrap()
                .to_vec::<f64>()
                .unwrap(),
            vec![2.0, 3.0]
        );
    }
}

#[test]
fn direct_zipup_validates_options_before_single_node_shortcuts() {
    let shared = DynIndex::new_dyn(2);
    let output = DynIndex::new_dyn(2);
    let scalar_left = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![shared.clone()], vec![1.0, 2.0]).unwrap()],
        vec![0],
    )
    .unwrap();
    let output_left = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![output, shared.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        ],
        vec![0],
    )
    .unwrap();
    let right = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![shared], vec![5.0, 6.0]).unwrap()],
        vec![0],
    )
    .unwrap();

    for (left, expected) in [(&scalar_left, vec![17.0]), (&output_left, vec![23.0, 34.0])] {
        assert_invalid_factorization_options(
            left.contract_zipup(&right, &0, None, Some(0)).unwrap_err(),
        );
        for form in [CanonicalForm::Unitary, CanonicalForm::LU, CanonicalForm::CI] {
            assert_invalid_factorization_options(
                left.contract_zipup_with(&right, &0, form, None, Some(0))
                    .unwrap_err(),
            );
            for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
                assert_invalid_factorization_options(
                    left.contract_zipup_with(
                        &right,
                        &0,
                        form,
                        Some(SvdTruncationPolicy::new(threshold)),
                        None,
                    )
                    .unwrap_err(),
                );
            }
            if form != CanonicalForm::Unitary {
                assert_invalid_factorization_options(
                    left.contract_zipup_with(
                        &right,
                        &0,
                        form,
                        Some(SvdTruncationPolicy::new(1e-8)),
                        None,
                    )
                    .unwrap_err(),
                );
            }
            let result = left
                .contract_zipup_with(&right, &0, form, None, Some(1))
                .unwrap();
            assert_eq!(
                result
                    .contract_to_tensor()
                    .unwrap()
                    .to_vec::<f64>()
                    .unwrap(),
                expected
            );
        }
    }
}

#[test]
fn direct_zipup_rejects_invalid_options_before_topology_errors() {
    let left = one_site(0, 2);
    let incompatible = one_site(1, 2);
    let empty = TreeTN::<IdxTensor, usize>::new();
    for (left, right) in [(&left, &incompatible), (&empty, &empty)] {
        assert_invalid_factorization_options(
            left.contract_zipup(right, &0, None, Some(0)).unwrap_err(),
        );
    }
}
