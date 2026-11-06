//! Every explicit-session entry that reads operand values refuses an AD-tracked operand.
//!
//! The runtime-free session routes read a host payload and publish plain host values, so a tracked
//! operand would enter them as a detached constant and its graph would be dropped silently. Each entry
//! therefore refuses it before reading values: factorize, QR and SVD with their typed unsupported-storage
//! variant, the indexed operations with an `IdxTensorError::Operation` naming the entry.
//!
//! The refusals must not mutate the operand: it stays tracked and its payload stays readable.

use tensor4all_core::qr_with_on_session;
use tensor4all_core::{
    contract_pair_on_session, contract_with_options_on_session, factorize_full_rank_on_session,
    factorize_on_session, index_select_on_session, inner_product_on_session, norm_on_session,
    norm_squared_on_session, permute_indices_on_session, permute_on_session,
    select_indices_on_session, stack_along_new_index_on_session, sum_on_session, Canonical,
    ContractionOptions, DynIndex, FactorizeAlg, FactorizeError, FactorizeOptions, IdxTensor,
    IdxTensorError, PairwiseContractionOptions, QrError, QrOptions,
};
use tensor4all_tensorbackend::explicit::Session;
use tensor4all_tensorbackend::CpuBackend;

fn session_backend() -> CpuBackend {
    CpuBackend::with_threads(1).expect("1T CPU backend")
}

/// A tracked dense operand on `[i = 2, k = 2]`, in a shape every indexed entry accepts.
fn tracked_operand(i: &DynIndex, k: &DynIndex) -> IdxTensor {
    IdxTensor::from_dense(vec![i.clone(), k.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
        .expect("dense operand")
        .enable_grad()
        .expect("tracked operand")
}

/// Assert the typed indexed-operation refusal names `operation` and the tracked representation.
fn assert_indexed_refusal(error: IdxTensorError, operation: &str) {
    match error {
        IdxTensorError::Operation {
            operation: reported,
            source,
        } => {
            assert_eq!(reported, operation);
            assert!(
                source.to_string().contains("tracked"),
                "{operation}: {source}"
            );
        }
        other => panic!("{operation}: expected a typed operation refusal, got {other:?}"),
    }
}

/// Assert factorize's typed unsupported-storage refusal still names the tracked representation.
fn assert_factorize_refusal(result: Result<impl std::fmt::Debug, FactorizeError>) {
    match result.expect_err("a tracked operand must be refused") {
        FactorizeError::UnsupportedStorage(message) => {
            assert!(message.contains("tracked"), "{message}");
        }
        other => panic!("expected an unsupported-storage refusal, got {other:?}"),
    }
}

#[test]
fn every_indexed_session_entry_refuses_a_tracked_operand() {
    let mut backend = session_backend();
    let i = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    // The operand is built before the engine is borrowed: a tracked operand is
    // created outside the session it will be refused by.
    let tracked = tracked_operand(&i, &k);
    let untracked = tracked.detach().expect("untracked sibling");
    // A second tracked operand whose index set differs, for the connected contraction.
    let right = IdxTensor::from_dense(vec![k.clone(), j.clone()], vec![1.0_f64; 4])
        .expect("dense operand")
        .enable_grad()
        .expect("tracked operand");
    let mut session = Session::new(backend.engine_mut());

    {
        assert_indexed_refusal(
            contract_pair_on_session(
                &mut session,
                &tracked,
                &untracked,
                PairwiseContractionOptions::new(),
            )
            .expect_err("contract_pair_on_session"),
            "contract_pair_on_session",
        );
        assert_indexed_refusal(
            contract_with_options_on_session(
                &mut session,
                &[&tracked, &right],
                ContractionOptions::new(),
            )
            .expect_err("contract_with_options_on_session"),
            "contract_with_options_on_session",
        );
        assert_indexed_refusal(
            inner_product_on_session(&mut session, &tracked, &untracked)
                .expect_err("inner_product_on_session"),
            "contract_pair_on_session",
        );
        assert_indexed_refusal(
            sum_on_session(&mut session, &tracked).expect_err("sum_on_session"),
            "sum_on_session",
        );
        assert_indexed_refusal(
            permute_on_session(&mut session, &tracked, &[1, 0]).expect_err("permute_on_session"),
            "permute_on_session",
        );
        // The index-alignment entry delegates its execution to `permute_on_session`, so the refusal
        // carries that route's operation name.
        assert_indexed_refusal(
            permute_indices_on_session(&mut session, &tracked, &[k.clone(), i.clone()])
                .expect_err("permute_indices_on_session"),
            "permute_on_session",
        );
        assert_indexed_refusal(
            norm_squared_on_session(&mut session, &tracked).expect_err("norm_squared_on_session"),
            "norm_squared_on_session",
        );
        assert_indexed_refusal(
            norm_on_session(&mut session, &tracked).expect_err("norm_on_session"),
            "norm_squared_on_session",
        );
        assert_indexed_refusal(
            select_indices_on_session(&mut session, &tracked, std::slice::from_ref(&k), &[1])
                .expect_err("select_indices_on_session"),
            "select_indices_on_session",
        );
        assert_indexed_refusal(
            stack_along_new_index_on_session(&mut session, &[&tracked, &untracked], j.clone(), 0)
                .expect_err("stack_along_new_index_on_session"),
            "stack_along_new_index_on_session",
        );
        assert_indexed_refusal(
            index_select_on_session(&mut session, &tracked, &i, DynIndex::new_dyn(2), &[1, 0])
                .expect_err("index_select_on_session"),
            "index_select_on_session",
        );
    }

    // No refusal detached the operand: both tracked values still hold their graph and payload.
    for operand in [&tracked, &right] {
        assert!(
            operand.tracks_grad(),
            "the refusal must not detach the graph"
        );
        assert!(
            operand.to_vec::<f64>().is_ok(),
            "the payload stays readable"
        );
    }
}

#[test]
fn every_factorization_session_entry_refuses_a_tracked_operand() {
    let mut backend = session_backend();
    let i = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(2);
    let tracked = tracked_operand(&i, &k);
    let untracked = tracked.detach().expect("untracked sibling");
    let left = [i.clone()];
    let mut session = Session::new(backend.engine_mut());

    {
        // The direct QR entry point: the public route a scheduler-owned lane calls.
        match qr_with_on_session::<f64>(&mut session, &tracked, &left, &QrOptions::new())
            .expect_err("qr_with_on_session")
        {
            QrError::UnsupportedStorage(message) => {
                assert!(message.contains("tracked"), "{message}");
            }
            other => panic!("expected a typed unsupported-storage refusal, got {other:?}"),
        }

        for options in [FactorizeOptions::qr(), FactorizeOptions::svd()] {
            assert_factorize_refusal(factorize_on_session(
                &mut session,
                &tracked,
                &left,
                &options,
            ));
        }
        for algorithm in [FactorizeAlg::QR, FactorizeAlg::SVD] {
            assert_factorize_refusal(factorize_full_rank_on_session(
                &mut session,
                &tracked,
                &left,
                algorithm,
                Canonical::Left,
            ));
        }

        // The held session still serves an untracked operand after the refusals.
        let (q, r) = qr_with_on_session::<f64>(&mut session, &untracked, &left, &QrOptions::new())
            .expect("untracked QR");
        assert_eq!(q.dims().last().copied(), r.dims().first().copied());
    }

    assert!(
        tracked.tracks_grad(),
        "the refusal must not detach the graph"
    );
}
