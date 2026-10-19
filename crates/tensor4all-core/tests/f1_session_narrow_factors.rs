//! Session factorization publishes narrow (`f32`/`c32`) factors as compact storage.
//!
//! The session QR and SVD routes read the host payload back from the session and publish the factors
//! through `IdxTensor::from_host_dense`. Compact storage holds `f32`/`Complex32`, so the factors keep
//! their dtype instead of being widened into `f64`/`Complex64`.

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tensor4all_core::{
    factorize_on_session, DynIndex, FactorizeOptions, IdxTensor, TensorContractionLike,
};
use tensor4all_tensorbackend::{
    explicit::{HeldSession, Session},
    CpuExecutionContext,
};

fn session_context() -> CpuExecutionContext {
    let backend = CpuBackend::with_threads(1).expect("1T CPU backend");
    CpuExecutionContext::from_backend(backend)
}

/// Run one session factorization and return `(left, right, rank)`.
fn factorize_in_session(
    tensor: &IdxTensor,
    left: &DynIndex,
    options: &FactorizeOptions,
) -> (IdxTensor, IdxTensor, usize) {
    let context = session_context();
    let backend: CpuBackend = context.with_backend(|backend| backend.clone());
    let mut held = HeldSession::open(&backend).expect("held session");
    let result = held
        .with_session(|view| {
            let mut session = Session::from_backend_session(view.backend());
            factorize_on_session(&mut session, tensor, std::slice::from_ref(left), options)
        })
        .expect("session entry");
    held.close().expect("close held session");
    (result.left, result.right, result.rank)
}

#[test]
fn session_qr_and_svd_publish_narrow_compact_factors() {
    let i = DynIndex::new_dyn(3);
    let j = DynIndex::new_dyn(2);

    let f32_tensor = IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        vec![1.0_f32, 2.0, 3.0, -1.0, 0.5, 4.0],
    )
    .unwrap();
    let c32_tensor = IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        vec![
            Complex32::new(1.0, -1.0),
            Complex32::new(2.0, 0.5),
            Complex32::new(-3.0, 0.0),
            Complex32::new(0.5, 2.0),
            Complex32::new(1.5, -0.5),
            Complex32::new(4.0, 1.0),
        ],
    )
    .unwrap();

    for options in [FactorizeOptions::qr(), FactorizeOptions::svd()] {
        let algorithm = format!("{:?}", options.alg);

        let (left, right, rank) = factorize_in_session(&f32_tensor, &i, &options);
        assert_eq!(rank, 2, "f32 {algorithm}");
        assert!(left.is_f32(), "f32 {algorithm}: left factor dtype");
        assert!(right.is_f32(), "f32 {algorithm}: right factor dtype");
        assert!(
            left.storage().expect("compact f32 factor").is_f32(),
            "f32 {algorithm}: compact payload dtype"
        );
        let reconstructed = left.contract_pair(&right).expect("reconstruction");
        let error = f32_tensor
            .sub(&reconstructed)
            .expect("f32 difference")
            .maxabs()
            .expect("f32 maxabs");
        assert!(error < 1.0e-5, "f32 {algorithm}: reconstruction {error}");

        let (left, right, rank) = factorize_in_session(&c32_tensor, &i, &options);
        assert_eq!(rank, 2, "c32 {algorithm}");
        assert!(left.is_c32(), "c32 {algorithm}: left factor dtype");
        assert!(right.is_c32(), "c32 {algorithm}: right factor dtype");
        assert!(
            left.storage().expect("compact c32 factor").is_c32(),
            "c32 {algorithm}: compact payload dtype"
        );
        let reconstructed = left.contract_pair(&right).expect("reconstruction");
        let error = c32_tensor
            .sub(&reconstructed)
            .expect("c32 difference")
            .maxabs()
            .expect("c32 maxabs");
        assert!(error < 1.0e-5, "c32 {algorithm}: reconstruction {error}");
    }
}

#[test]
fn session_svd_reports_the_singular_values_of_a_narrow_operand() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    // Diagonal matrix [[3, 0], [0, 4]] has singular values [4, 3].
    let tensor =
        IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![3.0_f32, 0.0, 0.0, 4.0]).unwrap();
    let (left, right, rank) = factorize_in_session(&tensor, &i, &FactorizeOptions::svd());
    assert_eq!(rank, 2);
    assert!(left.is_f32());
    assert!(right.is_f32());
    let reconstructed = left.contract_pair(&right).expect("reconstruction");
    let error = tensor
        .sub(&reconstructed)
        .expect("f32 difference")
        .maxabs()
        .expect("f32 maxabs");
    assert!(error < 1.0e-5, "reconstruction {error}");
}

#[test]
fn wide_operands_still_publish_wide_factors() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let tensor =
        IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let (left, right, _) = factorize_in_session(&tensor, &i, &FactorizeOptions::qr());
    assert!(left.is_f64());
    assert!(right.is_f64());
    let reconstructed = left.contract_pair(&right).expect("reconstruction");
    assert!(
        tensor
            .sub(&reconstructed)
            .expect("difference")
            .maxabs()
            .expect("maxabs")
            < 1.0e-12
    );

    let complex = IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(4.0, 0.5),
        ],
    )
    .unwrap();
    let (left, right, _) = factorize_in_session(&complex, &i, &FactorizeOptions::qr());
    assert!(left.is_c64());
    assert!(right.is_c64());
    let reconstructed = left.contract_pair(&right).expect("reconstruction");
    assert!(
        complex
            .sub(&reconstructed)
            .expect("difference")
            .maxabs()
            .expect("maxabs")
            < 1.0e-12
    );
}
