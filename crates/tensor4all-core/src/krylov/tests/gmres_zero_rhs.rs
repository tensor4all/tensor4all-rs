use super::*;

#[test]
fn gmres_variants_solve_homogeneous_systems_from_nonzero_guesses() {
    for complex in [false, true] {
        let idx = DynIndex::new_dyn(2);
        let (b, x0) = if complex {
            (
                IdxTensor::from_dense(vec![idx.clone()], vec![Complex64::new(0.0, 0.0); 2])
                    .unwrap(),
                IdxTensor::from_dense(
                    vec![idx],
                    vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
                )
                .unwrap(),
            )
        } else {
            (
                make_vector_with_index(vec![0.0, 0.0], &idx),
                make_vector_with_index(vec![1.0, -3.0], &idx),
            )
        };
        let identity = |x: &IdxTensor| Ok(x.clone());
        let options = GmresOptions::default();
        let plain = gmres(identity, &b, &x0, &options).unwrap();
        let affine = gmres_affine(identity, &b, &x0, 1.0.into(), 1.0.into(), &options).unwrap();
        let truncated = gmres_with_truncation(identity, &b, &x0, &options, |_| Ok(())).unwrap();
        for result in [plain, affine, truncated] {
            assert!(result.converged);
            assert_eq!(result.iterations, 0);
            assert_eq!(result.residual_norm, 0.0);
            assert_eq!(result.solution.maxabs().unwrap(), 0.0);
        }
        let restarted = restart_gmres_with_truncation(
            identity,
            &b,
            Some(&x0),
            &RestartGmresOptions::default(),
            |_| Ok(()),
        )
        .unwrap();
        assert!(restarted.converged);
        assert_eq!(restarted.solution.maxabs().unwrap(), 0.0);
    }
}

#[test]
fn gmres_affine_zero_rhs_still_rejects_zero_coefficients() {
    let idx = DynIndex::new_dyn(1);
    let b = make_vector_with_index(vec![0.0], &idx);
    let err = gmres_affine(
        |x: &IdxTensor| Ok(x.clone()),
        &b,
        &b,
        0.0.into(),
        0.0.into(),
        &GmresOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, KrylovError::NoAffineCoefficient));
}

#[test]
fn gmres_variants_do_not_discard_tiny_nonzero_rhs() {
    let idx = DynIndex::new_dyn(2);
    let b = make_vector_with_index(vec![1e-20, -2e-20], &idx);
    let x0 = make_vector_with_index(vec![0.0, 0.0], &idx);
    let identity = |x: &IdxTensor| Ok(x.clone());
    let options = GmresOptions::default();
    let plain = gmres(identity, &b, &x0, &options).unwrap();
    let affine = gmres_affine(identity, &b, &x0, 0.0.into(), 1.0.into(), &options).unwrap();
    let truncated = gmres_with_truncation(identity, &b, &x0, &options, |_| Ok(())).unwrap();
    for result in [plain, affine, truncated] {
        assert!(result.converged);
        assert!(result.solution.sub(&b).unwrap().maxabs().unwrap() < 1e-30);
    }
    let restarted = restart_gmres_with_truncation(
        identity,
        &b,
        Some(&x0),
        &RestartGmresOptions::default(),
        |_| Ok(()),
    )
    .unwrap();
    assert!(restarted.converged);
    assert!(restarted.solution.sub(&b).unwrap().maxabs().unwrap() < 1e-30);
}
