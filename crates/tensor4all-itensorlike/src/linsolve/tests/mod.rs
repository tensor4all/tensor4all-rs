use super::*;
use tensor4all_core::{DynIndex, IdxTensor};

fn make_tensor(indices: Vec<DynIndex>, data: Vec<f64>) -> IdxTensor {
    IdxTensor::from_dense(indices, data).unwrap()
}

#[test]
fn infer_index_mappings_returns_typed_error() {
    let op_in = DynIndex::new_dyn(2);
    let op_out = DynIndex::new_dyn(2);
    let init_site = DynIndex::new_dyn(2);

    let operator = TensorTrain::new(vec![make_tensor(
        vec![op_in, op_out],
        vec![1.0, 0.0, 0.0, 1.0],
    )])
    .unwrap();
    let init = TensorTrain::new(vec![make_tensor(vec![init_site], vec![1.0, 2.0])]).unwrap();

    let result: crate::error::Result<_> = infer_index_mappings(&operator, &init);
    let err = result.unwrap_err();

    assert!(matches!(err, TensorTrainError::OperationError { .. }));
    assert!(err.to_string().contains("Cannot auto-infer index mappings"));
}

#[test]
fn infer_index_mappings_handles_same_id_primed_output() {
    let init_site = DynIndex::new_dyn(2);
    let output_site = init_site.prime();

    let operator = TensorTrain::new(vec![make_tensor(
        vec![output_site.clone(), init_site.clone()],
        vec![1.0, 0.0, 0.0, 1.0],
    )])
    .unwrap();
    let init =
        TensorTrain::new(vec![make_tensor(vec![init_site.clone()], vec![1.0, 2.0])]).unwrap();

    let (input_mapping, output_mapping) = infer_index_mappings(&operator, &init).unwrap();
    let input_mapping = input_mapping.unwrap();
    let output_mapping = output_mapping.unwrap();

    assert_eq!(input_mapping[&0].true_index, init_site);
    assert_eq!(input_mapping[&0].internal_index, init_site);
    assert_eq!(output_mapping[&0].true_index, init_site);
    assert_eq!(output_mapping[&0].internal_index, output_site);
}

#[test]
fn local_solver_selection_wrapper_solves_real_and_complex_affine_systems() {
    use num_complex::Complex64;
    use tensor4all_core::AnyScalar;

    for complex in [false, true] {
        for nsites in [1, 2] {
            let z = |re, im| Complex64::new(re, if complex { im } else { 0.0 });
            let make = |indices, values: Vec<Complex64>| {
                if complex {
                    IdxTensor::from_dense(indices, values).unwrap()
                } else {
                    IdxTensor::from_dense(indices, values.iter().map(|x| x.re).collect::<Vec<_>>())
                        .unwrap()
                }
            };
            let sites: Vec<_> = (0..nsites).map(|_| DynIndex::new_dyn(2)).collect();
            let id = vec![z(1.0, 0.0), z(0.0, 0.0), z(0.0, 0.0), z(1.0, 0.0)];
            let bond = DynIndex::new_dyn(2);
            let values: Vec<_> = (0..1 << nsites)
                .map(|i| z(i as f64 + 1.0, i as f64 - 0.25))
                .collect();
            let rhs_tensors = if nsites == 1 {
                vec![make(sites.clone(), values.clone())]
            } else {
                vec![
                    make(vec![sites[0].clone(), bond.clone()], id.clone()),
                    make(vec![bond, sites[1].clone()], values.clone()),
                ]
            };
            let rhs = TensorTrain::new(rhs_tensors.clone()).unwrap();
            let init = TensorTrain::new(
                rhs_tensors
                    .iter()
                    .map(|t| t.scale(AnyScalar::new_real(0.0)).unwrap())
                    .collect(),
            )
            .unwrap();
            let op_bond = DynIndex::new_dyn(1);
            let operator = TensorTrain::new(
                sites
                    .iter()
                    .map(|site| {
                        let mut indices = vec![site.prime(), site.clone()];
                        if nsites == 2 {
                            indices.push(op_bond.clone());
                        }
                        make(indices, id.clone())
                    })
                    .collect(),
            )
            .unwrap();
            use crate::LocalSolver;
            use tensor4all_core::krylov::{BiCgStabOptions, CgOptions};
            for nsite in if nsites == 1 { vec![1] } else { vec![1, 2] } {
                for solver in [
                    LocalSolver::Gmres,
                    LocalSolver::Cg(CgOptions::default()),
                    LocalSolver::BiCgStab(BiCgStabOptions::default()),
                ] {
                    let a0 = z(0.25, 0.5);
                    let a1 = if matches!(&solver, LocalSolver::Cg(_)) {
                        z(1.5, -0.5)
                    } else {
                        z(1.5, -0.25)
                    };
                    let options = LinsolveOptions::new(2)
                        .with_nsite(nsite)
                        .with_local_solver(solver)
                        .with_coefficients(
                            if complex {
                                AnyScalar::from(a0)
                            } else {
                                AnyScalar::new_real(a0.re)
                            },
                            if complex {
                                AnyScalar::from(a1)
                            } else {
                                AnyScalar::new_real(a1.re)
                            },
                        );
                    let options = if matches!(options.local_solver(), LocalSolver::Gmres) {
                        options
                    } else {
                        options
                            .with_gmres_tol(f64::NAN)
                            .with_gmres_restart_dim(0)
                            .with_gmres_max_restarts(0)
                    };
                    assert_eq!(options.nsite(), nsite);
                    assert!(matches!(
                        LinsolveOptions::default().local_solver(),
                        LocalSolver::Gmres
                    ));
                    let result = linsolve(&operator, &rhs, init.clone(), &options).unwrap();
                    let expected = make(
                        sites.clone(),
                        values.iter().map(|v| v / (a0 + a1)).collect(),
                    );
                    assert!(
                        result
                            .to_dense()
                            .unwrap()
                            .sub(&expected)
                            .unwrap()
                            .maxabs()
                            .unwrap()
                            < 1e-8
                    );
                }
            }
        }
    }
}

#[test]
fn one_site_als_wrapper_rejects_invalid_update_modes_and_truncation() {
    use tensor4all_core::SvdTruncationPolicy;
    let site = DynIndex::new_dyn(2);
    let rhs = TensorTrain::new(vec![make_tensor(vec![site.clone()], vec![1.0, 2.0])]).unwrap();
    let operator = TensorTrain::new(vec![make_tensor(
        vec![site.prime(), site],
        vec![1.0, 0.0, 0.0, 1.0],
    )])
    .unwrap();
    for options in [
        LinsolveOptions::new(1).with_nsite(0),
        LinsolveOptions::new(1).with_nsite(3),
        LinsolveOptions::new(1).with_nsite(1).with_max_bond_dim(2),
        LinsolveOptions::new(1)
            .with_nsite(1)
            .with_svd_policy(SvdTruncationPolicy::new(0.0)),
    ] {
        let message = linsolve(&operator, &rhs, rhs.clone(), &options)
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("nsite") || message.contains("one-site"),
            "{message}"
        );
    }
}

#[test]
fn local_solver_selection_wrapper_validates_native_options_before_identity_shortcut() {
    use crate::LocalSolver;
    use tensor4all_core::krylov::{BiCgStabOptions, CgOptions};
    let site = DynIndex::new_dyn(2);
    let rhs = TensorTrain::new(vec![make_tensor(vec![site.clone()], vec![1.0, 2.0])]).unwrap();
    let operator = TensorTrain::new(vec![make_tensor(
        vec![site.prime(), site],
        vec![1.0, 0.0, 0.0, 1.0],
    )])
    .unwrap();
    for solver in [
        LocalSolver::Cg(CgOptions {
            hermitian_tol: f64::NAN,
            ..Default::default()
        }),
        LocalSolver::BiCgStab(BiCgStabOptions {
            atol: f64::INFINITY,
            ..Default::default()
        }),
    ] {
        let options = LinsolveOptions::new(0)
            .with_nsite(1)
            .with_coefficients(1.0, 0.0)
            .with_local_solver(solver);
        let error = linsolve(&operator, &rhs, rhs.clone(), &options).unwrap_err();
        let TensorTrainError::OperationErrorSource { source, .. } = &error else {
            panic!("expected OperationErrorSource, got {error:?}");
        };
        assert!(
            matches!(
                source.downcast_ref::<tensor4all_core::krylov::KrylovError>(),
                Some(tensor4all_core::krylov::KrylovError::InvalidOptions { .. })
            ),
            "{error:?}"
        );
    }
}

#[test]
fn local_solver_selection_wrapper_reports_invalid_gmres_options_as_operation_errors() {
    let site = DynIndex::new_dyn(2);
    let rhs = TensorTrain::new(vec![make_tensor(vec![site.clone()], vec![1.0, 2.0])]).unwrap();
    let operator = TensorTrain::new(vec![make_tensor(
        vec![site.prime(), site],
        vec![1.0, 0.0, 0.0, 1.0],
    )])
    .unwrap();
    for (options, expected) in [
        (
            LinsolveOptions::new(2).with_gmres_tol(f64::NAN),
            "gmres_tol must be finite and > 0",
        ),
        (
            LinsolveOptions::new(2).with_gmres_max_restarts(0),
            "gmres_max_restarts must be >= 1",
        ),
        (
            LinsolveOptions::new(2).with_gmres_restart_dim(0),
            "gmres_restart_dim must be >= 1",
        ),
    ] {
        let error = linsolve(&operator, &rhs, rhs.clone(), &options.with_nsite(1)).unwrap_err();
        assert!(
            matches!(&error, TensorTrainError::OperationError { .. }),
            "{error:?}"
        );
        assert!(error.to_string().contains(expected), "{error}");
    }
}
