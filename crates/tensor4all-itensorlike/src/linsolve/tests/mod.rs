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
    use crate::{BiCgStabOptions, CgOptions, LocalSolver};
    use num_complex::Complex64;
    use tensor4all_core::AnyScalar;

    for complex in [false, true] {
        for nsites in [1usize, 2] {
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
            // Column-major Hermitian positive-definite site operator L[out + 2 * in];
            // the MPO is L on every site.
            let local = [z(3.0, 0.0), z(0.25, 0.4), z(0.25, -0.4), z(2.0, 0.0)];
            let apply_dense = |x: &[Complex64]| -> Vec<Complex64> {
                (0..x.len())
                    .map(|out| {
                        (0..x.len())
                            .map(|inp| {
                                (0..nsites)
                                    .map(|k| {
                                        let (o, i) = ((out >> k) & 1, (inp >> k) & 1);
                                        local[o + 2 * i]
                                    })
                                    .product::<Complex64>()
                                    * x[inp]
                            })
                            .sum()
                    })
                    .collect()
            };
            let id = vec![z(1.0, 0.0), z(0.0, 0.0), z(0.0, 0.0), z(1.0, 0.0)];
            let bond = DynIndex::new_dyn(2);
            let solution: Vec<_> = (0..1 << nsites)
                .map(|i| z(i as f64 + 1.0, i as f64 - 0.25))
                .collect();
            // Trains whose dense value at [s0, s1] is values[s0 + 2 * s1].
            let train_tensors = |values: Vec<Complex64>| {
                if nsites == 1 {
                    vec![make(sites.clone(), values)]
                } else {
                    vec![
                        make(vec![sites[0].clone(), bond.clone()], id.clone()),
                        make(vec![bond.clone(), sites[1].clone()], values),
                    ]
                }
            };
            let init = TensorTrain::new(
                train_tensors(solution.clone())
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
                        make(indices, local.to_vec())
                    })
                    .collect(),
            )
            .unwrap();
            let expected = make(sites.clone(), solution.clone());
            for nsite in if nsites == 1 { vec![1] } else { vec![1, 2] } {
                for solver in [
                    LocalSolver::Gmres,
                    LocalSolver::Cg(CgOptions::default()),
                    LocalSolver::BiCgStab(BiCgStabOptions::default()),
                ] {
                    // CG needs a Hermitian positive-definite a0 I + a1 A, so its
                    // coefficients are real and positive.
                    let (a0, a1) = if matches!(&solver, LocalSolver::Cg(_)) {
                        (z(0.25, 0.0), z(1.5, 0.0))
                    } else {
                        (z(0.25, 0.5), z(1.5, -0.25))
                    };
                    let ax = apply_dense(&solution);
                    let rhs_values = solution
                        .iter()
                        .zip(&ax)
                        .map(|(x, ax)| a0 * x + a1 * ax)
                        .collect();
                    let rhs = TensorTrain::new(train_tensors(rhs_values)).unwrap();
                    let scalar = |c: Complex64| {
                        if complex {
                            AnyScalar::from(c)
                        } else {
                            AnyScalar::new_real(c.re)
                        }
                    };
                    let options = LinsolveOptions::new(2)
                        .with_nsite(nsite)
                        .with_local_solver(solver)
                        .with_coefficients(scalar(a0), scalar(a1));
                    let result = linsolve(&operator, &rhs, init.clone(), &options).unwrap();
                    let error = result
                        .to_dense()
                        .unwrap()
                        .sub(&expected)
                        .unwrap()
                        .maxabs()
                        .unwrap();
                    assert!(
                        error < 1e-8,
                        "complex={complex} nsites={nsites} nsite={nsite} {:?}: {error}",
                        options.local_solver()
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

fn identity_problem() -> (TensorTrain, TensorTrain) {
    let site = DynIndex::new_dyn(2);
    let rhs = TensorTrain::new(vec![make_tensor(vec![site.clone()], vec![1.0, 2.0])]).unwrap();
    let operator = TensorTrain::new(vec![make_tensor(
        vec![site.prime(), site],
        vec![1.0, 0.0, 0.0, 1.0],
    )])
    .unwrap();
    (operator, rhs)
}

fn treetn_error_chain_has<E: std::error::Error + 'static>(
    error: &TensorTrainError,
    predicate: impl Fn(&E) -> bool,
) -> bool {
    let TensorTrainError::TreeTn { source, .. } = error else {
        return false;
    };
    source
        .source
        .chain()
        .filter_map(|e| e.downcast_ref::<E>())
        .any(predicate)
}

#[test]
fn local_solver_selection_wrapper_validates_native_options_before_identity_shortcut() {
    use crate::{BiCgStabOptions, CgOptions, LocalSolver};
    use tensor4all_core::krylov::KrylovError;
    let (operator, rhs) = identity_problem();
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
        assert!(
            treetn_error_chain_has::<KrylovError>(&error, |e| matches!(
                e,
                KrylovError::InvalidOptions { .. }
            )),
            "{error:?}"
        );
    }
}

#[test]
fn local_solver_selection_wrapper_reports_invalid_gmres_options_as_treetn_errors() {
    use crate::{BiCgStabOptions, CgOptions, LocalSolver};
    let (operator, rhs) = identity_problem();
    for (options, expected) in [
        (
            LinsolveOptions::new(2).with_gmres_tol(f64::NAN),
            "gmres_tol must be finite and greater than zero",
        ),
        (
            LinsolveOptions::new(2).with_gmres_max_restarts(0),
            "gmres_max_restarts must be greater than zero",
        ),
        (
            LinsolveOptions::new(2).with_gmres_restart_dim(0),
            "gmres_restart_dim must be greater than zero",
        ),
        (
            LinsolveOptions::new(2)
                .with_local_solver(LocalSolver::Cg(CgOptions::default()))
                .with_gmres_tol(1e-12),
            "gmres_tol configures LocalSolver::Gmres only",
        ),
        (
            LinsolveOptions::new(2)
                .with_local_solver(LocalSolver::BiCgStab(BiCgStabOptions::default()))
                .with_gmres_restart_dim(10),
            "gmres_restart_dim configures LocalSolver::Gmres only",
        ),
    ] {
        let error = linsolve(&operator, &rhs, rhs.clone(), &options.with_nsite(1)).unwrap_err();
        let TensorTrainError::TreeTn { source, .. } = &error else {
            panic!("expected TensorTrainError::TreeTn, got {error:?}");
        };
        assert!(
            source
                .source
                .chain()
                .any(|e| e.to_string().contains(expected)),
            "{expected}: {error:?}"
        );
    }
}
