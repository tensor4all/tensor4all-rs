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
fn one_site_als_wrapper_solves_real_and_complex_affine_systems() {
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
            let a0 = z(0.25, 0.5);
            let a1 = z(1.5, -0.25);
            let options = LinsolveOptions::new(2).with_nsite(1).with_coefficients(
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
            assert_eq!(options.nsite(), 1);
            assert_eq!(LinsolveOptions::default().nsite(), 2);
            let result = linsolve(&operator, &rhs, init, &options).unwrap();
            let expected = make(sites, values.iter().map(|v| v / (a0 + a1)).collect());
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
