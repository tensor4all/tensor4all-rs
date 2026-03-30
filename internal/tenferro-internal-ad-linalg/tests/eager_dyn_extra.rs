use tenferro_internal_ad_core::{AdTensor, DynAdTensorRef};
use tenferro_internal_frontend_core::ScalarType;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn matrix_f64(values: &[f64], dims: &[usize]) -> AdTensor<f64> {
    AdTensor::new_primal(
        DenseTensor::<f64>::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap(),
    )
}

fn vector_f64(values: &[f64]) -> AdTensor<f64> {
    AdTensor::new_primal(
        DenseTensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap(),
    )
}

#[test]
fn eager_dyn_extra_entrypoints_cover_matrix_factorization_and_solve_paths() {
    let matrix = matrix_f64(&[4.0, 1.0, 1.0, 3.0], &[2, 2]);
    let rhs = vector_f64(&[1.0, 2.0]);

    let lu_factor =
        tenferro_internal_ad_linalg::eager::lu_factor_dyn(DynAdTensorRef::from(&matrix)).unwrap();
    assert_eq!(lu_factor.factors.scalar_type(), ScalarType::F64);
    assert_eq!(lu_factor.pivots.len(), 2);

    let lu_factor_ex = tenferro_internal_ad_linalg::eager::lu_factor_ex_dyn(DynAdTensorRef::from(
        &matrix,
    ))
    .unwrap();
    assert_eq!(lu_factor_ex.factors.scalar_type(), ScalarType::F64);
    assert_eq!(lu_factor_ex.pivots.len(), 2);
    assert_eq!(lu_factor_ex.info.len(), 1);

    let lu_solved = tenferro_internal_ad_linalg::eager::lu_solve_dyn(
        DynAdTensorRef::from(&lu_factor.factors),
        DynAdTensorRef::from(&rhs),
        &lu_factor.pivots,
    )
    .unwrap();
    assert_eq!(lu_solved.scalar_type(), ScalarType::F64);
    assert_eq!(lu_solved.dims(), &[2]);

    let solve_ex = tenferro_internal_ad_linalg::eager::solve_ex_dyn(
        DynAdTensorRef::from(&matrix),
        DynAdTensorRef::from(&rhs),
    )
    .unwrap();
    assert_eq!(solve_ex.solution.scalar_type(), ScalarType::F64);
    assert_eq!(solve_ex.info.len(), 1);

    let inv_ex = tenferro_internal_ad_linalg::eager::inv_ex_dyn(DynAdTensorRef::from(&matrix))
        .unwrap();
    assert_eq!(inv_ex.inverse.scalar_type(), ScalarType::F64);
    assert_eq!(inv_ex.info.len(), 1);

    let cholesky_ex =
        tenferro_internal_ad_linalg::eager::cholesky_ex_dyn(DynAdTensorRef::from(&matrix))
            .unwrap();
    assert_eq!(cholesky_ex.l.scalar_type(), ScalarType::F64);
    assert_eq!(cholesky_ex.info.len(), 1);
}
