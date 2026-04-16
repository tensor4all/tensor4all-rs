use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_treetn::{square_linsolve, LinsolveOptions, TreeTN};

#[test]
fn square_linsolve_handles_distinct_operator_input_output_indices() -> anyhow::Result<()> {
    let state_site0 = DynIndex::new_dyn(2);
    let state_site1 = DynIndex::new_dyn(2);
    let state_bond = DynIndex::new_dyn(1);

    let operator_site0_input = DynIndex::new_dyn(2);
    let operator_site0_output = DynIndex::new_dyn(2);
    let operator_site1_input = DynIndex::new_dyn(2);
    let operator_site1_output = DynIndex::new_dyn(2);
    let operator_bond = DynIndex::new_dyn(1);

    let operator = TreeTN::<TensorDynLen, usize>::from_tensors(
        vec![
            TensorDynLen::from_dense(
                vec![
                    operator_site0_output.clone(),
                    operator_site0_input.clone(),
                    operator_bond.clone(),
                ],
                vec![1.0_f64, 0.0, 0.0, 1.0],
            )?,
            TensorDynLen::from_dense(
                vec![
                    operator_bond,
                    operator_site1_output.clone(),
                    operator_site1_input.clone(),
                ],
                vec![1.0_f64, 0.0, 0.0, 1.0],
            )?,
        ],
        vec![0, 1],
    )?;

    let rhs = TreeTN::<TensorDynLen, usize>::from_tensors(
        vec![
            TensorDynLen::from_dense(
                vec![state_site0.clone(), state_bond.clone()],
                vec![1.0_f64, 2.0],
            )?,
            TensorDynLen::from_dense(
                vec![state_bond.clone(), state_site1.clone()],
                vec![1.0, -1.0],
            )?,
        ],
        vec![0, 1],
    )?;
    let init = TreeTN::<TensorDynLen, usize>::from_tensors(
        vec![
            TensorDynLen::from_dense(
                vec![state_site0.clone(), state_bond.clone()],
                vec![0.25_f64, 0.75],
            )?,
            TensorDynLen::from_dense(vec![state_bond, state_site1.clone()], vec![0.5, 1.5])?,
        ],
        vec![0, 1],
    )?;

    let result = square_linsolve(
        &operator,
        &rhs,
        init,
        &1usize,
        LinsolveOptions::new(1)
            .with_coefficients(0.0, 1.0)
            .with_krylov_tol(1e-12)
            .with_krylov_maxiter(16)
            .with_krylov_dim(8),
        None,
        None,
    )?;

    let dense = result.solution.to_dense()?;
    let rhs_dense = rhs.to_dense()?;
    let sol = dense.to_vec::<f64>()?;
    let expected = rhs_dense.to_vec::<f64>()?;
    assert_eq!(sol.len(), expected.len());
    for (a, b) in sol.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-10, "mismatch: {a} vs {b}");
    }
    Ok(())
}
