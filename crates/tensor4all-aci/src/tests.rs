use crate::validation::{validate_inputs, validate_options};
use crate::{
    initial_guess,
    random_tt::{
        initial_guess_core_entry_count, initial_guess_existing_entry_count,
        initial_guess_total_entry_count, MAX_INITIAL_GUESS_CORE_ENTRIES,
    },
    AciError, AciOptions, ElementwiseBatch, ElementwiseProblem, LocalBlockEvaluator,
};
use num_complex::Complex64;
use tensor4all_simplett::{
    tensor3_from_data, tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain,
};
use tensor4all_tensorbackend::Matrix;

fn tensor_train_with_link_dims(site_dims: &[usize], link_dims: &[usize]) -> TensorTrain<f64> {
    assert_eq!(link_dims.len(), site_dims.len().saturating_sub(1));
    let cores = site_dims
        .iter()
        .enumerate()
        .map(|(site, &site_dim)| {
            let left_dim = if site == 0 { 1 } else { link_dims[site - 1] };
            let right_dim = link_dims.get(site).copied().unwrap_or(1);
            tensor3_zeros(left_dim, site_dim, right_dim)
        })
        .collect();
    TensorTrain::new(cores).unwrap()
}

fn local_test_problem() -> ElementwiseProblem<f64> {
    let input_a = TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0, 10.0, 20.0], 1, 2, 2).unwrap(),
        tensor3_from_data(
            vec![
                1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
            ],
            2,
            2,
            3,
        )
        .unwrap(),
        tensor3_from_data(
            vec![
                5.0, 6.0, 7.0, 50.0, 60.0, 70.0, 500.0, 600.0, 700.0, 8.0, 9.0, 10.0,
            ],
            3,
            2,
            2,
        )
        .unwrap(),
        tensor3_from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2, 1).unwrap(),
    ])
    .unwrap();
    let input_b = TensorTrain::new(vec![
        tensor3_from_data(vec![2.0, 3.0, 20.0, 30.0], 1, 2, 2).unwrap(),
        tensor3_from_data(
            vec![
                2.0, 3.0, 4.0, 5.0, 20.0, 30.0, 40.0, 50.0, 200.0, 300.0, 400.0, 500.0,
            ],
            2,
            2,
            3,
        )
        .unwrap(),
        tensor3_from_data(
            vec![
                6.0, 7.0, 8.0, 60.0, 70.0, 80.0, 600.0, 700.0, 800.0, 9.0, 10.0, 11.0,
            ],
            3,
            2,
            2,
        )
        .unwrap(),
        tensor3_from_data(vec![2.0, 3.0, 4.0, 5.0], 2, 2, 1).unwrap(),
    ])
    .unwrap();
    let mut problem =
        ElementwiseProblem::new(vec![input_a, input_b], AciOptions::default()).unwrap();
    for input in 0..problem.n_inputs() {
        problem.left_frames[input][1] =
            Some(Matrix::from_col_major_vec(2, 2, vec![1.0, 2.0, 10.0, 20.0]));
        problem.right_frames[input][3] =
            Some(Matrix::from_col_major_vec(2, 2, vec![3.0, 4.0, 30.0, 40.0]));
    }
    problem
}

fn explicit_local_value(
    problem: &ElementwiseProblem<f64>,
    input: usize,
    bond: usize,
    row: usize,
    col: usize,
) -> f64 {
    let left_frame = problem.left_frames[input][bond].as_ref().unwrap();
    let right_frame = problem.right_frames[input][bond + 2].as_ref().unwrap();
    let left_core = problem.inputs[input].site_tensor(bond);
    let right_core = problem.inputs[input].site_tensor(bond + 1);
    let left_pivot = row % left_frame.nrows();
    let site_left = row / left_frame.nrows();
    let site_right = col % right_core.site_dim();
    let right_pivot = col / right_core.site_dim();
    let mut sum = 0.0;
    for a in 0..left_core.left_dim() {
        for m in 0..left_core.right_dim() {
            for b in 0..right_core.right_dim() {
                sum += left_frame[[left_pivot, a]]
                    * *left_core.get3(a, site_left, m)
                    * *right_core.get3(m, site_right, b)
                    * right_frame[[b, right_pivot]];
            }
        }
    }
    sum
}

fn multiply_batch(batch: ElementwiseBatch<'_, f64>, output: &mut [f64]) -> crate::Result<()> {
    assert_eq!(output.len(), batch.n_points());
    for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
        *value = batch.get(0, point).unwrap() * batch.get(1, point).unwrap();
    }
    Ok(())
}

fn zero_batch(batch: ElementwiseBatch<'_, f64>, output: &mut [f64]) -> crate::Result<()> {
    assert_eq!(output.len(), batch.n_points());
    output.fill(0.0);
    Ok(())
}

fn assert_solution_is_zero_on_binary_three_site_grid(problem: &ElementwiseProblem<f64>) {
    for indices in [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ] {
        assert_eq!(problem.solution.evaluate(&indices).unwrap(), 0.0);
    }
}

#[test]
fn default_options_are_conservative() {
    let options = AciOptions::<f64>::default();
    assert_eq!(options.max_iters, 20);
    assert_eq!(options.min_iters, 2);
    assert_eq!(options.max_bond_dim, usize::MAX);
    assert!((options.tolerance - 1e-12).abs() < 1e-15);
    assert!(!options.scale_tolerance);
    assert!(options.initial_guess.is_none());
}

#[test]
fn elementwise_batch_uses_column_major_input_point_layout() {
    let values = vec![10.0, 20.0, 11.0, 21.0, 12.0, 22.0];
    let batch = ElementwiseBatch::new(&values, 2, 3).unwrap();
    assert_eq!(batch.n_inputs(), 2);
    assert_eq!(batch.n_points(), 3);
    assert_eq!(batch.get(0, 0).unwrap(), 10.0);
    assert_eq!(batch.get(1, 0).unwrap(), 20.0);
    assert_eq!(batch.get(0, 2).unwrap(), 12.0);
    assert_eq!(batch.get(1, 2).unwrap(), 22.0);
    assert_eq!(batch.as_col_major_slice(), values.as_slice());
}

#[test]
fn elementwise_batch_rejects_bad_length() {
    let err = ElementwiseBatch::<f64>::new(&[1.0, 2.0, 3.0], 2, 2).unwrap_err();
    assert!(err.to_string().contains("length"));
}

#[test]
fn elementwise_batch_rejects_zero_inputs() {
    let err = ElementwiseBatch::<f64>::new(&[], 0, 1).unwrap_err();
    assert!(matches!(err, AciError::EmptyInputs));
    assert!(err.to_string().contains("at least one input"));
}

#[test]
fn elementwise_batch_rejects_zero_points() {
    let err = ElementwiseBatch::<f64>::new(&[], 1, 0).unwrap_err();
    assert!(matches!(err, AciError::EmptyInputs));
    assert!(err.to_string().contains("interpolation point"));
}

#[test]
fn elementwise_batch_rejects_shape_overflow() {
    let err = ElementwiseBatch::<f64>::new(&[], usize::MAX, 2).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("overflows usize"));
}

#[test]
fn elementwise_batch_rejects_out_of_range_input_index() {
    let values = [10.0, 20.0, 11.0, 21.0];
    let batch = ElementwiseBatch::new(&values, 2, 2).unwrap();

    let err = batch.get(2, 0).unwrap_err();
    assert!(matches!(
        err,
        AciError::BatchIndexOutOfBounds {
            axis: "input",
            index: 2,
            len: 2,
        }
    ));
    let message = err.to_string();
    assert!(message.contains("input index"));
    assert!(message.contains("out of bounds"));
    assert!(message.contains("len 2"));
}

#[test]
fn elementwise_batch_rejects_out_of_range_point_index() {
    let values = [10.0, 20.0, 11.0, 21.0];
    let batch = ElementwiseBatch::new(&values, 2, 2).unwrap();

    let err = batch.get(0, 2).unwrap_err();
    assert!(matches!(
        err,
        AciError::BatchIndexOutOfBounds {
            axis: "point",
            index: 2,
            len: 2,
        }
    ));
    let message = err.to_string();
    assert!(message.contains("point index"));
    assert!(message.contains("out of bounds"));
    assert!(message.contains("len 2"));
}

#[test]
fn elementwise_problem_initializes_boundary_frames() {
    let a = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 2, 2], 2.0);
    let problem = ElementwiseProblem::new(vec![a, b], AciOptions::default()).unwrap();
    assert_eq!(problem.len(), 3);
    assert_eq!(problem.n_inputs(), 2);
    assert_eq!(problem.left_frame_shape(0, 0), Some((1, 1)));
    assert_eq!(problem.right_frame_shape(0, 3), Some((1, 1)));
    assert_eq!(problem.pivot_errors, vec![0.0, 0.0]);
}

#[test]
fn elementwise_problem_initializes_all_rank_one_right_frames() {
    let a = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 2, 2], 2.0);
    let problem = ElementwiseProblem::new(vec![a, b], AciOptions::default()).unwrap();

    for input in 0..problem.n_inputs() {
        assert_eq!(
            problem.right_frame_shape(input, problem.len()),
            Some((1, 1))
        );
        for site in 1..problem.len() {
            assert_eq!(problem.right_frame_shape(input, site), Some((1, 1)));
        }
        assert_eq!(problem.right_frame_shape(input, 0), None);
    }
}

#[test]
fn elementwise_problem_handles_one_site_input() {
    let input = TensorTrain::<f64>::constant(&[2], 3.0);
    let problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();
    assert_eq!(problem.len(), 1);
    assert_eq!(problem.left_frame_shape(0, 0), Some((1, 1)));
    assert_eq!(problem.right_frame_shape(0, 1), Some((1, 1)));
    assert_eq!(problem.right_frame_shape(0, 0), None);
}

#[test]
fn elementwise_problem_preserves_initial_guess_values() {
    let input = TensorTrain::<f64>::constant(&[2, 2], 1.0);
    let tt = TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0, 3.0, 4.0], 1, 2, 2).unwrap(),
        tensor3_from_data(vec![5.0, 6.0, 7.0, 8.0], 2, 2, 1).unwrap(),
    ])
    .unwrap();
    let options = AciOptions {
        initial_guess: Some(tt.clone()),
        ..AciOptions::default()
    };

    let problem = ElementwiseProblem::new(vec![input], options).unwrap();

    for indices in [[0, 0], [0, 1], [1, 0], [1, 1]] {
        assert!(
            (problem.solution.evaluate(&indices).unwrap() - tt.evaluate(&indices).unwrap()).abs()
                < 1e-12
        );
    }
}

#[test]
fn elementwise_problem_updates_left_frame_for_selected_rows() {
    let input = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let mut problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();

    problem.update_left_frame(0, 0, &[0, 1]).unwrap();

    assert_eq!(problem.left_frame_shape(0, 1), Some((2, 1)));
}

#[test]
fn elementwise_problem_updates_left_frame_values() {
    let input = TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0, 10.0, 20.0], 1, 2, 2).unwrap(),
        tensor3_from_data(vec![3.0, 4.0, 5.0, 6.0], 2, 2, 1).unwrap(),
    ])
    .unwrap();
    let mut problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();

    problem.update_left_frame(0, 0, &[0, 1]).unwrap();

    assert_eq!(problem.left_frame_shape(0, 1), Some((2, 2)));
    assert_eq!(problem.left_frame_value(0, 1, 0, 0), Some(1.0));
    assert_eq!(problem.left_frame_value(0, 1, 1, 0), Some(2.0));
    assert_eq!(problem.left_frame_value(0, 1, 0, 1), Some(10.0));
    assert_eq!(problem.left_frame_value(0, 1, 1, 1), Some(20.0));
    assert_eq!(problem.left_frame_value(1, 1, 0, 0), None);
    assert_eq!(problem.left_frame_value(0, 2, 0, 0), None);
    assert_eq!(problem.left_frame_value(0, 1, 2, 0), None);
    assert_eq!(problem.left_frame_value(0, 1, 0, 2), None);
}

#[test]
fn elementwise_problem_updates_all_left_frames_for_selected_rows() {
    let a = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 2, 2], 2.0);
    let mut problem = ElementwiseProblem::new(vec![a, b], AciOptions::default()).unwrap();

    problem.update_left_frames(0, &[0, 1]).unwrap();

    assert_eq!(problem.left_frame_shape(0, 1), Some((2, 1)));
    assert_eq!(problem.left_frame_shape(1, 1), Some((2, 1)));
}

#[test]
fn elementwise_problem_updates_right_frame_for_selected_columns() {
    let input = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let mut problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();

    problem.update_right_frame(0, 2, &[0, 1]).unwrap();

    assert_eq!(problem.right_frame_shape(0, 2), Some((1, 2)));
}

#[test]
fn elementwise_problem_updates_right_frame_values() {
    let input = TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0, 10.0, 20.0], 1, 2, 2).unwrap(),
        tensor3_from_data(vec![3.0, 30.0, 4.0, 40.0], 2, 2, 1).unwrap(),
    ])
    .unwrap();
    let mut problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();

    problem.update_right_frame(0, 1, &[0, 1]).unwrap();

    assert_eq!(problem.right_frame_shape(0, 1), Some((2, 2)));
    assert_eq!(problem.right_frame_value(0, 1, 0, 0), Some(3.0));
    assert_eq!(problem.right_frame_value(0, 1, 1, 0), Some(30.0));
    assert_eq!(problem.right_frame_value(0, 1, 0, 1), Some(4.0));
    assert_eq!(problem.right_frame_value(0, 1, 1, 1), Some(40.0));
    assert_eq!(problem.right_frame_value(1, 1, 0, 0), None);
    assert_eq!(problem.right_frame_value(0, 3, 0, 0), None);
    assert_eq!(problem.right_frame_value(0, 1, 2, 0), None);
    assert_eq!(problem.right_frame_value(0, 1, 0, 2), None);
}

#[test]
fn elementwise_problem_rejects_invalid_frame_selection_indices() {
    let input = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let mut problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();

    let left_err = problem.update_left_frame(0, 0, &[2]).unwrap_err();
    assert!(matches!(left_err, AciError::InvalidInitialGuess { .. }));
    assert!(left_err.to_string().contains("row index"));

    let right_err = problem.update_right_frame(0, 2, &[2]).unwrap_err();
    assert!(matches!(right_err, AciError::InvalidInitialGuess { .. }));
    assert!(right_err.to_string().contains("column index"));
}

#[test]
fn elementwise_problem_one_bond_local_update_matches_dense_product_left_orthogonal() {
    let input_a = TensorTrain::<f64>::constant(&[2, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 2], 3.0);
    let options = AciOptions::<f64>::default();
    let mut problem = ElementwiseProblem::new(vec![input_a, input_b], options.clone()).unwrap();
    let mut operator = multiply_batch;

    problem
        .local_update(0, true, &options, &mut operator)
        .unwrap();

    for indices in [[0, 0], [1, 0], [0, 1], [1, 1]] {
        assert!((problem.solution.evaluate(&indices).unwrap() - 6.0).abs() < 1e-12);
    }
    assert_eq!(problem.left_frame_shape(0, 1), Some((1, 1)));
    assert_eq!(problem.left_frame_shape(1, 1), Some((1, 1)));
    assert_eq!(problem.pivot_errors.len(), 1);
    assert!(problem.pivot_errors[0] <= 1e-12);
}

#[test]
fn elementwise_problem_one_bond_local_update_matches_dense_product_right_orthogonal() {
    let input_a = TensorTrain::<f64>::constant(&[2, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 2], 3.0);
    let options = AciOptions::<f64>::default();
    let mut problem = ElementwiseProblem::new(vec![input_a, input_b], options.clone()).unwrap();
    for input in 0..problem.n_inputs() {
        problem.right_frames[input][1] = None;
    }
    let mut operator = multiply_batch;

    problem
        .local_update(0, false, &options, &mut operator)
        .unwrap();

    for indices in [[0, 0], [1, 0], [0, 1], [1, 1]] {
        assert!((problem.solution.evaluate(&indices).unwrap() - 6.0).abs() < 1e-12);
    }
    assert_eq!(problem.right_frame_shape(0, 1), Some((1, 1)));
    assert_eq!(problem.right_frame_shape(1, 1), Some((1, 1)));
    assert_eq!(problem.pivot_errors.len(), 1);
    assert!(problem.pivot_errors[0] <= 1e-12);
}

#[test]
fn elementwise_problem_one_bond_local_update_returns_operator_error_side_channel() {
    let input_a = TensorTrain::<f64>::constant(&[2, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 2], 3.0);
    let options = AciOptions::<f64>::default();
    let mut problem = ElementwiseProblem::new(vec![input_a, input_b], options.clone()).unwrap();
    let mut operator = |_batch: ElementwiseBatch<'_, f64>, _output: &mut [f64]| {
        Err(AciError::Operator {
            message: "side-channel operator failure".to_string(),
        })
    };

    let err = problem
        .local_update(0, true, &options, &mut operator)
        .unwrap_err();

    assert!(matches!(err, AciError::Operator { .. }));
    assert!(err.to_string().contains("side-channel operator failure"));
}

#[test]
fn elementwise_problem_one_bond_zero_update_keeps_left_frames_nonzero_dimensional() {
    let input_a = TensorTrain::<f64>::constant(&[2, 2, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 2, 2], 3.0);
    let options = AciOptions::<f64>::default();
    let mut problem = ElementwiseProblem::new(vec![input_a, input_b], options.clone()).unwrap();
    let mut operator = zero_batch;

    problem
        .local_update(0, true, &options, &mut operator)
        .unwrap();

    assert_solution_is_zero_on_binary_three_site_grid(&problem);
    assert_eq!(problem.left_frame_shape(0, 1), Some((1, 1)));
    assert_eq!(problem.left_frame_shape(1, 1), Some((1, 1)));
    assert!(problem.local_input_value(0, 1, 0, 0).is_ok());

    problem
        .local_update(1, true, &options, &mut operator)
        .unwrap();
    assert_solution_is_zero_on_binary_three_site_grid(&problem);
}

#[test]
fn elementwise_problem_one_bond_zero_update_keeps_right_frames_nonzero_dimensional() {
    let input_a = TensorTrain::<f64>::constant(&[2, 2, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 2, 2], 3.0);
    let options = AciOptions::<f64>::default();
    let mut problem = ElementwiseProblem::new(vec![input_a, input_b], options.clone()).unwrap();
    problem.update_left_frames(0, &[0]).unwrap();
    for input in 0..problem.n_inputs() {
        problem.right_frames[input][2] = None;
    }
    let mut operator = zero_batch;

    problem
        .local_update(1, false, &options, &mut operator)
        .unwrap();

    assert_solution_is_zero_on_binary_three_site_grid(&problem);
    assert_eq!(problem.right_frame_shape(0, 2), Some((1, 1)));
    assert_eq!(problem.right_frame_shape(1, 2), Some((1, 1)));
    assert!(problem.local_input_value(0, 0, 0, 0).is_ok());

    problem
        .local_update(0, false, &options, &mut operator)
        .unwrap();
    assert_solution_is_zero_on_binary_three_site_grid(&problem);
}

#[test]
fn local_input_value_matches_explicit_two_site_contraction() {
    let problem = local_test_problem();

    let value = problem.local_input_value(0, 1, 3, 2).unwrap();
    let expected = explicit_local_value(&problem, 0, 1, 3, 2);

    assert_eq!(value, expected);
    assert_eq!(value, 265_133_700.0);
}

#[test]
fn local_block_evaluator_uses_matrix_luci_point_order_and_batch_layout() {
    let problem = local_test_problem();
    let rows = [0, 3];
    let cols = [1, 2];
    let mut out = vec![0.0; rows.len() * cols.len()];
    let observed_batches = std::cell::RefCell::new(Vec::new());

    let operator = |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        assert_eq!(batch.n_inputs(), problem.n_inputs());
        assert_eq!(batch.n_points(), rows.len() * cols.len());
        observed_batches
            .borrow_mut()
            .push(batch.as_col_major_slice().to_vec());
        for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
            *value = batch.get(0, point).unwrap() + 10.0 * batch.get(1, point).unwrap();
        }
        Ok(())
    };
    let evaluator = LocalBlockEvaluator::new(&problem, 1, &operator).unwrap();

    evaluator.fill_local_block(&rows, &cols, &mut out).unwrap();

    let mut expected_values = Vec::new();
    for &col in &cols {
        for &row in &rows {
            for input in 0..problem.n_inputs() {
                expected_values.push(problem.local_input_value(input, 1, row, col).unwrap());
            }
        }
    }
    assert_eq!(observed_batches.into_inner(), vec![expected_values]);

    let mut expected_out = Vec::new();
    for &col in &cols {
        for &row in &rows {
            expected_out.push(
                problem.local_input_value(0, 1, row, col).unwrap()
                    + 10.0 * problem.local_input_value(1, 1, row, col).unwrap(),
            );
        }
    }
    assert_eq!(out, expected_out);
}

#[test]
fn local_block_evaluator_serves_duplicate_entries_from_cache() {
    let problem = local_test_problem();
    let rows = [0, 0, 1];
    let cols = [0];
    let mut first = vec![0.0; rows.len() * cols.len()];
    let mut second = vec![0.0; rows.len() * cols.len()];
    let observed_points = std::cell::RefCell::new(Vec::new());

    let operator = |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        observed_points.borrow_mut().push(batch.n_points());
        for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
            *value = batch.get(0, point).unwrap() + batch.get(1, point).unwrap();
        }
        Ok(())
    };
    let evaluator = LocalBlockEvaluator::new(&problem, 1, &operator).unwrap();

    evaluator
        .fill_local_block(&rows, &cols, &mut first)
        .unwrap();
    evaluator
        .fill_local_block(&rows, &cols, &mut second)
        .unwrap();

    assert_eq!(observed_points.into_inner(), vec![2]);
    assert_eq!(first[0], first[1]);
    assert_eq!(first, second);
}

#[test]
fn local_block_evaluator_cache_can_be_cleared() {
    let problem = local_test_problem();
    let rows = [0];
    let cols = [0];
    let mut out = vec![0.0];
    let observed_points = std::cell::RefCell::new(Vec::new());

    let operator = |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        observed_points.borrow_mut().push(batch.n_points());
        output[0] = batch.get(0, 0).unwrap();
        Ok(())
    };
    let evaluator = LocalBlockEvaluator::new(&problem, 1, &operator).unwrap();

    evaluator.fill_local_block(&rows, &cols, &mut out).unwrap();
    evaluator.clear_cache();
    evaluator.fill_local_block(&rows, &cols, &mut out).unwrap();

    assert_eq!(observed_points.into_inner(), vec![1, 1]);
}

#[test]
fn local_block_evaluators_with_different_operators_have_separate_caches() {
    let problem = local_test_problem();
    let rows = [0];
    let cols = [0];
    let mut first = vec![0.0];
    let mut second = vec![0.0];

    let first_operator = |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        output[0] = batch.get(0, 0).unwrap();
        Ok(())
    };
    let second_operator = |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        output[0] = batch.get(0, 0).unwrap() + 1.0;
        Ok(())
    };

    let first_evaluator = LocalBlockEvaluator::new(&problem, 1, &first_operator).unwrap();
    let second_evaluator = LocalBlockEvaluator::new(&problem, 1, &second_operator).unwrap();

    first_evaluator
        .fill_local_block(&rows, &cols, &mut first)
        .unwrap();
    second_evaluator
        .fill_local_block(&rows, &cols, &mut second)
        .unwrap();

    assert_eq!(second, vec![first[0] + 1.0]);
}

#[test]
fn local_block_evaluator_or_zero_records_operator_error_once() {
    let problem = local_test_problem();
    let rows = [0];
    let cols = [0];
    let mut out = vec![123.0];
    let calls = std::cell::Cell::new(0);
    let operator = |_batch: ElementwiseBatch<'_, f64>, _output: &mut [f64]| {
        calls.set(calls.get() + 1);
        Err(AciError::Operator {
            message: format!("operator failed on call {}", calls.get()),
        })
    };
    let evaluator = LocalBlockEvaluator::new(&problem, 1, &operator).unwrap();

    evaluator.fill_local_block_or_zero(&rows, &cols, &mut out);
    assert_eq!(out, vec![0.0]);
    evaluator.fill_local_block_or_zero(&rows, &cols, &mut out);
    assert_eq!(out, vec![0.0]);

    let err = evaluator.take_error().unwrap();
    assert!(err.to_string().contains("call 1"));
    assert!(evaluator.take_error().is_none());
}

#[test]
fn local_block_evaluator_or_zero_records_local_error_once() {
    let problem = local_test_problem();
    let rows = [4];
    let cols = [0];
    let mut out = vec![123.0];
    let operator = |_batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        output[0] = 1.0;
        Ok(())
    };
    let evaluator = LocalBlockEvaluator::new(&problem, 1, &operator).unwrap();

    evaluator.fill_local_block_or_zero(&rows, &cols, &mut out);

    assert_eq!(out, vec![0.0]);
    let err = evaluator.take_error().unwrap();
    assert!(err.to_string().contains("row index"));
}

#[test]
fn local_input_value_rejects_out_of_range_indices() {
    let problem = local_test_problem();

    let input_err = problem.local_input_value(2, 1, 0, 0).unwrap_err();
    assert!(matches!(input_err, AciError::InvalidInitialGuess { .. }));
    assert!(input_err.to_string().contains("input index"));

    let bond_err = problem.local_input_value(0, 3, 0, 0).unwrap_err();
    assert!(matches!(bond_err, AciError::InvalidInitialGuess { .. }));
    assert!(bond_err.to_string().contains("bond index"));

    let row_err = problem.local_input_value(0, 1, 4, 0).unwrap_err();
    assert!(matches!(row_err, AciError::InvalidInitialGuess { .. }));
    assert!(row_err.to_string().contains("row index"));

    let col_err = problem.local_input_value(0, 1, 0, 4).unwrap_err();
    assert!(matches!(col_err, AciError::InvalidInitialGuess { .. }));
    assert!(col_err.to_string().contains("column index"));
}

#[test]
fn local_input_value_rejects_missing_frames() {
    let input = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();

    let err = problem.local_input_value(0, 1, 0, 0).unwrap_err();

    assert!(matches!(err, AciError::InvalidInitialGuess { .. }));
    assert!(err.to_string().contains("left frame"));
}

#[test]
fn validate_inputs_rejects_empty_inputs() {
    let err = validate_inputs::<f64>(&[]).unwrap_err();
    assert!(err.to_string().contains("at least one"));
}

#[test]
fn validate_inputs_rejects_zero_site_tensor_train() {
    let input = TensorTrain::<f64>::new(vec![]).unwrap();
    let err = validate_inputs(&[input]).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("at least one site"));
}

#[test]
fn validate_inputs_rejects_zero_physical_dim_in_first_input() {
    let input = tensor_train_with_link_dims(&[0, 3], &[1]);
    let err = validate_inputs(&[input]).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    let message = err.to_string();
    assert!(message.contains("site 0"));
    assert!(message.contains("dimension must be positive"));
}

#[test]
fn validate_inputs_rejects_zero_physical_dim_in_later_input() {
    let first = tensor_train_with_link_dims(&[2, 3], &[1]);
    let later = tensor_train_with_link_dims(&[2, 0], &[1]);
    let err = validate_inputs(&[first, later]).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    let message = err.to_string();
    assert!(message.contains("site 1"));
    assert!(message.contains("dimension must be positive"));
}

#[test]
fn validate_inputs_rejects_zero_internal_bond_dim_in_first_input() {
    let input = tensor_train_with_link_dims(&[2, 3], &[0]);
    let err = validate_inputs(&[input]).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    let message = err.to_string();
    assert!(message.contains("bond dimension"));
    assert!(message.contains("positive"));
}

#[test]
fn validate_inputs_rejects_zero_internal_bond_dim_in_later_input() {
    let first = tensor_train_with_link_dims(&[2, 3], &[1]);
    let later = tensor_train_with_link_dims(&[2, 3], &[0]);
    let err = validate_inputs(&[first, later]).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    let message = err.to_string();
    assert!(message.contains("bond dimension"));
    assert!(message.contains("positive"));
}

#[test]
fn validate_inputs_rejects_length_mismatch() {
    let a = TensorTrain::<f64>::constant(&[2, 2], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let err = validate_inputs(&[a, b]).unwrap_err();
    assert!(err.to_string().contains("length mismatch"));
}

#[test]
fn validate_inputs_rejects_site_dim_mismatch() {
    let a = TensorTrain::<f64>::constant(&[2, 3], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 4], 1.0);
    let err = validate_inputs(&[a, b]).unwrap_err();
    assert!(err.to_string().contains("site dimension mismatch"));
}

#[test]
fn validate_options_rejects_zero_max_iters() {
    let options = AciOptions::<f64> {
        max_iters: 0,
        ..AciOptions::default()
    };
    let err = validate_options(&options).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("max_iters"));
}

#[test]
fn validate_options_rejects_zero_max_bond_dim() {
    let options = AciOptions::<f64> {
        max_bond_dim: 0,
        ..AciOptions::default()
    };
    let err = validate_options(&options).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("max_bond_dim"));
}

#[test]
fn validate_options_rejects_min_iters_above_max_iters() {
    let options = AciOptions::<f64> {
        min_iters: 3,
        max_iters: 2,
        ..AciOptions::default()
    };
    let err = validate_options(&options).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("min_iters"));
}

#[test]
fn validate_options_rejects_negative_tolerance() {
    let options = AciOptions::<f64> {
        tolerance: -1e-12,
        ..AciOptions::default()
    };
    let err = validate_options(&options).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("tolerance"));
}

#[test]
fn validate_options_rejects_nan_tolerance() {
    let options = AciOptions::<f64> {
        tolerance: f64::NAN,
        ..AciOptions::default()
    };
    let err = validate_options(&options).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("tolerance"));
}

#[test]
fn validate_options_rejects_infinite_tolerance() {
    let options = AciOptions::<f64> {
        tolerance: f64::INFINITY,
        ..AciOptions::default()
    };
    let err = validate_options(&options).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("tolerance"));
}

#[test]
fn default_initial_guess_matches_input_site_dims() {
    let a = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 3, 4], 2.0);
    let guess = initial_guess(&[a, b], &AciOptions::default()).unwrap();
    assert_eq!(guess.site_dims(), vec![2, 3, 4]);
}

#[test]
fn initial_guess_link_dims_are_empty_for_one_site_input() {
    let input = TensorTrain::<f64>::constant(&[5], 1.0);
    let guess = initial_guess(&[input], &AciOptions::default()).unwrap();
    assert_eq!(guess.link_dims(), Vec::<usize>::new());
}

#[test]
fn initial_guess_link_dims_are_limited_by_max_bond_dim() {
    let input = tensor_train_with_link_dims(&[8, 8, 8], &[8, 8]);
    let options = AciOptions::<f64> {
        max_bond_dim: 3,
        ..AciOptions::default()
    };
    let guess = initial_guess(&[input], &options).unwrap();
    assert_eq!(guess.link_dims(), vec![3, 3]);
}

#[test]
fn initial_guess_link_dims_are_limited_by_physical_left_right_products() {
    let input = tensor_train_with_link_dims(&[2, 3, 5, 7], &[20, 20, 20]);
    let guess = initial_guess(&[input], &AciOptions::default()).unwrap();
    assert_eq!(guess.link_dims(), vec![2, 6, 7]);
}

#[test]
fn initial_guess_link_dims_are_limited_by_minimum_input_link_dim() {
    let wide = tensor_train_with_link_dims(&[8, 8, 8], &[8, 8]);
    let narrow = tensor_train_with_link_dims(&[8, 8, 8], &[5, 2]);
    let guess = initial_guess(&[wide, narrow], &AciOptions::default()).unwrap();
    assert_eq!(guess.link_dims(), vec![5, 2]);
}

#[test]
fn initial_guess_is_deterministic_for_same_seed() {
    let a = TensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let options = AciOptions::<f64> {
        max_bond_dim: 2,
        rng_seed: 1234,
        ..AciOptions::default()
    };

    let guess_a = initial_guess(&[a.clone(), b.clone()], &options).unwrap();
    let guess_b = initial_guess(&[a, b], &options).unwrap();

    for indices in [[0, 0, 0], [1, 2, 1], [0, 1, 1]] {
        assert_eq!(
            guess_a.evaluate(&indices).unwrap(),
            guess_b.evaluate(&indices).unwrap()
        );
    }
}

#[test]
fn initial_guess_accepts_compatible_explicit_guess() {
    let input = TensorTrain::<f64>::constant(&[2, 3], 2.0);
    let explicit = TensorTrain::<f64>::constant(&[2, 3], 7.0);
    let options = AciOptions {
        initial_guess: Some(explicit),
        ..AciOptions::default()
    };

    let guess = initial_guess(&[input], &options).unwrap();

    assert_eq!(guess.site_dims(), vec![2, 3]);
    assert_eq!(guess.site_tensor(0).site_dim(), 2);
    assert_eq!(guess.site_tensor(1).site_dim(), 3);
    assert!((guess.evaluate(&[0, 0]).unwrap() - 7.0).abs() < 1e-12);
    assert!((guess.evaluate(&[1, 2]).unwrap() - 7.0).abs() < 1e-12);
}

#[test]
fn initial_guess_zero_initializes_nonzero_dimensional_right_frames() {
    let input_a = TensorTrain::<f64>::constant(&[2, 2, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 2, 2], 3.0);
    let zero_guess = tensor_train_with_link_dims(&[2, 2, 2], &[1, 1]);
    let options = AciOptions {
        initial_guess: Some(zero_guess),
        ..AciOptions::default()
    };

    let mut problem = ElementwiseProblem::new(vec![input_a, input_b], options.clone()).unwrap();

    assert_solution_is_zero_on_binary_three_site_grid(&problem);
    for input in 0..problem.n_inputs() {
        assert_eq!(problem.right_frame_shape(input, 1), Some((1, 1)));
        assert_eq!(problem.right_frame_shape(input, 2), Some((1, 1)));
        assert_eq!(problem.right_frame_shape(input, 3), Some((1, 1)));
    }
    problem.update_left_frames(0, &[0]).unwrap();
    assert!(problem.local_input_value(0, 1, 0, 0).is_ok());

    let mut operator = zero_batch;
    problem
        .local_update(1, false, &options, &mut operator)
        .unwrap();
    assert_solution_is_zero_on_binary_three_site_grid(&problem);
}

#[test]
fn initial_guess_rejects_incompatible_explicit_guess_site_dims() {
    let input = TensorTrain::<f64>::constant(&[2, 3], 2.0);
    let explicit = TensorTrain::<f64>::constant(&[2, 4], 7.0);
    let options = AciOptions {
        initial_guess: Some(explicit),
        ..AciOptions::default()
    };

    let err = initial_guess(&[input], &options).unwrap_err();

    assert!(matches!(err, AciError::InvalidInitialGuess { .. }));
    assert!(err.to_string().contains("site dimensions"));
}

#[test]
fn explicit_initial_guess_rejects_rank_above_max_bond_dim() {
    let input = TensorTrain::<f64>::constant(&[2, 2], 1.0);
    let explicit = TensorTrain::new(vec![
        tensor3_from_data(vec![1.0; 4], 1, 2, 2).unwrap(),
        tensor3_from_data(vec![1.0; 4], 2, 2, 1).unwrap(),
    ])
    .unwrap();
    let options = AciOptions {
        max_bond_dim: 1,
        initial_guess: Some(explicit),
        ..AciOptions::default()
    };

    let err = initial_guess(&[input], &options).unwrap_err();

    assert!(matches!(err, AciError::InvalidInitialGuess { .. }));
    let message = err.to_string();
    assert!(message.contains("bond dimension"));
    assert!(message.contains("max_bond_dim"));
}

#[test]
fn explicit_initial_guess_rejects_zero_bond_dimension() {
    let input = TensorTrain::<f64>::constant(&[2, 2], 1.0);
    let explicit = TensorTrain::new(vec![
        tensor3_from_data(Vec::<f64>::new(), 1, 2, 0).unwrap(),
        tensor3_from_data(Vec::<f64>::new(), 0, 2, 1).unwrap(),
    ])
    .unwrap();
    let options = AciOptions {
        initial_guess: Some(explicit),
        ..AciOptions::default()
    };

    let err = initial_guess(&[input], &options).unwrap_err();

    assert!(matches!(err, AciError::InvalidInitialGuess { .. }));
    let message = err.to_string();
    assert!(message.contains("dimension"));
    assert!(message.contains("positive"));
}

#[test]
fn complex_initial_guess_is_deterministic() {
    let a = TensorTrain::<Complex64>::constant(&[2, 3, 2], Complex64::new(1.0, 0.0));
    let b = TensorTrain::<Complex64>::constant(&[2, 3, 2], Complex64::new(2.0, 0.0));
    let options = AciOptions::<Complex64> {
        max_bond_dim: 2,
        rng_seed: 4321,
        ..AciOptions::default()
    };

    let guess_a = initial_guess(&[a.clone(), b.clone()], &options).unwrap();
    let guess_b = initial_guess(&[a, b], &options).unwrap();

    assert_eq!(guess_a.site_dims(), vec![2, 3, 2]);
    assert_eq!(guess_b.site_dims(), vec![2, 3, 2]);
    assert_eq!(
        guess_a.evaluate(&[1, 2, 1]).unwrap(),
        guess_b.evaluate(&[1, 2, 1]).unwrap()
    );
}

#[test]
fn initial_guess_existing_entry_count_matches_explicit_guess_cores() {
    let guess = tensor_train_with_link_dims(&[2, 3, 5], &[4, 6]);
    let entry_count = initial_guess_existing_entry_count(&guess).unwrap();
    assert_eq!(entry_count, 8 + 72 + 30);
}

#[test]
fn initial_guess_rejects_huge_non_overflowing_core_size() {
    let err = initial_guess_core_entry_count(MAX_INITIAL_GUESS_CORE_ENTRIES + 1, 1, 1).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("initial guess core size"));
}

#[test]
fn initial_guess_rejects_oversized_total_entries() {
    let err = initial_guess_total_entry_count(&[(MAX_INITIAL_GUESS_CORE_ENTRIES, 1, 1), (1, 1, 1)])
        .unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("initial guess total size"));
}
