use approx::assert_abs_diff_eq;
use tensor4all_aci::{elementwise, elementwise_batched, AciOptions, ElementwiseBatch};
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, TensorTrain};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn exact_options() -> AciOptions<f64> {
    AciOptions {
        max_iters: 8,
        min_iters: 2,
        max_bond_dim: 8,
        tolerance: 1e-12,
        ..AciOptions::default()
    }
}

fn assert_dense_close(actual: &TensorTrain<f64>, expected_shape: &[usize], expected: &[f64]) {
    let (actual_data, actual_shape) = actual.fulltensor();
    assert_eq!(actual_shape, expected_shape);
    assert_eq!(actual_data.len(), expected.len());
    for (actual_value, expected_value) in actual_data.iter().zip(expected) {
        assert_abs_diff_eq!(actual_value, expected_value, epsilon = 1e-9);
    }
}

fn dense_oracle2<F>(left: &TensorTrain<f64>, right: &TensorTrain<f64>, mut op: F) -> Vec<f64>
where
    F: FnMut(f64, f64) -> f64,
{
    let (left_data, left_shape) = left.fulltensor();
    let (right_data, right_shape) = right.fulltensor();
    assert_eq!(left_shape, right_shape);
    left_data
        .into_iter()
        .zip(right_data)
        .map(|(left_value, right_value)| op(left_value, right_value))
        .collect()
}

fn col_major_index3(
    left: usize,
    site: usize,
    right: usize,
    left_dim: usize,
    site_dim: usize,
) -> usize {
    left + left_dim * (site + site_dim * right)
}

fn two_site_tt_from_col_major(data: &[f64], shape: [usize; 2]) -> TestResult<TensorTrain<f64>> {
    assert_eq!(data.len(), shape[0] * shape[1]);
    let left_site_dim = shape[0];
    let right_site_dim = shape[1];
    let bond_dim = left_site_dim;

    let mut left_core = vec![0.0; left_site_dim * bond_dim];
    for site in 0..left_site_dim {
        left_core[col_major_index3(0, site, site, 1, left_site_dim)] = 1.0;
    }

    let mut right_core = vec![0.0; bond_dim * right_site_dim];
    for right_site in 0..right_site_dim {
        for bond in 0..bond_dim {
            let dense_index = bond + left_site_dim * right_site;
            right_core[col_major_index3(bond, right_site, 0, bond_dim, right_site_dim)] =
                data[dense_index];
        }
    }

    Ok(TensorTrain::new(vec![
        tensor3_from_data(left_core, 1, left_site_dim, bond_dim)?,
        tensor3_from_data(right_core, bond_dim, right_site_dim, 1)?,
    ])?)
}

#[test]
fn dense_oracle_matches_rank_one_constant_multiplication() -> TestResult {
    let left = TensorTrain::<f64>::constant(&[2, 3, 2], 2.5);
    let right = TensorTrain::<f64>::constant(&[2, 3, 2], -4.0);
    let expected = dense_oracle2(&left, &right, |left_value, right_value| {
        left_value * right_value
    });

    let result = elementwise_batched(
        |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
            for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
                *value = batch.get(0, point)? * batch.get(1, point)?;
            }
            Ok(())
        },
        &[left, right],
        &exact_options(),
    )?;

    assert_eq!(result.tensor_train.site_dims(), vec![2, 3, 2]);
    assert_dense_close(&result.tensor_train, &[2, 3, 2], &expected);
    Ok(())
}

#[test]
fn dense_oracle_matches_nonconstant_addition() -> TestResult {
    let left = two_site_tt_from_col_major(&[1.0, -2.0, 3.0, 4.0, -5.0, 6.0], [2, 3])?;
    let right = two_site_tt_from_col_major(&[0.5, 1.5, -3.0, 2.0, 8.0, -7.0], [2, 3])?;
    let expected = dense_oracle2(&left, &right, |left_value, right_value| {
        left_value + right_value
    });

    let result = elementwise(
        |values| values[0] + values[1],
        &[left, right],
        &exact_options(),
    )?;

    assert_eq!(result.tensor_train.site_dims(), vec![2, 3]);
    assert_dense_close(&result.tensor_train, &[2, 3], &expected);
    Ok(())
}

#[test]
fn dense_oracle_matches_nonlinear_scalar_operation() -> TestResult {
    let left = two_site_tt_from_col_major(&[0.0, 0.25, 0.5, 1.0, -0.75, 1.25], [2, 3])?;
    let right = two_site_tt_from_col_major(&[2.0, -1.0, 0.5, -0.25, 1.5, -2.0], [2, 3])?;
    let expected = dense_oracle2(&left, &right, |left_value, right_value| {
        left_value.sin() + right_value * right_value
    });

    let result = elementwise(
        |values| values[0].sin() + values[1] * values[1],
        &[left, right],
        &exact_options(),
    )?;

    assert_eq!(result.tensor_train.site_dims(), vec![2, 3]);
    assert_dense_close(&result.tensor_train, &[2, 3], &expected);
    Ok(())
}
