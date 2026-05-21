use crate::validation::{validate_inputs, validate_options};
use crate::{
    initial_guess,
    random_tt::{
        initial_guess_core_entry_count, initial_guess_existing_entry_count,
        initial_guess_total_entry_count, MAX_INITIAL_GUESS_CORE_ENTRIES,
    },
    AciError, AciOptions, ElementwiseBatch,
};
use tensor4all_simplett::{tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain};

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
