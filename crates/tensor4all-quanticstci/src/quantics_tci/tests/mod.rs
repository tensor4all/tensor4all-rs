use super::*;
use approx::assert_relative_eq;
use quanticsgrids::UnfoldingScheme;

#[test]
fn point_from_batch_rejects_out_of_bounds_point() {
    let data = vec![0, 1];
    let batch = GlobalIndexBatch::new(&data, 2, 1).unwrap();
    let err = point_from_batch(batch, 1).unwrap_err();

    assert!(err.to_string().contains("invalid batch index"));
    assert!(err.to_string().contains("site 0"));
    assert!(err.to_string().contains("point 1"));
}

#[test]
fn site_evaluator_propagates_coordinate_conversion_failure() {
    let called = std::cell::Cell::new(false);
    let cache: Rc<RefCell<HashMap<Vec<usize>, f64>>> = Rc::new(RefCell::new(HashMap::new()));
    let evaluate = site_evaluator(
        |_point: &[usize]| anyhow::bail!("synthetic coordinate failure"),
        |_batch: QuanticsBatch<'_, f64>| {
            called.set(true);
            Ok(vec![1.0_f64])
        },
        cache,
    );
    let batch = GlobalIndexBatch::new(&[3], 1, 1).unwrap();
    // The conversion error is propagated unchanged; the caller adds the
    // quantics-index context (see the entry points' own error messages).
    let error = evaluate(batch).unwrap_err().to_string();
    assert_eq!(error, "synthetic coordinate failure");
    assert!(!called.get());
}

#[test]
fn site_evaluator_passes_converted_coordinates_and_caches_points() {
    let cache: Rc<RefCell<HashMap<Vec<usize>, f64>>> = Rc::new(RefCell::new(HashMap::new()));
    let evaluate = site_evaluator(
        |_point: &[usize]| Ok(vec![0.25]),
        |batch: QuanticsBatch<'_, f64>| {
            assert_eq!(batch.n_dims(), 1);
            assert_eq!(batch.n_points(), 1);
            assert_eq!(batch.get(0, 0), Some(0.25));
            Ok(vec![batch.get(0, 0).unwrap() * 4.0])
        },
        Rc::clone(&cache),
    );
    let batch = GlobalIndexBatch::new(&[0], 1, 1).unwrap();
    assert_eq!(evaluate(batch).unwrap(), vec![1.0]);
    // Evaluated points are recorded for the returned QuanticsTensorCI2 cache.
    assert_eq!(cache.borrow().get(&vec![0]), Some(&1.0));
}

#[test]
fn site_evaluator_rejects_a_wrong_number_of_returned_values() {
    let cache: Rc<RefCell<HashMap<Vec<usize>, f64>>> = Rc::new(RefCell::new(HashMap::new()));
    let evaluate = site_evaluator(
        |_point: &[usize]| Ok(vec![0.0_f64]),
        |_batch: QuanticsBatch<'_, f64>| Ok(vec![1.0_f64, 2.0]),
        cache,
    );
    let batch = GlobalIndexBatch::new(&[0], 1, 1).unwrap();
    let error = evaluate(batch).unwrap_err().to_string();
    assert!(error.contains("returned 2 values for 1 points"));
}

#[test]
fn test_discrete_simple_function() {
    // f(i, j) = i + j (grididx are 0-indexed)
    // Use 4x4 grid which gives 2 sites with Fused scheme
    let f = |idx: &[usize]| (idx[0] + idx[1]) as f64;
    let sizes = vec![4, 4];

    // Use Fused to get 2 sites (4x4 = 2 bits, so 2 sites for 2D)
    let opts = QtciOptions::default()
        .with_tolerance(1e-10)
        .with_nrandominitpivot(3)
        .with_unfoldingscheme(UnfoldingScheme::Fused);

    let result = quanticscrossinterpolate_discrete_batch(
        &sizes,
        pointwise_index_batch(f),
        Some(vec![vec![1, 0]]),
        opts,
    );
    assert!(result.is_ok(), "Error: {:?}", result.err());

    let (qtci, _ranks, _errors) = result.unwrap();

    // Verify some evaluations (grididx are 0-indexed)
    let val = qtci.evaluate(&[2, 3]).unwrap();
    assert_relative_eq!(val, 5.0, epsilon = 1e-8);

    let val = qtci.evaluate(&[0, 0]).unwrap();
    assert_relative_eq!(val, 0.0, epsilon = 1e-8);

    // Rank should be low for this simple function (i + j is rank 2)
    assert!(qtci.rank() <= 3);
}

#[test]
fn test_discrete_tci_structure() {
    // Test that the QTCI structure (bonds, rank, cache, sum) works correctly.
    // f(i,j) = i + j on a 4x4 grid with Fused scheme.
    let f = |idx: &[usize]| (idx[0] + idx[1]) as f64;
    let sizes = vec![4, 4];

    let opts = QtciOptions::default()
        .with_tolerance(1e-10)
        .with_nrandominitpivot(3)
        .with_unfoldingscheme(UnfoldingScheme::Fused);

    let result = quanticscrossinterpolate_discrete_batch(
        &sizes,
        pointwise_index_batch(f),
        Some(vec![vec![1, 0]]),
        opts,
    );
    assert!(result.is_ok(), "Error: {:?}", result.err());

    let (qtci, _ranks, _errors) = result.unwrap();

    // Verify the structure is created correctly
    assert_eq!(qtci.link_dims().len(), 1); // 2 sites = 1 bond
    assert!(qtci.rank() > 0);

    // Verify that the function was called with correct grid indices by checking
    // all cached values. The cache maps quantics indices to function values.
    let grid = qtci.inherent_grid().unwrap();
    for (quantics_idx, &cached_val) in qtci.cachedata() {
        let grid_idx = grid.quantics_to_grididx(quantics_idx).unwrap();
        let expected = (grid_idx[0] + grid_idx[1]) as f64;
        assert_relative_eq!(cached_val, expected, epsilon = 1e-10);
    }
    assert!(!qtci.cachedata().is_empty());

    // Verify evaluate() matches f at known-exact points (same block in
    // quantics representation).
    let val = qtci.evaluate(&[0, 0]).unwrap();
    assert_relative_eq!(val, 0.0, epsilon = 1e-8);
    let val = qtci.evaluate(&[2, 3]).unwrap();
    assert_relative_eq!(val, 5.0, epsilon = 1e-8);
    let val = qtci.evaluate(&[3, 3]).unwrap();
    assert_relative_eq!(val, 6.0, epsilon = 1e-8);
}

#[test]
fn test_size_validation() {
    let f = |_idx: &[usize]| 1.0_f64;

    // Non-power of 2 should fail
    let sizes = vec![5, 5];
    let result = quanticscrossinterpolate_discrete_batch(
        &sizes,
        pointwise_index_batch(f),
        None,
        QtciOptions::default(),
    );
    assert!(result.is_err());
}

#[test]
fn test_from_arrays_empty_inputs() {
    let f = |_coords: &[f64]| 1.0_f64;
    let result = quanticscrossinterpolate_from_arrays_batch::<f64, _>(
        &[],
        pointwise_coordinate_batch(f),
        None,
        QtciOptions::default(),
    );
    assert!(result.is_err());
}

#[test]
fn discrete_interpolation_rejects_empty_size_without_panicking() {
    let f = |_point: &[usize]| 1.0_f64;
    let result = quanticscrossinterpolate_discrete_batch::<f64, _>(
        &[],
        pointwise_index_batch(f),
        None,
        QtciOptions::default(),
    );
    assert!(result.is_err());
    let error = result.err().unwrap().to_string();
    assert!(
        error.contains("at least one grid dimension"),
        "got: {error}"
    );
}

#[test]
fn test_from_arrays_empty_dimension() {
    let f = |_coords: &[f64]| 1.0_f64;
    let xvals = vec![vec![], vec![0.0, 1.0]];
    let result = quanticscrossinterpolate_from_arrays_batch::<f64, _>(
        &xvals,
        pointwise_coordinate_batch(f),
        None,
        QtciOptions::default(),
    );
    assert!(result.is_err());
}

#[test]
fn from_arrays_uses_interior_coordinates() {
    let xvals = vec![vec![0.0, 0.5, 2.0, 5.0]];
    let options = QtciOptions::default()
        .with_tolerance(1e-10)
        .with_nrandominitpivot(2)
        .with_unfoldingscheme(UnfoldingScheme::Fused);
    let (qtci, _, _) = quanticscrossinterpolate_from_arrays_batch::<f64, _>(
        &xvals,
        pointwise_coordinate_batch(|coords| coords[0] + 1.0),
        None,
        options,
    )
    .unwrap();

    assert_relative_eq!(qtci.evaluate(&[1]).unwrap(), 1.5, epsilon = 1e-8);
    assert_relative_eq!(qtci.evaluate(&[2]).unwrap(), 3.0, epsilon = 1e-8);
}

#[test]
fn from_arrays_rejects_invalid_coordinates() {
    let options = QtciOptions::default();
    let nan = quanticscrossinterpolate_from_arrays_batch::<f64, _>(
        &[vec![0.0, f64::NAN, 1.0, 2.0]],
        pointwise_coordinate_batch(|_| 1.0),
        None,
        options.clone(),
    )
    .err()
    .unwrap();
    assert!(nan.to_string().contains("finite"));

    let duplicate = quanticscrossinterpolate_from_arrays_batch::<f64, _>(
        &[vec![0.0, 1.0, 1.0, 2.0]],
        pointwise_coordinate_batch(|_| 1.0),
        None,
        options,
    )
    .err()
    .unwrap();
    assert!(duplicate.to_string().contains("strictly increasing"));
}

#[test]
fn test_options_builder() {
    let opts = QtciOptions::default()
        .with_tolerance(1e-6)
        .with_max_bond_dim(50)
        .with_unfoldingscheme(UnfoldingScheme::Fused);

    assert!((opts.tolerance - 1e-6).abs() < 1e-15);
    assert_eq!(opts.max_bond_dim, Some(50));
    assert_eq!(opts.unfolding_scheme, UnfoldingScheme::Fused);
}

#[test]
fn test_discrete_inherent_grid_accessor() {
    // quanticscrossinterpolate_discrete uses from_inherent internally.
    // Verify that inherent_grid() returns Some and discretized_grid() returns None.
    let f = |idx: &[usize]| (idx[0] + idx[1]) as f64;
    let sizes = vec![4, 4];

    let opts = QtciOptions::default()
        .with_tolerance(1e-10)
        .with_nrandominitpivot(3)
        .with_unfoldingscheme(UnfoldingScheme::Fused);

    let (qtci, _ranks, _errors) = quanticscrossinterpolate_discrete_batch(
        &sizes,
        pointwise_index_batch(f),
        Some(vec![vec![1, 0]]),
        opts,
    )
    .unwrap();

    // inherent_grid should be Some, discretized_grid should be None
    assert!(qtci.inherent_grid().is_some());
    assert!(qtci.discretized_grid().is_none());

    // Verify that cached function values are correct, proving the grid
    // coordinate mapping works for inherent discrete grids.
    let grid = qtci.inherent_grid().unwrap();
    for (quantics_idx, &cached_val) in qtci.cachedata() {
        let grid_idx = grid.quantics_to_grididx(quantics_idx).unwrap();
        let expected = (grid_idx[0] + grid_idx[1]) as f64;
        assert_relative_eq!(cached_val, expected, epsilon = 1e-10);
    }

    // Verify evaluate() at known-exact points
    let val = qtci.evaluate(&[0, 0]).unwrap();
    assert_relative_eq!(val, 0.0, epsilon = 1e-8);
    let val = qtci.evaluate(&[3, 3]).unwrap();
    assert_relative_eq!(val, 6.0, epsilon = 1e-8);
}

#[test]
fn test_discrete_cachedata_origcoord_error() {
    // cachedata_origcoord() should return an error for inherent discrete grids
    // because there are no original continuous coordinates.
    // f is non-zero at the default initial pivot (grid index 0).
    let f = |idx: &[usize]| idx[0] as f64 + 1.0;
    let sizes = vec![4];

    let opts = QtciOptions::default()
        .with_tolerance(1e-10)
        .with_nrandominitpivot(1)
        .with_unfoldingscheme(UnfoldingScheme::Fused);

    let (qtci, _ranks, _errors) =
        quanticscrossinterpolate_discrete_batch(&sizes, pointwise_index_batch(f), None, opts)
            .unwrap();

    let result = qtci.cachedata_origcoord();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("original coordinates are only available for discretized grids"));
}

#[test]
fn test_discrete_integral_returns_sum() {
    // For inherent discrete grids, integral() should just return the sum.
    // f(i) = 1 for all i, on a grid of size 4 => sum = 4
    let f = |_idx: &[usize]| 1.0_f64;
    let sizes = vec![4];

    let opts = QtciOptions::default()
        .with_tolerance(1e-10)
        .with_nrandominitpivot(3)
        .with_unfoldingscheme(UnfoldingScheme::Fused);

    let (qtci, _ranks, _errors) =
        quanticscrossinterpolate_discrete_batch(&sizes, pointwise_index_batch(f), None, opts)
            .unwrap();

    let integral = qtci.integral().unwrap();
    let sum = qtci.sum().unwrap();
    // For inherent grids, integral == sum
    assert_relative_eq!(integral, sum, epsilon = 1e-10);
    assert_relative_eq!(integral, 4.0, epsilon = 1e-8);
}

#[test]
fn test_continuous_grid_interpolation() {
    // Test quanticscrossinterpolate with a DiscretizedGrid.
    // f(x) = x^2 on [0, 1], 8 grid points (3 bits)
    let grid = DiscretizedGrid::builder(&[3])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .include_endpoint(true)
        .build()
        .unwrap();

    let f = |coords: &[f64]| coords[0] * coords[0];

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_nrandominitpivot(5)
        .with_unfoldingscheme(UnfoldingScheme::Interleaved);

    let (qtci, _ranks, _errors) =
        quanticscrossinterpolate_batch(&grid, pointwise_coordinate_batch(f), None, opts).unwrap();

    // Verify accessors
    assert!(qtci.discretized_grid().is_some());
    assert!(qtci.inherent_grid().is_none());
    assert!(qtci.rank() > 0);

    // cachedata should have some entries
    assert!(!qtci.cachedata().is_empty());

    // Verify cached function values via cachedata_origcoord.
    // Each cached point should have the correct original coordinate and
    // function value f(x) = x^2.
    let origcoord_data = qtci.cachedata_origcoord().unwrap();
    assert!(!origcoord_data.is_empty());
    for (coord, val) in &origcoord_data {
        assert_eq!(coord.len(), 1);
        let x = coord[0];
        let expected = x * x;
        assert!(
            (val - expected).abs() < 1e-10,
            "cached f({}) = {}, expected {}",
            x,
            val,
            expected
        );
    }

    // Verify evaluate() produces finite values at grid endpoints
    let val = qtci.evaluate(&[0]).unwrap();
    assert!(val.is_finite());
    let val = qtci.evaluate(&[7]).unwrap();
    assert!(val.is_finite());
}

#[test]
fn test_continuous_grid_integral() {
    // Integral of f(x) = 1 over [0, 1] with 16 points should be ~1.0
    // (sum of 16 ones * step = 16 * (1/16) = 1.0 for non-endpoint grids)
    let grid = DiscretizedGrid::builder(&[4]) // 2^4 = 16 points
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .build()
        .unwrap();

    let f = |_coords: &[f64]| 1.0_f64;

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_nrandominitpivot(3);

    let (qtci, _ranks, _errors) =
        quanticscrossinterpolate_batch(&grid, pointwise_coordinate_batch(f), None, opts).unwrap();

    let integral = qtci.integral().unwrap();
    // integral = sum * step_size
    // sum = 16.0, step_size = 1/16 = 0.0625 => integral = 1.0
    assert_relative_eq!(integral, 1.0, epsilon = 1e-8);
}

#[test]
fn test_discrete_with_initial_pivots() {
    // Test that initial pivots are correctly converted and the TCI runs successfully.
    let f = |idx: &[usize]| (idx[0] * idx[1]) as f64;
    let sizes = vec![4, 4];

    let opts = QtciOptions::default()
        .with_tolerance(1e-10)
        .with_nrandominitpivot(3)
        .with_unfoldingscheme(UnfoldingScheme::Fused);

    // Provide explicit initial pivots (0-indexed grid indices)
    let pivots = vec![vec![0, 0], vec![1, 2]];
    let result = quanticscrossinterpolate_discrete_batch(
        &sizes,
        pointwise_index_batch(f),
        Some(pivots),
        opts,
    );
    assert!(result.is_ok(), "Error: {:?}", result.err());

    let (qtci, _ranks, _errors) = result.unwrap();

    // Verify cached values match f(i,j) = i*j, proving the function was
    // called with correct grid indices from the initial pivots and TCI sweep.
    let grid = qtci.inherent_grid().unwrap();
    for (quantics_idx, &cached_val) in qtci.cachedata() {
        let grid_idx = grid.quantics_to_grididx(quantics_idx).unwrap();
        let expected = (grid_idx[0] * grid_idx[1]) as f64;
        assert_relative_eq!(cached_val, expected, epsilon = 1e-10);
    }
    assert!(!qtci.cachedata().is_empty());

    // Verify evaluate() at known-exact points
    let val = qtci.evaluate(&[0, 0]).unwrap();
    assert_relative_eq!(val, 0.0, epsilon = 1e-8);
    let val = qtci.evaluate(&[3, 3]).unwrap();
    assert_relative_eq!(val, 9.0, epsilon = 1e-8);
}

#[test]
fn test_discrete_rejects_invalid_initial_pivot() {
    let result = quanticscrossinterpolate_discrete_batch(
        &[4],
        pointwise_index_batch(|_idx: &[usize]| 1.0_f64),
        Some(vec![vec![4]]),
        QtciOptions::default().with_nrandominitpivot(0),
    );
    let message = match result {
        Ok(_) => panic!("invalid initial pivot unexpectedly succeeded"),
        Err(error) => error.to_string(),
    };

    assert!(
        message.contains("initial pivot [4] conversion failed"),
        "{message}"
    );
    assert!(message.contains("Grid index 4"), "{message}");
}

#[test]
fn test_continuous_grid_with_initial_pivots() {
    // Test quanticscrossinterpolate with initial pivots.
    let grid = DiscretizedGrid::builder(&[3])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .include_endpoint(true)
        .build()
        .unwrap();

    let f = |coords: &[f64]| coords[0];

    let opts = QtciOptions::default()
        .with_tolerance(1e-12)
        .with_nrandominitpivot(3);

    let pivots = vec![vec![1], vec![4]];
    let result =
        quanticscrossinterpolate_batch(&grid, pointwise_coordinate_batch(f), Some(pivots), opts);
    assert!(result.is_ok(), "Error: {:?}", result.err());

    let (qtci, _ranks, _errors) = result.unwrap();

    // Verify cached function values via cachedata_origcoord.
    // Each cached point should store f(x) = x correctly.
    let origcoord_data = qtci.cachedata_origcoord().unwrap();
    assert!(!origcoord_data.is_empty());
    for (coord, val) in &origcoord_data {
        assert_eq!(coord.len(), 1);
        let x = coord[0];
        assert!(
            (val - x).abs() < 1e-10,
            "cached f({}) = {}, expected {}",
            x,
            val,
            x
        );
    }

    // Verify evaluate() produces finite values
    let val = qtci.evaluate(&[0]).unwrap();
    assert!(val.is_finite());
    let val = qtci.evaluate(&[7]).unwrap();
    assert!(val.is_finite());
}

#[test]
fn test_continuous_grid_rejects_invalid_initial_pivot() {
    let grid = DiscretizedGrid::builder(&[3])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .include_endpoint(true)
        .build()
        .unwrap();
    let result = quanticscrossinterpolate_batch(
        &grid,
        pointwise_coordinate_batch(|_coords: &[f64]| 1.0_f64),
        Some(vec![vec![8]]),
        QtciOptions::default().with_nrandominitpivot(0),
    );
    let message = match result {
        Ok(_) => panic!("invalid initial pivot unexpectedly succeeded"),
        Err(error) => error.to_string(),
    };

    assert!(
        message.contains("initial pivot [8] conversion failed"),
        "{message}"
    );
    assert!(message.contains("Grid index 8"), "{message}");
}

#[test]
fn test_from_arrays_non_power_of_two() {
    let f = |_coords: &[f64]| 1.0_f64;
    let xvals = vec![vec![0.0, 1.0, 2.0]]; // 3 points, not power of 2
    let result = quanticscrossinterpolate_from_arrays_batch::<f64, _>(
        &xvals,
        pointwise_coordinate_batch(f),
        None,
        QtciOptions::default(),
    );
    assert!(result.is_err());
}

#[test]
fn test_from_arrays_unequal_dimensions() {
    let f = |_coords: &[f64]| 1.0_f64;
    // 4 points vs 8 points => different dimensions
    let xvals = vec![
        vec![0.0, 1.0, 2.0, 3.0],
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ];
    let result = quanticscrossinterpolate_from_arrays_batch::<f64, _>(
        &xvals,
        pointwise_coordinate_batch(f),
        None,
        QtciOptions::default(),
    );
    assert!(result.is_err());
}

#[test]
fn test_from_arrays_valid() {
    let f = |coords: &[f64]| coords[0] + coords[1];
    let xvals = vec![vec![0.0, 0.5, 2.0, 3.0], vec![0.0, 1.0, 2.0, 4.0]];

    let opts = QtciOptions::default()
        .with_tolerance(1e-10)
        .with_nrandominitpivot(3)
        .with_unfoldingscheme(UnfoldingScheme::Fused);

    let result = quanticscrossinterpolate_from_arrays_batch::<f64, _>(
        &xvals,
        pointwise_coordinate_batch(f),
        None,
        opts,
    );
    assert!(result.is_ok(), "Error: {:?}", result.err());

    let (qtci, _ranks, _errors) = result.unwrap();
    assert!(qtci.inherent_grid().is_some());
    assert!(qtci.cachedata_origcoord().is_err());
    assert!(qtci.rank() > 0);

    // Every cached point uses the supplied grid coordinates.
    let grid = qtci.inherent_grid().unwrap();
    let cache = qtci.cachedata();
    assert!(!cache.is_empty());
    for (quantics, val) in cache {
        let indices = grid.quantics_to_grididx(quantics).unwrap();
        let expected = xvals[0][indices[0]] + xvals[1][indices[1]];
        assert!((val - expected).abs() < 1e-10);
    }

    // Verify evaluate() at known-exact points.
    let val = qtci.evaluate(&[0, 0]).unwrap();
    assert_relative_eq!(val, 0.0, epsilon = 1e-8);
    let val = qtci.evaluate(&[3, 3]).unwrap();
    assert_relative_eq!(val, 7.0, epsilon = 1e-8);
}

#[test]
fn test_discrete_unequal_dimensions_error() {
    let f = |_idx: &[usize]| 1.0_f64;
    // 4 vs 8 => unequal
    let sizes = vec![4, 8];
    let result = quanticscrossinterpolate_discrete_batch(
        &sizes,
        pointwise_index_batch(f),
        None,
        QtciOptions::default(),
    );
    assert!(result.is_err());
}

/// Port of Julia test_tciinterface.jl: "quanticscrossinterpolate, 1d overload"
///
/// Tests that 1D functions can be interpolated via quanticscrossinterpolate_from_arrays.
#[test]
fn test_from_arrays_1d() {
    // f(x) = 0.1*x^2 - pi*x + 2
    let f_scalar = |x: f64| 0.1 * x * x - std::f64::consts::PI * x + 2.0;
    let f = move |coords: &[f64]| f_scalar(coords[0]);

    // 128 points from -3 to 2
    let n = 128;
    let xvals: Vec<f64> = (0..n)
        .map(|i| -3.0 + 5.0 * i as f64 / (n - 1) as f64)
        .collect();

    let options = QtciOptions::default().with_tolerance(1e-8);

    let xvals_ref = xvals.clone();
    let (qtci, _ranks, errors) = quanticscrossinterpolate_from_arrays_batch(
        &[xvals_ref],
        pointwise_coordinate_batch(f),
        None,
        options,
    )
    .unwrap();

    assert!(*errors.last().unwrap() < 1e-8);

    // Verify at grid points
    for (i, &x) in xvals.iter().enumerate() {
        let grid_idx = i; // 0-indexed
        let expected = f_scalar(x);
        let actual = qtci.evaluate(&[grid_idx]).unwrap();
        assert!(
            (actual - expected).abs() < 1e-6,
            "1D QTCI error at x={x}: expected={expected}, got={actual}"
        );
    }
}

#[test]
#[allow(deprecated)]
fn deprecated_pointwise_entry_point_matches_the_batched_path() {
    // The point-wise entry points are thin wrappers over the batched ones, so
    // they must keep producing the same ranks, errors, and values.
    let sizes = vec![8usize, 8];
    let pointwise = |index: &[usize]| (index[0] + 2 * index[1]) as f64;
    let options = QtciOptions::default()
        .with_tolerance(1e-10)
        .with_nrandominitpivot(0);

    let pivots = Some(vec![vec![1, 0]]);
    let (pointwise_qtci, pointwise_ranks, pointwise_errors) =
        quanticscrossinterpolate_discrete(&sizes, pointwise, pivots.clone(), options.clone())
            .unwrap();
    let (batched_qtci, batched_ranks, batched_errors) = quanticscrossinterpolate_discrete_batch(
        &sizes,
        pointwise_index_batch(pointwise),
        pivots,
        options,
    )
    .unwrap();

    assert_eq!(pointwise_ranks, batched_ranks);
    assert_eq!(pointwise_errors, batched_errors);
    assert_eq!(
        pointwise_qtci.evaluate(&[3, 5]).unwrap(),
        batched_qtci.evaluate(&[3, 5]).unwrap()
    );
}
