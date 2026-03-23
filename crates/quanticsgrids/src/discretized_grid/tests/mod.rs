use super::*;

#[test]
fn test_basic_1d_grid() {
    let grid = DiscretizedGrid::builder(&[3]).build().unwrap();
    assert_eq!(grid.ndims(), 1);
    assert_eq!(grid.lower_bound(), &[0.0]);
    assert_eq!(grid.upper_bound(), &[1.0]);
    assert_eq!(grid.len(), 3);
}

#[test]
fn test_basic_2d_grid() {
    let grid = DiscretizedGrid::builder(&[3, 2])
        .with_variable_names(&["x", "y"])
        .build()
        .unwrap();

    assert_eq!(grid.ndims(), 2);
    assert_eq!(grid.variable_names(), &["x", "y"]);
    assert_eq!(grid.lower_bound(), &[0.0, 0.0]);
    assert_eq!(grid.upper_bound(), &[1.0, 1.0]);
}

#[test]
fn test_custom_bounds() {
    let grid = DiscretizedGrid::builder(&[2])
        .with_lower_bound(&[-1.0])
        .with_upper_bound(&[1.0])
        .build()
        .unwrap();

    assert_eq!(grid.lower_bound(), &[-1.0]);
    assert_eq!(grid.upper_bound(), &[1.0]);

    // Grid step should be (1 - (-1)) / 2^2 = 0.5
    let step = grid.grid_step();
    assert!((step[0] - 0.5).abs() < 1e-10);
}

#[test]
fn test_origcoord_to_grididx() {
    let grid = DiscretizedGrid::builder(&[3]).build().unwrap();
    // Grid has 8 points: 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875

    let idx = grid.origcoord_to_grididx(&[0.0]).unwrap();
    assert_eq!(idx, vec![1]);

    let idx = grid.origcoord_to_grididx(&[0.5]).unwrap();
    assert_eq!(idx, vec![5]);
}

#[test]
fn test_grididx_to_origcoord() {
    let grid = DiscretizedGrid::builder(&[3]).build().unwrap();

    let coord = grid.grididx_to_origcoord(&[1]).unwrap();
    assert!((coord[0] - 0.0).abs() < 1e-10);

    let coord = grid.grididx_to_origcoord(&[5]).unwrap();
    assert!((coord[0] - 0.5).abs() < 1e-10);
}

#[test]
fn test_roundtrip_all_points() {
    let grid = DiscretizedGrid::builder(&[2, 2]).build().unwrap();

    for x in 1..=4 {
        for y in 1..=4 {
            let grididx = vec![x, y];
            let quantics = grid.grididx_to_quantics(&grididx).unwrap();
            let back = grid.quantics_to_grididx(&quantics).unwrap();
            assert_eq!(back, grididx);

            let coord = grid.grididx_to_origcoord(&grididx).unwrap();
            let back_idx = grid.origcoord_to_grididx(&coord).unwrap();
            assert_eq!(back_idx, grididx);
        }
    }
}

#[test]
fn test_include_endpoint() {
    let grid = DiscretizedGrid::builder(&[2])
        .include_endpoint(true)
        .build()
        .unwrap();

    // With endpoint, upper bound is adjusted to include the last point
    // 4 points, want last point at 1.0
    // Adjusted upper = 1.0 + (1.0 - 0.0) / (4 - 1) = 1.0 + 1/3 = 4/3
    let max_coord = grid.grid_max();
    assert!((max_coord[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_grid_origcoords() {
    let grid = DiscretizedGrid::builder(&[2]).build().unwrap();
    let coords = grid.grid_origcoords(0).unwrap();

    assert_eq!(coords.len(), 4);
    assert!((coords[0] - 0.0).abs() < 1e-10);
    assert!((coords[1] - 0.25).abs() < 1e-10);
    assert!((coords[2] - 0.5).abs() < 1e-10);
    assert!((coords[3] - 0.75).abs() < 1e-10);
}

#[test]
fn test_display() {
    let grid = DiscretizedGrid::builder(&[3, 2])
        .with_variable_names(&["x", "y"])
        .build()
        .unwrap();

    let display = format!("{}", grid);
    assert!(display.contains("DiscretizedGrid"));
    assert!(display.contains("x"));
    assert!(display.contains("y"));
}

#[test]
fn test_error_invalid_bounds() {
    let result = DiscretizedGrid::builder(&[2])
        .with_lower_bound(&[1.0])
        .with_upper_bound(&[0.0])
        .build();

    assert!(matches!(
        result,
        Err(QuanticsGridError::InvalidBounds { .. })
    ));
}

#[test]
fn test_error_coordinate_out_of_bounds() {
    let grid = DiscretizedGrid::builder(&[2]).build().unwrap();
    let result = grid.origcoord_to_grididx(&[1.5]);
    assert!(matches!(
        result,
        Err(QuanticsGridError::CoordinateOutOfBounds { .. })
    ));
}

#[test]
fn test_error_coordinate_equal_to_exclusive_upper_bound() {
    let grid = DiscretizedGrid::builder(&[3]).build().unwrap();
    let result = grid.origcoord_to_grididx(&[1.0]);
    assert!(matches!(
        result,
        Err(QuanticsGridError::CoordinateOutOfBounds { .. })
    ));
}

#[test]
fn test_include_endpoint_accepts_exact_upper_bound_only() {
    let grid = DiscretizedGrid::builder(&[2])
        .include_endpoint(true)
        .build()
        .unwrap();

    assert_eq!(grid.origcoord_to_grididx(&[1.0]).unwrap(), vec![4]);

    let result = grid.origcoord_to_grididx(&[1.1]);
    assert!(matches!(
        result,
        Err(QuanticsGridError::CoordinateOutOfBounds { .. })
    ));
}

#[test]
fn test_from_index_table() {
    let index_table = vec![
        vec![("a".to_string(), 1), ("b".to_string(), 2)],
        vec![("a".to_string(), 2)],
        vec![("b".to_string(), 1), ("a".to_string(), 3)],
    ];

    let grid = DiscretizedGrid::from_index_table(&["a", "b"], index_table)
        .build()
        .unwrap();

    assert_eq!(grid.ndims(), 2);
    assert_eq!(grid.rs(), &[3, 2]);

    // Test roundtrip
    for x in 1..=8 {
        for y in 1..=4 {
            let grididx = vec![x, y];
            let quantics = grid.grididx_to_quantics(&grididx).unwrap();
            let back = grid.quantics_to_grididx(&quantics).unwrap();
            assert_eq!(back, grididx);
        }
    }
}

#[test]
fn test_quantics_function() {
    let grid = DiscretizedGrid::builder(&[2]).build().unwrap();

    let f = |coords: &[f64]| coords[0] * 2.0;
    let qf = quantics_function(&grid, f);

    // grididx 1 -> coord 0.0 -> f = 0.0
    let quantics = grid.grididx_to_quantics(&[1]).unwrap();
    let result = qf(&quantics).unwrap();
    assert!((result - 0.0).abs() < 1e-10);

    // grididx 3 -> coord 0.5 -> f = 1.0
    let quantics = grid.grididx_to_quantics(&[3]).unwrap();
    let result = qf(&quantics).unwrap();
    assert!((result - 1.0).abs() < 1e-10);
}
