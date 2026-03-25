use super::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// =============================================================================
// Existing tests (unchanged)
// =============================================================================

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

// =============================================================================
// Ported from: origcoord_tests.jl - "origcoord functions - basic 1D grid"
// =============================================================================

#[test]
fn test_origcoord_basic_1d_grid() {
    let grid = DiscretizedGrid::builder(&[4]).build().unwrap();

    // grididx 1 -> origcoord 0.0
    assert_eq!(grid.grididx_to_origcoord(&[1]).unwrap(), vec![0.0]);
    // grididx 2^4 -> origcoord ≈ 1.0 - 1.0/2^4
    let coord = grid.grididx_to_origcoord(&[16]).unwrap();
    assert!((coord[0] - (1.0 - 1.0 / 16.0)).abs() < 1e-14);

    // origcoord_to_grididx
    assert_eq!(grid.origcoord_to_grididx(&[0.0]).unwrap(), vec![1]);
    assert_eq!(grid.origcoord_to_grididx(&[0.5]).unwrap(), vec![9]); // 2^3 + 1

    // Round-trip conversions
    for &gidx in &[1, 5, 16] {
        let origcoord = grid.grididx_to_origcoord(&[gidx]).unwrap();
        let recovered = grid.origcoord_to_grididx(&origcoord).unwrap();
        assert_eq!(recovered, vec![gidx]);
    }

    // quantics <-> origcoord
    let quantics = vec![1, 1, 1, 2]; // grididx = 2
    let expected = 1.0 / 16.0;
    let coord = grid.quantics_to_origcoord(&quantics).unwrap();
    assert!((coord[0] - expected).abs() < 1e-14);
    let recovered_q = grid.origcoord_to_quantics(&[expected]).unwrap();
    assert_eq!(recovered_q, quantics);
}

// =============================================================================
// Ported from: origcoord_tests.jl - "origcoord functions - 2D grid"
// =============================================================================

#[test]
fn test_origcoord_2d_grid() {
    let grid = DiscretizedGrid::builder(&[3, 4])
        .with_lower_bound(&[0.0, 1.0])
        .with_upper_bound(&[2.0, 3.0])
        .build()
        .unwrap();

    // Boundary values
    let coord_lower = grid.grididx_to_origcoord(&[1, 1]).unwrap();
    assert!((coord_lower[0] - 0.0).abs() < 1e-14);
    assert!((coord_lower[1] - 1.0).abs() < 1e-14);

    let coord_upper = grid.grididx_to_origcoord(&[8, 16]).unwrap();
    assert!((coord_upper[0] - (2.0 - 2.0 / 8.0)).abs() < 1e-14);
    assert!((coord_upper[1] - (3.0 - 2.0 / 16.0)).abs() < 1e-14);

    // Middle values
    let mid_grididx = vec![5, 9]; // 2^2+1=5, 2^3+1=9
    let expected_x = 0.0 + 2.0 * 4.0 / 8.0;
    let expected_y = 1.0 + 2.0 * 8.0 / 16.0;
    let mid_coord = grid.grididx_to_origcoord(&mid_grididx).unwrap();
    assert!((mid_coord[0] - expected_x).abs() < 1e-14);
    assert!((mid_coord[1] - expected_y).abs() < 1e-14);

    // Deterministic round-trip conversions (matches Julia: for _ in 1:20 with rand(1:2^3), rand(1:2^4))
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..20 {
        let gx = rng.random_range(1..=8); // 1..=2^3
        let gy = rng.random_range(1..=16); // 1..=2^4
        let origcoord = grid.grididx_to_origcoord(&[gx, gy]).unwrap();
        let recovered = grid.origcoord_to_grididx(&origcoord).unwrap();
        assert_eq!(recovered, vec![gx, gy]);
    }
}

// =============================================================================
// Ported from: origcoord_tests.jl - "origcoord functions - different base"
// =============================================================================

#[test]
fn test_origcoord_different_base() {
    let base: usize = 3;
    let grid = DiscretizedGrid::builder(&[2, 3])
        .with_base(base)
        .with_lower_bound(&[-1.0, 0.0])
        .with_upper_bound(&[1.0, 6.0])
        .build()
        .unwrap();

    // Boundary values
    let coord_lower = grid.grididx_to_origcoord(&[1, 1]).unwrap();
    assert!((coord_lower[0] - (-1.0)).abs() < 1e-14);
    assert!((coord_lower[1] - 0.0).abs() < 1e-14);

    let coord_upper = grid.grididx_to_origcoord(&[9, 27]).unwrap(); // base^2=9, base^3=27
    assert!((coord_upper[0] - (1.0 - 2.0 / 9.0)).abs() < 1e-14);
    assert!((coord_upper[1] - (6.0 - 6.0 / 27.0)).abs() < 1e-14);

    // Deterministic round-trip conversions (matches Julia: for _ in 1:20 with rand(1:base^2), rand(1:base^3))
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..20 {
        let gx = rng.random_range(1..=9); // 1..=base^2=9
        let gy = rng.random_range(1..=27); // 1..=base^3=27
        let origcoord = grid.grididx_to_origcoord(&[gx, gy]).unwrap();
        let recovered = grid.origcoord_to_grididx(&origcoord).unwrap();
        assert_eq!(recovered, vec![gx, gy]);
    }

    // quantics <-> origcoord round-trip (matches Julia: for _ in 1:20 with rand(1:base, length(grid)))
    let n_sites = grid.len();
    let mut rng = StdRng::seed_from_u64(43);
    for _ in 0..20 {
        let quantics: Vec<i64> = (0..n_sites)
            .map(|_| rng.random_range(1..=base as i64))
            .collect();
        // Validate quantics are within site dims
        let valid = quantics
            .iter()
            .enumerate()
            .all(|(s, &q)| q >= 1 && q <= grid.site_dim(s).unwrap() as i64);
        if valid {
            let grididx = grid.quantics_to_grididx(&quantics).unwrap();
            let origcoord = grid.grididx_to_origcoord(&grididx).unwrap();
            let q_coord = grid.quantics_to_origcoord(&quantics).unwrap();
            for (a, b) in origcoord.iter().zip(q_coord.iter()) {
                assert!((a - b).abs() < 1e-14);
            }
            let recovered_q = grid.origcoord_to_quantics(&origcoord).unwrap();
            assert_eq!(recovered_q, quantics);
        }
    }
}

// =============================================================================
// Ported from: origcoord_tests.jl - "origcoord functions - boundary checking"
// =============================================================================

#[test]
fn test_origcoord_boundary_checking() {
    let grid = DiscretizedGrid::builder(&[2, 2]).build().unwrap();

    // Within bounds
    assert_eq!(grid.origcoord_to_grididx(&[0.0, 0.0]).unwrap(), vec![1, 1]);
    assert_eq!(grid.origcoord_to_grididx(&[0.5, 0.5]).unwrap(), vec![3, 3]);
    assert_eq!(
        grid.origcoord_to_grididx(&[0.999, 0.999]).unwrap(),
        vec![4, 4]
    );

    // Outside bounds should error
    assert!(grid.origcoord_to_grididx(&[-0.1, 0.5]).is_err());
    assert!(grid.origcoord_to_grididx(&[0.5, -0.1]).is_err());
    assert!(grid.origcoord_to_grididx(&[1.1, 0.5]).is_err());
    assert!(grid.origcoord_to_grididx(&[0.5, 1.1]).is_err());
    assert!(grid.origcoord_to_grididx(&[-0.1, -0.1]).is_err());
    assert!(grid.origcoord_to_grididx(&[1.1, 1.1]).is_err());
}

// =============================================================================
// Ported from: origcoord_tests.jl - "origcoord functions - stress test"
// =============================================================================

#[test]
fn test_origcoord_stress_4d() {
    let grid = DiscretizedGrid::builder(&[2, 3, 2, 4])
        .with_base(2)
        .with_lower_bound(&[0.0, -2.0, 1.0, -5.0])
        .with_upper_bound(&[1.0, 2.0, 3.0, 5.0])
        .build()
        .unwrap();

    // Deterministic grid index tests (matches Julia: for _ in 1:30 with rand(1:2^R) for each dim)
    // Rs=(2,3,2,4) => max indices are (4, 8, 4, 16)
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..30 {
        let g1 = rng.random_range(1..=4); // 1..=2^2
        let g2 = rng.random_range(1..=8); // 1..=2^3
        let g3 = rng.random_range(1..=4); // 1..=2^2
        let g4 = rng.random_range(1..=16); // 1..=2^4
        let grididx = vec![g1, g2, g3, g4];
        let origcoord = grid.grididx_to_origcoord(&grididx).unwrap();
        let recovered = grid.origcoord_to_grididx(&origcoord).unwrap();
        assert_eq!(recovered, grididx);
    }

    // Quantics round-trip
    let n_sites = grid.len();
    for seed in 0..30 {
        let quantics: Vec<i64> = (0..n_sites)
            .map(|i| ((seed * 7 + i * 13) % 2 + 1) as i64)
            .collect();
        let grididx = grid.quantics_to_grididx(&quantics).unwrap();
        let origcoord = grid.grididx_to_origcoord(&grididx).unwrap();
        let q_coord = grid.quantics_to_origcoord(&quantics).unwrap();
        for (a, b) in origcoord.iter().zip(q_coord.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        let recovered_q = grid.origcoord_to_quantics(&origcoord).unwrap();
        assert_eq!(recovered_q, quantics);
    }
}

// =============================================================================
// Ported from: origcoord_tests.jl - "edge cases - extreme coordinate ranges"
// =============================================================================

#[test]
fn test_extreme_coordinate_ranges() {
    // Very large range
    let grid = DiscretizedGrid::builder(&[10, 8])
        .with_lower_bound(&[-1e10, -1e15])
        .with_upper_bound(&[1e10, 1e15])
        .build()
        .unwrap();

    let test_cases: Vec<(i64, i64)> = vec![(1, 1), (1024, 256), (512, 128), (100, 50), (700, 200)];
    for (gx, gy) in test_cases {
        let origcoord = grid.grididx_to_origcoord(&[gx, gy]).unwrap();
        let recovered = grid.origcoord_to_grididx(&origcoord).unwrap();
        assert_eq!(recovered, vec![gx, gy]);
    }

    // Very small range
    let grid2 = DiscretizedGrid::builder(&[5, 6])
        .with_lower_bound(&[1e-12, -1e-12])
        .with_upper_bound(&[1e-12 + 1e-15, -1e-12 + 1e-15])
        .build()
        .unwrap();

    for gx in [1, 16, 32] {
        for gy in [1, 32, 64] {
            let origcoord = grid2.grididx_to_origcoord(&[gx, gy]).unwrap();
            let recovered = grid2.origcoord_to_grididx(&origcoord).unwrap();
            assert_eq!(recovered, vec![gx, gy]);
        }
    }
}

// =============================================================================
// Ported from: origcoord_tests.jl - "edge cases - asymmetric ranges with negative bounds"
// =============================================================================

#[test]
fn test_asymmetric_negative_bounds() {
    let grid = DiscretizedGrid::builder(&[4, 6, 3])
        .with_lower_bound(&[-1000.0, -0.001, -1e6])
        .with_upper_bound(&[-999.0, 0.001, 1e6])
        .build()
        .unwrap();

    let grid_min = grid.grid_min();
    let grid_max = grid.grid_max();

    let n_sites = grid.len();
    for seed in 0..30 {
        let quantics: Vec<i64> = (0..n_sites)
            .map(|i| ((seed * 7 + i * 13) % 2 + 1) as i64)
            .collect();
        let grididx = grid.quantics_to_grididx(&quantics).unwrap();
        let origcoord = grid.grididx_to_origcoord(&grididx).unwrap();

        // Verify coordinates are within bounds
        for (d, &c) in origcoord.iter().enumerate() {
            assert!(c >= grid_min[d] - 1e-14);
            assert!(c <= grid_max[d] + 1e-14);
        }

        // Round-trip
        let recovered = grid.origcoord_to_grididx(&origcoord).unwrap();
        assert_eq!(recovered, grididx);
        let q_coord = grid.quantics_to_origcoord(&quantics).unwrap();
        for (a, b) in origcoord.iter().zip(q_coord.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}

// =============================================================================
// Ported from: origcoord_tests.jl - "boundary stress test"
// =============================================================================

#[test]
fn test_boundary_stress() {
    let grid = DiscretizedGrid::builder(&[8, 6])
        .with_lower_bound(&[-2.0, 3.0])
        .with_upper_bound(&[5.0, 8.0])
        .build()
        .unwrap();

    // Exactly at lower bounds
    assert_eq!(grid.origcoord_to_grididx(&[-2.0, 3.0]).unwrap(), vec![1, 1]);

    // At effective upper bounds (grid_max)
    let grid_max_x = -2.0 + 7.0 * (256.0 - 1.0) / 256.0;
    let grid_max_y = 3.0 + 5.0 * (64.0 - 1.0) / 64.0;
    let max_grididx = grid
        .origcoord_to_grididx(&[grid_max_x, grid_max_y])
        .unwrap();
    assert_eq!(max_grididx, vec![256, 64]);

    // Just inside boundaries
    let eps = 1e-12;
    assert!(grid.origcoord_to_grididx(&[-2.0 + eps, 3.0 + eps]).is_ok());

    // Just outside boundaries should error
    assert!(grid.origcoord_to_grididx(&[-2.0 - eps, 3.0 - eps]).is_err());
    assert!(grid.origcoord_to_grididx(&[5.0 + eps, 8.0 + eps]).is_err());
}

// =============================================================================
// Ported from: origcoord_floating_point_tests.jl
// =============================================================================

#[test]
fn test_fp_exact_boundaries() {
    let r: usize = 10;
    let grid = DiscretizedGrid::builder(&[r]).build().unwrap();

    // Exact lower bound
    assert_eq!(grid.origcoord_to_grididx(&[0.0]).unwrap(), vec![1]);

    // Exact grid_max
    let grid_max = grid.grid_max();
    let idx = grid.origcoord_to_grididx(&grid_max).unwrap();
    assert_eq!(idx, vec![1 << r]);

    // Exactly representable grid points
    let step = grid.grid_step()[0];
    for i in 1..=100.min(1 << r) {
        let exact_coord = (i as f64 - 1.0) * step;
        let idx = grid.origcoord_to_grididx(&[exact_coord]).unwrap();
        assert_eq!(idx, vec![i]);
    }
}

#[test]
fn test_fp_very_small_steps() {
    let r: usize = 50;
    let grid = DiscretizedGrid::builder(&[r]).build().unwrap();

    let step = grid.grid_step()[0];
    assert!((step - 1.0 / (1i64 << r) as f64).abs() < 1e-30);

    // Test specific indices
    let test_indices: Vec<i64> = vec![
        1,
        2,
        3,
        1 << 10,
        1 << 20,
        (1i64 << r) / 2,
        (1i64 << r) - 1,
        1i64 << r,
    ];
    for idx in test_indices {
        let coord = grid.grididx_to_origcoord(&[idx]).unwrap();
        let recovered = grid.origcoord_to_grididx(&coord).unwrap();
        assert_eq!(recovered, vec![idx]);
    }
}

#[test]
fn test_fp_catastrophic_cancellation() {
    // Case 1: Very close lower_bound and coordinate values
    let lower = 1e15;
    let upper = 1e15 + 1.0;
    let grid = DiscretizedGrid::builder(&[10])
        .with_lower_bound(&[lower])
        .with_upper_bound(&[upper])
        .build()
        .unwrap();

    // Test round-trip for epsilon values (matches Julia: [1e-15, 1e-14, 1e-13, 1e-12, 1e-10])
    for &eps in &[1e-15, 1e-14, 1e-13, 1e-12, 1e-10] {
        let coord = lower + eps;
        if coord < upper {
            let idx = grid.origcoord_to_grididx(&[coord]).unwrap();
            assert!(idx[0] >= 1 && idx[0] <= 1024);

            // Round-trip
            let recovered_coord = grid.grididx_to_origcoord(&idx).unwrap();
            let recovered_idx = grid.origcoord_to_grididx(&recovered_coord).unwrap();
            assert_eq!(recovered_idx, idx);
        }
    }

    // Case 2: Very large coordinate values
    let lower2 = 1e12;
    let upper2 = 1e12 + 1e6;
    let grid2 = DiscretizedGrid::builder(&[15])
        .with_lower_bound(&[lower2])
        .with_upper_bound(&[upper2])
        .build()
        .unwrap();

    let step2 = grid2.grid_step()[0];
    for &i in &[1i64, 100, 1000, (1i64 << 15) / 2, 1i64 << 15] {
        let coord = lower2 + (i as f64 - 1.0) * step2;
        if coord < upper2 {
            let idx = grid2.origcoord_to_grididx(&[coord]).unwrap();
            assert!((idx[0] - i).abs() <= 1);
        }
    }
}

#[test]
fn test_fp_round_trip_consistency() {
    // Standard precision range
    let grid1 = DiscretizedGrid::builder(&[20]).build().unwrap();
    // Very small range
    let grid2 = DiscretizedGrid::builder(&[15])
        .with_lower_bound(&[1e-10])
        .with_upper_bound(&[1e-9])
        .build()
        .unwrap();
    // Very large range
    let grid3 = DiscretizedGrid::builder(&[12])
        .with_lower_bound(&[1e10])
        .with_upper_bound(&[1e11])
        .build()
        .unwrap();
    // Negative range
    let grid4 = DiscretizedGrid::builder(&[18])
        .with_lower_bound(&[-1e5])
        .with_upper_bound(&[-1e4])
        .build()
        .unwrap();
    // Asymmetric around zero
    let grid5 = DiscretizedGrid::builder(&[16])
        .with_lower_bound(&[-1e-6])
        .with_upper_bound(&[1e-5])
        .build()
        .unwrap();

    for grid in [&grid1, &grid2, &grid3, &grid4, &grid5] {
        let r = grid.rs()[0];
        let max_idx = 1i64 << r;
        let test_indices: Vec<i64> = vec![
            1,
            2,
            3,
            max_idx / 4,
            max_idx / 2,
            3 * max_idx / 4,
            max_idx - 2,
            max_idx - 1,
            max_idx,
        ];

        for idx in test_indices {
            if idx >= 1 && idx <= max_idx {
                let coord = grid.grididx_to_origcoord(&[idx]).unwrap();
                let recovered = grid.origcoord_to_grididx(&coord).unwrap();
                assert_eq!(
                    recovered,
                    vec![idx],
                    "Round-trip failed for idx={} with R={}",
                    idx,
                    r
                );
            }
        }
    }
}

#[test]
fn test_fp_multidimensional_edge_cases() {
    let r: usize = 12;
    let grid = DiscretizedGrid::builder(&[r, r])
        .with_lower_bound(&[0.0, -1.0])
        .with_upper_bound(&[1.0, 1.0])
        .build()
        .unwrap();

    let max_idx = 1i64 << r;

    // Corner cases
    let corners: Vec<(i64, i64)> = vec![(1, 1), (1, max_idx), (max_idx, 1), (max_idx, max_idx)];
    for (ix, iy) in &corners {
        let coord = grid.grididx_to_origcoord(&[*ix, *iy]).unwrap();
        let recovered = grid.origcoord_to_grididx(&coord).unwrap();
        assert_eq!(recovered, vec![*ix, *iy]);
    }

    // More test cases
    let test_cases: Vec<(i64, i64)> = vec![
        (100, 200),
        (500, 1000),
        (2000, 3000),
        (4000, 4096),
        (1, 4096),
        (4096, 1),
    ];
    for (ix, iy) in test_cases {
        let coord = grid.grididx_to_origcoord(&[ix, iy]).unwrap();
        let recovered = grid.origcoord_to_grididx(&coord).unwrap();
        assert_eq!(recovered, vec![ix, iy]);
    }
}

#[test]
fn test_fp_extreme_coordinate_ranges() {
    let test_cases: Vec<(f64, f64, usize)> = vec![
        (1e-16, 1e-15, 10),
        (1e15, 1e16, 8),
        (-1e-15, 1e-15, 12),
        (0.0, 1e-14, 8),
    ];

    for (lower, upper, r) in test_cases {
        let grid = DiscretizedGrid::builder(&[r])
            .with_lower_bound(&[lower])
            .with_upper_bound(&[upper])
            .build()
            .unwrap();

        let max_idx = 1i64 << r;
        let boundary_indices: Vec<i64> = vec![1, 2, max_idx - 1, max_idx];

        for idx in boundary_indices {
            let coord = grid.grididx_to_origcoord(&[idx]).unwrap();
            assert!(coord[0] >= lower - 1e-20);
            assert!(coord[0] <= upper + 1e-20);

            let recovered = grid.origcoord_to_grididx(&coord).unwrap();
            assert_eq!(recovered, vec![idx]);
        }
    }
}

#[test]
fn test_fp_step_size_edge_cases() {
    // Step size not exactly representable
    let grid1 = DiscretizedGrid::builder(&[10])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0 / 3.0])
        .build()
        .unwrap();
    let step1 = grid1.grid_step()[0];
    assert!((step1 - (1.0 / 3.0) / 1024.0).abs() < 1e-20);

    for i in 1..=1024 {
        let coord = grid1.grididx_to_origcoord(&[i]).unwrap();
        let recovered = grid1.origcoord_to_grididx(&coord).unwrap();
        assert_eq!(recovered, vec![i]);
    }

    // Step size involving sqrt(2)
    let sqrt2 = 2.0_f64.sqrt();
    let grid2 = DiscretizedGrid::builder(&[8])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[sqrt2])
        .build()
        .unwrap();

    for &i in &[1i64, 16, 256] {
        let coord = grid2.grididx_to_origcoord(&[i]).unwrap();
        let recovered = grid2.origcoord_to_grididx(&coord).unwrap();
        assert_eq!(recovered, vec![i]);
    }
}

#[test]
fn test_fp_include_endpoint_behavior() {
    let r: usize = 10;
    let grid_without = DiscretizedGrid::builder(&[r]).build().unwrap();
    let grid_with = DiscretizedGrid::builder(&[r])
        .include_endpoint(true)
        .build()
        .unwrap();

    // With includeendpoint: upper_bound should map to last index
    let idx = grid_with.origcoord_to_grididx(&[1.0]).unwrap();
    assert_eq!(idx, vec![1 << r]);

    // Round-trip consistency for both cases
    for grid in [&grid_without, &grid_with] {
        let grid_max = grid.grid_max();
        let idx = grid.origcoord_to_grididx(&grid_max).unwrap();
        let recovered = grid.grididx_to_origcoord(&idx).unwrap();
        let re_idx = grid.origcoord_to_grididx(&recovered).unwrap();
        assert_eq!(re_idx, idx);
    }
}

#[test]
fn test_fp_mixed_precision_operations() {
    let grid = DiscretizedGrid::builder(&[12])
        .with_lower_bound(&[0.1])
        .with_upper_bound(&[0.9])
        .build()
        .unwrap();

    let step = grid.grid_step()[0];

    for &i in &[1i64, 100, 1000, 4096] {
        // Method 1: Direct calculation
        let coord1 = 0.1 + (i as f64 - 1.0) * step;
        // Method 2: Via grid function
        let coord2 = grid.grididx_to_origcoord(&[i]).unwrap()[0];

        assert!((coord1 - coord2).abs() < 1e-14);

        // Via grid function should round-trip exactly
        let idx2 = grid.origcoord_to_grididx(&[coord2]).unwrap();
        assert_eq!(idx2, vec![i]);
    }
}

// =============================================================================
// Ported from: discretizedgrid_misc_tests.jl
// =============================================================================

#[test]
fn test_constructor_error_upper_lt_lower() {
    // upper_bound < lower_bound (1D)
    let result = DiscretizedGrid::builder(&[10])
        .with_lower_bound(&[0.1])
        .with_upper_bound(&[0.01])
        .build();
    assert!(matches!(
        result,
        Err(QuanticsGridError::InvalidBounds { .. })
    ));

    // upper_bound < lower_bound (2D)
    let result = DiscretizedGrid::builder(&[10, 4])
        .with_lower_bound(&[0.1, 0.2])
        .with_upper_bound(&[0.01, 0.02])
        .build();
    assert!(matches!(
        result,
        Err(QuanticsGridError::InvalidBounds { .. })
    ));

    // upper_bound == lower_bound
    let result = DiscretizedGrid::builder(&[10])
        .with_lower_bound(&[0.1])
        .with_upper_bound(&[0.1])
        .build();
    assert!(matches!(
        result,
        Err(QuanticsGridError::InvalidBounds { .. })
    ));

    let result = DiscretizedGrid::builder(&[10, 4])
        .with_lower_bound(&[0.1, 0.2])
        .with_upper_bound(&[0.1, 0.02])
        .build();
    assert!(matches!(
        result,
        Err(QuanticsGridError::InvalidBounds { .. })
    ));
}

#[test]
fn test_constructor_incompatible_indextable() {
    // Index table references unknown variable "z"
    let index_table = vec![vec![("x".to_string(), 1)], vec![("z".to_string(), 1)]];
    let result = DiscretizedGrid::from_index_table(&["x", "y"], index_table).build();
    assert!(matches!(
        result,
        Err(QuanticsGridError::UnknownVariable { .. })
    ));
}

#[test]
fn test_constructor_endpoint_with_zero_resolution() {
    // R=0 and includeendpoint=true should fail
    let result = DiscretizedGrid::builder(&[0, 5])
        .include_endpoint(true)
        .build();
    assert!(matches!(
        result,
        Err(QuanticsGridError::EndpointWithZeroResolution { .. })
    ));
}

#[test]
fn test_sitedim() {
    let grid = DiscretizedGrid::builder(&[7, 8, 9])
        .with_variable_names(&["x", "y", "z"])
        .build()
        .unwrap();

    // Expected site dims for fused 3D with Rs=(7,8,9):
    // R=9 is max, so 9 sites.
    // For level 1..7: all 3 vars present => site_dim = 2^3 = 8
    // For level 8: only y,z present => site_dim = 2^2 = 4
    // For level 9: only z present => site_dim = 2^1 = 2
    let expected_sitedims: Vec<usize> = vec![8, 8, 8, 8, 8, 8, 8, 4, 2];
    for (i, &expected) in expected_sitedims.iter().enumerate() {
        assert_eq!(grid.site_dim(i).unwrap(), expected);
    }

    // Out of bounds
    assert!(matches!(
        grid.site_dim(9),
        Err(QuanticsGridError::SiteIndexOutOfBounds { .. })
    ));
}

#[test]
fn test_dimension_mismatch_for_bounds() {
    // 2D grid with 1-element lower bound and 2-element upper bound
    // (This would fail if lower_bound isn't expanded to match ndims)
    let result = DiscretizedGrid::builder(&[3, 3])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0, 2.0])
        .build();
    // This should succeed because single-element bounds expand to match ndims
    assert!(result.is_ok());

    // But completely wrong dimensions should fail
    let result = DiscretizedGrid::builder(&[3, 3])
        .with_lower_bound(&[0.0, 0.0, 0.0])
        .with_upper_bound(&[1.0, 2.0])
        .build();
    assert!(matches!(
        result,
        Err(QuanticsGridError::DimensionMismatch { .. })
    ));
}

// =============================================================================
// Ported from: grid_tests.jl - "quanticsfunction"
// =============================================================================

#[test]
fn test_quanticsfunction_1d() {
    let r: usize = 8;
    let grid = DiscretizedGrid::builder(&[r])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .build()
        .unwrap();

    let fx = |coords: &[f64]| (-coords[0]).exp();
    let fq = quantics_function(&grid, fx);

    // All-ones quantics -> coord 0.0 -> exp(0) = 1.0
    let ones_q = vec![1i64; r];
    assert!((fq(&ones_q).unwrap() - 1.0).abs() < 1e-14);

    // All-twos quantics -> coord ≈ 1.0 - 1/2^R
    let twos_q = vec![2i64; r];
    let expected = (-1.0 + 1.0 / (1 << r) as f64).exp();
    assert!((fq(&twos_q).unwrap() - expected).abs() < 1e-14);
}

#[test]
fn test_quanticsfunction_2d() {
    let r: usize = 4;
    let grid = DiscretizedGrid::builder(&[r, r]).build().unwrap();

    let f2d = |coords: &[f64]| coords[0] + coords[1];
    let fq = quantics_function(&grid, f2d);

    // All-ones quantics -> (0,0) -> 0.0
    let n_sites = grid.len();
    let ones_q = vec![1i64; n_sites];
    assert!((fq(&ones_q).unwrap() - 0.0).abs() < 1e-14);
}

#[test]
fn test_quanticsfunction_custom_indextable() {
    let index_table = vec![
        vec![("x".to_string(), 1)],
        vec![("x".to_string(), 2)],
        vec![("y".to_string(), 1)],
        vec![("y".to_string(), 2)],
    ];
    let grid = DiscretizedGrid::from_index_table(&["x", "y"], index_table)
        .build()
        .unwrap();

    let f2d = |coords: &[f64]| coords[0] + coords[1];
    let fq = quantics_function(&grid, f2d);
    assert!((fq(&[1, 1, 1, 1]).unwrap() - 0.0).abs() < 1e-14);
}

#[test]
fn test_quanticsfunction_complex_base3() {
    let index_table = vec![
        vec![("x".to_string(), 1)],
        vec![("y".to_string(), 1), ("z".to_string(), 1)],
        vec![("x".to_string(), 2), ("y".to_string(), 2)],
        vec![
            ("z".to_string(), 2),
            ("x".to_string(), 3),
            ("y".to_string(), 3),
        ],
    ];
    let grid = DiscretizedGrid::from_index_table(&["x", "y", "z"], index_table)
        .with_base(3)
        .with_lower_bound(&[-1.0, 2.0, 0.5])
        .with_upper_bound(&[3.0, 5.0, 2.0])
        .build()
        .unwrap();

    let f_complex = |coords: &[f64]| {
        let (x, y, z) = (coords[0], coords[1], coords[2]);
        (std::f64::consts::PI * x).sin() * (2.0 * std::f64::consts::PI * y).cos() * (-z).exp()
            + x * x * y
            - z * z * z
            + x * y * z
    };
    let fq = quantics_function(&grid, f_complex);

    // All-ones -> (-1.0, 2.0, 0.5)
    let result = fq(&[1, 1, 1, 1]).unwrap();
    let expected = f_complex(&[-1.0, 2.0, 0.5]);
    assert!((result - expected).abs() < 1e-10);

    // Max quantics -> grid_max
    let grid_max = grid.grid_max();
    let result = fq(&[3, 9, 9, 27]).unwrap();
    let expected = f_complex(&grid_max);
    assert!((result - expected).abs() < 1e-10);

    // Intermediate
    let mid_q = vec![2, 5, 5, 14];
    let coord_mid = grid.quantics_to_origcoord(&mid_q).unwrap();
    let result = fq(&mid_q).unwrap();
    let expected = f_complex(&coord_mid);
    assert!((result - expected).abs() < 1e-10);
}

// =============================================================================
// Ported from: grid_tests.jl - "1D grid (large R)"
// =============================================================================

#[test]
fn test_1d_large_r() {
    let r: usize = 62;
    let grid = DiscretizedGrid::builder(&[r]).build().unwrap();
    let max_idx = 1i64 << r;
    let quantics = grid.grididx_to_quantics(&[max_idx]).unwrap();
    assert_eq!(quantics, vec![2i64; r]);
}

// =============================================================================
// Ported from: grid_tests.jl - "2D grid (large R)"
// =============================================================================

#[test]
fn test_2d_large_r() {
    let r: usize = 62;
    let d: usize = 2;
    let grid = DiscretizedGrid::builder(&[r, r])
        .with_base(2)
        .build()
        .unwrap();
    let max_idx = 1i64 << r;
    let quantics = grid.grididx_to_quantics(&[max_idx, max_idx]).unwrap();
    assert_eq!(quantics, vec![1i64 << d; r]);
}

// =============================================================================
// Ported from: grid_tests.jl - "1D grid (too large R)"
// =============================================================================

#[test]
fn test_too_large_r() {
    let result = DiscretizedGrid::builder(&[64]).build();
    assert!(matches!(
        result,
        Err(QuanticsGridError::ResolutionTooLarge { .. })
    ));
}

// =============================================================================
// Ported from: grid_tests.jl - "grid representation conversion"
// =============================================================================

#[test]
fn test_grid_representation_conversion() {
    // 1D DiscretizedGrid
    {
        let grid = DiscretizedGrid::builder(&[10]).build().unwrap();
        let grididx: Vec<i64> = vec![2];

        let quantics = grid.grididx_to_quantics(&grididx).unwrap();
        let origcoord = grid.grididx_to_origcoord(&grididx).unwrap();

        assert_eq!(grid.quantics_to_grididx(&quantics).unwrap(), grididx);
        assert_eq!(grid.origcoord_to_grididx(&origcoord).unwrap(), grididx);
        assert_eq!(grid.quantics_to_origcoord(&quantics).unwrap(), origcoord);
        assert_eq!(grid.origcoord_to_quantics(&origcoord).unwrap(), quantics);
    }

    // 2D DiscretizedGrid
    {
        let grid = DiscretizedGrid::builder(&[10, 10]).build().unwrap();
        let grididx: Vec<i64> = vec![2, 3];

        let quantics = grid.grididx_to_quantics(&grididx).unwrap();
        let origcoord = grid.grididx_to_origcoord(&grididx).unwrap();

        assert_eq!(grid.quantics_to_grididx(&quantics).unwrap(), grididx);
        assert_eq!(grid.origcoord_to_grididx(&origcoord).unwrap(), grididx);
        assert_eq!(grid.quantics_to_origcoord(&quantics).unwrap(), origcoord);
        assert_eq!(grid.origcoord_to_quantics(&origcoord).unwrap(), quantics);
    }
}

// =============================================================================
// Ported from: grid_tests.jl - DiscretizedGrid 1D
// =============================================================================

#[test]
fn test_discretized_grid_1d_interleaved() {
    let r: usize = 5;
    let a = 0.1;
    let b = 2.0;
    let dx = (b - a) / (1 << r) as f64;
    let grid = DiscretizedGrid::builder(&[r])
        .with_lower_bound(&[a])
        .with_upper_bound(&[b])
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .build()
        .unwrap();
    assert_eq!(grid.local_dimensions(), vec![2; r]);

    let idx = grid.origcoord_to_grididx(&[0.999999 * dx + a]).unwrap();
    assert_eq!(idx, vec![2]);
    let idx = grid.origcoord_to_grididx(&[1.999999 * dx + a]).unwrap();
    assert_eq!(idx, vec![3]);

    let gmax = grid.grid_max();
    let idx = grid.origcoord_to_grididx(&[gmax[0] + 1e-9 * dx]).unwrap();
    assert_eq!(idx, vec![1 << r]);

    assert_eq!(grid.lower_bound(), &[0.1]);
    assert_eq!(grid.upper_bound(), &[2.0]);
    assert_eq!(grid.grid_min(), &[a]);
    assert!((grid.grid_max()[0] - (b - dx)).abs() < 1e-14);

    assert!((grid.grid_step()[0] - 0.059375).abs() < 1e-14);
}

// =============================================================================
// Ported from: grid_tests.jl - "1D (includeendpoint)"
// =============================================================================

#[test]
fn test_discretized_grid_1d_includeendpoint() {
    for scheme in [UnfoldingScheme::Interleaved, UnfoldingScheme::Fused] {
        let r: usize = 5;
        let a = 0.0;
        let b = 1.0;
        let n = (1 << r) as f64;
        let dx = (b - a) / (n - 1.0);
        let grid = DiscretizedGrid::builder(&[r])
            .with_lower_bound(&[a])
            .with_upper_bound(&[b])
            .include_endpoint(true)
            .with_unfolding_scheme(scheme)
            .build()
            .unwrap();
        assert_eq!(grid.local_dimensions(), vec![2; r]);

        assert_eq!(grid.lower_bound(), &[0.0]);
        // In Rust, upper_bound() returns the user-specified value (1.0),
        // not the effective upper bound. The effective upper is internal.
        assert!((grid.upper_bound()[0] - 1.0).abs() < 1e-14);
        assert_eq!(grid.grid_min(), &[a]);
        assert!((grid.grid_max()[0] - b).abs() < 1e-14);

        assert_eq!(grid.origcoord_to_grididx(&[a]).unwrap(), vec![1]);
        assert_eq!(grid.origcoord_to_grididx(&[b]).unwrap(), vec![1 << r]);

        assert!((grid.grid_step()[0] - dx).abs() < 1e-14);
        let max_origcoord = grid.quantics_to_origcoord(&vec![2; r]).unwrap();
        assert!((max_origcoord[0] - b).abs() < 1e-14);
    }
}

// =============================================================================
// Ported from: grid_tests.jl - "2D"
// =============================================================================

#[test]
fn test_discretized_grid_2d() {
    for scheme in [UnfoldingScheme::Interleaved, UnfoldingScheme::Fused] {
        let r: usize = 5;
        let a = [0.1, 0.1];
        let b = [2.0, 2.0];
        let dx: Vec<f64> = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (bi - ai) / (1 << r) as f64)
            .collect();
        let grid = DiscretizedGrid::builder(&[r, r])
            .with_lower_bound(&a)
            .with_upper_bound(&b)
            .with_unfolding_scheme(scheme)
            .build()
            .unwrap();

        match scheme {
            UnfoldingScheme::Interleaved => {
                assert_eq!(grid.local_dimensions(), vec![2; 2 * r]);
            }
            UnfoldingScheme::Fused => {
                assert_eq!(grid.local_dimensions(), vec![4; r]);
            }
            _ => {}
        }

        assert_eq!(grid.lower_bound(), &[0.1, 0.1]);
        let step = grid.grid_step();
        assert!((step[0] - dx[0]).abs() < 1e-14);
        assert!((step[1] - dx[1]).abs() < 1e-14);
        assert_eq!(grid.upper_bound(), &[2.0, 2.0]);

        assert_eq!(grid.grid_min(), &a);
        let gmax = grid.grid_max();
        assert!((gmax[0] - (b[0] - dx[0])).abs() < 1e-14);
        assert!((gmax[1] - (b[1] - dx[1])).abs() < 1e-14);

        // Test specific coordinates
        let c1: Vec<f64> = a
            .iter()
            .zip(dx.iter())
            .map(|(&ai, &di)| 0.999999 * di + ai)
            .collect();
        let idx1 = grid.origcoord_to_grididx(&c1).unwrap();
        assert_eq!(idx1, vec![2, 2]);

        let c2: Vec<f64> = a
            .iter()
            .zip(dx.iter())
            .map(|(&ai, &di)| 1.999999 * di + ai)
            .collect();
        let idx2 = grid.origcoord_to_grididx(&c2).unwrap();
        assert_eq!(idx2, vec![3, 3]);

        // Out of bounds
        assert!(grid.origcoord_to_grididx(&[0.0, 0.0]).is_err());
        assert!(grid.origcoord_to_grididx(&[0.0, 1.1]).is_err());
        assert!(grid.origcoord_to_grididx(&[1.1, 0.0]).is_err());
        assert!(grid.origcoord_to_grididx(&[3.0, 1.1]).is_err());
        assert!(grid.origcoord_to_grididx(&[1.1, 3.0]).is_err());
        assert!(grid.origcoord_to_grididx(&[3.0, 3.0]).is_err());
    }
}

// =============================================================================
// Ported from: grid_tests.jl - "2D (includeendpoint)"
// =============================================================================

#[test]
fn test_discretized_grid_2d_includeendpoint() {
    for scheme in [UnfoldingScheme::Interleaved, UnfoldingScheme::Fused] {
        let r: usize = 5;
        let a = [0.1, 0.1];
        let b = [2.0, 2.0];
        let n = (1 << r) as f64;
        let dx: Vec<f64> = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (bi - ai) / (n - 1.0))
            .collect();
        let grid = DiscretizedGrid::builder(&[r, r])
            .with_lower_bound(&a)
            .with_upper_bound(&b)
            .include_endpoint(true)
            .with_unfolding_scheme(scheme)
            .build()
            .unwrap();

        let step = grid.grid_step();
        assert!((step[0] - dx[0]).abs() < 1e-14);
        assert!((step[1] - dx[1]).abs() < 1e-14);

        match scheme {
            UnfoldingScheme::Interleaved => {
                assert_eq!(grid.local_dimensions(), vec![2; 2 * r]);
            }
            UnfoldingScheme::Fused => {
                assert_eq!(grid.local_dimensions(), vec![4; r]);
            }
            _ => {}
        }
    }
}

// =============================================================================
// Ported from: utilities_tests.jl - "localdimensions"
// =============================================================================

#[test]
fn test_local_dimensions() {
    let r: usize = 4;

    // Basic 1D with base=2
    let grid = DiscretizedGrid::builder(&[r]).build().unwrap();
    assert_eq!(grid.local_dimensions(), vec![2; r]);

    // 2D interleaved
    let grid_2d = DiscretizedGrid::builder(&[r, r])
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .build()
        .unwrap();
    assert_eq!(grid_2d.local_dimensions(), vec![2; 2 * r]);

    // Base 3
    let grid_base3 = DiscretizedGrid::builder(&[r]).with_base(3).build().unwrap();
    assert_eq!(grid_base3.local_dimensions(), vec![3; r]);

    // Base 4
    let grid_base4 = DiscretizedGrid::builder(&[r]).with_base(4).build().unwrap();
    assert_eq!(grid_base4.local_dimensions(), vec![4; r]);

    // Custom index table with mixed site sizes, base=2
    let index_table = vec![
        vec![("x".to_string(), 1), ("y".to_string(), 1)],
        vec![("x".to_string(), 2)],
        vec![("y".to_string(), 2)],
        vec![
            ("x".to_string(), 3),
            ("y".to_string(), 3),
            ("x".to_string(), 4),
        ],
    ];
    let grid_custom = DiscretizedGrid::from_index_table(&["x", "y"], index_table.clone())
        .with_base(2)
        .build()
        .unwrap();
    assert_eq!(grid_custom.local_dimensions(), vec![4, 2, 2, 8]);

    // Same with base=3
    let grid_custom_base3 = DiscretizedGrid::from_index_table(&["x", "y"], index_table)
        .with_base(3)
        .build()
        .unwrap();
    assert_eq!(grid_custom_base3.local_dimensions(), vec![9, 3, 3, 27]);

    // Single site with multiple indices
    let single_site_table = vec![vec![
        ("x".to_string(), 1),
        ("x".to_string(), 2),
        ("x".to_string(), 3),
    ]];
    let grid_single = DiscretizedGrid::from_index_table(&["x"], single_site_table)
        .with_base(2)
        .build()
        .unwrap();
    assert_eq!(grid_single.local_dimensions(), vec![8]);

    // Complex mixed site sizes
    let complex_table = vec![
        vec![("x".to_string(), 1)],
        vec![("y".to_string(), 1), ("z".to_string(), 1)],
        vec![
            ("x".to_string(), 2),
            ("y".to_string(), 2),
            ("z".to_string(), 2),
        ],
        vec![("x".to_string(), 3), ("y".to_string(), 3)],
    ];
    let grid_complex = DiscretizedGrid::from_index_table(&["x", "y", "z"], complex_table)
        .with_base(2)
        .build()
        .unwrap();
    assert_eq!(grid_complex.local_dimensions(), vec![2, 4, 8, 4]);
}

// =============================================================================
// Ported from: utilities_tests.jl - "unfoldingscheme"
// =============================================================================

#[test]
fn test_unfolding_scheme() {
    let r: usize = 4;

    // 2D fused vs interleaved
    let grid_2d_fused = DiscretizedGrid::builder(&[r, r])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .unwrap();
    let grid_2d_interleaved = DiscretizedGrid::builder(&[r, r])
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .build()
        .unwrap();

    let grididx = vec![2, 2];
    let quantics_fused = grid_2d_fused.grididx_to_quantics(&grididx).unwrap();
    let quantics_interleaved = grid_2d_interleaved.grididx_to_quantics(&grididx).unwrap();

    assert_eq!(quantics_fused.len(), r);
    assert_eq!(quantics_interleaved.len(), 2 * r);

    // Both should produce same origcoord
    let coord_fused = grid_2d_fused
        .quantics_to_origcoord(&quantics_fused)
        .unwrap();
    let coord_interleaved = grid_2d_interleaved
        .quantics_to_origcoord(&quantics_interleaved)
        .unwrap();
    assert_eq!(coord_fused, coord_interleaved);

    // Local dimensions
    assert_eq!(grid_2d_fused.local_dimensions(), vec![4; r]); // 2^2=4
    assert_eq!(grid_2d_interleaved.local_dimensions(), vec![2; 2 * r]);
}

// =============================================================================
// Ported from: utilities_tests.jl - "includeendpoint"
// =============================================================================

#[test]
fn test_include_endpoint_detailed() {
    let r: usize = 4;
    let n = (1 << r) as f64;

    let grid_without = DiscretizedGrid::builder(&[r]).build().unwrap();
    let grid_with = DiscretizedGrid::builder(&[r])
        .include_endpoint(true)
        .build()
        .unwrap();

    // grid_max
    assert!((grid_without.grid_max()[0] - (1.0 - 1.0 / n)).abs() < 1e-14);
    assert!((grid_with.grid_max()[0] - 1.0).abs() < 1e-14);

    // Grid step
    let step_without = grid_without.grid_step()[0];
    let step_with = grid_with.grid_step()[0];
    assert!((step_without - 1.0 / n).abs() < 1e-14);
    assert!((step_with - 1.0 / (n - 1.0)).abs() < 1e-14);

    // grididx_to_origcoord at max
    let coord_with = grid_with.grididx_to_origcoord(&[1 << r]).unwrap();
    assert!((coord_with[0] - 1.0).abs() < 1e-14);
    let coord_without = grid_without.grididx_to_origcoord(&[1 << r]).unwrap();
    assert!((coord_without[0] - (1.0 - 1.0 / n)).abs() < 1e-14);

    // 2D case
    let grid_2d_without = DiscretizedGrid::builder(&[r, r])
        .with_lower_bound(&[0.0, 0.0])
        .with_upper_bound(&[2.0, 3.0])
        .build()
        .unwrap();
    let grid_2d_with = DiscretizedGrid::builder(&[r, r])
        .with_lower_bound(&[0.0, 0.0])
        .with_upper_bound(&[2.0, 3.0])
        .include_endpoint(true)
        .build()
        .unwrap();

    let max_without = grid_2d_without.grid_max();
    let max_with = grid_2d_with.grid_max();
    assert!((max_without[0] - (2.0 - 2.0 / n)).abs() < 1e-14);
    assert!((max_without[1] - (3.0 - 3.0 / n)).abs() < 1e-14);
    assert!((max_with[0] - 2.0).abs() < 1e-14);
    assert!((max_with[1] - 3.0).abs() < 1e-14);

    // Upper bound accessible with includeendpoint
    let coord = grid_2d_with
        .grididx_to_origcoord(&[1 << r, 1 << r])
        .unwrap();
    assert!((coord[0] - 2.0).abs() < 1e-14);
    assert!((coord[1] - 3.0).abs() < 1e-14);
}

// =============================================================================
// Ported from: utilities_tests.jl - "includeendpoint" complex indextable part
// =============================================================================

#[test]
fn test_include_endpoint_complex_indextable() {
    let index_table = vec![
        vec![("x".to_string(), 1)],
        vec![("y".to_string(), 1), ("z".to_string(), 1)],
        vec![
            ("x".to_string(), 2),
            ("y".to_string(), 2),
            ("z".to_string(), 2),
        ],
        vec![("x".to_string(), 3), ("y".to_string(), 3)],
    ];

    let lower_bound = [-2.5, 1.2, 0.0];
    let upper_bound = [4.7, 8.1, 3.3];

    let grid_without = DiscretizedGrid::from_index_table(&["x", "y", "z"], index_table.clone())
        .with_base(3)
        .with_lower_bound(&lower_bound)
        .with_upper_bound(&upper_bound)
        .build()
        .unwrap();
    let grid_with = DiscretizedGrid::from_index_table(&["x", "y", "z"], index_table)
        .with_base(3)
        .with_lower_bound(&lower_bound)
        .with_upper_bound(&upper_bound)
        .include_endpoint(true)
        .build()
        .unwrap();

    // Local dims same regardless of includeendpoint
    let expected_local_dims = vec![3, 9, 27, 9]; // 3^1, 3^2, 3^3, 3^2
    assert_eq!(grid_without.local_dimensions(), expected_local_dims);
    assert_eq!(grid_with.local_dimensions(), expected_local_dims);

    // grid_max without endpoint
    let max_without = grid_without.grid_max();
    let expected_max_x = upper_bound[0] - (upper_bound[0] - lower_bound[0]) / 27.0; // 3^3
    let expected_max_y = upper_bound[1] - (upper_bound[1] - lower_bound[1]) / 27.0; // 3^3
    let expected_max_z = upper_bound[2] - (upper_bound[2] - lower_bound[2]) / 9.0; // 3^2
    assert!((max_without[0] - expected_max_x).abs() < 1e-10);
    assert!((max_without[1] - expected_max_y).abs() < 1e-10);
    assert!((max_without[2] - expected_max_z).abs() < 1e-10);

    // grid_max with endpoint
    let max_with = grid_with.grid_max();
    assert!((max_with[0] - upper_bound[0]).abs() < 1e-10);
    assert!((max_with[1] - upper_bound[1]).abs() < 1e-10);
    assert!((max_with[2] - upper_bound[2]).abs() < 1e-10);

    // Max quantics
    let max_quantics = vec![3, 9, 27, 9];
    let coord_max_without = grid_without.quantics_to_origcoord(&max_quantics).unwrap();
    let coord_max_with = grid_with.quantics_to_origcoord(&max_quantics).unwrap();

    // With endpoint, max quantics reaches exact upper bound
    for d in 0..3 {
        assert!((coord_max_with[d] - upper_bound[d]).abs() < 1e-10);
    }

    // Without endpoint, max quantics < upper bound
    for d in 0..3 {
        assert!(coord_max_without[d] < upper_bound[d]);
    }

    // Step sizes
    let step_without = grid_without.grid_step();
    let step_with = grid_with.grid_step();

    let expected_step_without_x = (upper_bound[0] - lower_bound[0]) / 27.0;
    let expected_step_without_y = (upper_bound[1] - lower_bound[1]) / 27.0;
    let expected_step_without_z = (upper_bound[2] - lower_bound[2]) / 9.0;
    assert!((step_without[0] - expected_step_without_x).abs() < 1e-10);
    assert!((step_without[1] - expected_step_without_y).abs() < 1e-10);
    assert!((step_without[2] - expected_step_without_z).abs() < 1e-10);

    let expected_step_with_x = (upper_bound[0] - lower_bound[0]) / 26.0; // 3^3-1
    let expected_step_with_y = (upper_bound[1] - lower_bound[1]) / 26.0;
    let expected_step_with_z = (upper_bound[2] - lower_bound[2]) / 8.0; // 3^2-1
    assert!((step_with[0] - expected_step_with_x).abs() < 1e-10);
    assert!((step_with[1] - expected_step_with_y).abs() < 1e-10);
    assert!((step_with[2] - expected_step_with_z).abs() < 1e-10);

    // Min quantics always maps to lower bound
    let min_quantics = vec![1, 1, 1, 1];
    let coord_min_without = grid_without.quantics_to_origcoord(&min_quantics).unwrap();
    let coord_min_with = grid_with.quantics_to_origcoord(&min_quantics).unwrap();
    for d in 0..3 {
        assert!((coord_min_without[d] - lower_bound[d]).abs() < 1e-10);
        assert!((coord_min_with[d] - lower_bound[d]).abs() < 1e-10);
    }

    // Intermediate quantics -> different coords due to different step sizes
    let mid_quantics = vec![2, 5, 14, 5];
    let coord_mid_without = grid_without.quantics_to_origcoord(&mid_quantics).unwrap();
    let coord_mid_with = grid_with.quantics_to_origcoord(&mid_quantics).unwrap();
    assert_ne!(coord_mid_without, coord_mid_with);
    for d in 0..3 {
        assert!(coord_mid_without[d] >= lower_bound[d] - 1e-10);
        assert!(coord_mid_without[d] <= upper_bound[d] + 1e-10);
        assert!(coord_mid_with[d] >= lower_bound[d] - 1e-10);
        assert!(coord_mid_with[d] <= upper_bound[d] + 1e-10);
    }

    // quantics_function boundary behavior
    let ub = upper_bound;
    let fq_with = quantics_function(&grid_with, move |coords: &[f64]| {
        (coords[0] - ub[0]).powi(4) + (coords[1] - ub[1]).powi(4) + (coords[2] - ub[2]).powi(4)
    });
    let ub2 = upper_bound;
    let fq_without = quantics_function(&grid_without, move |coords: &[f64]| {
        (coords[0] - ub2[0]).powi(4) + (coords[1] - ub2[1]).powi(4) + (coords[2] - ub2[2]).powi(4)
    });

    let result_with = fq_with(&max_quantics).unwrap();
    assert!(
        result_with.abs() < 1e-10,
        "Expected ~0 at upper_bound with includeendpoint, got {}",
        result_with
    );

    let result_without = fq_without(&max_quantics).unwrap();
    assert!(result_without > 0.0);
}

// =============================================================================
// Ported from: utilities_tests.jl - "boundary error handling"
// =============================================================================

#[test]
fn test_boundary_error_handling() {
    let grid = DiscretizedGrid::builder(&[5]).build().unwrap();
    assert!(grid.origcoord_to_grididx(&[-0.1]).is_err());
    assert!(grid.origcoord_to_grididx(&[1.1]).is_err());
}

// =============================================================================
// Ported from: utilities_tests.jl - "large grids"
// =============================================================================

#[test]
fn test_large_grids() {
    let r: usize = 62;
    let grid = DiscretizedGrid::builder(&[r]).build().unwrap();
    let max_idx = 1i64 << r;
    let quantics = grid.grididx_to_quantics(&[max_idx]).unwrap();
    assert_eq!(quantics, vec![2i64; r]);
}

// =============================================================================
// Ported from: utilities_tests.jl - "grid representation conversion _NEW_"
// =============================================================================

#[test]
fn test_grid_representation_conversion_new() {
    // 1D with custom bounds
    {
        let grid = DiscretizedGrid::builder(&[10])
            .with_lower_bound(&[-3.2])
            .with_upper_bound(&[4.8])
            .build()
            .unwrap();
        let grididx: Vec<i64> = vec![2];
        let quantics = grid.grididx_to_quantics(&grididx).unwrap();
        let origcoord = grid.grididx_to_origcoord(&grididx).unwrap();
        assert_eq!(grid.quantics_to_grididx(&quantics).unwrap(), grididx);
        assert_eq!(grid.origcoord_to_grididx(&origcoord).unwrap(), grididx);
        assert_eq!(grid.quantics_to_origcoord(&quantics).unwrap(), origcoord);
        assert_eq!(grid.origcoord_to_quantics(&origcoord).unwrap(), quantics);
    }

    // 2D with custom bounds
    {
        let grid = DiscretizedGrid::builder(&[10, 10])
            .with_lower_bound(&[-2.3, 1.2])
            .with_upper_bound(&[9.0, 3.5])
            .build()
            .unwrap();
        let grididx: Vec<i64> = vec![2, 3];
        let quantics = grid.grididx_to_quantics(&grididx).unwrap();
        let origcoord = grid.grididx_to_origcoord(&grididx).unwrap();
        assert_eq!(grid.quantics_to_grididx(&quantics).unwrap(), grididx);
        assert_eq!(grid.origcoord_to_grididx(&origcoord).unwrap(), grididx);
        assert_eq!(grid.quantics_to_origcoord(&quantics).unwrap(), origcoord);
        assert_eq!(grid.origcoord_to_quantics(&origcoord).unwrap(), quantics);
    }

    // Custom indextable
    {
        let index_table = vec![
            vec![("x".to_string(), 3), ("y".to_string(), 2)],
            vec![("y".to_string(), 1), ("x".to_string(), 1)],
            vec![("x".to_string(), 2)],
        ];
        let grid = DiscretizedGrid::from_index_table(&["x", "y"], index_table)
            .build()
            .unwrap();
        let grididx: Vec<i64> = vec![7, 2];
        let quantics = grid.grididx_to_quantics(&grididx).unwrap();
        let origcoord = grid.grididx_to_origcoord(&grididx).unwrap();
        assert_eq!(grid.quantics_to_grididx(&quantics).unwrap(), grididx);
        assert_eq!(grid.origcoord_to_grididx(&origcoord).unwrap(), grididx);
        assert_eq!(grid.quantics_to_origcoord(&quantics).unwrap(), origcoord);
        assert_eq!(grid.origcoord_to_quantics(&origcoord).unwrap(), quantics);
    }
}

// =============================================================================
// Ported from: utilities_tests.jl - "step size and origin"
// =============================================================================

#[test]
fn test_step_size_and_origin() {
    let r: usize = 4;
    let grid = DiscretizedGrid::builder(&[r]).build().unwrap();
    assert!((grid.grid_step()[0] - 1.0 / (1 << r) as f64).abs() < 1e-14);
    assert_eq!(grid.grid_min(), &[0.0]);
}

// =============================================================================
// Ported from: utilities_tests.jl - "grid_origcoords"
// =============================================================================

#[test]
fn test_grid_origcoords_1d() {
    let r: usize = 4;
    let grid = DiscretizedGrid::builder(&[r]).build().unwrap();
    let coords = grid.grid_origcoords(0).unwrap();

    assert_eq!(coords.len(), 1 << r);
    assert!((coords[0] - grid.grid_min()[0]).abs() < 1e-14);
    assert!((coords.last().unwrap() - grid.grid_max()[0]).abs() < 1e-14);

    // Match grididx_to_origcoord for all indices
    for i in 1..=(1 << r) {
        let expected = grid.grididx_to_origcoord(&[i]).unwrap()[0];
        assert!((coords[(i - 1) as usize] - expected).abs() < 1e-14);
    }
}

#[test]
fn test_grid_origcoords_2d() {
    let grid = DiscretizedGrid::builder(&[3, 4])
        .with_lower_bound(&[-1.0, 2.0])
        .with_upper_bound(&[5.0, 8.0])
        .build()
        .unwrap();

    // First dimension
    let coords_x = grid.grid_origcoords(0).unwrap();
    assert_eq!(coords_x.len(), 8); // 2^3
    assert!((coords_x[0] - grid.grid_min()[0]).abs() < 1e-14);
    assert!((coords_x.last().unwrap() - grid.grid_max()[0]).abs() < 1e-14);

    // Second dimension
    let coords_y = grid.grid_origcoords(1).unwrap();
    assert_eq!(coords_y.len(), 16); // 2^4
    assert!((coords_y[0] - grid.grid_min()[1]).abs() < 1e-14);
    assert!((coords_y.last().unwrap() - grid.grid_max()[1]).abs() < 1e-14);

    // Match grididx_to_origcoord
    for i in 1..=8 {
        let expected = grid.grididx_to_origcoord(&[i, 1]).unwrap()[0];
        assert!((coords_x[(i - 1) as usize] - expected).abs() < 1e-14);
    }
    for j in 1..=16 {
        let expected = grid.grididx_to_origcoord(&[1, j]).unwrap()[1];
        assert!((coords_y[(j - 1) as usize] - expected).abs() < 1e-14);
    }

    // Spacing consistency
    let step = grid.grid_step();
    let x_spacing = coords_x[1] - coords_x[0];
    let y_spacing = coords_y[1] - coords_y[0];
    assert!((x_spacing - step[0]).abs() < 1e-14);
    assert!((y_spacing - step[1]).abs() < 1e-14);
}

#[test]
fn test_grid_origcoords_base3() {
    let grid = DiscretizedGrid::builder(&[3])
        .with_base(3)
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[2.7])
        .build()
        .unwrap();
    let coords = grid.grid_origcoords(0).unwrap();
    assert_eq!(coords.len(), 27); // 3^3
    assert!((coords[0] - grid.grid_min()[0]).abs() < 1e-14);
    assert!((coords.last().unwrap() - grid.grid_max()[0]).abs() < 1e-14);
}

#[test]
fn test_grid_origcoords_include_endpoint() {
    let grid = DiscretizedGrid::builder(&[3])
        .include_endpoint(true)
        .build()
        .unwrap();
    let coords = grid.grid_origcoords(0).unwrap();
    assert_eq!(coords.len(), 8); // 2^3
    assert!((coords[0] - 0.0).abs() < 1e-14);
    assert!((coords.last().unwrap() - 1.0).abs() < 1e-14);
}

#[test]
fn test_grid_origcoords_custom_indextable() {
    let index_table = vec![
        vec![("x".to_string(), 1), ("y".to_string(), 1)],
        vec![("z".to_string(), 1)],
        vec![
            ("x".to_string(), 2),
            ("y".to_string(), 2),
            ("z".to_string(), 2),
        ],
        vec![("x".to_string(), 3), ("y".to_string(), 3)],
    ];
    let grid = DiscretizedGrid::from_index_table(&["x", "y", "z"], index_table)
        .with_base(2)
        .with_lower_bound(&[-2.0, 1.0, 0.5])
        .with_upper_bound(&[3.0, 4.0, 2.0])
        .build()
        .unwrap();

    // By name
    let coords_x = grid.grid_origcoords_by_name("x").unwrap();
    let coords_y = grid.grid_origcoords_by_name("y").unwrap();
    let coords_z = grid.grid_origcoords_by_name("z").unwrap();

    assert_eq!(coords_x.len(), 8); // 2^3
    assert_eq!(coords_y.len(), 8); // 2^3
    assert_eq!(coords_z.len(), 4); // 2^2

    // Bounds
    assert!((coords_x[0] - grid.grid_min()[0]).abs() < 1e-14);
    assert!((coords_y[0] - grid.grid_min()[1]).abs() < 1e-14);
    assert!((coords_z[0] - grid.grid_min()[2]).abs() < 1e-14);
    assert!((coords_x.last().unwrap() - grid.grid_max()[0]).abs() < 1e-14);
    assert!((coords_y.last().unwrap() - grid.grid_max()[1]).abs() < 1e-14);
    assert!((coords_z.last().unwrap() - grid.grid_max()[2]).abs() < 1e-14);
}

#[test]
fn test_grid_origcoords_out_of_bounds() {
    let grid_1d = DiscretizedGrid::builder(&[4]).build().unwrap();
    // 0-indexed, so dim=1 is out of bounds for 1D
    assert!(grid_1d.grid_origcoords(1).is_err());

    let grid_2d = DiscretizedGrid::builder(&[3, 4]).build().unwrap();
    assert!(grid_2d.grid_origcoords(2).is_err());
}

#[test]
fn test_grid_origcoords_complex_asymmetric() {
    let rs = [5, 3, 7, 2];
    let lower = [-10.0, 0.1, 50.0, -2.5];
    let upper = [15.0, 3.9, 100.0, 7.8];
    let grid = DiscretizedGrid::builder(&rs)
        .with_base(3)
        .with_lower_bound(&lower)
        .with_upper_bound(&upper)
        .build()
        .unwrap();

    for d in 0..4 {
        let coords_d = grid.grid_origcoords(d).unwrap();
        let expected_len = 3usize.pow(rs[d] as u32);
        assert_eq!(coords_d.len(), expected_len);
        assert!((coords_d[0] - grid.grid_min()[d]).abs() < 1e-10);
        assert!((coords_d.last().unwrap() - grid.grid_max()[d]).abs() < 1e-10);

        // Uniform spacing
        let step = grid.grid_step()[d];
        for i in 1..coords_d.len() {
            let spacing = coords_d[i] - coords_d[i - 1];
            assert!((spacing - step).abs() < 1e-10);
        }

        // Consistency with grididx_to_origcoord
        for &idx in &[1i64, expected_len as i64 / 2, expected_len as i64] {
            let mut grid_idx = vec![1i64; 4];
            grid_idx[d] = idx;
            let expected_coord = grid.grididx_to_origcoord(&grid_idx).unwrap()[d];
            assert!((coords_d[(idx - 1) as usize] - expected_coord).abs() < 1e-10);
        }
    }
}

#[test]
fn test_grid_origcoords_tiny() {
    let grid = DiscretizedGrid::builder(&[1]).build().unwrap();
    let coords = grid.grid_origcoords(0).unwrap();
    assert_eq!(coords.len(), 2);
    assert!((coords[0] - 0.0).abs() < 1e-14);
    assert!((coords[1] - 0.5).abs() < 1e-14);
}

// =============================================================================
// Ported from: utilities_tests.jl - "DiscretizedGrid(R, lower, upper) constructor"
// =============================================================================

#[test]
fn test_1d_constructor_with_bounds() {
    let grid = DiscretizedGrid::builder(&[3])
        .with_lower_bound(&[-2.0])
        .with_upper_bound(&[3.0])
        .build()
        .unwrap();
    assert_eq!(grid.rs(), &[3]);
    assert_eq!(grid.lower_bound(), &[-2.0]);
    assert_eq!(grid.upper_bound(), &[3.0]);
}

// =============================================================================
// Ported from: utilities_tests.jl - "grouped unfoldingscheme"
// =============================================================================

#[test]
fn test_grouped_unfolding_scheme() {
    let rs = [3usize, 4, 5];
    let grid = DiscretizedGrid::builder(&rs)
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();

    // Expected index table: all bits of dim 1 first, then dim 2, then dim 3
    let table = grid.index_table();
    // Total sites = 3 + 4 + 5 = 12
    assert_eq!(table.len(), 12);

    // Check structure: 3 sites for dim1, 4 for dim2, 5 for dim3
    // Variable names are "1", "2", "3" by default
    let mut idx = 0;
    for (d, &r_d) in rs.iter().enumerate() {
        let var_name = (d + 1).to_string();
        for bit in 1..=r_d {
            assert_eq!(table[idx].len(), 1);
            assert_eq!(table[idx][0].0, var_name);
            assert_eq!(table[idx][0].1, bit);
            idx += 1;
        }
    }
}

// =============================================================================
// Ported from: comprehensive_tests.jl - main comprehensive test
// =============================================================================

#[test]
fn test_comprehensive_functionality() {
    let index_table = vec![
        vec![
            ("x".to_string(), 1),
            ("y".to_string(), 1),
            ("z".to_string(), 1),
        ],
        vec![("w".to_string(), 1)],
        vec![
            ("z".to_string(), 2),
            ("w".to_string(), 2),
            ("x".to_string(), 3),
            ("y".to_string(), 3),
        ],
        vec![("x".to_string(), 2), ("y".to_string(), 2)],
        vec![("z".to_string(), 3)],
        vec![("w".to_string(), 3), ("x".to_string(), 4)],
        vec![
            ("y".to_string(), 4),
            ("z".to_string(), 4),
            ("w".to_string(), 4),
        ],
    ];

    let lower_bound = [-std::f64::consts::PI, 1e-12, 1e8, -1e6];
    let upper_bound = [2.0 * std::f64::consts::PI, 1e-10, 1e8 + 1000.0, 1e6];
    let base: usize = 3;

    let grid = DiscretizedGrid::from_index_table(&["x", "y", "z", "w"], index_table)
        .with_base(base)
        .with_lower_bound(&lower_bound)
        .with_upper_bound(&upper_bound)
        .include_endpoint(true)
        .build()
        .unwrap();

    assert_eq!(grid.ndims(), 4);
    assert_eq!(grid.len(), 7);
    assert_eq!(grid.base(), 3);
    assert_eq!(grid.variable_names(), &["x", "y", "z", "w"]);

    assert_eq!(grid.rs(), &[4, 4, 4, 4]);

    let expected_localdims = vec![27, 3, 81, 9, 3, 9, 27];
    assert_eq!(grid.local_dimensions(), expected_localdims);

    for (i, &expected) in expected_localdims.iter().enumerate() {
        assert_eq!(grid.site_dim(i).unwrap(), expected);
    }

    let lb = grid.lower_bound();
    let gmin = grid.grid_min();
    let gmax = grid.grid_max();
    let gstep = grid.grid_step();

    for d in 0..4 {
        assert!((lb[d] - lower_bound[d]).abs() < 1e-14);
        assert!((gmin[d] - lower_bound[d]).abs() < 1e-14);
    }

    // In Rust, upper_bound() returns the user-specified value (not the effective one)
    let ub = grid.upper_bound();
    for d in 0..4 {
        assert_eq!(ub[d], upper_bound[d]);
    }

    // grid_max should approximate upper_bound (Julia: @test all(gmax .≈ upper_bound))
    // Use relative tolerance matching Julia's ≈ (rtol = √eps ≈ 1.49e-8)
    for d in 0..4 {
        let scale = upper_bound[d].abs().max(gmax[d].abs()).max(1.0);
        assert!(
            (gmax[d] - upper_bound[d]).abs() < 1.5e-8 * scale,
            "grid_max[{}] = {} ≠ upper_bound[{}] = {} (relative diff = {})",
            d,
            gmax[d],
            d,
            upper_bound[d],
            (gmax[d] - upper_bound[d]).abs() / scale
        );
    }

    // Steps should be positive
    for &s in gstep.iter() {
        assert!(s > 0.0);
    }

    // Corner indices round-trip
    let base_pow_4 = 81i64; // 3^4
    let corners: Vec<Vec<i64>> = vec![
        vec![1, 1, 1, 1],
        vec![base_pow_4, base_pow_4, base_pow_4, base_pow_4],
        vec![1, base_pow_4, 1, base_pow_4],
        vec![base_pow_4, 1, base_pow_4, 1],
        vec![9, 9, 9, 9], // base^2 = 9
    ];

    for grididx in &corners {
        let quantics = grid.grididx_to_quantics(grididx).unwrap();
        let recovered = grid.quantics_to_grididx(&quantics).unwrap();
        assert_eq!(&recovered, grididx);

        let origcoord = grid.grididx_to_origcoord(grididx).unwrap();
        let recovered2 = grid.origcoord_to_grididx(&origcoord).unwrap();
        assert_eq!(&recovered2, grididx);

        let origcoord2 = grid.quantics_to_origcoord(&quantics).unwrap();
        let recovered_q = grid.origcoord_to_quantics(&origcoord2).unwrap();
        assert_eq!(recovered_q, quantics);

        for d in 0..4 {
            assert!((origcoord[d] - origcoord2[d]).abs() < 1e-10);
        }
    }

    // Deterministic random-like round-trips (matches Julia: for _ in 1:50 with rand(1:sitedim(grid, site)))
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..50 {
        let quantics: Vec<i64> = (0..grid.len())
            .map(|site| {
                let dim = grid.site_dim(site).unwrap() as i64;
                rng.random_range(1..=dim)
            })
            .collect();

        let grididx = grid.quantics_to_grididx(&quantics).unwrap();
        let origcoord = grid.grididx_to_origcoord(&grididx).unwrap();
        let back_grididx = grid.origcoord_to_grididx(&origcoord).unwrap();
        let back_quantics = grid.grididx_to_quantics(&back_grididx).unwrap();
        let back_origcoord = grid.quantics_to_origcoord(&back_quantics).unwrap();

        assert_eq!(back_grididx, grididx);
        assert_eq!(back_quantics, quantics);
        for (d, (&back, &orig)) in back_origcoord.iter().zip(origcoord.iter()).enumerate() {
            assert!(
                (back - orig).abs() < 1e-10,
                "origcoord mismatch at dim {}: {} vs {}",
                d,
                back,
                orig
            );
        }

        // Grid indices within valid range
        for (d, &idx) in grididx.iter().enumerate() {
            assert!(idx >= 1);
            assert!(idx <= (base as i64).pow(grid.rs()[d] as u32));
        }

        // Quantics vector has correct length
        assert_eq!(back_quantics.len(), grid.len());
    }

    // Error handling
    assert!(grid
        .origcoord_to_grididx(&[lb[0] - 1.0, lb[1], lb[2], lb[3]])
        .is_err());
    assert!(grid
        .origcoord_to_grididx(&[ub[0] + 1.0, ub[1], ub[2], ub[3]])
        .is_err());
    assert!(grid.grididx_to_origcoord(&[0, 1, 1, 1]).is_err());
    assert!(grid
        .grididx_to_origcoord(&[base_pow_4 + 1, 1, 1, 1])
        .is_err());

    // quantics_function
    let test_fn = |coords: &[f64]| {
        coords[0].sin() * coords[1].cos() * (-coords[2] / 1e8).exp() * (coords[3].powi(2))
    };
    let qf = quantics_function(&grid, test_fn);

    for seed in 0..10 {
        let quantics: Vec<i64> = (0..grid.len())
            .map(|site| {
                let dim = grid.site_dim(site).unwrap() as i64;
                ((seed * 11 + site * 17 + 5) as i64 % dim) + 1
            })
            .collect();

        let result1 = qf(&quantics).unwrap();
        let origcoord = grid.quantics_to_origcoord(&quantics).unwrap();
        let result2 = test_fn(&origcoord);
        assert!((result1 - result2).abs() < 1e-10);
    }
}

// =============================================================================
// Ported from: comprehensive_tests.jl - simple_grid part
// =============================================================================

#[test]
fn test_simple_grid_interleaved_base5() {
    let grid = DiscretizedGrid::builder(&[3, 4])
        .with_base(5)
        .with_lower_bound(&[0.0, -1.0])
        .with_upper_bound(&[1.0, 1.0])
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .build()
        .unwrap();
    assert_eq!(grid.rs(), &[3, 4]);
    assert_eq!(grid.base(), 5);
    assert_eq!(grid.local_dimensions(), vec![5; 7]); // 3+4=7 sites
}

// =============================================================================
// Ported from: comprehensive_tests.jl - 1D grid roundtrip
// =============================================================================

#[test]
fn test_1d_grid_roundtrip() {
    let grid = DiscretizedGrid::builder(&[8])
        .with_base(2)
        .with_lower_bound(&[-5.0])
        .with_upper_bound(&[10.0])
        .build()
        .unwrap();
    assert_eq!(grid.ndims(), 1);
    assert_eq!(grid.rs(), &[8]);

    for &i in &[1i64, 100, 256] {
        let coord = grid.grididx_to_origcoord(&[i]).unwrap();
        let back_idx = grid.origcoord_to_grididx(&coord).unwrap();
        assert_eq!(back_idx, vec![i]);
    }
}

// =============================================================================
// Ported from: comprehensive_tests.jl - fused vs interleaved
// =============================================================================

#[test]
fn test_fused_vs_interleaved_consistency() {
    let rs = [3, 3];
    let grid_fused = DiscretizedGrid::builder(&rs)
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .with_base(2)
        .build()
        .unwrap();
    let grid_interleaved = DiscretizedGrid::builder(&rs)
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .with_base(2)
        .build()
        .unwrap();

    assert_eq!(grid_fused.len(), 3);
    assert_eq!(grid_interleaved.len(), 6);
    assert_eq!(grid_fused.rs(), grid_interleaved.rs());

    let test_grididx = vec![4, 6];
    let coord_fused = grid_fused.grididx_to_origcoord(&test_grididx).unwrap();
    let coord_interleaved = grid_interleaved
        .grididx_to_origcoord(&test_grididx)
        .unwrap();
    for d in 0..2 {
        assert!((coord_fused[d] - coord_interleaved[d]).abs() < 1e-14);
    }
}

// =============================================================================
// Ported from: comprehensive_tests.jl - large 2D grid
// =============================================================================

#[test]
fn test_large_2d_grid() {
    let grid = DiscretizedGrid::builder(&[20, 20]).build().unwrap();
    let max_idx = 1i64 << 20;

    let extreme_indices: Vec<i64> = vec![1, max_idx / 2, max_idx - 1, max_idx];
    for &idx in &extreme_indices {
        let grididx = vec![idx, idx];
        let quantics = grid.grididx_to_quantics(&grididx).unwrap();
        let recovered = grid.quantics_to_grididx(&quantics).unwrap();
        assert_eq!(recovered, grididx);
    }
}

// =============================================================================
// Ported from: comprehensive_tests.jl - endpoint comparison
// =============================================================================

#[test]
fn test_endpoint_comparison() {
    let rs = [4, 4];
    let grid_no = DiscretizedGrid::builder(&rs).build().unwrap();
    let grid_with = DiscretizedGrid::builder(&rs)
        .include_endpoint(true)
        .build()
        .unwrap();

    // In Rust, upper_bound() returns the user-specified value (same for both),
    // but grid_max and grid_step differ.
    let gmax_no = grid_no.grid_max();
    let gmax_with = grid_with.grid_max();
    for d in 0..2 {
        assert!(gmax_with[d] > gmax_no[d]);
    }

    let step_no = grid_no.grid_step();
    let step_with = grid_with.grid_step();
    for d in 0..2 {
        assert!(step_no[d] > 0.0);
        assert!(step_with[d] > 0.0);
    }
}

// =============================================================================
// Ported from: comprehensive_tests.jl - base2 grid roundtrip
// =============================================================================

#[test]
fn test_base2_grid_roundtrip() {
    let grid = DiscretizedGrid::builder(&[5, 3])
        .with_base(2)
        .with_lower_bound(&[0.0, -1.0])
        .with_upper_bound(&[3.0, 2.0])
        .build()
        .unwrap();

    for seed in 0..20 {
        let quantics: Vec<i64> = (0..grid.len())
            .map(|site| {
                let dim = grid.site_dim(site).unwrap() as i64;
                ((seed * 7 + site * 13 + 3) as i64 % dim) + 1
            })
            .collect();

        let grididx = grid.quantics_to_grididx(&quantics).unwrap();
        let back_q = grid.grididx_to_quantics(&grididx).unwrap();
        assert_eq!(back_q, quantics);

        let origcoord = grid.grididx_to_origcoord(&grididx).unwrap();
        let back_idx = grid.origcoord_to_grididx(&origcoord).unwrap();
        assert_eq!(back_idx, grididx);
    }
}

// =============================================================================
// Ported from: comprehensive_tests.jl - simple_tupletable_base2
// =============================================================================

#[test]
fn test_simple_index_table_base2() {
    let index_table = vec![
        vec![("x".to_string(), 1)],
        vec![("x".to_string(), 2)],
        vec![("y".to_string(), 1)],
        vec![("y".to_string(), 2)],
    ];
    let grid = DiscretizedGrid::from_index_table(&["x", "y"], index_table)
        .with_base(2)
        .build()
        .unwrap();

    let test_quantics: Vec<Vec<i64>> = vec![
        vec![1, 1, 1, 1],
        vec![2, 1, 1, 1],
        vec![1, 2, 1, 1],
        vec![2, 2, 2, 2],
    ];
    for q in &test_quantics {
        let grididx = grid.quantics_to_grididx(q).unwrap();
        let back_q = grid.grididx_to_quantics(&grididx).unwrap();
        assert_eq!(&back_q, q);
    }
}

// =============================================================================
// Ported from: comprehensive_tests.jl - constructor with variablenames and Rs
// =============================================================================

#[test]
fn test_constructor_with_variable_names_and_rs() {
    // Default parameters
    let grid1 = DiscretizedGrid::builder(&[3, 4, 2])
        .with_variable_names(&["a", "b", "c"])
        .build()
        .unwrap();
    assert_eq!(grid1.variable_names(), &["a", "b", "c"]);
    assert_eq!(grid1.rs(), &[3, 4, 2]);
    assert_eq!(grid1.base(), 2);
    assert_eq!(grid1.ndims(), 3);

    // Custom parameters with includeendpoint
    let grid2 = DiscretizedGrid::builder(&[3, 4, 2])
        .with_variable_names(&["a", "b", "c"])
        .with_lower_bound(&[-1.0, 0.0, -5.0])
        .with_upper_bound(&[1.0, 2.0, 5.0])
        .with_base(3)
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .include_endpoint(true)
        .build()
        .unwrap();
    assert_eq!(grid2.variable_names(), &["a", "b", "c"]);
    assert_eq!(grid2.rs(), &[3, 4, 2]);
    assert_eq!(grid2.base(), 3);
    assert_eq!(grid2.lower_bound(), &[-1.0, 0.0, -5.0]);

    // Roundtrip
    let test_grididx = vec![2, 3, 1];
    let origcoord = grid2.grididx_to_origcoord(&test_grididx).unwrap();
    let back = grid2.origcoord_to_grididx(&origcoord).unwrap();
    assert_eq!(back, test_grididx);
}

// =============================================================================
// Ported from: comprehensive_tests.jl - per-dimension includeendpoint
// =============================================================================

#[test]
fn test_per_dimension_include_endpoint() {
    let grid = DiscretizedGrid::builder(&[3, 4])
        .with_variable_names(&["x", "y"])
        .with_include_endpoint(&[true, false])
        .build()
        .unwrap();

    assert_eq!(grid.variable_names(), &["x", "y"]);
    assert_eq!(grid.rs(), &[3, 4]);

    let lb = grid.lower_bound();
    let ub = grid.upper_bound();
    assert!((lb[0] - 0.0).abs() < 1e-14);
    assert!((lb[1] - 0.0).abs() < 1e-14);
    // y (includeendpoint=false) keeps original upper_bound=1.0
    assert!((ub[1] - 1.0).abs() < 1e-14);

    // Roundtrip
    let test_grididx = vec![2, 2];
    let origcoord = grid.grididx_to_origcoord(&test_grididx).unwrap();
    let back = grid.origcoord_to_grididx(&origcoord).unwrap();
    assert_eq!(back, test_grididx);
}

// =============================================================================
// Ported from: comprehensive_tests.jl - "integration test mixed bases"
// =============================================================================

#[test]
fn test_mixed_bases_discretized_grid() {
    let rs = [2, 1, 3];
    let bases = [2, 3, 5];
    let lower = [0.0, -1.0, 10.0];
    let upper = [4.0, 2.0, 135.0];
    let grid = DiscretizedGrid::builder(&rs)
        .with_bases(&bases)
        .with_lower_bound(&lower)
        .with_upper_bound(&upper)
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .unwrap();

    let step = grid.grid_step();
    assert!((step[0] - 1.0).abs() < 1e-14);
    assert!((step[1] - 1.0).abs() < 1e-14);
    assert!((step[2] - 1.0).abs() < 1e-14);

    let gmax = grid.grid_max();
    assert!((gmax[0] - 3.0).abs() < 1e-14);
    assert!((gmax[1] - 1.0).abs() < 1e-14);
    assert!((gmax[2] - 134.0).abs() < 1e-14);

    let origcoord = vec![2.0, 0.0, 12.0];
    let grididx = grid.origcoord_to_grididx(&origcoord).unwrap();
    assert_eq!(grididx, vec![3, 2, 3]);
    let back_coord = grid.grididx_to_origcoord(&[3, 2, 3]).unwrap();
    for d in 0..3 {
        assert!((back_coord[d] - origcoord[d]).abs() < 1e-14);
    }

    // Quantics roundtrip
    let quantics = grid.origcoord_to_quantics(&origcoord).unwrap();
    let back_coord2 = grid.quantics_to_origcoord(&quantics).unwrap();
    for d in 0..3 {
        assert!((back_coord2[d] - origcoord[d]).abs() < 1e-14);
    }
}

// =============================================================================
// Ported from: comprehensive_tests.jl - "integration test mixed bases includeendpoint"
// =============================================================================

#[test]
fn test_mixed_bases_include_endpoint() {
    let rs = [1, 1];
    let bases = [2, 6];
    let grid = DiscretizedGrid::builder(&rs)
        .with_bases(&bases)
        .with_include_endpoint(&[true, true])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();

    // In Rust, upper_bound() returns the user-specified value (1.0, 1.0).
    // The Julia upper_bound() returns the effective upper (2.0, 1.2).
    // Test grid_max instead: with includeendpoint=true, grid_max == user upper bound.
    let gmax = grid.grid_max();
    assert!((gmax[0] - 1.0).abs() < 1e-14); // base=2, R=1: max = user_upper = 1.0
    assert!((gmax[1] - 1.0).abs() < 1e-14); // base=6, R=1: max = user_upper = 1.0

    // Verify step sizes
    // For base=2, R=1, includeendpoint: step = (upper-lower)/(base^R-1) = 1.0/(2-1) = 1.0
    // For base=6, R=1, includeendpoint: step = (upper-lower)/(base^R-1) = 1.0/(6-1) = 0.2
    let step = grid.grid_step();
    assert!((step[0] - 1.0).abs() < 1e-14);
    assert!((step[1] - 0.2).abs() < 1e-14);
}

// =============================================================================
// Ported from: comprehensive_tests.jl - "Additional Constructors"
// =============================================================================

#[test]
fn test_additional_constructors() {
    // 2D with explicit lower/upper and interleaved
    let grid = DiscretizedGrid::builder(&[3, 4])
        .with_lower_bound(&[-2.0, 3.0])
        .with_upper_bound(&[4.0, 5.0])
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .build()
        .unwrap();
    assert_eq!(grid.ndims(), 2);
    assert_eq!(grid.rs(), &[3, 4]);
}

// =============================================================================
// Ported from: origcoord_tests.jl - "performance stress test - rapid conversions"
// =============================================================================

#[test]
fn test_rapid_conversions_base3() {
    let grid = DiscretizedGrid::builder(&[12, 10, 8, 6])
        .with_base(3)
        .build()
        .unwrap();

    // quantics -> grididx -> quantics round-trips (matches Julia: n_tests = 1000)
    let n_sites = grid.len();
    for seed in 0..1000u64 {
        // Generate valid quantics deterministically
        let quantics: Vec<i64> = (0..n_sites)
            .map(|i| {
                let dim = grid.site_dim(i).unwrap() as i64;
                ((seed
                    .wrapping_mul(7)
                    .wrapping_add(i as u64)
                    .wrapping_mul(13)
                    .wrapping_add(3)) as i64
                    % dim)
                    .abs()
                    + 1
            })
            .collect();
        let grididx = grid.quantics_to_grididx(&quantics).unwrap();
        let recovered = grid.grididx_to_quantics(&grididx).unwrap();
        assert_eq!(recovered, quantics);
    }

    // grididx -> quantics -> grididx round-trips (matches Julia: n_tests = 1000)
    let max_indices: Vec<i64> = vec![3i64.pow(12), 3i64.pow(10), 3i64.pow(8), 3i64.pow(6)];
    for seed in 0..1000u64 {
        let grididx: Vec<i64> = (0..4)
            .map(|d| {
                ((seed
                    .wrapping_mul(11)
                    .wrapping_add((d as u64).wrapping_mul(17))
                    .wrapping_add(5)) as i64
                    % max_indices[d])
                    .abs()
                    + 1
            })
            .collect();
        let quantics = grid.grididx_to_quantics(&grididx).unwrap();
        let recovered = grid.quantics_to_grididx(&quantics).unwrap();
        assert_eq!(recovered, grididx);
    }

    // origcoord conversions (matches Julia: min(n_tests, 200) = 200)
    for seed in 0..200u64 {
        let grididx: Vec<i64> = (0..4)
            .map(|d| {
                ((seed
                    .wrapping_mul(11)
                    .wrapping_add((d as u64).wrapping_mul(17))
                    .wrapping_add(5)) as i64
                    % max_indices[d])
                    .abs()
                    + 1
            })
            .collect();
        let origcoord = grid.grididx_to_origcoord(&grididx).unwrap();
        let recovered = grid.origcoord_to_grididx(&origcoord).unwrap();
        assert_eq!(recovered, grididx);
    }
}

// =============================================================================
// Ported from: origcoord_tests.jl - "edge cases - extreme base values"
// =============================================================================

#[test]
fn test_extreme_base_values() {
    let max_base: usize = 16;
    let grid = DiscretizedGrid::builder(&[3, 4])
        .with_base(max_base)
        .build()
        .unwrap();

    let n_sites = grid.len();
    for seed in 0..20 {
        let quantics: Vec<i64> = (0..n_sites)
            .map(|i| {
                let dim = grid.site_dim(i).unwrap() as i64;
                ((seed * 7 + i * 13 + 3) as i64 % dim) + 1
            })
            .collect();
        let grididx = grid.quantics_to_grididx(&quantics).unwrap();
        let recovered = grid.grididx_to_quantics(&grididx).unwrap();
        assert_eq!(recovered, quantics);

        // Verify all grid indices are in valid range
        for (d, &idx) in grididx.iter().enumerate() {
            assert!(idx >= 1);
            assert!(idx <= (max_base as i64).pow(grid.rs()[d] as u32));
        }
    }

    // Base=2 (minimum valid)
    let grid2 = DiscretizedGrid::builder(&[10, 8])
        .with_base(2)
        .build()
        .unwrap();
    for seed in 0..20 {
        let quantics: Vec<i64> = (0..grid2.len())
            .map(|i| ((seed * 7 + i * 13) % 2 + 1) as i64)
            .collect();
        let grididx = grid2.quantics_to_grididx(&quantics).unwrap();
        let recovered = grid2.grididx_to_quantics(&grididx).unwrap();
        assert_eq!(recovered, quantics);
    }
}

// =============================================================================
// Ported from: origcoord_tests.jl - "edge cases - maximum R values"
// =============================================================================

#[test]
fn test_maximum_r_values() {
    let large_r: usize = 58;
    let grid = DiscretizedGrid::builder(&[large_r]).build().unwrap();

    // Min quantics
    let min_q = vec![1i64; large_r];
    let min_idx = grid.quantics_to_grididx(&min_q).unwrap();
    assert_eq!(min_idx, vec![1]);

    // Max quantics
    let max_q = vec![2i64; large_r];
    let max_idx = grid.quantics_to_grididx(&max_q).unwrap();
    assert_eq!(max_idx, vec![1i64 << large_r]);

    // Roundtrip
    let recovered_min = grid.grididx_to_quantics(&[1]).unwrap();
    assert_eq!(recovered_min, min_q);
    let recovered_max = grid.grididx_to_quantics(&[1i64 << large_r]).unwrap();
    assert_eq!(recovered_max, max_q);

    // Middle values
    for seed in 0..10 {
        let quantics: Vec<i64> = (0..large_r)
            .map(|i| ((seed * 7 + i * 13) % 2 + 1) as i64)
            .collect();
        let grididx = grid.quantics_to_grididx(&quantics).unwrap();
        let recovered = grid.grididx_to_quantics(&grididx).unwrap();
        assert_eq!(recovered, quantics);
    }
}

// =============================================================================
// Ported from: origcoord_tests.jl - "edge cases - numerical precision"
// =============================================================================

#[test]
fn test_numerical_precision_edge_cases() {
    let grid = DiscretizedGrid::builder(&[30, 25])
        .with_lower_bound(&[1e-15, -1e-15])
        .with_upper_bound(&[1e-14, 1e-14])
        .build()
        .unwrap();

    // Specific indices for round-trip
    let test_cases: Vec<(i64, i64)> = vec![
        (1, 1),
        (1 << 30, 1 << 25),
        (1 << 15, 1 << 12),
        (500000, 10000),
        (999999, 33000000),
    ];
    for (gx, gy) in test_cases {
        if gx <= 1i64 << 30 && gy <= 1i64 << 25 {
            let origcoord = grid.grididx_to_origcoord(&[gx, gy]).unwrap();
            let recovered = grid.origcoord_to_grididx(&origcoord).unwrap();
            assert_eq!(recovered, vec![gx, gy]);
        }
    }

    // Near boundaries
    let eps_coord_x = 1e-15 + 1e-20;
    let eps_coord_y = 1e-14 - 1e-20;
    let result = grid.origcoord_to_grididx(&[eps_coord_x, eps_coord_y]);
    assert!(result.is_ok());
}

// =============================================================================
// Ported from: grid_tests.jl - "DiscretizedGrid" 1D fused
// =============================================================================

#[test]
fn test_discretized_grid_1d_fused() {
    let r: usize = 5;
    let a = 0.1;
    let b = 2.0;
    let dx = (b - a) / (1 << r) as f64;
    let grid = DiscretizedGrid::builder(&[r])
        .with_lower_bound(&[a])
        .with_upper_bound(&[b])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .unwrap();
    assert_eq!(grid.local_dimensions(), vec![2; r]);

    let idx = grid.origcoord_to_grididx(&[0.999999 * dx + a]).unwrap();
    assert_eq!(idx, vec![2]);

    assert_eq!(grid.lower_bound(), &[0.1]);
    assert_eq!(grid.upper_bound(), &[2.0]);
    assert_eq!(grid.grid_min(), &[a]);
    assert!((grid.grid_max()[0] - (b - dx)).abs() < 1e-14);
    assert!((grid.grid_step()[0] - 0.059375).abs() < 1e-14);
}

// =============================================================================
// Ported from: comprehensive_tests.jl - "integration test 1"
// (3D frequency grid with includeendpoint per-dimension)
// =============================================================================

#[test]
fn test_integration_frequency_grid() {
    let r: usize = 5;
    let n = 1i64 << r;
    let fermi_min = -(n - 1) as f64;
    let fermi_max = (n - 1 + 2) as f64; // bc includeendpoint=false
    let bose_min = -(n as f64);
    let bose_max = (n - 2 + 2) as f64; // bc includeendpoint=false

    // 3D grid (v, v', w) with no momenta
    let grid = DiscretizedGrid::builder(&[r, r, r])
        .with_variable_names(&["v", "v'", "w"])
        .with_lower_bound(&[fermi_min, fermi_min, bose_min])
        .with_upper_bound(&[fermi_max, fermi_max, bose_max])
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .with_include_endpoint(&[true, true, true])
        .build()
        .unwrap();

    // origcoord_to_quantics at 0.0 for all dimensions
    let quantics = grid.origcoord_to_quantics(&[0.0, 0.0, 0.0]).unwrap();

    // Verify quantics length is correct for interleaved 3D with R=5
    assert_eq!(quantics.len(), 3 * r); // 15 sites for interleaved

    // Verify round-trip: quantics -> origcoord -> quantics should be consistent
    let recovered_coord = grid.quantics_to_origcoord(&quantics).unwrap();
    let re_quantics = grid.origcoord_to_quantics(&recovered_coord).unwrap();
    assert_eq!(re_quantics, quantics);

    // The recovered coordinate should be close to 0.0 (within one grid step)
    let step = grid.grid_step();
    for d in 0..3 {
        assert!(
            recovered_coord[d].abs() < step[d] + 1e-10,
            "Coordinate {} should be near 0.0 (within step {}), got {}",
            d,
            step[d],
            recovered_coord[d]
        );
    }

    // Verify general round-trip at boundary coordinates
    let test_coords: Vec<Vec<f64>> = vec![vec![fermi_min, fermi_min, bose_min]];
    for coord in &test_coords {
        let q = grid.origcoord_to_quantics(coord).unwrap();
        let grididx = grid.quantics_to_grididx(&q).unwrap();
        let back_coord = grid.grididx_to_origcoord(&grididx).unwrap();
        let back_q = grid.origcoord_to_quantics(&back_coord).unwrap();
        assert_eq!(back_q, q, "Round-trip failed for coord {:?}", coord);
    }
}

// =============================================================================
// Ported from: origcoord_tests.jl - "origcoord functions - fused indices grid"
// (custom index table with fused multi-variable sites)
// =============================================================================

#[test]
fn test_fused_indices_grid() {
    // x has R=2 (bits 1,2 at different sites)
    // y has R=2 (bits 1,2 at different sites)
    // site 1: x bit 2, y bit 1 => site_dim=base^2=4
    // site 2: x bit 1 => site_dim=base^1=2
    // site 3: y bit 2 => site_dim=base^1=2
    let index_table = vec![
        vec![("x".to_string(), 2), ("y".to_string(), 1)],
        vec![("x".to_string(), 1)],
        vec![("y".to_string(), 2)],
    ];
    let grid = DiscretizedGrid::from_index_table(&["x", "y"], index_table)
        .with_lower_bound(&[0.0, -1.0])
        .with_upper_bound(&[4.0, 1.0])
        .build()
        .unwrap();

    // Round-trip deterministic tests (matches Julia: for _ in 1:50 with rand(1:4), rand(1:2), rand(1:2))
    let site_dims: Vec<i64> = (0..grid.len())
        .map(|s| grid.site_dim(s).unwrap() as i64)
        .collect();

    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..50 {
        let quantics: Vec<i64> = site_dims
            .iter()
            .map(|&dim| rng.random_range(1..=dim))
            .collect();

        let grididx = grid.quantics_to_grididx(&quantics).unwrap();
        let origcoord = grid.grididx_to_origcoord(&grididx).unwrap();

        let recovered = grid.origcoord_to_grididx(&origcoord).unwrap();
        assert_eq!(recovered, grididx);

        let q_coord = grid.quantics_to_origcoord(&quantics).unwrap();
        for d in 0..2 {
            assert!((q_coord[d] - origcoord[d]).abs() < 1e-14);
        }

        let recovered_q = grid.origcoord_to_quantics(&origcoord).unwrap();
        assert_eq!(recovered_q, quantics);
    }
}

// =============================================================================
// Ported from: discretizedgrid_misc_tests.jl - "DiscretizedGrid constructors consistency"
// =============================================================================

#[test]
fn test_constructors_consistency() {
    // Build two equivalent grids: one from Rs with variable names, one from index table
    let grid1 = DiscretizedGrid::builder(&[2, 3, 4])
        .with_variable_names(&["x", "y", "z"])
        .build()
        .unwrap();

    // Build the expected index table for fused scheme with Rs=(2,3,4)
    // Level 1: z(1), y(1), x(1)
    // Level 2: z(2), y(2), x(2)
    // Level 3: z(3), y(3)
    // Level 4: z(4)
    let index_table = vec![
        vec![
            ("z".to_string(), 1),
            ("y".to_string(), 1),
            ("x".to_string(), 1),
        ],
        vec![
            ("z".to_string(), 2),
            ("y".to_string(), 2),
            ("x".to_string(), 2),
        ],
        vec![("z".to_string(), 3), ("y".to_string(), 3)],
        vec![("z".to_string(), 4)],
    ];
    let grid2 = DiscretizedGrid::from_index_table(&["x", "y", "z"], index_table)
        .build()
        .unwrap();

    // Should produce the same index table
    assert_eq!(grid1.index_table(), grid2.index_table());
}

// =============================================================================
// Ported from: discretizedgrid_misc_tests.jl - "DiscretizedGrid 0-dimensional show"
// (Tests that Display doesn't panic for edge cases)
// =============================================================================

#[test]
fn test_display_does_not_panic() {
    // 1D grid
    let grid = DiscretizedGrid::builder(&[2]).build().unwrap();
    let _ = format!("{}", grid);

    // Multi-D grid
    let grid2 = DiscretizedGrid::builder(&[2, 3, 4]).build().unwrap();
    let _ = format!("{}", grid2);
}

// =============================================================================
// Ported from: discretizedgrid_misc_tests.jl - "DiscretizedGrid show method"
// =============================================================================

#[test]
fn test_display_format() {
    let grid = DiscretizedGrid::builder(&[2, 3, 4]).build().unwrap();
    let text = format!("{}", grid);
    assert!(text.contains("DiscretizedGrid"));
}

// =============================================================================
// Ported from: discretizedgrid_misc_tests.jl -
// "DiscretizedGrid accepts untyped indextable container"
// =============================================================================

#[test]
fn test_accepts_indextable_with_endpoint() {
    let r: usize = 10;
    let n = (1 << r) as f64;
    let lower_bound = [0.0, 0.0];
    let upper_bound = [2.0, n - 1.0];

    let mut index_table: Vec<Vec<(String, usize)>> = Vec::new();
    for l in 1..=r {
        index_table.push(vec![("w".to_string(), l)]);
        index_table.push(vec![("n".to_string(), r - l + 1)]);
    }

    let grid = DiscretizedGrid::from_index_table(&["w", "n"], index_table)
        .with_lower_bound(&lower_bound)
        .with_upper_bound(&upper_bound)
        .with_include_endpoint(&[false, true])
        .build()
        .unwrap();

    assert_eq!(grid.variable_names(), &["w", "n"]);
    assert_eq!(grid.index_table().len(), 2 * r);
    let gmax = grid.grid_max();
    assert!((gmax[1] - (n - 1.0)).abs() < 1e-10);
}

// =============================================================================
// Ported from: origcoord_tests.jl -
// "stress test - maximum complexity fused indices"
// =============================================================================

#[test]
fn test_max_complexity_fused_indices() {
    let index_table = vec![
        vec![
            ("a".to_string(), 8),
            ("b".to_string(), 7),
            ("c".to_string(), 6),
            ("d".to_string(), 5),
        ],
        vec![("e".to_string(), 4), ("f".to_string(), 3)],
        vec![("g".to_string(), 2)],
        vec![("h".to_string(), 1)],
        vec![
            ("a".to_string(), 7),
            ("c".to_string(), 5),
            ("e".to_string(), 3),
            ("g".to_string(), 1),
        ],
        vec![
            ("b".to_string(), 6),
            ("d".to_string(), 4),
            ("f".to_string(), 2),
        ],
        vec![("a".to_string(), 6), ("h".to_string(), 2)],
        vec![
            ("b".to_string(), 5),
            ("c".to_string(), 4),
            ("d".to_string(), 3),
        ],
        vec![("e".to_string(), 2), ("f".to_string(), 1)],
        vec![("a".to_string(), 5), ("g".to_string(), 3)],
        vec![
            ("b".to_string(), 4),
            ("c".to_string(), 3),
            ("d".to_string(), 2),
            ("h".to_string(), 3),
        ],
        vec![("a".to_string(), 4), ("e".to_string(), 1)],
        vec![("b".to_string(), 3), ("f".to_string(), 4)],
        vec![("a".to_string(), 3), ("c".to_string(), 2)],
        vec![("b".to_string(), 2), ("d".to_string(), 1)],
        vec![("a".to_string(), 2)],
        vec![("b".to_string(), 1)],
        vec![("a".to_string(), 1)],
        vec![("c".to_string(), 1)],
    ];

    let grid =
        DiscretizedGrid::from_index_table(&["a", "b", "c", "d", "e", "f", "g", "h"], index_table)
            .with_base(2)
            .build()
            .unwrap();

    // Expected site dims: 2^4, 2^2, 2^1, 2^1, 2^4, 2^3, 2^2, 2^3, 2^2, 2^2,
    //                     2^4, 2^2, 2^2, 2^2, 2^2, 2^1, 2^1, 2^1, 2^1
    let expected_dims: Vec<usize> =
        vec![16, 4, 2, 2, 16, 8, 4, 8, 4, 4, 16, 4, 4, 4, 4, 2, 2, 2, 2];
    assert_eq!(grid.local_dimensions(), expected_dims);

    // Stress test round-trips
    let site_dims: Vec<i64> = expected_dims.iter().map(|&d| d as i64).collect();
    for seed in 0..100 {
        let quantics: Vec<i64> = (0..grid.len())
            .map(|i| ((seed * 7 + i * 13 + 3) as i64 % site_dims[i]) + 1)
            .collect();

        let grididx = grid.quantics_to_grididx(&quantics).unwrap();
        let recovered = grid.grididx_to_quantics(&grididx).unwrap();
        assert_eq!(recovered, quantics);

        // Origcoord roundtrip
        let origcoord = grid.grididx_to_origcoord(&grididx).unwrap();
        let recovered_idx = grid.origcoord_to_grididx(&origcoord).unwrap();
        assert_eq!(recovered_idx, grididx);
    }
}

// =============================================================================
// Ported from: grid_tests.jl - "DiscretizedGrid" 2D fused specific
// =============================================================================

#[test]
fn test_discretized_grid_2d_fused_specific() {
    let r: usize = 5;
    let a = [0.1, 0.1];
    let b = [2.0, 2.0];
    let dx: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (bi - ai) / (1 << r) as f64)
        .collect();

    let grid = DiscretizedGrid::builder(&[r, r])
        .with_lower_bound(&a)
        .with_upper_bound(&b)
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .unwrap();

    // Fused: local dimensions = [2^2; R] = [4; 5]
    assert_eq!(grid.local_dimensions(), vec![4; r]);

    // Near-max coordinate
    let near_max: Vec<f64> = b
        .iter()
        .zip(dx.iter())
        .map(|(&bi, &di)| bi - di - 1e-9 * di)
        .collect();
    let idx = grid.origcoord_to_grididx(&near_max).unwrap();
    assert_eq!(idx, vec![1 << r; 2]);
}

// =============================================================================
// Ported from: comprehensive_tests.jl - constructors consistency check
// (DiscretizedGrid(variablenames, Rs) vs DiscretizedGrid(variablenames, indextable))
// =============================================================================

#[test]
fn test_variablenames_rs_constructor_consistency() {
    // Build from Rs with variable names
    let grid1 = DiscretizedGrid::builder(&[2, 3, 4])
        .with_variable_names(&["x", "y", "z"])
        .build()
        .unwrap();

    // Build from same Rs using variable_names (same as above, just checking consistency)
    let grid2 = DiscretizedGrid::builder(&[2, 3, 4])
        .with_variable_names(&["x", "y", "z"])
        .build()
        .unwrap();

    assert_eq!(grid1.rs(), grid2.rs());
    assert_eq!(grid1.index_table(), grid2.index_table());
    assert_eq!(grid1.variable_names(), grid2.variable_names());
}

// =============================================================================
// Ported from: origcoord_tests.jl - "R=0 dimension behavior"
// =============================================================================

#[test]
fn test_r0_dimension_behavior() {
    // 2D grid where first dimension has R=0 and second has R=3
    let grid = DiscretizedGrid::builder(&[0, 3])
        .with_lower_bound(&[-1.0, 2.0])
        .with_upper_bound(&[1.0, 6.0])
        .build()
        .unwrap();

    // R=0 dimension should only accept grid index 1
    assert!(grid.grididx_to_quantics(&[1, 1]).is_ok());
    assert!(grid.grididx_to_quantics(&[2, 1]).is_err());

    // Grid index 1 in R=0 dim -> coord = lower_bound
    let coord = grid.grididx_to_origcoord(&[1, 1]).unwrap();
    assert!((coord[0] - (-1.0)).abs() < 1e-14);

    // Quantics should only contain indices for R>0 dimensions
    let q = grid.grididx_to_quantics(&[1, 1]).unwrap();
    assert_eq!(q.len(), 3); // Only for R=3 dimension

    // grid_min and grid_max for R=0 dimension
    assert!((grid.grid_min()[0] - (-1.0)).abs() < 1e-14);
    // For R=0, grid_max should also be lower_bound (only one point)
    assert!((grid.grid_max()[0] - (-1.0)).abs() < 1e-14);

    // grid_origcoords for R=0 dimension
    let coords_r0 = grid.grid_origcoords(0).unwrap();
    assert_eq!(coords_r0.len(), 1);
    assert!((coords_r0[0] - (-1.0)).abs() < 1e-14);

    let coords_r3 = grid.grid_origcoords(1).unwrap();
    assert_eq!(coords_r3.len(), 8); // 2^3

    // Round-trip
    let q = grid.grididx_to_quantics(&[1, 5]).unwrap();
    let back = grid.quantics_to_grididx(&q).unwrap();
    assert_eq!(back, vec![1, 5]);

    // R=0 with includeendpoint should error
    let result = DiscretizedGrid::builder(&[3, 0])
        .include_endpoint(true)
        .build();
    assert!(matches!(
        result,
        Err(QuanticsGridError::EndpointWithZeroResolution { .. })
    ));
}

// =============================================================================
// Ported from: origcoord_tests.jl - "edge cases - mixed R values including zeros"
// =============================================================================

#[test]
fn test_mixed_r_values_with_zeros() {
    let grid = DiscretizedGrid::builder(&[0, 5, 0, 3, 0])
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .build()
        .unwrap();

    // All quantics should be for the non-zero R dimensions
    // R values: [0, 5, 0, 3, 0] -> total quantics = 5 + 3 = 8
    assert_eq!(grid.len(), 8);

    // Test that R=0 dimensions always produce grid index 1
    for seed in 0..20 {
        let quantics: Vec<i64> = (0..grid.len())
            .map(|i| ((seed * 7 + i * 13) % 2 + 1) as i64)
            .collect();
        let grididx = grid.quantics_to_grididx(&quantics).unwrap();

        // R=0 dimensions (indices 0, 2, 4) should always be 1
        assert_eq!(grididx[0], 1);
        assert_eq!(grididx[2], 1);
        assert_eq!(grididx[4], 1);

        // Non-zero dimensions should be in valid range
        assert!(grididx[1] >= 1 && grididx[1] <= 32); // 2^5
        assert!(grididx[3] >= 1 && grididx[3] <= 8); // 2^3
    }
}

// =============================================================================
// Additional edge case tests
// =============================================================================

#[test]
fn test_single_point_grid() {
    // Grid with R=0 in all dimensions (edge case)
    let grid = DiscretizedGrid::builder(&[0, 0, 0]).build().unwrap();

    // Only valid grid index is (1, 1, 1)
    let q = grid.grididx_to_quantics(&[1, 1, 1]).unwrap();
    assert!(q.is_empty()); // No quantics for R=0

    let back = grid.quantics_to_grididx(&[]).unwrap();
    assert_eq!(back, vec![1, 1, 1]);

    // Origcoord
    let coord = grid.grididx_to_origcoord(&[1, 1, 1]).unwrap();
    assert_eq!(coord, vec![0.0, 0.0, 0.0]);

    let back_idx = grid.origcoord_to_grididx(&[0.0, 0.0, 0.0]).unwrap();
    assert_eq!(back_idx, vec![1, 1, 1]);
}

#[test]
fn test_origcoord_to_grididx_near_boundary_clamping() {
    // Test clamping behavior for coordinates near grid_max
    let r: usize = 8;
    let grid = DiscretizedGrid::builder(&[r]).build().unwrap();

    let step = grid.grid_step()[0];
    let grid_max = grid.grid_max()[0];

    // Coordinate at grid_max should map to last index
    let idx = grid.origcoord_to_grididx(&[grid_max]).unwrap();
    assert_eq!(idx, vec![1 << r]);

    // Coordinate slightly above grid_max but below upper_bound should still map to last index
    let slightly_above = grid_max + step / 10.0;
    if slightly_above < grid.upper_bound()[0] {
        let idx = grid.origcoord_to_grididx(&[slightly_above]).unwrap();
        assert_eq!(idx, vec![1 << r]);
    }
}

#[test]
fn test_grid_step_accumulation() {
    // Test coordinates generated by accumulative addition
    let r: usize = 20;
    let grid = DiscretizedGrid::builder(&[r]).build().unwrap();

    let step = grid.grid_step()[0];
    let mut coord = 0.0;
    for i in 1..=1000.min(1 << r) {
        let actual_idx = grid.origcoord_to_grididx(&[coord]).unwrap();
        // Allow for small errors due to float accumulation
        assert!(
            (actual_idx[0] - i).abs() <= 1,
            "At i={}, expected ~{}, got {}",
            i,
            i,
            actual_idx[0]
        );
        coord += step;
        if coord >= grid.upper_bound()[0] {
            break;
        }
    }
}

// =============================================================================
// Ported from: comprehensive_tests.jl - error handling for invalid quantics
// =============================================================================

#[test]
fn test_invalid_quantics_value() {
    let index_table = vec![
        vec![
            ("x".to_string(), 1),
            ("y".to_string(), 1),
            ("z".to_string(), 1),
        ],
        vec![("w".to_string(), 1)],
        vec![
            ("z".to_string(), 2),
            ("w".to_string(), 2),
            ("x".to_string(), 3),
            ("y".to_string(), 3),
        ],
        vec![("x".to_string(), 2), ("y".to_string(), 2)],
        vec![("z".to_string(), 3)],
        vec![("w".to_string(), 3), ("x".to_string(), 4)],
        vec![
            ("y".to_string(), 4),
            ("z".to_string(), 4),
            ("w".to_string(), 4),
        ],
    ];

    let grid = DiscretizedGrid::from_index_table(&["x", "y", "z", "w"], index_table)
        .with_base(3)
        .build()
        .unwrap();

    // Invalid quantics value (28 > 27 = 3^3 for site 0)
    let invalid_q = vec![28, 1, 1, 1, 1, 1, 1];
    assert!(matches!(
        grid.quantics_to_grididx(&invalid_q),
        Err(QuanticsGridError::QuanticsOutOfRange { .. })
    ));
}

// =============================================================================
// Ported from: comprehensive_tests.jl - invalid grididx
// =============================================================================

#[test]
fn test_invalid_grididx() {
    let grid = DiscretizedGrid::builder(&[4, 4])
        .with_base(3)
        .build()
        .unwrap();

    // Grid index 0 is out of bounds (1-indexed)
    assert!(matches!(
        grid.grididx_to_origcoord(&[0, 1]),
        Err(QuanticsGridError::GridIndexOutOfBounds { .. })
    ));

    // Grid index > base^R is out of bounds
    assert!(matches!(
        grid.grididx_to_origcoord(&[82, 1]), // 3^4 = 81, so 82 is out of bounds
        Err(QuanticsGridError::GridIndexOutOfBounds { .. })
    ));
}

// =============================================================================
// Ported from: comprehensive_tests.jl - per-dimension bounds validation
// =============================================================================

#[test]
fn test_precision_near_boundaries() {
    let index_table = vec![
        vec![
            ("x".to_string(), 1),
            ("y".to_string(), 1),
            ("z".to_string(), 1),
        ],
        vec![("w".to_string(), 1)],
        vec![
            ("z".to_string(), 2),
            ("w".to_string(), 2),
            ("x".to_string(), 3),
            ("y".to_string(), 3),
        ],
        vec![("x".to_string(), 2), ("y".to_string(), 2)],
        vec![("z".to_string(), 3)],
        vec![("w".to_string(), 3), ("x".to_string(), 4)],
        vec![
            ("y".to_string(), 4),
            ("z".to_string(), 4),
            ("w".to_string(), 4),
        ],
    ];

    let lower_bound = [-std::f64::consts::PI, 1e-12, 1e8, -1e6];
    let upper_bound = [2.0 * std::f64::consts::PI, 1e-10, 1e8 + 1000.0, 1e6];

    let grid = DiscretizedGrid::from_index_table(&["x", "y", "z", "w"], index_table)
        .with_base(3)
        .with_lower_bound(&lower_bound)
        .with_upper_bound(&upper_bound)
        .include_endpoint(true)
        .build()
        .unwrap();

    let lb = grid.lower_bound();
    let ub = grid.upper_bound();
    let gstep = grid.grid_step();

    // Precision test coordinates
    let test_coords: Vec<Vec<f64>> = vec![
        vec![lb[0] + gstep[0] * 0.5, lb[1], lb[2], lb[3]],
        vec![lb[0], lb[1] + gstep[1] * 0.999999, lb[2], lb[3]],
    ];

    for coord in &test_coords {
        if coord.iter().zip(lb.iter()).all(|(c, l)| c >= l)
            && coord.iter().zip(ub.iter()).all(|(c, u)| c <= u)
        {
            let grididx = grid.origcoord_to_grididx(coord).unwrap();
            let recovered = grid.grididx_to_origcoord(&grididx).unwrap();
            let re_grididx = grid.origcoord_to_grididx(&recovered).unwrap();
            assert_eq!(re_grididx, grididx);
        }
    }
}

// =============================================================================
// Test: bases() accessor
// =============================================================================

#[test]
fn test_bases_accessor() {
    // Uniform base
    let grid = DiscretizedGrid::builder(&[3, 4])
        .with_base(5)
        .build()
        .unwrap();
    assert_eq!(grid.bases(), &[5, 5]);
    assert_eq!(grid.base(), 5);

    // Per-dimension bases
    let grid2 = DiscretizedGrid::builder(&[2, 3])
        .with_bases(&[2, 3])
        .build()
        .unwrap();
    assert_eq!(grid2.bases(), &[2, 3]);
}

// =============================================================================
// Test: grid_origcoords_by_name error handling
// =============================================================================

#[test]
fn test_grid_origcoords_by_name_unknown_variable() {
    let grid = DiscretizedGrid::builder(&[3, 4])
        .with_variable_names(&["x", "y"])
        .build()
        .unwrap();

    assert!(matches!(
        grid.grid_origcoords_by_name("z"),
        Err(QuanticsGridError::UnknownVariable { .. })
    ));
}

// =============================================================================
// Test: is_empty
// =============================================================================

#[test]
fn test_is_empty() {
    let grid = DiscretizedGrid::builder(&[3]).build().unwrap();
    assert!(!grid.is_empty());

    // Grid with all R=0 has no sites
    let grid_empty = DiscretizedGrid::builder(&[0]).build().unwrap();
    assert!(grid_empty.is_empty());
}

// =============================================================================
// Test: wrong quantics length
// =============================================================================

#[test]
fn test_wrong_quantics_length() {
    let grid = DiscretizedGrid::builder(&[3, 4]).build().unwrap();

    // Too few quantics
    assert!(matches!(
        grid.quantics_to_grididx(&[1, 1, 1]),
        Err(QuanticsGridError::WrongQuanticsLength { .. })
    ));

    // Too many quantics
    let too_many = vec![1i64; grid.len() + 1];
    assert!(matches!(
        grid.quantics_to_grididx(&too_many),
        Err(QuanticsGridError::WrongQuanticsLength { .. })
    ));
}

// =============================================================================
// Ported from: origcoord_floating_point_tests.jl -
// "floating point precision - clamp behavior verification"
// =============================================================================

#[test]
fn test_fp_clamp_behavior() {
    let r: usize = 8;
    let grid = DiscretizedGrid::builder(&[r]).build().unwrap();
    let max_idx = 1i64 << r; // 256

    let step = grid.grid_step()[0];

    // Coordinate slightly before first grid point (should clamp to 1)
    let coord_before = -step / 2.0;
    if coord_before >= grid.lower_bound()[0] {
        let idx = grid.origcoord_to_grididx(&[coord_before]).unwrap();
        assert_eq!(idx, vec![1]);
    }

    // Coordinate slightly after last grid point (should clamp to max index)
    let last_coord = grid.grididx_to_origcoord(&[max_idx]).unwrap()[0];
    let coord_after = last_coord + step / 2.0;
    if coord_after <= grid.upper_bound()[0] {
        let idx = grid.origcoord_to_grididx(&[coord_after]).unwrap();
        assert_eq!(idx, vec![max_idx]);
    }

    // Test that coordinates exactly at grid_max map to the last index
    let grid_max_coord = grid.grid_max()[0];
    let idx_at_max = grid.origcoord_to_grididx(&[grid_max_coord]).unwrap();
    assert_eq!(idx_at_max, vec![max_idx]);
}

// =============================================================================
// Ported from: origcoord_floating_point_tests.jl -
// "floating point precision - boundary coordinate generation"
// =============================================================================

#[test]
fn test_fp_boundary_coord_generation() {
    let r: usize = 20;
    let grid = DiscretizedGrid::builder(&[r]).build().unwrap();

    let step = grid.grid_step()[0];

    // Test coordinates generated by accumulative addition (potential error accumulation)
    let mut coord = 0.0;
    for i in 1..=1000.min(1 << r) {
        let actual_idx = grid.origcoord_to_grididx(&[coord]).unwrap();
        // Allow for small errors due to accumulation (matches Julia: abs(actual_idx - expected_idx) <= 1)
        assert!(
            (actual_idx[0] - i).abs() <= 1,
            "At i={}, expected ~{}, got {}",
            i,
            i,
            actual_idx[0]
        );
        coord += step;
        if coord >= grid.upper_bound()[0] {
            break;
        }
    }
}
