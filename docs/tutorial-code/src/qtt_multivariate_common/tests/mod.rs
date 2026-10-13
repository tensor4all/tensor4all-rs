use super::*;
use std::f64::consts::PI;

fn test_multivariate_target(x: f64, y: f64) -> f64 {
    (20.0 * PI * x * y).cos() / 1000.0
}

#[test]
fn multivariate_target_matches_known_values() {
    assert!((test_multivariate_target(0.0, 0.75) - 0.001).abs() < 1e-15);
    assert!((test_multivariate_target(0.25, 0.2) + 0.001).abs() < 1e-15);
}

#[test]
fn default_multivariate_config_uses_smoke_resolution() {
    assert_eq!(DEFAULT_MULTIVARIATE_CONFIG.bits, 7);
    assert_eq!(point_count(&DEFAULT_MULTIVARIATE_CONFIG), 128);
}

#[test]
fn point_count_uses_power_of_two_resolution() {
    let config = MultivariateTutorialConfig {
        bits: 4,
        ..DEFAULT_MULTIVARIATE_CONFIG
    };
    assert_eq!(point_count(&config), 16);
}

#[test]
fn multivariate_grid_is_two_dimensional_and_half_open() -> Result<(), Box<dyn Error>> {
    let config = MultivariateTutorialConfig {
        bits: 3,
        ..DEFAULT_MULTIVARIATE_CONFIG
    };
    let grid = build_multivariate_grid(&config, UnfoldingScheme::Interleaved)?;
    let x = grid.grid_origcoords(0)?;
    let y = grid.grid_origcoords(1)?;

    assert_eq!(grid.ndims(), 2);
    assert_eq!(grid.rs(), &[3, 3]);
    assert_eq!(grid.variable_names(), &["x".to_string(), "y".to_string()]);
    assert_eq!(x.len(), 8);
    assert_eq!(y.len(), 8);
    assert!((x[0] - config.lower_bound).abs() < 1e-15);
    assert!((y[0] - config.lower_bound).abs() < 1e-15);
    assert!(x.last().copied().expect("x has points") < config.upper_bound);
    assert!(y.last().copied().expect("y has points") < config.upper_bound);

    Ok(())
}

#[test]
fn multivariate_grid_uses_requested_unfolding_scheme() -> Result<(), Box<dyn Error>> {
    let config = MultivariateTutorialConfig {
        bits: 3,
        ..DEFAULT_MULTIVARIATE_CONFIG
    };
    let interleaved = build_multivariate_grid(&config, UnfoldingScheme::Interleaved)?;
    let grouped = build_multivariate_grid(&config, UnfoldingScheme::Grouped)?;

    assert_eq!(interleaved.local_dimensions(), vec![2; 6]);
    assert_eq!(grouped.local_dimensions(), vec![2; 6]);
    assert_ne!(
        interleaved.grididx_to_quantics(&[2, 5])?,
        grouped.grididx_to_quantics(&[2, 5])?
    );

    Ok(())
}

#[test]
fn both_unfolding_schemes_build_accurate_small_qtts() -> Result<(), Box<dyn Error>> {
    let config = MultivariateTutorialConfig {
        bits: 4,
        max_bond_dim: 32,
        ..DEFAULT_MULTIVARIATE_CONFIG
    };

    for scheme in [UnfoldingScheme::Interleaved, UnfoldingScheme::Grouped] {
        let grid = build_multivariate_grid(&config, scheme)?;
        let (qtci, ranks, errors) =
            build_multivariate_qtt(&grid, test_multivariate_target, &config)?;

        assert!(qtci.rank() > 0);
        assert!(!ranks.is_empty());
        assert!(!errors.is_empty());

        let value = qtci.evaluate(&[0, 0])?;
        let exact = test_multivariate_target(config.lower_bound, config.lower_bound);
        assert!((value - exact).abs() < 1e-9);
    }

    Ok(())
}

#[test]
fn multivariate_samples_cover_the_cartesian_grid() -> Result<(), Box<dyn Error>> {
    let config = MultivariateTutorialConfig {
        bits: 3,
        max_bond_dim: 32,
        ..DEFAULT_MULTIVARIATE_CONFIG
    };
    let interleaved_grid = build_multivariate_grid(&config, UnfoldingScheme::Interleaved)?;
    let grouped_grid = build_multivariate_grid(&config, UnfoldingScheme::Grouped)?;
    let (interleaved, _, _) =
        build_multivariate_qtt(&interleaved_grid, test_multivariate_target, &config)?;
    let (grouped, _, _) = build_multivariate_qtt(&grouped_grid, test_multivariate_target, &config)?;

    let samples = collect_samples(&interleaved, &grouped, test_multivariate_target)?;

    assert_eq!(samples.len(), 64);
    let first = samples.first().expect("samples are not empty");
    assert_eq!(first.x_index, 1);
    assert_eq!(first.y_index, 1);
    assert!((first.exact - 0.001).abs() < 1e-15);
    assert!(first.interleaved_abs_error < 1e-9);
    assert!(first.grouped_abs_error < 1e-9);

    Ok(())
}

#[test]
fn dense_qtt_batch_evaluation_matches_direct_evaluation() -> Result<(), Box<dyn Error>> {
    let config = MultivariateTutorialConfig {
        bits: 3,
        max_bond_dim: 32,
        ..DEFAULT_MULTIVARIATE_CONFIG
    };
    let grid = build_multivariate_grid(&config, UnfoldingScheme::Grouped)?;
    let (qtt, _, _) = build_multivariate_qtt(&grid, test_multivariate_target, &config)?;

    let dense_values = evaluate_dense_qtt(&qtt)?;

    for (x_index, y_index) in [(0, 0), (1, 4), (7, 7)] {
        let flat_index = x_index * point_count(&config) + y_index;
        let direct = qtt.evaluate(&[x_index, y_index])?;
        assert!((dense_values[flat_index] - direct).abs() < 1e-12);
    }

    Ok(())
}

#[test]
fn multivariate_bond_rows_pair_layout_profiles() {
    let rows = collect_bond_dims(&[2, 3, 2], &[2, 4]);
    assert_eq!(
        rows,
        vec![
            (1, Some(2), Some(2)),
            (2, Some(3), Some(4)),
            (3, Some(2), None)
        ]
    );
}
