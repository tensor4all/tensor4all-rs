use std::error::Error;
use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use tensor4all_core::{IndexLike, SvdTruncationPolicy};
use tensor4all_quanticstci::{
    quanticscrossinterpolate_discrete, DiscretizedGrid, InherentDiscreteGrid, QtciOptions,
    UnfoldingScheme,
};
use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};
use tensor4all_treetn::{
    contraction::{ContractionMethod, ContractionOptions},
    partial_contract, tensor_train_to_treetn, PartialContractionSpec, TruncationOptions,
};

use tensor4all_treetn::Operator;

use tensor4all_tutorial_code::{
    output_paths, qtt_affine_common, qtt_elementwise_product_utils, qtt_fourier_common,
    qtt_function_utils, qtt_integral_sweep_utils, qtt_interval_common, qtt_interval_utils,
    qtt_multivariate_common, qtt_partial_fourier2d_common, qtt_r_sweep_utils,
};

const FUNCTION_BITS: usize = 7;
const FUNCTION_NPOINTS: usize = 1 << FUNCTION_BITS;
const FUNCTION_TOLERANCE: f64 = 1e-12;
const FUNCTION_MAX_BOND_DIM: usize = 32;
const SINE_FREQUENCY: f64 = 10.0;

type QttDemoOutput = (
    tensor4all_quanticstci::QuanticsTensorCI2<f64>,
    Vec<usize>,
    Vec<f64>,
);

fn scratch_dir(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX_EPOCH")
        .as_nanos();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("verification")
        .join(format!("{name}-{nonce}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn read_lines(path: &Path) -> Vec<String> {
    fs::read_to_string(path)
        .expect("read generated csv")
        .lines()
        .map(|line| line.to_owned())
        .collect()
}

fn fourier_input_grid(
    config: &qtt_fourier_common::FourierTutorialConfig,
) -> Result<DiscretizedGrid, Box<dyn Error>> {
    Ok(DiscretizedGrid::builder(&[config.bits])
        .with_variable_names(&["x"])
        .with_bounds(config.x_lower_bound, config.x_upper_bound)
        .include_endpoint(config.include_endpoint)
        .build()?)
}

fn fourier_frequency_grid(
    config: &qtt_fourier_common::FourierTutorialConfig,
) -> Result<DiscretizedGrid, Box<dyn Error>> {
    let (k_lower_bound, k_upper_bound) = qtt_fourier_common::physical_frequency_bounds(config);

    Ok(DiscretizedGrid::builder(&[config.bits])
        .with_variable_names(&["k"])
        .with_bounds(k_lower_bound, k_upper_bound)
        .include_endpoint(config.include_endpoint)
        .build()?)
}

fn function_target(x: f64) -> f64 {
    x.cosh()
}

fn multivariate_target(x: f64, y: f64) -> f64 {
    (20.0 * PI * x * y).cos() / 1000.0
}

fn factor_b_target(x: f64) -> f64 {
    (SINE_FREQUENCY * x).sin()
}

fn build_function_qtt(target_fn: fn(f64) -> f64) -> Result<QttDemoOutput, Box<dyn Error>> {
    let sizes = [FUNCTION_NPOINTS];
    let options = QtciOptions::default()
        .with_tolerance(FUNCTION_TOLERANCE)
        .with_maxbonddim(FUNCTION_MAX_BOND_DIM)
        .with_nrandominitpivot(0)
        .with_unfoldingscheme(UnfoldingScheme::Interleaved)
        .with_verbosity(0);

    let callback = move |idx: &[i64]| -> f64 {
        let x = (idx[0] as f64 - 1.0) / FUNCTION_NPOINTS as f64;
        target_fn(x)
    };

    let initial_pivots = vec![
        vec![1],
        vec![(FUNCTION_NPOINTS / 2) as i64],
        vec![FUNCTION_NPOINTS as i64],
    ];

    Ok(quanticscrossinterpolate_discrete(
        &sizes,
        callback,
        Some(initial_pivots),
        options,
    )?)
}

#[test]
fn output_paths_support_default_and_override() {
    let manifest_dir = Path::new("/tmp/project-root");

    let default_dir = output_paths::resolve_data_dir(manifest_dir, None);
    assert_eq!(default_dir, manifest_dir.join("docs").join("data"));

    let override_dir = output_paths::resolve_data_dir(manifest_dir, Some("/tmp/check/data"));
    assert_eq!(override_dir, PathBuf::from("/tmp/check/data"));
}

#[test]
fn qtt_function_demo_stays_accurate_and_exports_stable_csv() -> Result<(), Box<dyn Error>> {
    let (qtci, ranks, errors) = build_function_qtt(function_target)?;
    let tt = qtci.tensor_train();
    let samples = qtt_function_utils::collect_samples(&qtci, FUNCTION_NPOINTS, function_target)?;

    assert_eq!(samples.len(), FUNCTION_NPOINTS);
    assert!(!ranks.is_empty());
    assert!(!errors.is_empty());
    assert!(qtci.rank() > 0);
    assert!(tt.len() > 0);

    let max_abs_error = samples
        .iter()
        .map(|sample| sample.abs_error)
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_error < 1e-9,
        "unexpectedly large function-demo error: {max_abs_error}"
    );

    let first = samples.first().expect("at least one sample");
    assert!((first.x - 0.0).abs() < f64::EPSILON);
    assert!((first.exact - 1.0).abs() < 1e-12);
    assert!((first.qtt - first.exact).abs() < 1e-9);

    let middle = &samples[FUNCTION_NPOINTS / 2];
    assert!(middle.x > 0.0);
    assert!(middle.exact.is_finite());
    assert!((middle.exact - middle.qtt).abs() < 1e-9);

    let scratch = scratch_dir("qtt-function");
    let samples_path = scratch.join("samples.csv");
    let bond_dims_path = scratch.join("bond_dims.csv");
    qtt_function_utils::write_samples_csv(&samples_path, &samples)?;
    qtt_function_utils::write_bond_dims_csv(&bond_dims_path, &qtci.link_dims())?;

    let sample_lines = read_lines(&samples_path);
    assert_eq!(
        sample_lines.first().map(String::as_str),
        Some("index,x,exact,qtt,abs_error")
    );
    assert_eq!(sample_lines.len(), samples.len() + 1);
    assert!(sample_lines
        .get(1)
        .expect("first sample row")
        .starts_with("1,0.0000000000000000,1.0000000000000000,"));

    let bond_lines = read_lines(&bond_dims_path);
    assert_eq!(
        bond_lines.first().map(String::as_str),
        Some("bond_index,bond_dim")
    );
    assert_eq!(bond_lines.len(), qtci.link_dims().len() + 1);

    Ok(())
}

#[test]
fn qtt_interval_demo_tracks_the_interval_and_exports_stable_csv() -> Result<(), Box<dyn Error>> {
    let config = qtt_interval_common::DEFAULT_INTERVAL_CONFIG;
    let grid = qtt_interval_common::build_interval_grid(&config)?;
    let (qtci, ranks, errors) = qtt_interval_common::build_interval_qtt(
        &grid,
        qtt_interval_common::interval_target,
        &config,
    )?;
    let tt = qtci.tensor_train();
    let samples =
        qtt_interval_utils::collect_samples(&qtci, &grid, qtt_interval_common::interval_target)?;

    let coords = grid.grid_origcoords(0)?;
    assert_eq!(samples.len(), coords.len());
    assert!(!ranks.is_empty());
    assert!(!errors.is_empty());
    assert!(qtci.rank() > 0);
    assert!(tt.len() > 0);

    let first_coord = coords
        .first()
        .copied()
        .expect("interval grid starts with a coordinate");
    let last_coord = coords
        .last()
        .copied()
        .expect("interval grid ends with a coordinate");
    assert!((first_coord - config.lower_bound).abs() < 1e-12);
    assert!((last_coord - config.upper_bound).abs() < 1e-12);

    let max_abs_error = samples
        .iter()
        .map(|sample| sample.abs_error)
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_error < 1e-9,
        "unexpectedly large interval-demo error: {max_abs_error}"
    );

    let first = samples.first().expect("at least one sample");
    assert!((first.x - config.lower_bound).abs() < 1e-12);
    assert!((first.exact - qtt_interval_common::interval_target(first.x)).abs() < 1e-12);
    assert!((first.qtt - first.exact).abs() < 1e-9);

    let last = samples.last().expect("at least one sample");
    assert!((last.x - config.upper_bound).abs() < 1e-12);
    assert!((last.exact - qtt_interval_common::interval_target(last.x)).abs() < 1e-12);
    assert!((last.qtt - last.exact).abs() < 1e-9);

    let scratch = scratch_dir("qtt-interval");
    let samples_path = scratch.join("samples.csv");
    let bond_dims_path = scratch.join("bond_dims.csv");
    qtt_interval_utils::write_samples_csv(&samples_path, &samples)?;
    qtt_interval_utils::write_bond_dims_csv(&bond_dims_path, &qtci.link_dims())?;

    let sample_lines = read_lines(&samples_path);
    assert_eq!(
        sample_lines.first().map(String::as_str),
        Some("index,x,exact,qtt,abs_error")
    );
    assert_eq!(sample_lines.len(), samples.len() + 1);
    assert!(sample_lines
        .get(1)
        .expect("first interval sample row")
        .starts_with("1,-1.0000000000000000,1.0000000000000000,"));

    let bond_lines = read_lines(&bond_dims_path);
    assert_eq!(
        bond_lines.first().map(String::as_str),
        Some("bond_index,bond_dim")
    );
    assert_eq!(bond_lines.len(), qtci.link_dims().len() + 1);

    Ok(())
}

#[test]
fn qtt_interval_integral_matches_the_analytic_answer() -> Result<(), Box<dyn Error>> {
    let config = qtt_interval_common::DEFAULT_INTERVAL_CONFIG;
    let grid = qtt_interval_common::build_interval_grid(&config)?;
    let (qtci, ranks, errors) = qtt_interval_common::build_interval_qtt(
        &grid,
        qtt_interval_common::interval_target,
        &config,
    )?;

    assert!(!ranks.is_empty());
    assert!(!errors.is_empty());

    let integral = qtci.integral()?;
    let expected = qtt_interval_common::exact_integral(&config);
    let abs_error = (integral - expected).abs();

    assert!(
        abs_error < 8e-2,
        "integral = {integral}, expected = {expected}, abs_error = {abs_error}"
    );

    Ok(())
}

#[test]
fn qtt_interval_exact_integral_uses_the_configured_bounds() {
    let config = qtt_interval_common::IntervalTutorialConfig {
        bits: 7,
        lower_bound: 0.0,
        upper_bound: 3.0,
        include_endpoint: true,
    };

    assert!((qtt_interval_common::exact_integral(&config) - 9.0).abs() < 1e-12);
}

#[test]
fn qtt_fourier_gaussian_input_builds_stable_qtt() -> Result<(), Box<dyn Error>> {
    let config = qtt_fourier_common::DEFAULT_FOURIER_CONFIG;
    let input_grid = fourier_input_grid(&config)?;
    let frequency_grid = fourier_frequency_grid(&config)?;
    let (qtci, ranks, errors) = qtt_fourier_common::build_gaussian_qtt(&input_grid, &config)?;

    let input_coords = input_grid.grid_origcoords(0)?;
    let frequency_coords = frequency_grid.grid_origcoords(0)?;

    assert_eq!(input_coords.len(), 1usize << config.bits);
    assert_eq!(frequency_coords.len(), 1usize << config.bits);
    assert!(!ranks.is_empty());
    assert!(!errors.is_empty());

    let first_x = input_coords
        .first()
        .copied()
        .expect("input grid has points");
    let first_value = qtci.evaluate(&[1])?;
    let exact_first = qtt_fourier_common::gaussian_target(first_x);
    assert!((first_value - exact_first).abs() < 1e-9);

    let reference = qtt_fourier_common::gaussian_fourier_reference(0.0);
    assert!((reference.re - (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-12);
    assert!(reference.im.abs() < 1e-12);

    Ok(())
}

#[test]
fn qtt_fourier_operator_matches_gaussian_transform() -> Result<(), Box<dyn Error>> {
    let config = qtt_fourier_common::DEFAULT_FOURIER_CONFIG;
    let input_grid = fourier_input_grid(&config)?;
    let frequency_grid = fourier_frequency_grid(&config)?;
    let (qtci, ranks, errors) = qtt_fourier_common::build_gaussian_qtt(&input_grid, &config)?;
    let operator = qtt_fourier_common::build_fourier_operator(&config)?;
    let (transformed, site_indices) =
        qtt_fourier_common::transform_gaussian(&qtci, &operator, &config)?;
    let samples = qtt_fourier_common::collect_samples(
        &transformed,
        &site_indices,
        &input_grid,
        &frequency_grid,
        &config,
    )?;
    let input_bond_dims = qtci.link_dims();
    let transformed_bond_dims = qtt_fourier_common::tree_link_dims(&transformed);
    let bond_dims = qtt_fourier_common::collect_bond_dims_from_profiles(
        &input_bond_dims,
        &transformed_bond_dims,
    );

    assert!(!ranks.is_empty());
    assert!(!errors.is_empty());
    assert_eq!(samples.len(), 1usize << config.bits);
    assert_eq!(bond_dims.len(), input_bond_dims.len());

    let max_abs_error = samples
        .iter()
        .map(|sample| sample.abs_error)
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_error < 1e-9,
        "unexpectedly large Fourier error: {max_abs_error}"
    );

    let center = samples
        .iter()
        .find(|sample| sample.k.abs() < 1e-12)
        .expect("frequency grid should include zero");
    assert!((center.qtt_re - center.analytic_re).abs() < 1e-9);
    assert!(center.qtt_im.abs() < 1e-9);

    let scratch = scratch_dir("qtt-fourier");
    let samples_path = scratch.join("samples.csv");
    let bond_dims_path = scratch.join("bond_dims.csv");
    qtt_fourier_common::write_samples_csv(&samples_path, &samples)?;
    qtt_fourier_common::write_bond_dims_csv(&bond_dims_path, &bond_dims)?;

    let sample_lines = read_lines(&samples_path);
    assert_eq!(
        sample_lines.first().map(String::as_str),
        Some("index,x,k,analytic_re,analytic_im,qtt_re,qtt_im,abs_error")
    );
    assert_eq!(sample_lines.len(), samples.len() + 1);

    let bond_lines = read_lines(&bond_dims_path);
    assert_eq!(
        bond_lines.first().map(String::as_str),
        Some("bond_index,input_bond_dim,transformed_bond_dim")
    );
    assert_eq!(bond_lines.len(), bond_dims.len() + 1);

    Ok(())
}

#[test]
fn qtt_fourier_transform_uses_aligned_state_indices() -> Result<(), Box<dyn Error>> {
    let config = qtt_fourier_common::DEFAULT_FOURIER_CONFIG;
    let input_grid = fourier_input_grid(&config)?;
    let (qtci, _, _) = qtt_fourier_common::build_gaussian_qtt(&input_grid, &config)?;
    let operator = qtt_fourier_common::build_fourier_operator(&config)?;

    let (_transformed, output_site_indices) =
        qtt_fourier_common::transform_gaussian(&qtci, &operator, &config)?;

    assert_eq!(output_site_indices.len(), config.bits);
    for (site, output_index) in output_site_indices.iter().enumerate() {
        let operator_output_index = &operator
            .get_output_mapping(&site)
            .expect("Fourier output mapping should exist")
            .true_index;
        assert!(
            !output_index.same_id(operator_output_index),
            "transform_gaussian should not expose the operator's original output index IDs"
        );
    }

    Ok(())
}

#[test]
fn qtt_fourier_operator_bond_profile_is_exportable() -> Result<(), Box<dyn Error>> {
    let config = qtt_fourier_common::DEFAULT_FOURIER_CONFIG;
    let operator = qtt_fourier_common::build_fourier_operator(&config)?;
    let bond_dims = qtt_fourier_common::tree_link_dims(&operator.mpo);

    assert_eq!(bond_dims.len(), config.bits.saturating_sub(1));
    assert!(bond_dims.iter().all(|&bond_dim| bond_dim >= 1));

    let scratch = scratch_dir("qtt-fourier-operator-bonds");
    let path = scratch.join("operator_bond_dims.csv");
    qtt_fourier_common::write_fourier_operator_bond_dims_csv(&path, &bond_dims)?;

    let lines = read_lines(&path);
    assert_eq!(
        lines.first().map(String::as_str),
        Some("bond_index,bond_dim")
    );
    assert_eq!(lines.len(), bond_dims.len() + 1);

    Ok(())
}

#[test]
fn qtt_integral_sweep_utils_export_stable_csv_rows() -> Result<(), Box<dyn Error>> {
    use qtt_integral_sweep_utils::{write_sweep_csv, IntegralSweepRow};

    let scratch = scratch_dir("qtt-integral-sweep");
    let csv_path = scratch.join("integral_sweep.csv");
    write_sweep_csv(
        &csv_path,
        &[IntegralSweepRow {
            r: 7,
            npoints: 128,
            integral: 3.125,
            exact_integral: 3.0,
            abs_error: 0.125,
            rank: 3,
        }],
    )?;

    let lines = read_lines(&csv_path);
    assert_eq!(
        lines.first().map(String::as_str),
        Some("r,npoints,integral,exact_integral,abs_error,rank")
    );
    assert_eq!(lines.len(), 2);
    assert_eq!(
        lines.get(1).map(String::as_str),
        Some("7,128,3.1250000000000000,3.0000000000000000,0.1250000000000000,3")
    );

    Ok(())
}

#[test]
fn qtt_r_sweep_helpers_cover_index_conversion_and_error_stats() -> Result<(), Box<dyn Error>> {
    use qtt_r_sweep_utils::{max_abs_error, mean_abs_error, write_samples_csv, write_stats_csv};
    use qtt_r_sweep_utils::{SweepSample, SweepStats};

    let samples = vec![
        SweepSample {
            r: 3,
            npoints: 8,
            index: 1,
            x: 0.0,
            exact: 0.0,
            qtt: 0.1,
            abs_error: 0.1,
        },
        SweepSample {
            r: 3,
            npoints: 8,
            index: 8,
            x: 0.875,
            exact: 0.5,
            qtt: 0.25,
            abs_error: 0.25,
        },
    ];

    assert!((qtt_r_sweep_utils::discrete_index_to_unit_interval(1, 8) - 0.0).abs() < 1e-12);
    assert!((qtt_r_sweep_utils::discrete_index_to_unit_interval(8, 8) - 0.875).abs() < 1e-12);
    assert!((mean_abs_error(&samples) - 0.175).abs() < 1e-12);
    assert!((max_abs_error(&samples) - 0.25).abs() < 1e-12);

    let scratch = scratch_dir("qtt-r-sweep");
    let samples_path = scratch.join("samples.csv");
    let stats_path = scratch.join("stats.csv");
    write_samples_csv(&samples_path, &samples)?;
    write_stats_csv(
        &stats_path,
        &[SweepStats {
            r: 3,
            npoints: 8,
            build_time_sec: 0.123,
            mean_abs_error: 0.175,
            max_abs_error: 0.25,
            rank: 4,
        }],
    )?;

    let sample_lines = read_lines(&samples_path);
    assert_eq!(
        sample_lines.first().map(String::as_str),
        Some("r,npoints,index,x,exact,qtt,abs_error")
    );
    assert_eq!(sample_lines.len(), samples.len() + 1);
    assert_eq!(
        sample_lines.get(1).map(String::as_str),
        Some("3,8,1,0.0000000000000000,0.0000000000000000,0.1000000000000000,0.1000000000000000")
    );

    let stats_lines = read_lines(&stats_path);
    assert_eq!(
        stats_lines.first().map(String::as_str),
        Some("r,npoints,build_time_sec,mean_abs_error,max_abs_error,rank")
    );
    assert_eq!(stats_lines.len(), 2);
    assert_eq!(
        stats_lines.get(1).map(String::as_str),
        Some("3,8,0.1230000000000000,0.1750000000000000,0.2500000000000000,4")
    );

    Ok(())
}

#[test]
fn qtt_elementwise_product_helpers_shape_bond_profiles_and_csv_output() -> Result<(), Box<dyn Error>>
{
    use qtt_elementwise_product_utils::{
        collect_bond_profile, discrete_index_to_unit_interval, global_index_to_quantics_sites,
        write_bond_dims_csv, write_samples_csv, SamplePoint,
    };

    assert!((discrete_index_to_unit_interval(1, 8) - 0.0).abs() < 1e-12);
    assert!((discrete_index_to_unit_interval(8, 8) - 0.875).abs() < 1e-12);
    assert_eq!(global_index_to_quantics_sites(1, 4), vec![0, 0, 0, 0]);
    assert_eq!(global_index_to_quantics_sites(6, 4), vec![0, 1, 0, 1]);
    assert_eq!(global_index_to_quantics_sites(16, 4), vec![1, 1, 1, 1]);

    let tt = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let (treetn, _site_indices) = tensor_train_to_treetn(&tt)?;
    let rows = collect_bond_profile(&tt, &tt, &treetn, &treetn)?;

    assert!(!rows.is_empty());
    assert_eq!(rows[0].bond_index, 1);
    assert!(rows.iter().all(|row| row.cosh >= 1));
    assert!(rows.iter().all(|row| row.factor_b >= 1));
    assert!(rows.iter().all(|row| row.product_raw >= 1));
    assert!(rows.iter().all(|row| row.product_compressed >= 1));

    let scratch = scratch_dir("qtt-elementwise-product");
    let samples_path = scratch.join("samples.csv");
    let bond_dims_path = scratch.join("bond_dims.csv");

    write_samples_csv(
        &samples_path,
        &[SamplePoint {
            index: 1,
            x: 0.0,
            cosh_exact: 1.0,
            cosh_qtt: 1.0,
            factor_b_exact: 0.0,
            factor_b_qtt: 0.0,
            product_exact: 0.0,
            product_raw: 0.0,
            product_compressed: 0.0,
            abs_error_raw: 0.0,
            abs_error_compressed: 0.0,
        }],
    )?;
    write_bond_dims_csv(&bond_dims_path, &rows)?;

    let sample_lines = read_lines(&samples_path);
    assert_eq!(
        sample_lines.first().map(String::as_str),
        Some("index,x,cosh_exact,cosh_qtt,factor_b_exact,factor_b_qtt,product_exact,product_raw,product_compressed,abs_error_raw,abs_error_compressed")
    );
    assert_eq!(sample_lines.len(), 2);

    let bond_lines = read_lines(&bond_dims_path);
    assert_eq!(
        bond_lines.first().map(String::as_str),
        Some("bond_index,cosh,factor_b,product_raw,product_compressed")
    );
    assert_eq!(bond_lines.len(), rows.len() + 1);

    Ok(())
}

#[test]
fn qtt_partial_fourier2d_reference_is_gaussian_in_k_and_periodic_in_t() {
    let config = qtt_partial_fourier2d_common::DEFAULT_PARTIAL_FOURIER2D_CONFIG;

    assert!((qtt_partial_fourier2d_common::source_function(0.0, 0.0, &config) - 1.0).abs() < 1e-12);
    assert!(
        (qtt_partial_fourier2d_common::source_function(0.0, 0.5, &config)
            - (-1.0f64).powi(config.t_frequency as i32))
        .abs()
            < 1e-12
    );

    let reference = qtt_partial_fourier2d_common::partial_fourier_reference(0.0, 0.0, &config);
    assert!((reference.re - (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-12);
    assert!(reference.im.abs() < 1e-12);

    let left = qtt_partial_fourier2d_common::partial_fourier_reference(-0.25, 0.25, &config);
    let right = qtt_partial_fourier2d_common::partial_fourier_reference(0.25, 0.25, &config);
    assert!((left.re - right.re).abs() < 1e-12);
    assert!(left.im.abs() < 1e-12);
    assert!(right.im.abs() < 1e-12);
}

#[test]
fn qtt_partial_fourier2d_interleaved_x_nodes_are_even_positions() {
    assert_eq!(
        qtt_partial_fourier2d_common::x_site_node_mapping(4),
        vec![(0, 0), (1, 2), (2, 4), (3, 6)]
    );
    assert_eq!(
        qtt_partial_fourier2d_common::interleaved_site_values(0b101, 0b011, 3),
        vec![1, 0, 0, 1, 1, 1]
    );
}

#[test]
fn qtt_elementwise_product_demo_tracks_the_pointwise_product() -> Result<(), Box<dyn Error>> {
    let (cosh_qtci, cosh_ranks, cosh_errors) = build_function_qtt(function_target)?;
    let (factor_b_qtci, factor_b_ranks, factor_b_errors) = build_function_qtt(factor_b_target)?;

    let cosh_tt = cosh_qtci.tensor_train();
    let factor_b_tt = factor_b_qtci.tensor_train();
    let (cosh_treetn, cosh_site_indices) = tensor_train_to_treetn(&cosh_tt)?;
    let (factor_b_treetn, factor_b_site_indices) = tensor_train_to_treetn(&factor_b_tt)?;

    let diagonal_pairs = cosh_site_indices
        .iter()
        .cloned()
        .zip(factor_b_site_indices.iter().cloned())
        .collect();
    let spec = PartialContractionSpec {
        contract_pairs: vec![],
        diagonal_pairs,
        output_order: Some(cosh_site_indices.clone()),
    };

    let mut center_nodes = cosh_treetn.node_names();
    center_nodes.sort();
    let center = center_nodes[center_nodes.len() / 2];

    let raw_product_options = ContractionOptions::new(ContractionMethod::Naive);
    let product_raw_tn = partial_contract(
        &cosh_treetn,
        &factor_b_treetn,
        &spec,
        &center,
        raw_product_options,
    )?;

    let compression_options = TruncationOptions::default()
        .with_svd_policy(SvdTruncationPolicy::new(1e-12))
        .with_max_rank(64);
    let product_compressed_tn = product_raw_tn
        .clone()
        .truncate([center], compression_options)?;

    let samples = qtt_elementwise_product_utils::collect_samples(
        &cosh_qtci,
        &factor_b_qtci,
        &product_raw_tn,
        &product_compressed_tn,
        &cosh_site_indices,
        FUNCTION_BITS,
        FUNCTION_NPOINTS,
        function_target,
        factor_b_target,
    )?;

    let max_raw_error = samples
        .iter()
        .map(|sample| sample.abs_error_raw)
        .fold(0.0_f64, f64::max);
    let max_compressed_error = samples
        .iter()
        .map(|sample| sample.abs_error_compressed)
        .fold(0.0_f64, f64::max);
    let max_raw_vs_factor_error = samples
        .iter()
        .map(|sample| (sample.cosh_qtt * sample.factor_b_qtt - sample.product_raw).abs())
        .fold(0.0_f64, f64::max);

    assert!(!cosh_ranks.is_empty());
    assert!(!factor_b_ranks.is_empty());
    assert!(!cosh_errors.is_empty());
    assert!(!factor_b_errors.is_empty());
    assert!(
        max_raw_vs_factor_error < 1e-12,
        "raw TreeTN product differs from QTT factor product: {max_raw_vs_factor_error}"
    );
    assert!(
        max_raw_error < 1e-14,
        "raw TreeTN product error was too large: {max_raw_error}"
    );
    assert!(
        max_compressed_error < 2e-14,
        "compressed TreeTN product error was too large: {max_compressed_error}"
    );

    Ok(())
}

#[test]
fn qtt_multivariate_demo_compares_layouts_and_exports_stable_csv() -> Result<(), Box<dyn Error>> {
    let config = qtt_multivariate_common::MultivariateTutorialConfig {
        bits: 3,
        maxbonddim: 16,
        maxiter: 12,
        ..qtt_multivariate_common::DEFAULT_MULTIVARIATE_CONFIG
    };
    let interleaved_grid =
        qtt_multivariate_common::build_multivariate_grid(&config, UnfoldingScheme::Interleaved)?;
    let grouped_grid =
        qtt_multivariate_common::build_multivariate_grid(&config, UnfoldingScheme::Grouped)?;
    let (interleaved, interleaved_ranks, interleaved_errors) =
        qtt_multivariate_common::build_multivariate_qtt(
            &interleaved_grid,
            multivariate_target,
            &config,
        )?;
    let (grouped, grouped_ranks, grouped_errors) = qtt_multivariate_common::build_multivariate_qtt(
        &grouped_grid,
        multivariate_target,
        &config,
    )?;

    let samples =
        qtt_multivariate_common::collect_samples(&interleaved, &grouped, multivariate_target)?;
    let bond_dims =
        qtt_multivariate_common::collect_bond_dims(&interleaved.link_dims(), &grouped.link_dims());

    assert_eq!(samples.len(), 64);
    assert!(!interleaved_ranks.is_empty());
    assert!(!grouped_ranks.is_empty());
    assert!(!interleaved_errors.is_empty());
    assert!(!grouped_errors.is_empty());
    assert!(interleaved.rank() > 0);
    assert!(grouped.rank() > 0);
    assert!(!bond_dims.is_empty());

    let max_interleaved_error = samples
        .iter()
        .map(|sample| sample.interleaved_abs_error)
        .fold(0.0_f64, f64::max);
    let max_grouped_error = samples
        .iter()
        .map(|sample| sample.grouped_abs_error)
        .fold(0.0_f64, f64::max);

    assert!(
        max_interleaved_error < 1e-3,
        "unexpectedly large interleaved error: {max_interleaved_error}"
    );
    assert!(
        max_grouped_error < 1e-3,
        "unexpectedly large grouped error: {max_grouped_error}"
    );

    let scratch = scratch_dir("qtt-multivariate");
    let samples_path = scratch.join("samples.csv");
    let bond_dims_path = scratch.join("bond_dims.csv");
    qtt_multivariate_common::write_samples_csv(&samples_path, &samples)?;
    qtt_multivariate_common::write_bond_dims_csv(&bond_dims_path, &bond_dims)?;

    let sample_lines = read_lines(&samples_path);
    assert_eq!(
        sample_lines.first().map(String::as_str),
        Some("x_index,y_index,x,y,exact,interleaved_qtt,grouped_qtt,interleaved_abs_error,grouped_abs_error")
    );
    assert_eq!(sample_lines.len(), samples.len() + 1);

    let bond_lines = read_lines(&bond_dims_path);
    assert_eq!(
        bond_lines.first().map(String::as_str),
        Some("bond_index,interleaved_bond_dim,grouped_bond_dim")
    );
    assert_eq!(bond_lines.len(), bond_dims.len() + 1);

    Ok(())
}

#[test]
fn qtt_affine_fused_grid_uses_library_site_mapping() -> Result<(), Box<dyn Error>> {
    let grid = InherentDiscreteGrid::builder(&[3, 3])
        .with_variable_names(&["x", "y"])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()?;

    assert_eq!(grid.local_dimensions(), vec![4, 4, 4]);
    assert_eq!(
        grid.grididx_to_quantics(&[0b101 + 1, 0b011 + 1])?
            .into_iter()
            .map(|site| (site - 1) as usize)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );

    Ok(())
}

#[test]
fn qtt_affine_references_distinguish_periodic_and_open_boundaries() {
    let bits = 3;
    let n = 1usize << bits;

    let periodic = qtt_affine_common::transformed_reference(
        n - 1,
        2,
        bits,
        qtt_affine_common::AffineBoundaryMode::Periodic,
    );
    let wrapped = qtt_affine_common::source_function(1, 2, n);
    assert!((periodic - wrapped).abs() < 1e-12);

    let open = qtt_affine_common::transformed_reference(
        n - 1,
        2,
        bits,
        qtt_affine_common::AffineBoundaryMode::Open,
    );
    assert_eq!(open, 0.0);

    let inside_open = qtt_affine_common::transformed_reference(
        1,
        2,
        bits,
        qtt_affine_common::AffineBoundaryMode::Open,
    );
    let inside_expected = qtt_affine_common::source_function(3, 2, n);
    assert!((inside_open - inside_expected).abs() < 1e-12);
}

#[test]
fn qtt_affine_source_qtt_matches_source_function() -> Result<(), Box<dyn Error>> {
    let config = qtt_affine_common::AffineTutorialConfig {
        bits: 4,
        tolerance: 1e-12,
        maxbonddim: 32,
        maxiter: 20,
    };
    let (qtci, ranks, errors) = qtt_affine_common::build_source_qtt(&config)?;

    assert!(!ranks.is_empty());
    assert!(!errors.is_empty());
    assert!(qtci.rank() > 0);

    let evaluation_grid = qtci
        .inherent_grid()
        .expect("source QTT uses an inherent discrete grid");
    let n = qtt_affine_common::point_count(config.bits);
    let grid_fused_sites: Vec<usize> = evaluation_grid
        .grididx_to_quantics(&[0b0101 + 1, 0b0011 + 1])?
        .into_iter()
        .map(|site| (site - 1) as usize)
        .collect();
    assert_eq!(grid_fused_sites, vec![0, 1, 2, 3]);

    let mut max_abs_error = 0.0_f64;
    for x in 0..n {
        for y in 0..n {
            let sites: Vec<usize> = evaluation_grid
                .grididx_to_quantics(&[(x + 1) as i64, (y + 1) as i64])?
                .into_iter()
                .map(|site| (site - 1) as usize)
                .collect();
            let qtt_direct = qtci.evaluate(&[(x + 1) as i64, (y + 1) as i64])?;
            let qtt_fused = qtci.tensor_train().evaluate(&sites)?;
            let exact = qtt_affine_common::source_function(x, y, n);
            max_abs_error = max_abs_error.max((qtt_direct - exact).abs());
            max_abs_error = max_abs_error.max((qtt_fused - exact).abs());
        }
    }

    assert!(
        max_abs_error < 1e-9,
        "unexpectedly large affine source error: {max_abs_error}"
    );

    Ok(())
}

#[test]
fn qtt_affine_pullback_matches_periodic_and_open_ground_truth() -> Result<(), Box<dyn Error>> {
    let config = qtt_affine_common::AffineTutorialConfig {
        bits: 4,
        tolerance: 1e-12,
        maxbonddim: 32,
        maxiter: 20,
    };

    let (source, _, _) = qtt_affine_common::build_source_qtt(&config)?;
    let periodic_operator = qtt_affine_common::build_affine_operator(
        &config,
        qtt_affine_common::AffineBoundaryMode::Periodic,
    )?;
    let open_operator = qtt_affine_common::build_affine_operator(
        &config,
        qtt_affine_common::AffineBoundaryMode::Open,
    )?;
    let evaluation_grid = source
        .inherent_grid()
        .expect("source QTT uses an inherent discrete grid");

    let (periodic, periodic_sites) =
        qtt_affine_common::apply_affine_operator(&source, &periodic_operator)?;
    let (open, open_sites) = qtt_affine_common::apply_affine_operator(&source, &open_operator)?;

    let samples = qtt_affine_common::collect_samples(
        &periodic,
        &periodic_sites,
        &open,
        &open_sites,
        evaluation_grid,
        &config,
    )?;

    let max_periodic_error = samples
        .iter()
        .map(|sample| sample.periodic_abs_error)
        .fold(0.0_f64, f64::max);
    let max_open_error = samples
        .iter()
        .map(|sample| sample.open_abs_error)
        .fold(0.0_f64, f64::max);

    assert!(
        max_periodic_error < 1e-8,
        "unexpectedly large periodic affine error: {max_periodic_error}"
    );
    assert!(
        max_open_error < 1e-8,
        "unexpectedly large open affine error: {max_open_error}"
    );

    let boundary_sample = samples
        .iter()
        .find(|sample| sample.x + sample.y >= qtt_affine_common::point_count(config.bits))
        .expect("open-boundary region exists");
    assert_eq!(boundary_sample.open_exact, 0.0);
    assert!(boundary_sample.open_qtt.abs() < 1e-8);

    Ok(())
}

#[test]
fn qtt_affine_exports_stable_csv_headers() -> Result<(), Box<dyn Error>> {
    let config = qtt_affine_common::AffineTutorialConfig {
        bits: 3,
        tolerance: 1e-12,
        maxbonddim: 32,
        maxiter: 20,
    };

    let (source, _, _) = qtt_affine_common::build_source_qtt(&config)?;
    let periodic_operator = qtt_affine_common::build_affine_operator(
        &config,
        qtt_affine_common::AffineBoundaryMode::Periodic,
    )?;
    let open_operator = qtt_affine_common::build_affine_operator(
        &config,
        qtt_affine_common::AffineBoundaryMode::Open,
    )?;
    let evaluation_grid = source
        .inherent_grid()
        .expect("source QTT uses an inherent discrete grid");
    let (periodic, periodic_sites) =
        qtt_affine_common::apply_affine_operator(&source, &periodic_operator)?;
    let (open, open_sites) = qtt_affine_common::apply_affine_operator(&source, &open_operator)?;

    let samples = qtt_affine_common::collect_samples(
        &periodic,
        &periodic_sites,
        &open,
        &open_sites,
        evaluation_grid,
        &config,
    )?;
    let source_tt = source.tensor_train();
    let (source_treetn, _) = tensor_train_to_treetn(&source_tt)?;
    let bond_dims = qtt_affine_common::collect_bond_dims(
        &qtt_affine_common::tree_link_dims(&source_treetn),
        &qtt_affine_common::tree_link_dims(&periodic),
        &qtt_affine_common::tree_link_dims(&open),
    );
    let operator_bond_dims = qtt_affine_common::collect_operator_bond_dims(
        &qtt_affine_common::tree_link_dims(&periodic_operator.mpo),
        &qtt_affine_common::tree_link_dims(&open_operator.mpo),
    );

    let scratch = scratch_dir("qtt-affine");
    let samples_path = scratch.join("samples.csv");
    let bonds_path = scratch.join("bond_dims.csv");
    let operator_bonds_path = scratch.join("operator_bond_dims.csv");

    qtt_affine_common::write_samples_csv(&samples_path, &samples)?;
    qtt_affine_common::write_bond_dims_csv(&bonds_path, &bond_dims)?;
    qtt_affine_common::write_operator_bond_dims_csv(&operator_bonds_path, &operator_bond_dims)?;

    assert_eq!(
        read_lines(&samples_path).first().map(String::as_str),
        Some("x_index,y_index,x,y,source_u_periodic,source_v,source_exact,periodic_exact,periodic_qtt,periodic_abs_error,open_exact,open_qtt,open_abs_error")
    );
    let first_sample = samples.first().expect("affine samples are not empty");
    assert!(
        (first_sample.source_exact
            - qtt_affine_common::source_function(
                first_sample.x,
                first_sample.y,
                qtt_affine_common::point_count(config.bits),
            ))
        .abs()
            < 1e-12
    );
    assert_eq!(
        read_lines(&bonds_path).first().map(String::as_str),
        Some("bond_index,input_bond_dim,periodic_transformed_bond_dim,open_transformed_bond_dim")
    );
    assert_eq!(
        read_lines(&operator_bonds_path).first().map(String::as_str),
        Some("bond_index,periodic_operator_bond_dim,open_operator_bond_dim")
    );

    Ok(())
}
#[test]
fn qtt_partial_fourier2d_source_qtt_matches_input_function() -> Result<(), Box<dyn Error>> {
    let config = qtt_partial_fourier2d_common::PartialFourier2dConfig {
        bits: 5,
        maxbonddim: 32,
        maxiter: 12,
        ..qtt_partial_fourier2d_common::DEFAULT_PARTIAL_FOURIER2D_CONFIG
    };
    let input_grid = qtt_partial_fourier2d_common::build_input_grid(&config)?;
    let (qtci, ranks, errors) =
        qtt_partial_fourier2d_common::build_source_qtt(&input_grid, &config)?;

    assert!(!ranks.is_empty());
    assert!(!errors.is_empty());
    assert!(qtci.rank() > 0);

    let x_coords = input_grid.grid_origcoords(0)?;
    let t_coords = input_grid.grid_origcoords(1)?;
    let x_index = x_coords.len() / 2;
    let t_index = t_coords.len() / 3;
    let qtt = qtci.evaluate(&[(x_index + 1) as i64, (t_index + 1) as i64])?;
    let exact = qtt_partial_fourier2d_common::source_function(
        x_coords[x_index],
        t_coords[t_index],
        &config,
    );

    assert!((qtt - exact).abs() < 1e-8);

    Ok(())
}

#[test]
fn qtt_partial_fourier2d_operator_targets_only_x_nodes() -> Result<(), Box<dyn Error>> {
    let config = qtt_partial_fourier2d_common::PartialFourier2dConfig {
        bits: 4,
        ..qtt_partial_fourier2d_common::DEFAULT_PARTIAL_FOURIER2D_CONFIG
    };
    let operator = qtt_partial_fourier2d_common::build_partial_fourier_operator(&config)?;
    let nodes = operator.node_names();

    assert_eq!(nodes.len(), config.bits);
    for node in [0usize, 2, 4, 6] {
        assert!(nodes.contains(&node));
    }
    for node in [1usize, 3, 5, 7] {
        assert!(!nodes.contains(&node));
    }

    Ok(())
}

#[test]
fn qtt_partial_fourier2d_operator_matches_analytic_transform() -> Result<(), Box<dyn Error>> {
    let config = qtt_partial_fourier2d_common::PartialFourier2dConfig {
        bits: 6,
        maxbonddim: 64,
        maxiter: 20,
        ..qtt_partial_fourier2d_common::DEFAULT_PARTIAL_FOURIER2D_CONFIG
    };
    let input_grid = qtt_partial_fourier2d_common::build_input_grid(&config)?;
    let frequency_grid = qtt_partial_fourier2d_common::build_frequency_grid(&config)?;
    let (qtci, ranks, errors) =
        qtt_partial_fourier2d_common::build_source_qtt(&input_grid, &config)?;
    let operator = qtt_partial_fourier2d_common::build_partial_fourier_operator(&config)?;
    let (transformed, site_indices) =
        qtt_partial_fourier2d_common::transform_x_dimension(&qtci, &operator)?;
    let samples = qtt_partial_fourier2d_common::collect_samples(
        &transformed,
        &site_indices,
        &frequency_grid,
        &config,
    )?;

    assert!(!ranks.is_empty());
    assert!(!errors.is_empty());
    assert_eq!(site_indices.len(), 2 * config.bits);
    assert_eq!(
        samples.len(),
        (1usize << config.bits) * (1usize << config.bits)
    );

    let max_abs_error = samples
        .iter()
        .map(|sample| sample.abs_error)
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_error < 5e-8,
        "unexpectedly large partial Fourier error: {max_abs_error}"
    );

    let center = samples
        .iter()
        .find(|sample| sample.k.abs() < 1e-12 && sample.t.abs() < 1e-12)
        .expect("frequency/time grid should include k=0 and t=0");
    assert!((center.qtt_re - center.analytic_re).abs() < 5e-8);
    assert!(center.qtt_im.abs() < 5e-8);

    Ok(())
}

#[test]
fn qtt_partial_fourier2d_output_sites_are_interleaved_once() {
    let mut k_sites = qtt_partial_fourier2d_common::global_index_to_quantics_sites(0b110 + 1, 3);
    k_sites.reverse();
    let t_sites = qtt_partial_fourier2d_common::global_index_to_quantics_sites(0b011 + 1, 3);
    let values: Vec<usize> = k_sites
        .into_iter()
        .zip(t_sites)
        .flat_map(|(k_site, t_site)| [k_site, t_site])
        .collect();

    assert_eq!(values, vec![0, 0, 1, 1, 1, 1]);
    assert_eq!(values.len(), 6);
}

#[test]
fn qtt_partial_fourier2d_exports_stable_csv_headers() -> Result<(), Box<dyn Error>> {
    use qtt_partial_fourier2d_common::{
        write_bond_dims_csv, write_operator_bond_dims_csv, write_samples_csv,
        PartialFourier2dSamplePoint,
    };

    let scratch = scratch_dir("qtt-partial-fourier2d");
    let samples_path = scratch.join("samples.csv");
    let bond_dims_path = scratch.join("bond_dims.csv");
    let operator_bond_dims_path = scratch.join("operator_bond_dims.csv");

    write_samples_csv(
        &samples_path,
        &[PartialFourier2dSamplePoint {
            k_index: 1,
            t_index: 1,
            source_x_index: 1,
            k: 0.0,
            t: 0.0,
            analytic_re: 1.0,
            analytic_im: 0.0,
            qtt_re: 1.0,
            qtt_im: 0.0,
            abs_error: 0.0,
        }],
    )?;
    write_bond_dims_csv(&bond_dims_path, &[(1, Some(2), Some(3))])?;
    write_operator_bond_dims_csv(&operator_bond_dims_path, &[4])?;

    assert_eq!(
        read_lines(&samples_path).first().map(String::as_str),
        Some("k_index,t_index,source_x_index,k,t,analytic_re,analytic_im,qtt_re,qtt_im,abs_error")
    );
    assert_eq!(
        read_lines(&bond_dims_path).first().map(String::as_str),
        Some("bond_index,input_bond_dim,transformed_bond_dim")
    );
    assert_eq!(
        read_lines(&operator_bond_dims_path)
            .first()
            .map(String::as_str),
        Some("bond_index,bond_dim")
    );

    Ok(())
}
