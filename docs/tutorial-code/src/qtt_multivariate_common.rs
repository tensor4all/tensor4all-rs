//! Shared setup for the multivariate QTT tutorial.
//!
//! This helper implements a focused two-dimensional tutorial: grid creation,
//! QTT construction for two unfolding schemes, dense sampling, bond-dimension
//! pairing, CSV writers and a compact terminal summary.

use std::error::Error;
use std::fs::File;
use std::io;
use std::io::{BufWriter, Write};
use std::path::Path;

use tensor4all_quanticstci::{
    pointwise_coordinate_batch, quanticscrossinterpolate_batch, DiscretizedGrid, QtciOptions,
    QuanticsTensorCI2, UnfoldingScheme,
};
use tensor4all_simplett::{MultiIndex, TTCache};

/// Configuration for the multivariate tutorial.
#[derive(Debug, Clone, Copy)]
pub struct MultivariateTutorialConfig {
    pub bits: usize,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub include_endpoint: bool,
    pub tolerance: f64,
    pub max_bond_dim: usize,
    pub maxiter: usize,
}

/// Default configuration used throughout the tutorial.
pub const DEFAULT_MULTIVARIATE_CONFIG: MultivariateTutorialConfig = MultivariateTutorialConfig {
    bits: 7,
    lower_bound: -2.0,
    upper_bound: 2.0,
    include_endpoint: false,
    tolerance: 1e-12,
    max_bond_dim: 64,
    maxiter: 20,
};

/// QTT construction result returned by the multivariate helper.
pub type MultivariateQttOutput = (QuanticsTensorCI2<f64>, Vec<usize>, Vec<f64>);

pub fn point_count(config: &MultivariateTutorialConfig) -> usize {
    1usize << config.bits
}

/// Build the two-dimensional discretized grid used by the tutorial.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_multivariate_grid(
    config: &MultivariateTutorialConfig,
    scheme: UnfoldingScheme,
) -> Result<DiscretizedGrid, Box<dyn Error>> {
    Ok(DiscretizedGrid::builder(&[config.bits, config.bits])
        .with_variable_names(&["x", "y"])
        .with_bounds(config.lower_bound, config.upper_bound)
        .with_unfolding_scheme(scheme)
        .include_endpoint(config.include_endpoint)
        .build()?)
}

/// Build a QTT approximation for the provided 2D `target_fn`.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_multivariate_qtt<F>(
    grid: &DiscretizedGrid,
    target_fn: F,
    config: &MultivariateTutorialConfig,
) -> Result<MultivariateQttOutput, Box<dyn Error>>
where
    F: Fn(f64, f64) -> f64 + Copy + 'static,
{
    let options = QtciOptions::default()
        .with_tolerance(config.tolerance)
        .with_max_bond_dim(config.max_bond_dim)
        .with_maxiter(config.maxiter)
        .with_nrandominitpivot(0)
        .with_verbosity(0);

    let f = move |coords: &[f64]| -> f64 { target_fn(coords[0], coords[1]) };
    let npoints = point_count(config);
    // Grid indices are 0-based.
    let initial_pivots = vec![
        vec![0, 0],
        vec![npoints / 2 - 1, npoints / 2 - 1],
        vec![npoints - 1, npoints - 1],
    ];

    Ok(quanticscrossinterpolate_batch(
        grid,
        pointwise_coordinate_batch(f),
        Some(initial_pivots),
        options,
    )?)
}

/// One sample row in the exported dense table.
#[derive(Debug, Clone)]
pub struct MultivariateSamplePoint {
    pub x_index: usize,
    pub y_index: usize,
    pub x: f64,
    pub y: f64,
    pub exact: f64,
    pub interleaved_qtt: f64,
    pub grouped_qtt: f64,
    pub interleaved_abs_error: f64,
    pub grouped_abs_error: f64,
}

pub type BondDimRow = (usize, Option<usize>, Option<usize>);

/// Summary data for one QTT layout.
pub struct LayoutSummary<'a> {
    pub name: &'static str,
    pub qtt: &'a QuanticsTensorCI2<f64>,
    pub ranks: &'a [usize],
    pub errors: &'a [f64],
}

const DENSE_EVALUATION_BATCH_POINTS: usize = 8192;

fn invalid_multivariate_input(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn discretized_grid(qtt: &QuanticsTensorCI2<f64>) -> Result<&DiscretizedGrid, Box<dyn Error>> {
    qtt.discretized_grid()
        .ok_or_else(|| invalid_multivariate_input("QTT result does not contain a discretized grid"))
}

fn validate_dense_grid(grid: &DiscretizedGrid) -> Result<(), Box<dyn Error>> {
    if grid.ndims() != 2 {
        return Err(invalid_multivariate_input(format!(
            "multivariate tutorial expects a 2D grid, got {} dimensions",
            grid.ndims()
        )));
    }

    Ok(())
}

fn dense_quantics_batch(
    grid: &DiscretizedGrid,
    y_count: usize,
    start: usize,
    len: usize,
) -> Result<Vec<MultiIndex>, Box<dyn Error>> {
    let mut indices = Vec::with_capacity(len);

    for flat_index in start..start + len {
        let x_index = flat_index / y_count;
        let y_index = flat_index % y_count;
        let quantics = grid.grididx_to_quantics(&[x_index, y_index])?;
        indices.push(quantics);
    }

    Ok(indices)
}

fn evaluate_dense_qtt(qtt: &QuanticsTensorCI2<f64>) -> Result<Vec<f64>, Box<dyn Error>> {
    let grid = discretized_grid(qtt)?;
    validate_dense_grid(grid)?;

    let x_count = grid.grid_origcoords(0)?.len();
    let y_count = grid.grid_origcoords(1)?.len();
    let total_points = x_count * y_count;

    let tensor_train = qtt.tensor_train();
    let mut cache = TTCache::new(&tensor_train);
    let split = Some(cache.len().max(1) / 2).filter(|&split| split > 0);
    let mut values = Vec::with_capacity(total_points);

    let mut start = 0;
    while start < total_points {
        let batch_len = DENSE_EVALUATION_BATCH_POINTS.min(total_points - start);
        let quantics = dense_quantics_batch(grid, y_count, start, batch_len)?;
        let batch_values = cache.evaluate_many(&quantics, split)?;
        values.extend(batch_values);
        start += batch_len;
    }

    Ok(values)
}

/// Collect dense samples for both layouts on the full cartesian grid.
/// # Errors
///
/// Returns an error when the collection fails (a shape mismatch or
/// /// backend failure).
///
pub fn collect_samples<F>(
    interleaved: &QuanticsTensorCI2<f64>,
    grouped: &QuanticsTensorCI2<f64>,
    exact_fn: F,
) -> Result<Vec<MultivariateSamplePoint>, Box<dyn Error>>
where
    F: Fn(f64, f64) -> f64,
{
    let grid = discretized_grid(interleaved)?;
    validate_dense_grid(grid)?;

    let x_coords = grid.grid_origcoords(0)?;
    let y_coords = grid.grid_origcoords(1)?;
    let interleaved_values = evaluate_dense_qtt(interleaved)?;
    let grouped_values = evaluate_dense_qtt(grouped)?;
    let expected_count = x_coords.len() * y_coords.len();

    if interleaved_values.len() != expected_count || grouped_values.len() != expected_count {
        return Err(invalid_multivariate_input(
            "dense QTT evaluation did not produce the expected sample count",
        ));
    }

    let mut samples = Vec::with_capacity(x_coords.len() * y_coords.len());

    for (x_offset, &x) in x_coords.iter().enumerate() {
        for (y_offset, &y) in y_coords.iter().enumerate() {
            let x_index = x_offset + 1;
            let y_index = y_offset + 1;
            let flat_index = x_offset * y_coords.len() + y_offset;
            let exact = exact_fn(x, y);
            let interleaved_qtt = interleaved_values[flat_index];
            let grouped_qtt = grouped_values[flat_index];

            samples.push(MultivariateSamplePoint {
                x_index,
                y_index,
                x,
                y,
                exact,
                interleaved_qtt,
                grouped_qtt,
                interleaved_abs_error: (exact - interleaved_qtt).abs(),
                grouped_abs_error: (exact - grouped_qtt).abs(),
            });
        }
    }

    Ok(samples)
}

/// Pair two bond-dimension profiles into rows for CSV output.
pub fn collect_bond_dims(interleaved: &[usize], grouped: &[usize]) -> Vec<BondDimRow> {
    let row_count = interleaved.len().max(grouped.len());
    (0..row_count)
        .map(|i| (i + 1, interleaved.get(i).copied(), grouped.get(i).copied()))
        .collect()
}

/// Write dense samples to CSV.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_samples_csv(
    path: &Path,
    samples: &[MultivariateSamplePoint],
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "x_index,y_index,x,y,exact,interleaved_qtt,grouped_qtt,interleaved_abs_error,grouped_abs_error"
    )?;
    for sample in samples {
        writeln!(
            w,
            "{},{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16}",
            sample.x_index,
            sample.y_index,
            sample.x,
            sample.y,
            sample.exact,
            sample.interleaved_qtt,
            sample.grouped_qtt,
            sample.interleaved_abs_error,
            sample.grouped_abs_error
        )?;
    }

    Ok(())
}

/// Write bond-dimension rows to CSV.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_bond_dims_csv(path: &Path, rows: &[BondDimRow]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "bond_index,interleaved_bond_dim,grouped_bond_dim")?;
    for (bond_index, interleaved, grouped) in rows {
        let interleaved_text = interleaved.map(|v| v.to_string()).unwrap_or_default();
        let grouped_text = grouped.map(|v| v.to_string()).unwrap_or_default();
        writeln!(w, "{},{},{}", bond_index, interleaved_text, grouped_text)?;
    }

    Ok(())
}

/// Print a compact summary to the terminal used by the binary.
pub fn print_summary(
    interleaved: LayoutSummary<'_>,
    grouped: LayoutSummary<'_>,
    samples: &[MultivariateSamplePoint],
    config: &MultivariateTutorialConfig,
) {
    let max_interleaved_error = samples
        .iter()
        .map(|sample| sample.interleaved_abs_error)
        .fold(0.0_f64, f64::max);
    let max_grouped_error = samples
        .iter()
        .map(|sample| sample.grouped_abs_error)
        .fold(0.0_f64, f64::max);

    println!("QTT multivariate tutorial");
    println!("bits per dimension = {}", config.bits);
    println!("max QTCI sweeps = {}", config.maxiter);
    println!(
        "domain = [{:.3}, {:.3}) x [{:.3}, {:.3})",
        config.lower_bound, config.upper_bound, config.lower_bound, config.upper_bound
    );
    println!("sample count = {}", samples.len());
    println!("{} rank = {}", interleaved.name, interleaved.qtt.rank());
    println!("{} rank = {}", grouped.name, grouped.qtt.rank());
    println!(
        "{} rank history length = {}",
        interleaved.name,
        interleaved.ranks.len()
    );
    println!(
        "{} rank history length = {}",
        grouped.name,
        grouped.ranks.len()
    );
    println!(
        "{} error history length = {}",
        interleaved.name,
        interleaved.errors.len()
    );
    println!(
        "{} error history length = {}",
        grouped.name,
        grouped.errors.len()
    );
    println!("max interleaved abs error = {:.3e}", max_interleaved_error);
    println!("max grouped abs error = {:.3e}", max_grouped_error);
    println!();
}

#[cfg(test)]
mod tests;
