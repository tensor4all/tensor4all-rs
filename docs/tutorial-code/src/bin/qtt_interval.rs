//! Build a QTT for a scalar function on a physical interval and export sample
//! data for Julia plotting.
//!
//! This tutorial is the first step beyond the simple `[0,1]` example.  It uses
//! `DiscretizedGrid` and `quanticscrossinterpolate(...)` so the function is
//! defined on a real interval instead of only on discrete integer indices.

use std::error::Error;
use std::fs;
use std::path::Path;

use tensor4all_tutorial_code::output_paths;
use tensor4all_tutorial_code::qtt_interval_common::{
    build_interval_grid, build_interval_qtt, interval_target, DEFAULT_INTERVAL_CONFIG,
};
use tensor4all_tutorial_code::qtt_interval_utils::{
    collect_bond_dims, collect_samples, print_summary, write_bond_dims_csv, write_samples_csv,
};

/// How many sample rows to print to the terminal for a quick sanity check.
const SAMPLE_PRINT_COUNT: usize = 8;

fn main() -> Result<(), Box<dyn Error>> {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = output_paths::data_dir();
    let config = DEFAULT_INTERVAL_CONFIG;

    // The Julia script writes plots to docs/plots. Rust only needs the data
    // directory, and it can be overridden with TENSOR4ALL_DATA_DIR for smoke
    // tests that should not touch docs/data.
    fs::create_dir_all(&data_dir)?;

    // Build the continuous interval grid first so the rest of the demo can
    // reuse the same domain definition everywhere.
    let grid = build_interval_grid(&config)?;
    let demo_label = "QTT interval demo";

    // Library call: `quanticscrossinterpolate(...)`
    // Inputs:
    // - `&grid`: the physical interval and resolution
    // - `target_function`: callback `Fn(f64) -> f64` that maps coordinates to values
    // - `None`: no custom initial pivots
    // - `options`: tolerance, bond-dimension limits, and TCI settings
    //
    // Output:
    // - the QTT approximation
    // - a rank history for the interpolation sweeps
    // - an error history for the interpolation sweeps
    let (qtci, ranks, errors) = build_interval_qtt(&grid, interval_target, &config)?;

    // Library call: `tensor_train()`
    // This exposes the underlying TT structure so we can inspect cores,
    // site dimensions, and bond dimensions in a human-readable way.
    let tt = qtci.tensor_train();

    // Library call: `evaluate(...)`
    // We sample the QTT back on the interval and compare each value with the
    // analytic function.
    let samples = collect_samples(&qtci, &grid, interval_target)?;

    // Library call: `link_dims()`
    // These are the internal bond dimensions between TT cores.
    let bond_dims = collect_bond_dims(&qtci);
    let max_abs_error = samples
        .iter()
        .map(|sample| sample.abs_error)
        .fold(0.0_f64, f64::max);

    print_summary(
        &qtci,
        &tt,
        &grid,
        &ranks,
        &errors,
        &samples,
        demo_label,
        config.bits,
        SAMPLE_PRINT_COUNT,
        max_abs_error,
    );

    write_samples_csv(&data_dir.join("qtt_interval_samples.csv"), &samples)?;
    write_bond_dims_csv(&data_dir.join("qtt_interval_bond_dims.csv"), &bond_dims)?;

    println!(
        "wrote {}",
        data_dir.join("qtt_interval_samples.csv").display()
    );
    println!(
        "wrote {}",
        data_dir.join("qtt_interval_bond_dims.csv").display()
    );
    println!(
        "next: run the Julia plotting script at {}",
        project_root
            .join("docs")
            .join("plotting")
            .join("qtt_interval_plot.jl")
            .display()
    );

    Ok(())
}
