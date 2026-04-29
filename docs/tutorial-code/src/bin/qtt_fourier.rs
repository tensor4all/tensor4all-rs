//! Compute a Fourier transform of a Gaussian QTT on a quantics grid.
//!
//! This tutorial builds the input Gaussian, applies the built-in quantics
//! Fourier operator, and exports the samples for Julia plotting.

use std::error::Error;
use std::fs;
use std::path::Path;

use tensor4all_quanticstci::DiscretizedGrid;
use tensor4all_tutorial_code::output_paths;
use tensor4all_tutorial_code::qtt_fourier_common::{
    build_fourier_operator, build_gaussian_qtt, collect_bond_dims_from_profiles, collect_samples,
    physical_frequency_bounds, print_summary, transform_gaussian, tree_link_dims,
    write_bond_dims_csv, write_fourier_operator_bond_dims_csv, write_samples_csv,
    DEFAULT_FOURIER_CONFIG,
};

fn main() -> Result<(), Box<dyn Error>> {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = output_paths::data_dir();
    let config = DEFAULT_FOURIER_CONFIG;

    fs::create_dir_all(&data_dir)?;

    let input_grid = DiscretizedGrid::builder(&[config.bits])
        .with_variable_names(&["x"])
        .with_bounds(config.x_lower_bound, config.x_upper_bound)
        .include_endpoint(config.include_endpoint)
        .build()?;

    // k-grid bounds are determined by the input grid bounds and the Fourier transform convention.
    let (k_lower_bound, k_upper_bound) = physical_frequency_bounds(&config);
    let frequency_grid = DiscretizedGrid::builder(&[config.bits])
        .with_variable_names(&["k"])
        .with_bounds(k_lower_bound, k_upper_bound)
        .include_endpoint(config.include_endpoint)
        .build()?;

    // Approximate the Gaussian on the quantics grid as a QTT tensor train.
    let (qtci, ranks, errors) = build_gaussian_qtt(&input_grid, &config)?;

    // Build the quantics Fourier MPO once, then use and inspect the same operator.
    let operator = build_fourier_operator(&config)?;

    // Apply the Fourier MPO and keep the output site indices for evaluation.
    let (transformed, site_indices) = transform_gaussian(&qtci, &operator, &config)?;

    // Evaluate selected coefficients of the transformed QTT on the frequency grid.
    let samples = collect_samples(
        &transformed,
        &site_indices,
        &input_grid,
        &frequency_grid,
        &config,
    )?;

    // `link_dims` gives the bond dimensions of the input Gaussian QTT.
    let input_bond_dims = qtci.link_dims();

    // The transformed state is a TreeTN, so inspect its internal bonds separately.
    let transformed_bond_dims = tree_link_dims(&transformed);

    // The Fourier operator is represented as an MPO; its bonds are the operator ranks.
    let operator_bond_dims = tree_link_dims(&operator.mpo);

    let bond_dims = collect_bond_dims_from_profiles(&input_bond_dims, &transformed_bond_dims);

    print_summary(&qtci, &transformed, &ranks, &errors, &samples, &config);
    write_samples_csv(&data_dir.join("qtt_fourier_samples.csv"), &samples)?;
    write_bond_dims_csv(&data_dir.join("qtt_fourier_bond_dims.csv"), &bond_dims)?;
    write_fourier_operator_bond_dims_csv(
        &data_dir.join("qtt_fourier_operator_bond_dims.csv"),
        &operator_bond_dims,
    )?;

    println!(
        "wrote {}",
        data_dir.join("qtt_fourier_samples.csv").display()
    );
    println!(
        "wrote {}",
        data_dir.join("qtt_fourier_bond_dims.csv").display()
    );
    println!(
        "wrote {}",
        data_dir
            .join("qtt_fourier_operator_bond_dims.csv")
            .display()
    );
    println!(
        "next: run the Julia plotting script at {}",
        project_root
            .join("docs")
            .join("plotting")
            .join("qtt_fourier_plot.jl")
            .display()
    );

    Ok(())
}
