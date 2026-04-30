//! Apply a Fourier transform to only one dimension of a 2D QTT.

use std::error::Error;
use std::fs;
use std::path::Path;

use tensor4all_tutorial_code::output_paths;
use tensor4all_tutorial_code::qtt_partial_fourier2d_common::{
    build_frequency_grid, build_input_grid, build_partial_fourier_operator, build_source_qtt,
    collect_bond_dims, collect_samples, print_summary, transform_x_dimension, tree_link_dims,
    write_bond_dims_csv, write_operator_bond_dims_csv, write_samples_csv,
    DEFAULT_PARTIAL_FOURIER2D_CONFIG,
};

fn main() -> Result<(), Box<dyn Error>> {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = output_paths::data_dir();
    let config = DEFAULT_PARTIAL_FOURIER2D_CONFIG;

    fs::create_dir_all(&data_dir)?;

    let input_grid = build_input_grid(&config)?;
    let frequency_grid = build_frequency_grid(&config)?;
    let (qtci, ranks, errors) = build_source_qtt(&input_grid, &config)?;
    let operator = build_partial_fourier_operator(&config)?;
    let (transformed, site_indices) = transform_x_dimension(&qtci, &operator)?;
    let samples = collect_samples(&transformed, &site_indices, &frequency_grid, &config)?;

    let bond_dims = collect_bond_dims(&qtci.link_dims(), &tree_link_dims(&transformed));
    let operator_bond_dims = tree_link_dims(&operator.mpo);

    print_summary(&qtci, &transformed, &ranks, &errors, &samples, &config);

    write_samples_csv(
        &data_dir.join("qtt_partial_fourier2d_samples.csv"),
        &samples,
    )?;
    write_bond_dims_csv(
        &data_dir.join("qtt_partial_fourier2d_bond_dims.csv"),
        &bond_dims,
    )?;
    write_operator_bond_dims_csv(
        &data_dir.join("qtt_partial_fourier2d_operator_bond_dims.csv"),
        &operator_bond_dims,
    )?;

    println!(
        "wrote {}",
        data_dir.join("qtt_partial_fourier2d_samples.csv").display()
    );
    println!(
        "wrote {}",
        data_dir
            .join("qtt_partial_fourier2d_bond_dims.csv")
            .display()
    );
    println!(
        "wrote {}",
        data_dir
            .join("qtt_partial_fourier2d_operator_bond_dims.csv")
            .display()
    );
    println!(
        "next: run the Julia plotting script at {}",
        project_root
            .join("docs")
            .join("plotting")
            .join("qtt_partial_fourier2d_plot.jl")
            .display()
    );

    Ok(())
}
