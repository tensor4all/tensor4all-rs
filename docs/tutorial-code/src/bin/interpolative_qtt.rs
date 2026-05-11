//! Build single-scale and multiscale Interpolative QTT examples.
//!
//! This tutorial exports one-dimensional samples, two-dimensional samples, and
//! bond dimensions for the mdBook plots.

use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;

use tensor4all_interpolativeqtt::AbstractTensorTrain;
use tensor4all_tutorial_code::interpolative_qtt_common::{
    build_multi_scale_inverse_square_1d_qtt, build_multi_scale_inverse_square_2d_qtt,
    build_single_scale_smooth_qtt, collect_1d_samples, collect_2d_samples, collect_bond_dims,
    max_abs_error_1d, max_abs_error_2d, print_summary, smooth_target, softened_inverse_square_1d,
    softened_inverse_square_2d, write_1d_samples_csv, write_2d_samples_csv, write_bond_dims_csv,
    InterpolativeQttTutorialConfig, DEFAULT_INTERPOLATIVE_QTT_CONFIG,
};
use tensor4all_tutorial_code::output_paths;

const BITS_1D_ENV: &str = "INTERPOLATIVE_QTT_BITS_1D";
const BITS_2D_ENV: &str = "INTERPOLATIVE_QTT_BITS_2D";
const DEGREE_1D_ENV: &str = "INTERPOLATIVE_QTT_DEGREE_1D";
const DEGREE_2D_ENV: &str = "INTERPOLATIVE_QTT_DEGREE_2D";

fn parse_env_usize(name: &str) -> Result<Option<usize>, Box<dyn Error>> {
    let Ok(value) = std::env::var(name) else {
        return Ok(None);
    };

    let parsed = value.parse::<usize>().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{name} must be a non-negative integer, got {value:?}: {err}"),
        )
    })?;

    Ok(Some(parsed))
}

fn config_from_env() -> Result<InterpolativeQttTutorialConfig, Box<dyn Error>> {
    let mut config = DEFAULT_INTERPOLATIVE_QTT_CONFIG;

    if let Some(bits_1d) = parse_env_usize(BITS_1D_ENV)? {
        config.bits_1d = bits_1d;
    }
    if let Some(bits_2d) = parse_env_usize(BITS_2D_ENV)? {
        config.bits_2d = bits_2d;
    }
    if let Some(degree_1d) = parse_env_usize(DEGREE_1D_ENV)? {
        config.single_scale_degree = degree_1d;
        config.multi_scale_degree_1d = degree_1d;
    }
    if let Some(degree_2d) = parse_env_usize(DEGREE_2D_ENV)? {
        config.multi_scale_degree_2d = degree_2d;
    }

    Ok(config)
}

fn main() -> Result<(), Box<dyn Error>> {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = output_paths::data_dir();
    let config = config_from_env()?;

    fs::create_dir_all(&data_dir)?;

    let single_scale = build_single_scale_smooth_qtt(&config)?;
    let multi_scale_1d = build_multi_scale_inverse_square_1d_qtt(&config)?;
    let multi_scale_2d = build_multi_scale_inverse_square_2d_qtt(&config)?;

    let smooth_samples = collect_1d_samples(
        "single_scale_1d",
        &single_scale,
        config.bits_1d,
        config.lower_bound,
        config.upper_bound,
        smooth_target,
    )?;
    let inverse_samples_1d = collect_1d_samples(
        "multi_scale_1d",
        &multi_scale_1d,
        config.bits_1d,
        config.lower_bound,
        config.upper_bound,
        |x| softened_inverse_square_1d(x, config.epsilon),
    )?;

    let lower = [config.lower_bound, config.lower_bound];
    let upper = [config.upper_bound, config.upper_bound];
    let samples_2d =
        collect_2d_samples(&multi_scale_2d, config.bits_2d, &lower, &upper, |coords| {
            softened_inverse_square_2d(coords, config.epsilon)
        })?;

    let smooth_max_error = max_abs_error_1d(&smooth_samples);
    let multi_1d_max_error = max_abs_error_1d(&inverse_samples_1d);
    let multi_2d_max_error = max_abs_error_2d(&samples_2d);

    let mut samples_1d = smooth_samples.clone();
    samples_1d.extend(inverse_samples_1d);

    let single_scale_bonds = single_scale.link_dims();
    let multi_1d_bonds = multi_scale_1d.link_dims();
    let multi_2d_bonds = multi_scale_2d.link_dims();
    let bond_dims = collect_bond_dims(&single_scale_bonds, &multi_1d_bonds, &multi_2d_bonds);

    print_summary(
        &config,
        &single_scale,
        &multi_scale_1d,
        &multi_scale_2d,
        smooth_max_error,
        multi_1d_max_error,
        multi_2d_max_error,
    );

    write_1d_samples_csv(
        &data_dir.join("interpolative_qtt_1d_samples.csv"),
        &samples_1d,
    )?;
    write_2d_samples_csv(
        &data_dir.join("interpolative_qtt_2d_samples.csv"),
        &samples_2d,
    )?;
    write_bond_dims_csv(
        &data_dir.join("interpolative_qtt_bond_dims.csv"),
        &bond_dims,
    )?;

    println!(
        "wrote {}",
        data_dir.join("interpolative_qtt_1d_samples.csv").display()
    );
    println!(
        "wrote {}",
        data_dir.join("interpolative_qtt_2d_samples.csv").display()
    );
    println!(
        "wrote {}",
        data_dir.join("interpolative_qtt_bond_dims.csv").display()
    );
    println!(
        "next: run the Julia plotting script at {}",
        project_root
            .join("docs")
            .join("plotting")
            .join("interpolative_qtt_plot.jl")
            .display()
    );

    Ok(())
}
