//! Compute a definite integral from a QTT on a physical interval.
//!
//! This is the smallest interval-integral tutorial: build the same QTT as the
//! interval demo, call `integral()`, and compare with the analytic answer.

use std::error::Error;

use tensor4all_tutorial_code::qtt_interval_common::{
    build_interval_grid, build_interval_qtt, exact_integral, interval_target,
    DEFAULT_INTERVAL_CONFIG,
};

fn main() -> Result<(), Box<dyn Error>> {
    let config = DEFAULT_INTERVAL_CONFIG;
    let grid = build_interval_grid(&config)?;
    let (qtci, ranks, errors) = build_interval_qtt(&grid, interval_target, &config)?;

    let integral = qtci.integral()?;
    let exact = exact_integral(&config);
    let abs_error = (integral - exact).abs();

    println!("QTT definite integral tutorial");
    println!(
        "interval = [{:.3}, {:.3}]",
        config.lower_bound, config.upper_bound
    );
    println!("bits R = {}", config.bits);
    println!("grid step = {:?}", grid.grid_step());
    println!("target function = x^2");
    println!("rank = {}", qtci.rank());
    println!("interpolation steps = {}", ranks.len());
    if let Some(final_error) = errors.last() {
        println!("final interpolation error = {:.3e}", final_error);
    }
    println!("integral() = {:.12}", integral);
    println!("exact integral = {:.12}", exact);
    println!("absolute error = {:.3e}", abs_error);

    Ok(())
}
