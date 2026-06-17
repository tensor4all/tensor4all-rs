//! Build a periodic difference-kernel MPO from a one-dimensional kernel QTT.

use std::error::Error;
use std::fs;
use std::path::Path;

use tensor4all_quanticstransform::{difference_kernel_mpo, BoundaryCondition};
use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_tutorial_code::output_paths;
use tensor4all_tutorial_code::qtt_difference_kernel_common::{
    build_kernel_qtt, collect_bond_dims, collect_samples, point_count, print_summary,
    real_qtt_to_complex, write_bond_dims_csv, write_samples_csv, DEFAULT_DIFFERENCE_KERNEL_CONFIG,
};

fn main() -> Result<(), Box<dyn Error>> {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = output_paths::data_dir();
    let config = DEFAULT_DIFFERENCE_KERNEL_CONFIG;

    fs::create_dir_all(&data_dir)?;

    // Build a QTT for the one-dimensional periodic kernel f(z).
    let (kernel, ranks, errors) = build_kernel_qtt(&config)?;
    let kernel_tt = kernel.tensor_train();

    // `difference_kernel_mpo(...)` expects a complex-valued binary QTT.
    let complex_kernel = real_qtt_to_complex(&kernel_tt)?;

    // This constructs A[x, x'] = f((x - x') mod 2^R) as a fused-site MPO.
    let mpo = difference_kernel_mpo(&complex_kernel, BoundaryCondition::Periodic)?;

    let samples = collect_samples(&kernel, &mpo, &config)?;
    let bond_dims = collect_bond_dims(&kernel.link_dims(), &mpo.link_dims());

    print_summary(&kernel, &mpo, &ranks, &errors, &samples, &config);

    write_samples_csv(
        &data_dir.join("qtt_difference_kernel_samples.csv"),
        &samples,
    )?;
    write_bond_dims_csv(
        &data_dir.join("qtt_difference_kernel_bond_dims.csv"),
        &bond_dims,
    )?;

    println!(
        "wrote {}",
        data_dir.join("qtt_difference_kernel_samples.csv").display()
    );
    println!(
        "wrote {}",
        data_dir
            .join("qtt_difference_kernel_bond_dims.csv")
            .display()
    );
    println!(
        "matrix size = {} x {}",
        point_count(config.bits),
        point_count(config.bits)
    );
    println!(
        "next: run the Julia plotting script at {}",
        project_root
            .join("docs")
            .join("plotting")
            .join("qtt_difference_kernel_plot.jl")
            .display()
    );

    Ok(())
}
