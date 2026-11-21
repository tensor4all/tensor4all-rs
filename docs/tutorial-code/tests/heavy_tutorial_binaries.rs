//! Full tutorial runs retained outside the ordinary suite.

use super::tutorial_binaries::{assert_csv_has_data, run_binary, scratch_dir};
use std::{error::Error, fs};

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn tutorial_binaries_run_current_api_flows() -> Result<(), Box<dyn Error>> {
    struct TutorialBinary {
        name: &'static str,
        path: &'static str,
        env: &'static [(&'static str, &'static str)],
        expected_csvs: &'static [&'static str],
    }

    let binaries = [
        TutorialBinary {
            name: "tensor4all_tutorial_code",
            path: env!("CARGO_BIN_EXE_tensor4all-tutorial-code"),
            env: &[],
            expected_csvs: &[],
        },
        TutorialBinary {
            name: "qtt_function",
            path: env!("CARGO_BIN_EXE_qtt_function"),
            env: &[],
            expected_csvs: &["qtt_function_samples.csv", "qtt_function_bond_dims.csv"],
        },
        TutorialBinary {
            name: "qtt_interval",
            path: env!("CARGO_BIN_EXE_qtt_interval"),
            env: &[],
            expected_csvs: &["qtt_interval_samples.csv", "qtt_interval_bond_dims.csv"],
        },
        TutorialBinary {
            name: "qtt_integral",
            path: env!("CARGO_BIN_EXE_qtt_integral"),
            env: &[],
            expected_csvs: &[],
        },
        TutorialBinary {
            name: "qtt_integral_sweep",
            path: env!("CARGO_BIN_EXE_qtt_integral_sweep"),
            env: &[],
            expected_csvs: &["qtt_integral_sweep.csv"],
        },
        TutorialBinary {
            name: "qtt_r_sweep",
            path: env!("CARGO_BIN_EXE_qtt_r_sweep"),
            env: &[("QTT_R_SWEEP_MAX_BITS", "10")],
            expected_csvs: &["qtt_r_sweep_samples.csv", "qtt_r_sweep_stats.csv"],
        },
        TutorialBinary {
            name: "qtt_multivariate",
            path: env!("CARGO_BIN_EXE_qtt_multivariate"),
            env: &[
                ("QTT_MULTIVARIATE_BITS", "5"),
                ("QTT_MULTIVARIATE_MAXBONDDIM", "32"),
                ("QTT_MULTIVARIATE_MAXITER", "10"),
            ],
            expected_csvs: &[
                "qtt_multivariate_samples.csv",
                "qtt_multivariate_bond_dims.csv",
            ],
        },
        TutorialBinary {
            name: "interpolative_qtt",
            path: env!("CARGO_BIN_EXE_interpolative_qtt"),
            env: &[
                ("INTERPOLATIVE_QTT_BITS_1D", "5"),
                ("INTERPOLATIVE_QTT_BITS_2D", "4"),
                ("INTERPOLATIVE_QTT_DEGREE_1D", "12"),
                ("INTERPOLATIVE_QTT_DEGREE_2D", "6"),
            ],
            expected_csvs: &[
                "interpolative_qtt_1d_samples.csv",
                "interpolative_qtt_2d_samples.csv",
                "interpolative_qtt_bond_dims.csv",
            ],
        },
        TutorialBinary {
            name: "patched_interpolative_qtt",
            path: env!("CARGO_BIN_EXE_patched_interpolative_qtt"),
            env: &[],
            expected_csvs: &[],
        },
        TutorialBinary {
            name: "qtt_elementwise_product",
            path: env!("CARGO_BIN_EXE_qtt_elementwise_product"),
            env: &[],
            expected_csvs: &[
                "qtt_elementwise_product_samples.csv",
                "qtt_elementwise_product_bond_dims.csv",
            ],
        },
        TutorialBinary {
            name: "qtt_affine",
            path: env!("CARGO_BIN_EXE_qtt_affine"),
            env: &[],
            expected_csvs: &[
                "qtt_affine_samples.csv",
                "qtt_affine_bond_dims.csv",
                "qtt_affine_operator_bond_dims.csv",
            ],
        },
        TutorialBinary {
            name: "qtt_difference_kernel",
            path: env!("CARGO_BIN_EXE_qtt_difference_kernel"),
            env: &[],
            expected_csvs: &[
                "qtt_difference_kernel_samples.csv",
                "qtt_difference_kernel_bond_dims.csv",
            ],
        },
        TutorialBinary {
            name: "qtt_fourier",
            path: env!("CARGO_BIN_EXE_qtt_fourier"),
            env: &[],
            expected_csvs: &[
                "qtt_fourier_samples.csv",
                "qtt_fourier_bond_dims.csv",
                "qtt_fourier_operator_bond_dims.csv",
            ],
        },
        TutorialBinary {
            name: "qtt_partial_fourier2d",
            path: env!("CARGO_BIN_EXE_qtt_partial_fourier2d"),
            env: &[],
            expected_csvs: &[
                "qtt_partial_fourier2d_samples.csv",
                "qtt_partial_fourier2d_bond_dims.csv",
                "qtt_partial_fourier2d_operator_bond_dims.csv",
            ],
        },
    ];

    for binary in binaries {
        let data_dir = scratch_dir(binary.name);
        run_binary(binary.path, &data_dir, binary.env);
        for filename in binary.expected_csvs {
            assert_csv_has_data(&data_dir, filename);
        }
        fs::remove_dir_all(data_dir)?;
    }

    Ok(())
}

#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn qtt_r_sweep_defaults_to_full_depth() -> Result<(), Box<dyn Error>> {
    let binary = env!("CARGO_BIN_EXE_qtt_r_sweep");
    let data_dir = scratch_dir("qtt_r_sweep_default");

    // The helper explicitly removes this variable before launching the binary,
    // so this run exercises the documented default rather than inheriting a
    // CI override from the parent process.
    run_binary(binary, &data_dir, &[]);

    let stats = fs::read_to_string(data_dir.join("qtt_r_sweep_stats.csv"))?;
    let rows: Vec<_> = stats.lines().collect();
    assert_eq!(
        rows.len(),
        15,
        "the default sweep covers depths 2 through 15"
    );
    let final_row: Vec<_> = rows
        .last()
        .expect("depth-15 statistics row")
        .split(',')
        .take(2)
        .collect();
    assert_eq!(
        final_row,
        ["15", "32768"],
        "the default sweep must emit its depth-15, 32768-point result"
    );

    fs::remove_dir_all(data_dir)?;
    Ok(())
}
