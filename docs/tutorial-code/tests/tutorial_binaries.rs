use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn scratch_dir(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX_EPOCH")
        .as_nanos();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("binary-smoke")
        .join(format!("{name}-{nonce}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn run_binary(binary_path: &str, data_dir: &Path, extra_env: &[(&str, &str)]) {
    let mut command = Command::new(binary_path);
    command.env("TENSOR4ALL_DATA_DIR", data_dir);
    for (key, value) in extra_env {
        command.env(key, value);
    }

    let output = command.output().expect("run tutorial binary");
    assert!(
        output.status.success(),
        "tutorial binary failed\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_csv_has_data(data_dir: &Path, filename: &str) {
    let path = data_dir.join(filename);
    let contents =
        fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let mut lines = contents.lines();
    let header = lines
        .next()
        .unwrap_or_else(|| panic!("{} has no header", path.display()));
    assert!(
        !header.trim().is_empty(),
        "{} has empty header",
        path.display()
    );
    assert!(
        lines.next().is_some(),
        "{} has no data rows after header",
        path.display()
    );
}

#[test]
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
            env: &[],
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
    }

    Ok(())
}
