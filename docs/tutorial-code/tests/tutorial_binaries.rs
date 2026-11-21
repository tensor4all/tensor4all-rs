use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn scratch_dir(name: &str) -> PathBuf {
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

pub(super) fn run_binary(binary_path: &str, data_dir: &Path, extra_env: &[(&str, &str)]) {
    let mut command = Command::new(binary_path);
    command
        .env("TENSOR4ALL_DATA_DIR", data_dir)
        .env_remove("QTT_R_SWEEP_MAX_BITS");
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

pub(super) fn assert_csv_has_data(data_dir: &Path, filename: &str) {
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
fn qtt_r_sweep_depth_override_validates_bounds() -> Result<(), Box<dyn Error>> {
    let binary = env!("CARGO_BIN_EXE_qtt_r_sweep");
    let data_dir = scratch_dir("qtt_r_sweep_bounds");
    for value in ["not-an-integer", "0", "16"] {
        let output = Command::new(binary)
            .env("TENSOR4ALL_DATA_DIR", &data_dir)
            .env("QTT_R_SWEEP_MAX_BITS", value)
            .output()?;
        assert!(
            !output.status.success(),
            "accepted invalid bit depth {value}"
        );
        if value != "not-an-integer" {
            assert!(String::from_utf8_lossy(&output.stderr).contains("must be in 2..=15"));
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        let output = Command::new(binary)
            .env("TENSOR4ALL_DATA_DIR", &data_dir)
            .env(
                "QTT_R_SWEEP_MAX_BITS",
                std::ffi::OsString::from_vec(vec![0xff]),
            )
            .output()?;
        assert!(!output.status.success(), "accepted a non-UTF-8 bit depth");
    }

    run_binary(binary, &data_dir, &[("QTT_R_SWEEP_MAX_BITS", "2")]);
    let stats = fs::read_to_string(data_dir.join("qtt_r_sweep_stats.csv"))?;
    let rows: Vec<_> = stats.lines().collect();
    assert_eq!(
        rows.len(),
        2,
        "the minimum depth produces one statistics row"
    );
    assert!(
        rows[1].starts_with("2,4,"),
        "R=2 must contain four grid points"
    );

    let samples = fs::read_to_string(data_dir.join("qtt_r_sweep_samples.csv"))?;
    let mut samples = samples.lines();
    assert_eq!(
        samples.next(),
        Some("r,npoints,index,x,exact,qtt,abs_error")
    );
    let samples: Vec<_> = samples.collect();
    assert_eq!(samples.len(), 4);
    for (index, row) in samples.iter().enumerate() {
        let fields: Vec<_> = row.split(',').collect();
        assert_eq!(fields.len(), 7);
        assert_eq!(&fields[..2], ["2", "4"]);
        assert_eq!(fields[2].parse::<usize>()?, index + 1);
        let x = index as f64 / 4.0;
        let exact = (30.0 * x).sin();
        assert_eq!(fields[3].parse::<f64>()?, x);
        assert_eq!(fields[4], format!("{exact:.16}"));
        let qtt = fields[5].parse::<f64>()?;
        // Same accuracy bound as the binary and the live bit-depth tutorial.
        assert!(
            (qtt - exact).abs() < 1e-8,
            "index {index}: {qtt} != {exact}"
        );
        assert!(fields[6].parse::<f64>()? < 1e-8);
    }

    fs::remove_dir_all(data_dir)?;
    Ok(())
}
