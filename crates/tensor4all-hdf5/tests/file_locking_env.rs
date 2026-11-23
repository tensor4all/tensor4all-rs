//! #1386: the library must not mutate the process-global
//! `HDF5_USE_FILE_LOCKING`; the caller chooses it before the first HDF5 call.
//!
//! The probe runs in a child process (this test binary re-executed) so the
//! variable's state at HDF5 library initialization is controlled exactly,
//! independent of whatever the harness exported to the parent.

use std::process::Command;
use tensor4all_core::index::DynIndex;
use tensor4all_core::IdxTensor;
use tensor4all_hdf5::{load_itensor, save_itensor};

const VAR: &str = "HDF5_USE_FILE_LOCKING";
const PROBE_ENV: &str = "T4A_HDF5_FILE_LOCKING_PROBE_DIR";

fn env_snapshot() -> Vec<(std::ffi::OsString, std::ffi::OsString)> {
    let mut vars: Vec<_> = std::env::vars_os().collect();
    vars.sort();
    vars
}

/// Same-path save-then-load, repeated; returns the final roundtrip data.
fn same_path_roundtrips(dir: &std::path::Path) -> Vec<f64> {
    let file = dir.join("probe.h5");
    let file = file.to_str().unwrap();
    let mut last = Vec::new();
    for i in 0..20 {
        let data: Vec<f64> = (0..4).map(|k| (i * 10 + k) as f64).collect();
        let tensor = IdxTensor::from_dense(
            vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)],
            data.clone(),
        )
        .unwrap();
        save_itensor(file, "t", &tensor).unwrap();
        last = load_itensor(file, "t").unwrap().to_vec::<f64>().unwrap();
        assert_eq!(last, data);
    }
    last
}

/// In-process: no backend operation changes the process environment at all.
#[test]
fn first_use_does_not_modify_process_environment() {
    let dir = tempfile::tempdir().unwrap();
    let before = env_snapshot();
    let roundtrip = same_path_roundtrips(dir.path());
    let after = env_snapshot();
    assert_eq!(roundtrip, vec![190.0, 191.0, 192.0, 193.0]);
    assert_eq!(before, after, "library mutated the process environment");
}

/// Child entry point: only active when spawned by the parent tests below.
#[test]
fn probe_child() {
    let Some(dir) = std::env::var_os(PROBE_ENV) else {
        return;
    };
    let before = std::env::var_os(VAR);
    same_path_roundtrips(std::path::Path::new(&dir));
    let after = std::env::var_os(VAR);
    assert_eq!(before, after);
    println!("PROBE {VAR}={:?}", after);
}

fn run_probe(configure: impl FnOnce(&mut Command)) -> String {
    let dir = tempfile::tempdir().unwrap();
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", "probe_child", "--nocapture", "--test-threads=1"])
        .env(PROBE_ENV, dir.path());
    configure(&mut cmd);
    let out = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "probe failed: {stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
}

/// Caller left the variable unset: the library does not set it on first use.
#[test]
fn library_does_not_set_variable_on_first_use() {
    let stdout = run_probe(|cmd| {
        cmd.env_remove(VAR);
    });
    assert!(
        stdout.contains(&format!("PROBE {VAR}=None")),
        "variable was written by the library: {stdout}"
    );
}

/// Opt-out path: the caller sets `FALSE` up front; same-path reopen works and
/// the caller's value is left untouched.
#[test]
fn caller_opt_out_roundtrips_same_path() {
    let stdout = run_probe(|cmd| {
        cmd.env(VAR, "FALSE");
    });
    assert!(
        stdout.contains(&format!("PROBE {VAR}=Some(\"FALSE\")")),
        "{stdout}"
    );
}
