use super::{api_file_names, require_success, validate_inventory};
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_inventory() -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the Unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("tensor4all-api-dump-{suffix}"));
    fs::create_dir(&path).unwrap();
    path
}

#[test]
fn command_failures_are_reported() {
    let error = require_success(false, "test command").unwrap_err();
    assert!(error.to_string().contains("test command failed"));
}

#[test]
fn successful_commands_are_accepted() {
    require_success(true, "test command").unwrap();
}

#[test]
fn inventory_requires_every_expected_file_and_rejects_stale_files() {
    let output = temp_inventory();
    fs::write(output.join("present.md"), "# present").unwrap();
    fs::write(output.join("stale.md"), "# stale").unwrap();
    let expected = BTreeSet::from(["present.md".to_string(), "missing.md".to_string()]);

    let error = validate_inventory(&expected, &output).unwrap_err();
    assert!(error.to_string().contains("missing.md"));
    assert!(error.to_string().contains("stale.md"));
    fs::remove_dir_all(output).unwrap();
}

#[test]
fn inventory_rejects_duplicate_public_crate_filenames() {
    let error = api_file_names(["tensor4all-a", "tensor4all_a"]).unwrap_err();
    assert!(error.to_string().contains("tensor4all_a.md"));
}
