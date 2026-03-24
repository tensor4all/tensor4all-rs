use super::*;
use crate::backend::File;

#[cfg(feature = "runtime-loading")]
fn init_hdf5() {
    // Initialize HDF5 library for tests (runtime-loading mode)
    if !hdf5_rt::sys::is_initialized() {
        let paths = [
            "/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so",
            "/usr/lib/libhdf5.so",
            "/opt/homebrew/lib/libhdf5.dylib",
            "/usr/local/lib/libhdf5.dylib",
        ];
        for path in &paths {
            if std::path::Path::new(path).exists() {
                let _ = hdf5_rt::sys::init(Some(path));
                break;
            }
        }
    }
}

#[cfg(all(feature = "link", not(feature = "runtime-loading")))]
fn init_hdf5() {
    // No initialization needed for link mode
}

#[test]
#[ignore = "requires HDF5 library"]
fn test_write_read_type_version() {
    init_hdf5();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let file = File::create(path.to_str().unwrap()).unwrap();
    let group = file.create_group("test").unwrap();

    write_type_version(&group, "ITensor", 1).unwrap();
    let (t, v) = read_type_version(&group).unwrap();
    assert_eq!(t, "ITensor");
    assert_eq!(v, 1);
}

#[test]
#[ignore = "requires HDF5 library"]
fn test_require_type_version_ok() {
    init_hdf5();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let file = File::create(path.to_str().unwrap()).unwrap();
    let group = file.create_group("test").unwrap();

    write_type_version(&group, "MPS", 1).unwrap();
    let v = require_type_version(&group, "MPS", 1).unwrap();
    assert_eq!(v, 1);
}

#[test]
#[ignore = "requires HDF5 library"]
fn test_require_type_version_wrong_type() {
    init_hdf5();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let file = File::create(path.to_str().unwrap()).unwrap();
    let group = file.create_group("test").unwrap();

    write_type_version(&group, "ITensor", 1).unwrap();
    let err = require_type_version(&group, "MPS", 1).unwrap_err();
    assert!(err.to_string().contains("Expected HDF5 type 'MPS'"));
}

#[test]
#[ignore = "requires HDF5 library"]
fn test_require_type_version_too_new() {
    init_hdf5();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let file = File::create(path.to_str().unwrap()).unwrap();
    let group = file.create_group("test").unwrap();

    write_type_version(&group, "ITensor", 99).unwrap();
    let err = require_type_version(&group, "ITensor", 1).unwrap_err();
    assert!(err.to_string().contains("Unsupported ITensor version 99"));
}
