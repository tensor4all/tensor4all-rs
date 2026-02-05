//! Diagnostic test to investigate H5Iis_valid behavior on different HDF5 builds.
//!
//! This test is designed to run on CI and print detailed information about
//! how HDF5 ID validation functions behave.
//!
//! Note: This test requires HDF5 to be linked at build time. In runtime-loading
//! mode without the link feature, the test is skipped unless HDF5_LIB is set.

// Only compile this test when link feature is available, or when testing runtime-loading
#![cfg(any(feature = "link", feature = "runtime-loading"))]

use std::ffi::CString;
use std::ptr;

use tensor4all_hdf5_ffi::globals::{self, H5P_DEFAULT};
use tensor4all_hdf5_ffi::sync;
use tensor4all_hdf5_ffi::sys::{
    self, hid_t, H5Acreate2, H5Awrite, H5Fclose, H5Fcreate, H5Gclose, H5Gcreate2, H5Idec_ref,
    H5Iget_type, H5Iis_valid, H5Sclose, H5Screate_simple,
};

/// Initialize HDF5 if in runtime-loading mode
#[cfg(all(feature = "runtime-loading", not(feature = "link")))]
fn ensure_hdf5_init() -> bool {
    use tensor4all_hdf5_ffi::hdf5_init;

    if tensor4all_hdf5_ffi::hdf5_is_initialized() {
        return true;
    }

    match std::env::var("HDF5_LIB") {
        Ok(path) => {
            hdf5_init(&path).expect("Failed to initialize HDF5");
            true
        }
        Err(_) => {
            eprintln!("Skipping test: HDF5_LIB environment variable not set");
            false
        }
    }
}

#[cfg(feature = "link")]
fn ensure_hdf5_init() -> bool {
    true
}

fn get_hdf5_version() -> (u32, u32, u32) {
    let mut major: u32 = 0;
    let mut minor: u32 = 0;
    let mut release: u32 = 0;
    unsafe {
        sys::H5get_libversion(&mut major, &mut minor, &mut release);
    }
    (major, minor, release)
}

/// Print diagnostic info about an HDF5 ID
fn diagnose_id(name: &str, id: hid_t) {
    let is_valid = unsafe { H5Iis_valid(id) };
    let id_type = unsafe { H5Iget_type(id) };

    println!("=== {} ===", name);
    println!("  id value: {} (0x{:016x})", id, id as u64);
    println!("  H5Iis_valid: {}", is_valid);
    println!("  H5Iget_type: {}", id_type);
    println!("  id > 0: {}", id > 0);

    // Extract type from ID bits (upper 8 bits in HDF5 1.10+)
    let type_from_bits = (id as u64 >> 56) & 0xFF;
    println!("  type from bits (upper 8): {}", type_from_bits);
}

#[test]
fn test_diagnose_h5iis_valid() {
    if !ensure_hdf5_init() {
        return; // Skip test if HDF5 is not available
    }

    let (major, minor, release) = get_hdf5_version();
    println!("\n");
    println!("========================================");
    println!("HDF5 Library Version: {}.{}.{}", major, minor, release);
    println!("========================================\n");

    // Create a temporary file
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("diagnose_h5i_test.h5");
    let file_path_cstr = CString::new(file_path.to_str().unwrap()).unwrap();

    sync(|| {
        // Create file
        let file_id = unsafe {
            H5Fcreate(
                file_path_cstr.as_ptr(),
                sys::H5F_ACC_TRUNC,
                H5P_DEFAULT,
                H5P_DEFAULT,
            )
        };
        println!("After H5Fcreate:");
        diagnose_id("file_id", file_id);
        assert!(file_id > 0, "H5Fcreate failed");

        // Create group
        let group_name = CString::new("test_group").unwrap();
        let group_id = unsafe {
            H5Gcreate2(
                file_id,
                group_name.as_ptr(),
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT,
            )
        };
        println!("\nAfter H5Gcreate2:");
        diagnose_id("group_id", group_id);
        assert!(group_id > 0, "H5Gcreate2 failed");

        // Create dataspace for attribute
        let dims: [u64; 1] = [1];
        let space_id = unsafe { H5Screate_simple(1, dims.as_ptr(), ptr::null()) };
        println!("\nAfter H5Screate_simple:");
        diagnose_id("space_id", space_id);
        assert!(space_id > 0, "H5Screate_simple failed");

        // Create attribute - this is where the issue was observed
        let attr_name = CString::new("test_attr").unwrap();
        let attr_id = unsafe {
            H5Acreate2(
                group_id,
                attr_name.as_ptr(),
                globals::H5T_NATIVE_INT32(),
                space_id,
                H5P_DEFAULT,
                H5P_DEFAULT,
            )
        };
        println!("\nAfter H5Acreate2 (this was failing):");
        diagnose_id("attr_id", attr_id);
        assert!(attr_id > 0, "H5Acreate2 failed");

        // Write to attribute
        let value: i32 = 42;
        let write_result = unsafe {
            H5Awrite(
                attr_id,
                globals::H5T_NATIVE_INT32(),
                &value as *const i32 as *const _,
            )
        };
        println!("\nAfter H5Awrite:");
        println!("  write result: {}", write_result);
        diagnose_id("attr_id (after write)", attr_id);

        // Check validity again after some operations
        println!("\n--- Checking validity after operations ---");
        diagnose_id("file_id (recheck)", file_id);
        diagnose_id("group_id (recheck)", group_id);
        diagnose_id("attr_id (recheck)", attr_id);

        // Cleanup
        unsafe {
            H5Idec_ref(attr_id);
            H5Sclose(space_id);
            H5Gclose(group_id);
            H5Fclose(file_id);
        }

        println!("\n--- After cleanup ---");
        diagnose_id("attr_id (after close)", attr_id);
    });

    // Remove temp file
    let _ = std::fs::remove_file(&file_path);

    println!("\n========================================");
    println!("Diagnostic test completed");
    println!("========================================\n");
}
