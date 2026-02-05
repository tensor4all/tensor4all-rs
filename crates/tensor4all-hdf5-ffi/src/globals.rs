//! HDF5 global constants with support for both build-time linking and runtime loading.
//!
//! Two modes are supported:
//! - `link` feature (default): Uses constants from hdf5-sys
//! - `runtime-loading` feature: Loads constants at runtime from the library

#![allow(non_upper_case_globals, non_snake_case)]

use crate::sys::hid_t;
use std::sync::LazyLock;

/// H5P_DEFAULT constant (always 0 in HDF5)
pub const H5P_DEFAULT: hid_t = 0;

/// Invalid ID constant
pub const H5I_INVALID_HID: hid_t = -1;

// =============================================================================
// Link mode: use hdf5-sys constants
// Link mode takes precedence when both features are enabled.
// =============================================================================
#[cfg(feature = "link")]
mod link_impl {
    use super::*;

    /// Force library initialization before accessing any global constant.
    /// This matches hdf5-metno's pattern.
    #[inline]
    fn ensure_init() {
        LazyLock::force(&crate::sync::LIBRARY_INIT);
    }

    pub fn H5T_NATIVE_INT() -> hid_t {
        ensure_init();
        *hdf5_sys::h5t::H5T_NATIVE_INT
    }

    pub fn H5T_NATIVE_FLOAT() -> hid_t {
        ensure_init();
        *hdf5_sys::h5t::H5T_NATIVE_FLOAT
    }

    pub fn H5T_NATIVE_DOUBLE() -> hid_t {
        ensure_init();
        *hdf5_sys::h5t::H5T_NATIVE_DOUBLE
    }

    pub fn H5T_NATIVE_INT64() -> hid_t {
        ensure_init();
        *hdf5_sys::h5t::H5T_NATIVE_INT64
    }

    pub fn H5T_NATIVE_UINT64() -> hid_t {
        ensure_init();
        *hdf5_sys::h5t::H5T_NATIVE_UINT64
    }

    pub fn H5T_NATIVE_INT8() -> hid_t {
        ensure_init();
        *hdf5_sys::h5t::H5T_NATIVE_INT8
    }

    pub fn H5T_NATIVE_INT16() -> hid_t {
        ensure_init();
        *hdf5_sys::h5t::H5T_NATIVE_INT16
    }

    pub fn H5T_NATIVE_INT32() -> hid_t {
        ensure_init();
        *hdf5_sys::h5t::H5T_NATIVE_INT32
    }

    pub fn H5T_NATIVE_UINT8() -> hid_t {
        ensure_init();
        *hdf5_sys::h5t::H5T_NATIVE_UINT8
    }

    pub fn H5T_NATIVE_UINT16() -> hid_t {
        ensure_init();
        *hdf5_sys::h5t::H5T_NATIVE_UINT16
    }

    pub fn H5T_NATIVE_UINT32() -> hid_t {
        ensure_init();
        *hdf5_sys::h5t::H5T_NATIVE_UINT32
    }

    pub fn H5T_C_S1() -> hid_t {
        ensure_init();
        *hdf5_sys::h5t::H5T_C_S1
    }

    pub fn H5P_FILE_ACCESS() -> hid_t {
        ensure_init();
        *hdf5_sys::h5p::H5P_CLS_FILE_ACCESS
    }

    pub fn H5P_DATASET_CREATE() -> hid_t {
        ensure_init();
        *hdf5_sys::h5p::H5P_CLS_DATASET_CREATE
    }

    pub fn H5P_ATTRIBUTE_CREATE() -> hid_t {
        ensure_init();
        *hdf5_sys::h5p::H5P_CLS_ATTRIBUTE_CREATE
    }
}

// =============================================================================
// Runtime loading mode: load constants from library
// Only used when link feature is not enabled.
// =============================================================================
#[cfg(all(feature = "runtime-loading", not(feature = "link")))]
mod runtime_impl {
    use super::*;
    use libloading::Symbol;
    use std::sync::OnceLock;

    /// Container for lazily-loaded HDF5 global constants.
    struct GlobalConstants {
        // Native types
        h5t_native_int: hid_t,
        h5t_native_float: hid_t,
        h5t_native_double: hid_t,
        h5t_native_int8: hid_t,
        h5t_native_int16: hid_t,
        h5t_native_int32: hid_t,
        h5t_native_int64: hid_t,
        h5t_native_uint8: hid_t,
        h5t_native_uint16: hid_t,
        h5t_native_uint32: hid_t,
        h5t_native_uint64: hid_t,
        // String type
        h5t_c_s1: hid_t,
        // Property list classes
        h5p_file_access: hid_t,
        h5p_dataset_create: hid_t,
        h5p_attribute_create: hid_t,
    }

    static GLOBALS: OnceLock<GlobalConstants> = OnceLock::new();

    fn get_globals() -> &'static GlobalConstants {
        GLOBALS.get_or_init(|| {
            // Force library initialization first
            LazyLock::force(&crate::sync::LIBRARY_INIT);

            // Access the library through our sys module
            let lib = crate::sys::funcs::get_library()
                .expect("HDF5 library not loaded - call hdf5_init() first");

            // HDF5 exports predefined type IDs with _g suffix (e.g., H5T_NATIVE_INT_g)
            unsafe {
                GlobalConstants {
                    h5t_native_int: load_global(lib, "H5T_NATIVE_INT_g"),
                    h5t_native_float: load_global(lib, "H5T_NATIVE_FLOAT_g"),
                    h5t_native_double: load_global(lib, "H5T_NATIVE_DOUBLE_g"),
                    h5t_native_int8: load_global(lib, "H5T_NATIVE_INT8_g"),
                    h5t_native_int16: load_global(lib, "H5T_NATIVE_INT16_g"),
                    h5t_native_int32: load_global(lib, "H5T_NATIVE_INT32_g"),
                    h5t_native_int64: load_global(lib, "H5T_NATIVE_INT64_g"),
                    h5t_native_uint8: load_global(lib, "H5T_NATIVE_UINT8_g"),
                    h5t_native_uint16: load_global(lib, "H5T_NATIVE_UINT16_g"),
                    h5t_native_uint32: load_global(lib, "H5T_NATIVE_UINT32_g"),
                    h5t_native_uint64: load_global(lib, "H5T_NATIVE_UINT64_g"),
                    h5t_c_s1: load_global(lib, "H5T_C_S1_g"),
                    h5p_file_access: load_global(lib, "H5P_CLS_FILE_ACCESS_ID_g"),
                    h5p_dataset_create: load_global(lib, "H5P_CLS_DATASET_CREATE_ID_g"),
                    h5p_attribute_create: load_global(lib, "H5P_CLS_ATTRIBUTE_CREATE_ID_g"),
                }
            }
        })
    }

    /// Load a global variable from the HDF5 library.
    ///
    /// HDF5's predefined type IDs are actually pointers to hid_t values.
    unsafe fn load_global(lib: &libloading::Library, name: &str) -> hid_t {
        // Try to load as a pointer to hid_t (how HDF5 exports these)
        let sym: Result<Symbol<*const hid_t>, _> = lib.get(name.as_bytes());
        match sym {
            Ok(ptr) => {
                let ptr = *ptr;
                if ptr.is_null() {
                    H5I_INVALID_HID
                } else {
                    *ptr
                }
            }
            Err(_) => H5I_INVALID_HID,
        }
    }

    pub fn H5T_NATIVE_INT() -> hid_t {
        get_globals().h5t_native_int
    }

    pub fn H5T_NATIVE_FLOAT() -> hid_t {
        get_globals().h5t_native_float
    }

    pub fn H5T_NATIVE_DOUBLE() -> hid_t {
        get_globals().h5t_native_double
    }

    pub fn H5T_NATIVE_INT64() -> hid_t {
        get_globals().h5t_native_int64
    }

    pub fn H5T_NATIVE_UINT64() -> hid_t {
        get_globals().h5t_native_uint64
    }

    pub fn H5T_NATIVE_INT8() -> hid_t {
        get_globals().h5t_native_int8
    }

    pub fn H5T_NATIVE_INT16() -> hid_t {
        get_globals().h5t_native_int16
    }

    pub fn H5T_NATIVE_INT32() -> hid_t {
        get_globals().h5t_native_int32
    }

    pub fn H5T_NATIVE_UINT8() -> hid_t {
        get_globals().h5t_native_uint8
    }

    pub fn H5T_NATIVE_UINT16() -> hid_t {
        get_globals().h5t_native_uint16
    }

    pub fn H5T_NATIVE_UINT32() -> hid_t {
        get_globals().h5t_native_uint32
    }

    pub fn H5T_C_S1() -> hid_t {
        get_globals().h5t_c_s1
    }

    pub fn H5P_FILE_ACCESS() -> hid_t {
        get_globals().h5p_file_access
    }

    pub fn H5P_DATASET_CREATE() -> hid_t {
        get_globals().h5p_dataset_create
    }

    pub fn H5P_ATTRIBUTE_CREATE() -> hid_t {
        get_globals().h5p_attribute_create
    }
}

// =============================================================================
// Public API: re-export from the appropriate implementation
// Link mode takes precedence when both features are enabled.
// =============================================================================

#[cfg(feature = "link")]
pub use link_impl::*;

#[cfg(all(feature = "runtime-loading", not(feature = "link")))]
pub use runtime_impl::*;
