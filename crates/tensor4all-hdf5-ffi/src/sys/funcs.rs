//! HDF5 function loading - supports both build-time linking and runtime loading.
//!
//! Two modes are supported:
//! - `link` feature (default): Links to HDF5 at build time via hdf5-sys
//! - `runtime-loading` feature: Loads HDF5 at runtime via dlopen
//!
//! For Julia/Python bindings, use `runtime-loading` feature.

#![allow(non_snake_case)]

use super::types::*;

// =============================================================================
// Link mode: use hdf5-sys (build-time linking)
// =============================================================================
#[cfg(feature = "link")]
mod link_impl {
    use super::*;

    /// Check if the library is initialized (always true for link mode).
    pub fn is_initialized() -> bool {
        true
    }

    /// Get the library path (not applicable for link mode).
    pub fn library_path() -> Option<String> {
        None
    }

    // Wrapper functions that use our local types but call hdf5-sys
    // This ensures type consistency across both modes

    pub unsafe fn H5open() -> herr_t {
        hdf5_sys::h5::H5open()
    }

    pub unsafe fn H5close() -> herr_t {
        hdf5_sys::h5::H5close()
    }

    pub unsafe fn H5dont_atexit() -> herr_t {
        hdf5_sys::h5::H5dont_atexit()
    }

    pub unsafe fn H5get_libversion(
        majnum: *mut c_uint,
        minnum: *mut c_uint,
        relnum: *mut c_uint,
    ) -> herr_t {
        hdf5_sys::h5::H5get_libversion(majnum, minnum, relnum)
    }

    pub unsafe fn H5Fcreate(
        filename: *const c_char,
        flags: c_uint,
        create_plist: hid_t,
        access_plist: hid_t,
    ) -> hid_t {
        hdf5_sys::h5f::H5Fcreate(filename, flags, create_plist, access_plist)
    }

    pub unsafe fn H5Fopen(filename: *const c_char, flags: c_uint, access_plist: hid_t) -> hid_t {
        hdf5_sys::h5f::H5Fopen(filename, flags, access_plist)
    }

    pub unsafe fn H5Fclose(file_id: hid_t) -> herr_t {
        hdf5_sys::h5f::H5Fclose(file_id)
    }

    pub unsafe fn H5Gcreate2(
        loc_id: hid_t,
        name: *const c_char,
        lcpl_id: hid_t,
        gcpl_id: hid_t,
        gapl_id: hid_t,
    ) -> hid_t {
        hdf5_sys::h5g::H5Gcreate2(loc_id, name, lcpl_id, gcpl_id, gapl_id)
    }

    pub unsafe fn H5Gopen2(loc_id: hid_t, name: *const c_char, gapl_id: hid_t) -> hid_t {
        hdf5_sys::h5g::H5Gopen2(loc_id, name, gapl_id)
    }

    pub unsafe fn H5Gclose(group_id: hid_t) -> herr_t {
        hdf5_sys::h5g::H5Gclose(group_id)
    }

    pub unsafe fn H5Dcreate2(
        loc_id: hid_t,
        name: *const c_char,
        type_id: hid_t,
        space_id: hid_t,
        lcpl_id: hid_t,
        dcpl_id: hid_t,
        dapl_id: hid_t,
    ) -> hid_t {
        hdf5_sys::h5d::H5Dcreate2(loc_id, name, type_id, space_id, lcpl_id, dcpl_id, dapl_id)
    }

    pub unsafe fn H5Dopen2(loc_id: hid_t, name: *const c_char, dapl_id: hid_t) -> hid_t {
        hdf5_sys::h5d::H5Dopen2(loc_id, name, dapl_id)
    }

    pub unsafe fn H5Dclose(dset_id: hid_t) -> herr_t {
        hdf5_sys::h5d::H5Dclose(dset_id)
    }

    pub unsafe fn H5Dread(
        dset_id: hid_t,
        mem_type_id: hid_t,
        mem_space_id: hid_t,
        file_space_id: hid_t,
        plist_id: hid_t,
        buf: *mut c_void,
    ) -> herr_t {
        hdf5_sys::h5d::H5Dread(
            dset_id,
            mem_type_id,
            mem_space_id,
            file_space_id,
            plist_id,
            buf,
        )
    }

    pub unsafe fn H5Dwrite(
        dset_id: hid_t,
        mem_type_id: hid_t,
        mem_space_id: hid_t,
        file_space_id: hid_t,
        plist_id: hid_t,
        buf: *const c_void,
    ) -> herr_t {
        hdf5_sys::h5d::H5Dwrite(
            dset_id,
            mem_type_id,
            mem_space_id,
            file_space_id,
            plist_id,
            buf,
        )
    }

    pub unsafe fn H5Dget_space(dset_id: hid_t) -> hid_t {
        hdf5_sys::h5d::H5Dget_space(dset_id)
    }

    pub unsafe fn H5Dget_type(dset_id: hid_t) -> hid_t {
        hdf5_sys::h5d::H5Dget_type(dset_id)
    }

    pub unsafe fn H5Acreate2(
        loc_id: hid_t,
        attr_name: *const c_char,
        type_id: hid_t,
        space_id: hid_t,
        acpl_id: hid_t,
        aapl_id: hid_t,
    ) -> hid_t {
        hdf5_sys::h5a::H5Acreate2(loc_id, attr_name, type_id, space_id, acpl_id, aapl_id)
    }

    pub unsafe fn H5Aopen(obj_id: hid_t, attr_name: *const c_char, aapl_id: hid_t) -> hid_t {
        hdf5_sys::h5a::H5Aopen(obj_id, attr_name, aapl_id)
    }

    pub unsafe fn H5Aclose(attr_id: hid_t) -> herr_t {
        hdf5_sys::h5a::H5Aclose(attr_id)
    }

    pub unsafe fn H5Aread(attr_id: hid_t, type_id: hid_t, buf: *mut c_void) -> herr_t {
        hdf5_sys::h5a::H5Aread(attr_id, type_id, buf)
    }

    pub unsafe fn H5Awrite(attr_id: hid_t, type_id: hid_t, buf: *const c_void) -> herr_t {
        hdf5_sys::h5a::H5Awrite(attr_id, type_id, buf)
    }

    pub unsafe fn H5Aget_space(attr_id: hid_t) -> hid_t {
        hdf5_sys::h5a::H5Aget_space(attr_id)
    }

    pub unsafe fn H5Aget_type(attr_id: hid_t) -> hid_t {
        hdf5_sys::h5a::H5Aget_type(attr_id)
    }

    pub unsafe fn H5Screate(type_: c_int) -> hid_t {
        hdf5_sys::h5s::H5Screate(std::mem::transmute(type_))
    }

    pub unsafe fn H5Screate_simple(
        rank: c_int,
        dims: *const hsize_t,
        maxdims: *const hsize_t,
    ) -> hid_t {
        hdf5_sys::h5s::H5Screate_simple(rank, dims, maxdims)
    }

    pub unsafe fn H5Sclose(space_id: hid_t) -> herr_t {
        hdf5_sys::h5s::H5Sclose(space_id)
    }

    pub unsafe fn H5Sget_simple_extent_ndims(space_id: hid_t) -> c_int {
        hdf5_sys::h5s::H5Sget_simple_extent_ndims(space_id)
    }

    pub unsafe fn H5Sget_simple_extent_dims(
        space_id: hid_t,
        dims: *mut hsize_t,
        maxdims: *mut hsize_t,
    ) -> c_int {
        hdf5_sys::h5s::H5Sget_simple_extent_dims(space_id, dims, maxdims)
    }

    pub unsafe fn H5Tcopy(type_id: hid_t) -> hid_t {
        hdf5_sys::h5t::H5Tcopy(type_id)
    }

    pub unsafe fn H5Tclose(type_id: hid_t) -> herr_t {
        hdf5_sys::h5t::H5Tclose(type_id)
    }

    pub unsafe fn H5Tget_class(type_id: hid_t) -> c_int {
        hdf5_sys::h5t::H5Tget_class(type_id) as c_int
    }

    pub unsafe fn H5Tget_size(type_id: hid_t) -> size_t {
        hdf5_sys::h5t::H5Tget_size(type_id)
    }

    pub unsafe fn H5Tcreate(type_class: c_int, size: size_t) -> hid_t {
        hdf5_sys::h5t::H5Tcreate(std::mem::transmute(type_class), size)
    }

    pub unsafe fn H5Tinsert(
        parent_id: hid_t,
        name: *const c_char,
        offset: size_t,
        member_id: hid_t,
    ) -> herr_t {
        hdf5_sys::h5t::H5Tinsert(parent_id, name, offset, member_id)
    }

    pub unsafe fn H5Tvlen_create(base_id: hid_t) -> hid_t {
        hdf5_sys::h5t::H5Tvlen_create(base_id)
    }

    pub unsafe fn H5Tset_size(type_id: hid_t, size: size_t) -> herr_t {
        hdf5_sys::h5t::H5Tset_size(type_id, size)
    }

    pub unsafe fn H5Tset_cset(type_id: hid_t, cset: c_int) -> herr_t {
        hdf5_sys::h5t::H5Tset_cset(type_id, std::mem::transmute(cset))
    }

    pub unsafe fn H5Tset_strpad(type_id: hid_t, strpad: c_int) -> herr_t {
        hdf5_sys::h5t::H5Tset_strpad(type_id, std::mem::transmute(strpad))
    }

    pub unsafe fn H5Iget_type(id: hid_t) -> c_int {
        hdf5_sys::h5i::H5Iget_type(id) as c_int
    }

    pub unsafe fn H5Iis_valid(id: hid_t) -> htri_t {
        hdf5_sys::h5i::H5Iis_valid(id)
    }

    pub unsafe fn H5Iinc_ref(id: hid_t) -> c_int {
        hdf5_sys::h5i::H5Iinc_ref(id)
    }

    pub unsafe fn H5Idec_ref(id: hid_t) -> c_int {
        hdf5_sys::h5i::H5Idec_ref(id)
    }

    pub unsafe fn H5Iget_ref(id: hid_t) -> c_int {
        hdf5_sys::h5i::H5Iget_ref(id)
    }

    pub unsafe fn H5Eget_current_stack() -> hid_t {
        hdf5_sys::h5e::H5Eget_current_stack()
    }

    pub unsafe fn H5Eget_msg(
        msg_id: hid_t,
        type_: *mut c_int,
        msg: *mut c_char,
        size: size_t,
    ) -> ssize_t {
        hdf5_sys::h5e::H5Eget_msg(msg_id, type_ as *mut _, msg, size)
    }

    pub unsafe fn H5Eset_auto2(
        estack_id: hid_t,
        func: *const c_void,
        client_data: *mut c_void,
    ) -> herr_t {
        hdf5_sys::h5e::H5Eset_auto2(estack_id, std::mem::transmute(func), client_data)
    }

    pub unsafe fn H5Eprint2(estack_id: hid_t, stream: *mut c_void) -> herr_t {
        hdf5_sys::h5e::H5Eprint2(estack_id, stream as *mut _)
    }

    pub unsafe fn H5Ewalk2(
        estack_id: hid_t,
        direction: c_int,
        func: *const c_void,
        client_data: *mut c_void,
    ) -> herr_t {
        hdf5_sys::h5e::H5Ewalk2(
            estack_id,
            std::mem::transmute(direction),
            std::mem::transmute(func),
            client_data,
        )
    }

    pub unsafe fn H5Pcreate(cls_id: hid_t) -> hid_t {
        hdf5_sys::h5p::H5Pcreate(cls_id)
    }

    pub unsafe fn H5Pclose(plist_id: hid_t) -> herr_t {
        hdf5_sys::h5p::H5Pclose(plist_id)
    }

    pub unsafe fn H5Lexists(loc_id: hid_t, name: *const c_char, lapl_id: hid_t) -> htri_t {
        hdf5_sys::h5l::H5Lexists(loc_id, name, lapl_id)
    }
}

// =============================================================================
// Runtime loading mode: use libloading (dlopen)
// Only compiled when link feature is not enabled.
// =============================================================================
#[cfg(all(feature = "runtime-loading", not(feature = "link")))]
#[allow(clippy::useless_transmute)] // Required for generic function pointer casting in load_func macro
mod runtime_impl {
    use super::*;
    use libloading::{Library, Symbol};
    use std::sync::{Mutex, OnceLock};

    /// Library state holder
    struct LibState {
        _lib: Library,
        path: String,
        funcs: Functions,
    }

    // SAFETY: Functions contain only function pointers which are Send + Sync
    unsafe impl Send for LibState {}
    unsafe impl Sync for LibState {}

    static LIB: OnceLock<LibState> = OnceLock::new();
    static INIT_LOCK: Mutex<()> = Mutex::new(());

    /// Load the HDF5 library from the given path.
    pub fn load_library(path: &str) -> crate::Result<()> {
        // Fast path: already initialized
        if let Some(state) = LIB.get() {
            if state.path == path {
                return Ok(());
            }
            return Err(crate::Error::AlreadyInitialized(state.path.clone()));
        }

        // Slow path: need to initialize
        let _guard = INIT_LOCK.lock().unwrap();

        // Double-check after acquiring lock
        if let Some(state) = LIB.get() {
            if state.path == path {
                return Ok(());
            }
            return Err(crate::Error::AlreadyInitialized(state.path.clone()));
        }

        // Actually load the library
        let lib = unsafe { Library::new(path) }.map_err(|e| crate::Error::LibraryLoad {
            path: path.to_string(),
            source: e.to_string(),
        })?;

        let funcs = unsafe { Functions::load(&lib)? };

        // Initialize the library
        unsafe {
            let ret = (funcs.H5open)();
            if ret < 0 {
                return Err(crate::Error::Internal("H5open failed".to_string()));
            }
        }

        let state = LibState {
            _lib: lib,
            path: path.to_string(),
            funcs,
        };

        // This should succeed since we hold the lock and checked above
        let _ = LIB.set(state);

        Ok(())
    }

    /// Check if the library is initialized.
    pub fn is_initialized() -> bool {
        LIB.get().is_some()
    }

    /// Get the library path.
    pub fn library_path() -> Option<String> {
        LIB.get().map(|s| s.path.clone())
    }

    /// Get a reference to the loaded library (for loading global constants).
    pub fn get_library() -> Option<&'static Library> {
        LIB.get().map(|s| &s._lib)
    }

    /// Get the function table (panics if not initialized).
    fn funcs() -> &'static Functions {
        &LIB.get()
            .expect("HDF5 not initialized - call hdf5_init() first")
            .funcs
    }

    /// HDF5 function pointers
    pub struct Functions {
        // H5 - General library functions
        pub H5open: unsafe extern "C" fn() -> herr_t,
        pub H5close: unsafe extern "C" fn() -> herr_t,
        pub H5dont_atexit: unsafe extern "C" fn() -> herr_t,
        pub H5get_libversion: unsafe extern "C" fn(*mut c_uint, *mut c_uint, *mut c_uint) -> herr_t,

        // H5F - File operations
        pub H5Fcreate: unsafe extern "C" fn(*const c_char, c_uint, hid_t, hid_t) -> hid_t,
        pub H5Fopen: unsafe extern "C" fn(*const c_char, c_uint, hid_t) -> hid_t,
        pub H5Fclose: unsafe extern "C" fn(hid_t) -> herr_t,
        pub H5Fflush: unsafe extern "C" fn(hid_t, c_int) -> herr_t,
        pub H5Fget_name: unsafe extern "C" fn(hid_t, *mut c_char, size_t) -> ssize_t,

        // H5G - Group operations
        pub H5Gcreate2: unsafe extern "C" fn(hid_t, *const c_char, hid_t, hid_t, hid_t) -> hid_t,
        pub H5Gopen2: unsafe extern "C" fn(hid_t, *const c_char, hid_t) -> hid_t,
        pub H5Gclose: unsafe extern "C" fn(hid_t) -> herr_t,

        // H5D - Dataset operations
        pub H5Dcreate2:
            unsafe extern "C" fn(hid_t, *const c_char, hid_t, hid_t, hid_t, hid_t, hid_t) -> hid_t,
        pub H5Dopen2: unsafe extern "C" fn(hid_t, *const c_char, hid_t) -> hid_t,
        pub H5Dclose: unsafe extern "C" fn(hid_t) -> herr_t,
        pub H5Dread: unsafe extern "C" fn(hid_t, hid_t, hid_t, hid_t, hid_t, *mut c_void) -> herr_t,
        pub H5Dwrite:
            unsafe extern "C" fn(hid_t, hid_t, hid_t, hid_t, hid_t, *const c_void) -> herr_t,
        pub H5Dget_space: unsafe extern "C" fn(hid_t) -> hid_t,
        pub H5Dget_type: unsafe extern "C" fn(hid_t) -> hid_t,

        // H5A - Attribute operations
        pub H5Acreate2:
            unsafe extern "C" fn(hid_t, *const c_char, hid_t, hid_t, hid_t, hid_t) -> hid_t,
        pub H5Aopen: unsafe extern "C" fn(hid_t, *const c_char, hid_t) -> hid_t,
        pub H5Aclose: unsafe extern "C" fn(hid_t) -> herr_t,
        pub H5Aread: unsafe extern "C" fn(hid_t, hid_t, *mut c_void) -> herr_t,
        pub H5Awrite: unsafe extern "C" fn(hid_t, hid_t, *const c_void) -> herr_t,
        pub H5Aget_space: unsafe extern "C" fn(hid_t) -> hid_t,
        pub H5Aget_type: unsafe extern "C" fn(hid_t) -> hid_t,

        // H5S - Dataspace operations
        pub H5Screate: unsafe extern "C" fn(c_int) -> hid_t,
        pub H5Screate_simple: unsafe extern "C" fn(c_int, *const hsize_t, *const hsize_t) -> hid_t,
        pub H5Sclose: unsafe extern "C" fn(hid_t) -> herr_t,
        pub H5Sget_simple_extent_ndims: unsafe extern "C" fn(hid_t) -> c_int,
        pub H5Sget_simple_extent_dims:
            unsafe extern "C" fn(hid_t, *mut hsize_t, *mut hsize_t) -> c_int,

        // H5T - Datatype operations
        pub H5Tcopy: unsafe extern "C" fn(hid_t) -> hid_t,
        pub H5Tclose: unsafe extern "C" fn(hid_t) -> herr_t,
        pub H5Tget_class: unsafe extern "C" fn(hid_t) -> c_int,
        pub H5Tget_size: unsafe extern "C" fn(hid_t) -> size_t,
        pub H5Tcreate: unsafe extern "C" fn(c_int, size_t) -> hid_t,
        pub H5Tinsert: unsafe extern "C" fn(hid_t, *const c_char, size_t, hid_t) -> herr_t,
        pub H5Tvlen_create: unsafe extern "C" fn(hid_t) -> hid_t,
        pub H5Tset_size: unsafe extern "C" fn(hid_t, size_t) -> herr_t,
        pub H5Tset_cset: unsafe extern "C" fn(hid_t, c_int) -> herr_t,
        pub H5Tset_strpad: unsafe extern "C" fn(hid_t, c_int) -> herr_t,

        // H5I - Identifier operations
        pub H5Iget_type: unsafe extern "C" fn(hid_t) -> c_int,
        pub H5Iis_valid: unsafe extern "C" fn(hid_t) -> htri_t,
        pub H5Iinc_ref: unsafe extern "C" fn(hid_t) -> c_int,
        pub H5Idec_ref: unsafe extern "C" fn(hid_t) -> c_int,
        pub H5Iget_ref: unsafe extern "C" fn(hid_t) -> c_int,

        // H5E - Error operations
        pub H5Eget_current_stack: unsafe extern "C" fn() -> hid_t,
        pub H5Eget_msg: unsafe extern "C" fn(hid_t, *mut c_int, *mut c_char, size_t) -> ssize_t,
        pub H5Eset_auto2: unsafe extern "C" fn(hid_t, *const c_void, *mut c_void) -> herr_t,
        pub H5Eprint2: unsafe extern "C" fn(hid_t, *mut c_void) -> herr_t,
        pub H5Ewalk2: unsafe extern "C" fn(hid_t, c_int, *const c_void, *mut c_void) -> herr_t,

        // H5P - Property list operations
        pub H5Pcreate: unsafe extern "C" fn(hid_t) -> hid_t,
        pub H5Pclose: unsafe extern "C" fn(hid_t) -> herr_t,

        // H5L - Link operations
        pub H5Lexists: unsafe extern "C" fn(hid_t, *const c_char, hid_t) -> htri_t,
    }

    impl Functions {
        unsafe fn load(lib: &Library) -> crate::Result<Self> {
            macro_rules! load_func {
                ($lib:expr, $name:ident) => {{
                    let sym: Symbol<unsafe extern "C" fn() -> herr_t> =
                        $lib.get(stringify!($name).as_bytes()).map_err(|e| {
                            crate::Error::Internal(format!(
                                "Failed to load {}: {}",
                                stringify!($name),
                                e
                            ))
                        })?;
                    std::mem::transmute(*sym)
                }};
            }

            Ok(Self {
                H5open: load_func!(lib, H5open),
                H5close: load_func!(lib, H5close),
                H5dont_atexit: load_func!(lib, H5dont_atexit),
                H5get_libversion: load_func!(lib, H5get_libversion),

                H5Fcreate: load_func!(lib, H5Fcreate),
                H5Fopen: load_func!(lib, H5Fopen),
                H5Fclose: load_func!(lib, H5Fclose),
                H5Fflush: load_func!(lib, H5Fflush),
                H5Fget_name: load_func!(lib, H5Fget_name),

                H5Gcreate2: load_func!(lib, H5Gcreate2),
                H5Gopen2: load_func!(lib, H5Gopen2),
                H5Gclose: load_func!(lib, H5Gclose),

                H5Dcreate2: load_func!(lib, H5Dcreate2),
                H5Dopen2: load_func!(lib, H5Dopen2),
                H5Dclose: load_func!(lib, H5Dclose),
                H5Dread: load_func!(lib, H5Dread),
                H5Dwrite: load_func!(lib, H5Dwrite),
                H5Dget_space: load_func!(lib, H5Dget_space),
                H5Dget_type: load_func!(lib, H5Dget_type),

                H5Acreate2: load_func!(lib, H5Acreate2),
                H5Aopen: load_func!(lib, H5Aopen),
                H5Aclose: load_func!(lib, H5Aclose),
                H5Aread: load_func!(lib, H5Aread),
                H5Awrite: load_func!(lib, H5Awrite),
                H5Aget_space: load_func!(lib, H5Aget_space),
                H5Aget_type: load_func!(lib, H5Aget_type),

                H5Screate: load_func!(lib, H5Screate),
                H5Screate_simple: load_func!(lib, H5Screate_simple),
                H5Sclose: load_func!(lib, H5Sclose),
                H5Sget_simple_extent_ndims: load_func!(lib, H5Sget_simple_extent_ndims),
                H5Sget_simple_extent_dims: load_func!(lib, H5Sget_simple_extent_dims),

                H5Tcopy: load_func!(lib, H5Tcopy),
                H5Tclose: load_func!(lib, H5Tclose),
                H5Tget_class: load_func!(lib, H5Tget_class),
                H5Tget_size: load_func!(lib, H5Tget_size),
                H5Tcreate: load_func!(lib, H5Tcreate),
                H5Tinsert: load_func!(lib, H5Tinsert),
                H5Tvlen_create: load_func!(lib, H5Tvlen_create),
                H5Tset_size: load_func!(lib, H5Tset_size),
                H5Tset_cset: load_func!(lib, H5Tset_cset),
                H5Tset_strpad: load_func!(lib, H5Tset_strpad),

                H5Iget_type: load_func!(lib, H5Iget_type),
                H5Iis_valid: load_func!(lib, H5Iis_valid),
                H5Iinc_ref: load_func!(lib, H5Iinc_ref),
                H5Idec_ref: load_func!(lib, H5Idec_ref),
                H5Iget_ref: load_func!(lib, H5Iget_ref),

                H5Eget_current_stack: load_func!(lib, H5Eget_current_stack),
                H5Eget_msg: load_func!(lib, H5Eget_msg),
                H5Eset_auto2: load_func!(lib, H5Eset_auto2),
                H5Eprint2: load_func!(lib, H5Eprint2),
                H5Ewalk2: load_func!(lib, H5Ewalk2),

                H5Pcreate: load_func!(lib, H5Pcreate),
                H5Pclose: load_func!(lib, H5Pclose),

                H5Lexists: load_func!(lib, H5Lexists),
            })
        }
    }

    // Wrapper functions that match hdf5-sys signatures

    pub unsafe fn H5open() -> herr_t {
        (funcs().H5open)()
    }

    pub unsafe fn H5close() -> herr_t {
        (funcs().H5close)()
    }

    pub unsafe fn H5dont_atexit() -> herr_t {
        (funcs().H5dont_atexit)()
    }

    pub unsafe fn H5get_libversion(
        majnum: *mut c_uint,
        minnum: *mut c_uint,
        relnum: *mut c_uint,
    ) -> herr_t {
        (funcs().H5get_libversion)(majnum, minnum, relnum)
    }

    pub unsafe fn H5Fcreate(
        filename: *const c_char,
        flags: c_uint,
        create_plist: hid_t,
        access_plist: hid_t,
    ) -> hid_t {
        (funcs().H5Fcreate)(filename, flags, create_plist, access_plist)
    }

    pub unsafe fn H5Fopen(filename: *const c_char, flags: c_uint, access_plist: hid_t) -> hid_t {
        (funcs().H5Fopen)(filename, flags, access_plist)
    }

    pub unsafe fn H5Fclose(file_id: hid_t) -> herr_t {
        (funcs().H5Fclose)(file_id)
    }

    pub unsafe fn H5Gcreate2(
        loc_id: hid_t,
        name: *const c_char,
        lcpl_id: hid_t,
        gcpl_id: hid_t,
        gapl_id: hid_t,
    ) -> hid_t {
        (funcs().H5Gcreate2)(loc_id, name, lcpl_id, gcpl_id, gapl_id)
    }

    pub unsafe fn H5Gopen2(loc_id: hid_t, name: *const c_char, gapl_id: hid_t) -> hid_t {
        (funcs().H5Gopen2)(loc_id, name, gapl_id)
    }

    pub unsafe fn H5Gclose(group_id: hid_t) -> herr_t {
        (funcs().H5Gclose)(group_id)
    }

    pub unsafe fn H5Dcreate2(
        loc_id: hid_t,
        name: *const c_char,
        type_id: hid_t,
        space_id: hid_t,
        lcpl_id: hid_t,
        dcpl_id: hid_t,
        dapl_id: hid_t,
    ) -> hid_t {
        (funcs().H5Dcreate2)(loc_id, name, type_id, space_id, lcpl_id, dcpl_id, dapl_id)
    }

    pub unsafe fn H5Dopen2(loc_id: hid_t, name: *const c_char, dapl_id: hid_t) -> hid_t {
        (funcs().H5Dopen2)(loc_id, name, dapl_id)
    }

    pub unsafe fn H5Dclose(dset_id: hid_t) -> herr_t {
        (funcs().H5Dclose)(dset_id)
    }

    pub unsafe fn H5Dread(
        dset_id: hid_t,
        mem_type_id: hid_t,
        mem_space_id: hid_t,
        file_space_id: hid_t,
        plist_id: hid_t,
        buf: *mut c_void,
    ) -> herr_t {
        (funcs().H5Dread)(
            dset_id,
            mem_type_id,
            mem_space_id,
            file_space_id,
            plist_id,
            buf,
        )
    }

    pub unsafe fn H5Dwrite(
        dset_id: hid_t,
        mem_type_id: hid_t,
        mem_space_id: hid_t,
        file_space_id: hid_t,
        plist_id: hid_t,
        buf: *const c_void,
    ) -> herr_t {
        (funcs().H5Dwrite)(
            dset_id,
            mem_type_id,
            mem_space_id,
            file_space_id,
            plist_id,
            buf,
        )
    }

    pub unsafe fn H5Dget_space(dset_id: hid_t) -> hid_t {
        (funcs().H5Dget_space)(dset_id)
    }

    pub unsafe fn H5Dget_type(dset_id: hid_t) -> hid_t {
        (funcs().H5Dget_type)(dset_id)
    }

    pub unsafe fn H5Acreate2(
        loc_id: hid_t,
        attr_name: *const c_char,
        type_id: hid_t,
        space_id: hid_t,
        acpl_id: hid_t,
        aapl_id: hid_t,
    ) -> hid_t {
        (funcs().H5Acreate2)(loc_id, attr_name, type_id, space_id, acpl_id, aapl_id)
    }

    pub unsafe fn H5Aopen(obj_id: hid_t, attr_name: *const c_char, aapl_id: hid_t) -> hid_t {
        (funcs().H5Aopen)(obj_id, attr_name, aapl_id)
    }

    pub unsafe fn H5Aclose(attr_id: hid_t) -> herr_t {
        (funcs().H5Aclose)(attr_id)
    }

    pub unsafe fn H5Aread(attr_id: hid_t, type_id: hid_t, buf: *mut c_void) -> herr_t {
        (funcs().H5Aread)(attr_id, type_id, buf)
    }

    pub unsafe fn H5Awrite(attr_id: hid_t, type_id: hid_t, buf: *const c_void) -> herr_t {
        (funcs().H5Awrite)(attr_id, type_id, buf)
    }

    pub unsafe fn H5Aget_space(attr_id: hid_t) -> hid_t {
        (funcs().H5Aget_space)(attr_id)
    }

    pub unsafe fn H5Aget_type(attr_id: hid_t) -> hid_t {
        (funcs().H5Aget_type)(attr_id)
    }

    pub unsafe fn H5Screate(type_: c_int) -> hid_t {
        (funcs().H5Screate)(type_)
    }

    pub unsafe fn H5Screate_simple(
        rank: c_int,
        dims: *const hsize_t,
        maxdims: *const hsize_t,
    ) -> hid_t {
        (funcs().H5Screate_simple)(rank, dims, maxdims)
    }

    pub unsafe fn H5Sclose(space_id: hid_t) -> herr_t {
        (funcs().H5Sclose)(space_id)
    }

    pub unsafe fn H5Sget_simple_extent_ndims(space_id: hid_t) -> c_int {
        (funcs().H5Sget_simple_extent_ndims)(space_id)
    }

    pub unsafe fn H5Sget_simple_extent_dims(
        space_id: hid_t,
        dims: *mut hsize_t,
        maxdims: *mut hsize_t,
    ) -> c_int {
        (funcs().H5Sget_simple_extent_dims)(space_id, dims, maxdims)
    }

    pub unsafe fn H5Tcopy(type_id: hid_t) -> hid_t {
        (funcs().H5Tcopy)(type_id)
    }

    pub unsafe fn H5Tclose(type_id: hid_t) -> herr_t {
        (funcs().H5Tclose)(type_id)
    }

    pub unsafe fn H5Tget_class(type_id: hid_t) -> c_int {
        (funcs().H5Tget_class)(type_id)
    }

    pub unsafe fn H5Tget_size(type_id: hid_t) -> size_t {
        (funcs().H5Tget_size)(type_id)
    }

    pub unsafe fn H5Tcreate(type_class: c_int, size: size_t) -> hid_t {
        (funcs().H5Tcreate)(type_class, size)
    }

    pub unsafe fn H5Tinsert(
        parent_id: hid_t,
        name: *const c_char,
        offset: size_t,
        member_id: hid_t,
    ) -> herr_t {
        (funcs().H5Tinsert)(parent_id, name, offset, member_id)
    }

    pub unsafe fn H5Tvlen_create(base_id: hid_t) -> hid_t {
        (funcs().H5Tvlen_create)(base_id)
    }

    pub unsafe fn H5Tset_size(type_id: hid_t, size: size_t) -> herr_t {
        (funcs().H5Tset_size)(type_id, size)
    }

    pub unsafe fn H5Tset_cset(type_id: hid_t, cset: c_int) -> herr_t {
        (funcs().H5Tset_cset)(type_id, cset)
    }

    pub unsafe fn H5Tset_strpad(type_id: hid_t, strpad: c_int) -> herr_t {
        (funcs().H5Tset_strpad)(type_id, strpad)
    }

    pub unsafe fn H5Iget_type(id: hid_t) -> c_int {
        (funcs().H5Iget_type)(id)
    }

    pub unsafe fn H5Iis_valid(id: hid_t) -> htri_t {
        (funcs().H5Iis_valid)(id)
    }

    pub unsafe fn H5Iinc_ref(id: hid_t) -> c_int {
        (funcs().H5Iinc_ref)(id)
    }

    pub unsafe fn H5Idec_ref(id: hid_t) -> c_int {
        (funcs().H5Idec_ref)(id)
    }

    pub unsafe fn H5Iget_ref(id: hid_t) -> c_int {
        (funcs().H5Iget_ref)(id)
    }

    pub unsafe fn H5Eget_current_stack() -> hid_t {
        (funcs().H5Eget_current_stack)()
    }

    pub unsafe fn H5Eget_msg(
        msg_id: hid_t,
        type_: *mut c_int,
        msg: *mut c_char,
        size: size_t,
    ) -> ssize_t {
        (funcs().H5Eget_msg)(msg_id, type_, msg, size)
    }

    pub unsafe fn H5Eset_auto2(
        estack_id: hid_t,
        func: *const c_void,
        client_data: *mut c_void,
    ) -> herr_t {
        (funcs().H5Eset_auto2)(estack_id, func, client_data)
    }

    pub unsafe fn H5Eprint2(estack_id: hid_t, stream: *mut c_void) -> herr_t {
        (funcs().H5Eprint2)(estack_id, stream)
    }

    pub unsafe fn H5Ewalk2(
        estack_id: hid_t,
        direction: c_int,
        func: *const c_void,
        client_data: *mut c_void,
    ) -> herr_t {
        (funcs().H5Ewalk2)(estack_id, direction, func, client_data)
    }

    pub unsafe fn H5Pcreate(cls_id: hid_t) -> hid_t {
        (funcs().H5Pcreate)(cls_id)
    }

    pub unsafe fn H5Pclose(plist_id: hid_t) -> herr_t {
        (funcs().H5Pclose)(plist_id)
    }

    pub unsafe fn H5Lexists(loc_id: hid_t, name: *const c_char, lapl_id: hid_t) -> htri_t {
        (funcs().H5Lexists)(loc_id, name, lapl_id)
    }
}

// =============================================================================
// Public API: re-export from the appropriate implementation
// =============================================================================

// Link mode takes precedence when both features are enabled
#[cfg(feature = "link")]
pub use link_impl::*;

// Runtime-loading mode only when link is not enabled
#[cfg(all(feature = "runtime-loading", not(feature = "link")))]
pub use runtime_impl::*;

#[cfg(all(feature = "runtime-loading", not(feature = "link")))]
pub use runtime_impl::{get_library, load_library};

// Compile error if neither feature is enabled
#[cfg(not(any(feature = "link", feature = "runtime-loading")))]
compile_error!("Either 'link' or 'runtime-loading' feature must be enabled");
