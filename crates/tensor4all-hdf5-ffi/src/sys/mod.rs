//! Low-level HDF5 FFI bindings.
//!
//! Two modes are supported:
//! - `link` feature (default): Re-exports from hdf5-sys (hdf5-metno-sys)
//! - `runtime-loading` feature: Uses libloading for runtime loading

// Local type definitions (used by both modes)
pub mod h5;
pub mod h5a;
pub mod h5d;
pub mod h5e;
pub mod h5f;
pub mod h5g;
pub mod h5i;
pub mod h5l;
pub mod h5o;
pub mod h5p;
pub mod h5r;
pub mod h5s;
pub mod h5t;
pub mod h5z;
pub mod types;

// Function implementations
pub mod funcs;

// Re-export all types from local modules
pub use h5::*;
pub use h5a::*;
pub use h5d::*;
pub use h5e::*;
pub use h5f::*;
pub use h5g::*;
pub use h5i::*;
pub use h5l::*;
pub use h5o::*;
pub use h5p::*;
pub use h5r::*;
pub use h5s::*;
pub use h5t::*;
pub use h5z::*;
pub use types::*;

// Re-export function loading (conditional)
pub use funcs::{is_initialized, library_path};

#[cfg(feature = "runtime-loading")]
pub use funcs::load_library;

// Re-export wrapper functions
pub use funcs::{
    H5Aclose, H5Acreate2, H5Aget_space, H5Aget_type, H5Aopen, H5Aread, H5Awrite, H5Dclose,
    H5Dcreate2, H5Dget_space, H5Dget_type, H5Dopen2, H5Dread, H5Dwrite, H5Eget_current_stack,
    H5Eget_msg, H5Eprint2, H5Eset_auto2, H5Ewalk2, H5Fclose, H5Fcreate, H5Fopen, H5Gclose,
    H5Gcreate2, H5Gopen2, H5Idec_ref, H5Iget_ref, H5Iget_type, H5Iinc_ref, H5Iis_valid, H5Lexists,
    H5Pclose, H5Pcreate, H5Sclose, H5Screate, H5Screate_simple, H5Sget_simple_extent_dims,
    H5Sget_simple_extent_ndims, H5Tclose, H5Tcopy, H5Tcreate, H5Tget_class, H5Tget_size, H5Tinsert,
    H5Tset_cset, H5Tset_size, H5Tset_strpad, H5Tvlen_create, H5close, H5dont_atexit,
    H5get_libversion, H5open,
};
