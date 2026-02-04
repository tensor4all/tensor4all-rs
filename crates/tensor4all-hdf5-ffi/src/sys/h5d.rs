#![allow(non_camel_case_types, non_upper_case_globals)]
//! Creating and manipulating scientific datasets (H5D)
//!
//! Types and constants for HDF5 dataset operations.

use super::types::*;

pub use H5D_alloc_time_t::*;
pub use H5D_fill_time_t::*;
pub use H5D_fill_value_t::*;
pub use H5D_layout_t::*;
pub use H5D_mpio_actual_chunk_opt_mode_t::*;
pub use H5D_mpio_actual_io_mode_t::*;
pub use H5D_mpio_no_collective_cause_t::*;
pub use H5D_space_status_t::*;
pub use H5D_vds_view_t::*;

// Default chunk cache settings
pub const H5D_CHUNK_CACHE_NSLOTS_DEFAULT: size_t = !0;
pub const H5D_CHUNK_CACHE_NBYTES_DEFAULT: size_t = !0;
pub const H5D_CHUNK_CACHE_W0_DEFAULT: c_float = -1.0;

// Chunk options
pub const H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS: c_uint = 0x0002;

// Chunk index types
pub type H5D_chunk_index_t = c_uint;
pub const H5D_CHUNK_BTREE: H5D_chunk_index_t = 0;
pub const H5D_CHUNK_IDX_BTREE: H5D_chunk_index_t = H5D_CHUNK_BTREE;

/// Dataset layout
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5D_layout_t {
    H5D_LAYOUT_ERROR = -1,
    H5D_COMPACT = 0,
    H5D_CONTIGUOUS = 1,
    H5D_CHUNKED = 2,
    H5D_VIRTUAL = 3,
    H5D_NLAYOUTS = 4,
}

impl Default for H5D_layout_t {
    fn default() -> Self {
        Self::H5D_CONTIGUOUS
    }
}

/// Storage allocation time
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5D_alloc_time_t {
    H5D_ALLOC_TIME_ERROR = -1,
    H5D_ALLOC_TIME_DEFAULT = 0,
    H5D_ALLOC_TIME_EARLY = 1,
    H5D_ALLOC_TIME_LATE = 2,
    H5D_ALLOC_TIME_INCR = 3,
}

impl Default for H5D_alloc_time_t {
    fn default() -> Self {
        Self::H5D_ALLOC_TIME_DEFAULT
    }
}

/// Space allocation status
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5D_space_status_t {
    H5D_SPACE_STATUS_ERROR = -1,
    H5D_SPACE_STATUS_NOT_ALLOCATED = 0,
    H5D_SPACE_STATUS_PART_ALLOCATED = 1,
    H5D_SPACE_STATUS_ALLOCATED = 2,
}

/// Fill value write time
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5D_fill_time_t {
    H5D_FILL_TIME_ERROR = -1,
    H5D_FILL_TIME_ALLOC = 0,
    H5D_FILL_TIME_NEVER = 1,
    H5D_FILL_TIME_IFSET = 2,
}

impl Default for H5D_fill_time_t {
    fn default() -> Self {
        Self::H5D_FILL_TIME_IFSET
    }
}

/// Fill value status
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5D_fill_value_t {
    H5D_FILL_VALUE_ERROR = -1,
    H5D_FILL_VALUE_UNDEFINED = 0,
    H5D_FILL_VALUE_DEFAULT = 1,
    H5D_FILL_VALUE_USER_DEFINED = 2,
}

impl Default for H5D_fill_value_t {
    fn default() -> Self {
        Self::H5D_FILL_VALUE_DEFAULT
    }
}

/// MPI-IO chunk optimization mode
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5D_mpio_actual_chunk_opt_mode_t {
    H5D_MPIO_NO_CHUNK_OPTIMIZATION = 0,
    H5D_MPIO_LINK_CHUNK = 1,
    H5D_MPIO_MULTI_CHUNK = 2,
}

/// MPI-IO actual I/O mode
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5D_mpio_actual_io_mode_t {
    H5D_MPIO_NO_COLLECTIVE = 0,
    H5D_MPIO_CHUNK_INDEPENDENT = 1,
    H5D_MPIO_CHUNK_COLLECTIVE = 2,
    H5D_MPIO_CHUNK_MIXED = 3,
    H5D_MPIO_CONTIGUOUS_COLLECTIVE = 4,
}

/// Reason for not performing collective I/O
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5D_mpio_no_collective_cause_t {
    H5D_MPIO_COLLECTIVE = 0,
    H5D_MPIO_SET_INDEPENDENT = 1,
    H5D_MPIO_DATATYPE_CONVERSION = 2,
    H5D_MPIO_DATA_TRANSFORMS = 4,
    H5D_MPIO_MPI_OPT_TYPES_ENV_VAR_DISABLED = 8,
    H5D_MPIO_NOT_SIMPLE_OR_SCALAR_DATASPACES = 16,
    H5D_MPIO_NOT_CONTIGUOUS_OR_CHUNKED_DATASET = 32,
    H5D_MPIO_FILTERS = 64,
}

/// VDS view type
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
pub enum H5D_vds_view_t {
    H5D_VDS_ERROR = -1,
    H5D_VDS_FIRST_MISSING = 0,
    H5D_VDS_LAST_AVAILABLE = 1,
}

impl Default for H5D_vds_view_t {
    fn default() -> Self {
        Self::H5D_VDS_LAST_AVAILABLE
    }
}

/// Dataset operator type
pub type H5D_operator_t = Option<
    unsafe extern "C" fn(
        elem: *mut c_void,
        type_id: hid_t,
        ndim: c_uint,
        point: *const hsize_t,
        operator_data: *mut c_void,
    ) -> herr_t,
>;

/// Scatter function type
pub type H5D_scatter_func_t = Option<
    unsafe extern "C" fn(
        src_buf: *mut *const c_void,
        src_buf_bytes_used: *mut size_t,
        op_data: *mut c_void,
    ) -> herr_t,
>;

/// Gather function type
pub type H5D_gather_func_t = Option<
    unsafe extern "C" fn(
        dst_buf: *const c_void,
        dst_buf_bytes_used: size_t,
        op_data: *mut c_void,
    ) -> herr_t,
>;

/// Append callback type
pub type H5D_append_cb_t = Option<
    unsafe extern "C" fn(dataset_id: hid_t, cur_dims: *mut hsize_t, op_data: *mut c_void) -> herr_t,
>;

/// Chunk iterator callback type
pub type H5D_chunk_iter_op_t = Option<
    unsafe extern "C" fn(
        offset: *const hsize_t,
        filter_mask: c_uint,
        addr: haddr_t,
        size: hsize_t,
        op_data: *mut c_void,
    ) -> c_int,
>;
