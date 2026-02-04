//! HDF5 type definitions (from hdf5-types).
//!
//! This module provides native Rust equivalents of HDF5 types.

pub mod array;
#[cfg(feature = "complex")]
pub mod complex;
pub mod dyn_value;
pub mod h5type;
pub mod string;

// Re-export libc functions for memory management
pub use libc::{free, malloc};

// Re-export main types
pub use array::*;
pub use dyn_value::*;
pub use h5type::*;
pub use string::*;
