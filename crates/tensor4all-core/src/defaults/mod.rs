//! Default concrete type implementations.
//!
//! This module provides the default concrete types for tensor network operations:
//!
//! - [`DynIndex`]: Default index type (`Index<DynId, NoSymmSpace, TagSet>`)
//!
//! These types are suitable for most tensor network applications and provide
//! a good balance of flexibility and performance.

mod dyn_index;

pub use dyn_index::DynIndex;
