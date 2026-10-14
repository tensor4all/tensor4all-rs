//! Integration tests for `tensor4all-tensorci` in one target.
//! Each former test target is a module, so test names keep the file name as
//! their prefix (filter with `<file>::`). Files that install process-global
//! state stay separate `[[test]]` targets in `Cargo.toml`.

mod global_pivot_coordinate_search;
mod rng_stream_contract;
mod rook_zero_fibers;
mod session_entry;
mod small_values;
mod tci2_tolerance;
