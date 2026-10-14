//! Integration tests for `tensor4all-partitionedtreetn` in one target.
//! Each former test target is a module, so test names keep the file name as
//! their prefix (filter with `<file>::`). Files that install process-global
//! state stay separate `[[test]]` targets in `Cargo.toml`.

mod adaptive_common;

mod adaptive_cache_candidates;
mod adaptive_interpolation;
mod adaptive_interpolation_engines;
mod adaptive_l2;
mod adaptive_l2_corner_miss;
mod adaptive_l2_diagnostics;
mod adaptive_l2_treetci;
mod adaptive_m2_golden;
mod adaptive_patch_size;
mod adaptive_rounding_calibration;
mod partitioned_tree_tn;
mod patching;
mod qft_reconstruction;
mod reconstruction;
mod subdomain_tree_tn;
