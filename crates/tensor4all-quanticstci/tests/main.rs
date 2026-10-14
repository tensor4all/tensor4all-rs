//! Integration tests for `tensor4all-quanticstci` in one target.
//! Each former test target is a module, so test names keep the file name as
//! their prefix (filter with `<file>::`). Files that install process-global
//! state stay separate `[[test]]` targets in `Cargo.toml`.

mod context_entry;
mod feature_test_physicist;
mod qft_2d_test;
mod readme_example;
