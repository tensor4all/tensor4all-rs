//! Integration tests for `tensor4all-treersi` in one target.
//! Each former test target is a module, so test names keep the file name as
//! their prefix (filter with `<file>::`). Files that install process-global
//! state stay separate `[[test]]` targets in `Cargo.toml`.

#[path = "../examples/support/mod.rs"]
mod support;

mod boundaries;
mod products;
