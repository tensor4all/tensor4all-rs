//! Integration tests for `tensor4all-itensorlike` in one target.
//! Each former test target is a module, so test names keep the file name as
//! their prefix (filter with `<file>::`). Files that install process-global
//! state stay separate `[[test]]` targets in `Cargo.toml`.

mod bench_basic_ops;
mod bug_complex_inner;
mod bug_fit_bond_growth;
mod bug_fit_elementwise;
mod bug_norm_oom_large_tt;
mod bug_zipup_inflated_bonds;
mod contract_context;
mod fit_adaptive_initializer;
mod fit_bond_capping;
mod linsolve_mpo;
mod panic_paths;
mod tensortrain_inner;
