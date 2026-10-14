//! Integration tests for `tensor4all-treeaci` in one target.
//! Each former test target is a module, so test names keep the file name as
//! their prefix (filter with `<file>::`). Files that install process-global
//! state stay separate `[[test]]` targets in `Cargo.toml`.

#[path = "support/g0.rs"]
mod g0;

mod audit_regressions;
mod branch_degree;
mod branch_diagnostics;
mod f7_session_context;
mod g0_convergence;
mod heavy_g0_convergence;
mod public_api;
mod rank_scaling;
mod resource_limits;
mod rng_stream_contract;
