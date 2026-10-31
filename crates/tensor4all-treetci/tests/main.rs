//! Integration tests for `tensor4all-treetci` in one target.
//! Each former test target is a module, so test names keep the file name as
//! their prefix (filter with `<file>::`). Files that install process-global
//! state stay separate `[[test]]` targets in `Cargo.toml`.

mod common;

mod advanced_quantics;
mod assembly_error_paths;
mod continuation;
mod evaluation_memo;
mod iteration_visitor;
mod materialize_context;
mod optimizer_context;
mod rng_stream_contract;
mod simple_parity;
mod stage_context;
mod termination;
mod tree_interpolator;
mod truncated_proposer_junction;
