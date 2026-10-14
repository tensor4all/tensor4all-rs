//! Integration tests for `tensor4all-treetn` in one target.
//! Each former test target is a module, so test names keep the file name as
//! their prefix (filter with `<file>::`). Files that install process-global
//! state stay separate `[[test]]` targets in `Cargo.toml`.

mod ad_treetn;
mod basic;
mod bug_swap_values;
mod cached_evaluator_determinism;
mod cuda_component_transfer;
mod cuda_src_contraction;
mod cuda_tree_contraction;
mod deterministic_order;
mod dmrg;
mod f7_zipup_context;
mod fit_sum;
mod gse;
mod issue192_regression;
mod linsolve;
mod linsolve_mpo_xb;
mod ops;
mod random_rng_stream;
mod repro_swap_truncation;
mod simplett_bridge;
mod site_free_nodes;
mod src_context_seam;
mod swap_test;
mod tdvp;
mod tensor_like;
mod truncate_validation;
mod unfuse;
