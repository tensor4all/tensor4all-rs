//! Integration tests for `tensor4all-treetci` in one target.
//! Each former test target is a module, so test names keep the file name as
//! their prefix (filter with `<file>::`). Files that install process-global
//! state stay separate `[[test]]` targets in `Cargo.toml`.

mod common;

mod advanced_quantics;
mod assembly_error_paths;
mod callback_error_type;
mod continuation;
mod edge_visitor_passes;
mod evaluation_memo;
mod external_proposer;
mod heavy_issue1371_local_threshold;
mod heavy_issue1433_box;
mod heavy_issue1441_enumeration;
mod issue1078_context_adoption;
mod issue1153_degenerate_junction;
mod issue1159_rank1_junction;
mod issue1264_rank_slack;
mod issue1352_rank_deficient;
mod issue1352_unit_site;
mod issue1371_local_threshold;
mod issue1433_leaf_side_enumeration;
mod iteration_visitor;
mod materialize_context;
mod optimizer_context;
mod rng_stream_contract;
mod simple_parity;
mod stage_context;
mod termination;
mod tree_interpolator;
mod truncated_proposer_junction;
