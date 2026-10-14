//! Integration tests for `tensor4all-core` in one target.
//! Each former test target is a module, so test names keep the file name as
//! their prefix (filter with `<file>::`). Files that install process-global
//! state stay separate `[[test]]` targets in `Cargo.toml`.

mod ad_integration;
mod bench_contract_fit_patterns;
mod bug_qr_after_permute;
mod ci_single_precision;
mod context_foundation;
mod cuda_context_factorization;
mod cuda_transfer;
mod error_paths;
mod f1_session_contraction;
mod f1_session_inner_product;
mod f1_session_narrow_floats;
mod f1_session_norm;
mod f1_session_permute;
mod f1_session_permute_indices;
mod f1_session_sum;
mod index_suite;
mod linalg_eigh;
mod linalg_factorize;
mod linalg_qr;
mod prepared_contraction;
mod random_rng_stream;
mod resident_rrqr;
mod send_sync;
mod small_pivots;
mod tensor_any_scalar;
mod tensor_api_correction;
mod tensor_basic;
mod tensor_comparison;
mod tensor_contract_nary_pair_equivalence;
mod tensor_contraction;
mod tensor_diag;
mod tensor_mask;
mod tensor_native_ad;
mod tensor_permute;
mod tensor_shape_packing;
mod tensor_tensor_like;
mod tensor_unfuse;
