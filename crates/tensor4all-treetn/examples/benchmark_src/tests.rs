use super::*;

#[test]
fn chain_memory_estimate_is_checked_and_exact() {
    let estimate = estimate_memory("mpo-mps", 3, 2, 2, 2);
    assert_eq!(estimate.total_input_bytes, 384);
    assert_eq!(estimate.largest_input_tensor_bytes, 128);
    assert_eq!(estimate.dense_output_bytes, 64);
    assert_eq!(estimate.max_degree, 2);
}

#[test]
fn binary_tree_estimate_has_bounded_degree_and_core_size() {
    let estimate = estimate_memory("tree", 7, 2, 4, 4);
    assert_eq!(estimate.total_input_bytes, 10_240);
    assert_eq!(estimate.largest_input_tensor_bytes, 2_048);
    assert_eq!(estimate.dense_output_bytes, 131_072);
    assert_eq!(estimate.max_degree, 3);
    assert_eq!(binary_tree_shape_counts(15).3, 3);
}

#[test]
#[should_panic(expected = "dense output elements overflow")]
fn memory_estimate_rejects_dense_output_overflow() {
    let _ = estimate_memory("mpo-mpo", usize::BITS as usize, 2, 2, 2);
}
