use crate::{CandidateMatrixSource, DenseMatrixSource};

#[test]
fn dense_source_get_block_is_column_major() {
    let src = DenseMatrixSource::from_column_major(&[1.0, 3.0, 2.0, 4.0], 2, 2);
    let mut out = [0.0; 4];
    src.get_block(&[0, 1], &[0, 1], &mut out);
    assert_eq!(out, [1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn dense_source_get_block_uses_cross_product_for_noncontiguous_indices() {
    let src = DenseMatrixSource::from_column_major(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, 3);
    let mut out = [0.0; 4];
    src.get_block(&[1, 0], &[2, 0], &mut out);
    assert_eq!(out, [6.0, 3.0, 4.0, 1.0]);
}

#[test]
fn scalar_get_delegates_to_get_block() {
    let src = DenseMatrixSource::from_column_major(&[1.0, 3.0, 2.0, 4.0], 2, 2);
    assert_eq!(src.get(1, 0), 3.0);
}
