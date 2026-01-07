//! 2D matrix multiplication for blocked arrays.

use std::collections::HashMap;

use mdarray::DTensor;
use mdarray_linalg::matmul::{MatMul, MatMulBuilder};
use mdarray_linalg_faer::Faer;

use crate::block_data::{BlockData, BlockTensor2};
use crate::blocked_array::{BlockedArray, BlockedArrayLike};
use crate::error::{BlockedArrayError, Result};
use crate::scalar::Scalar;

/// Build index of non-zero blocks grouped by a specific axis.
///
/// For a 2D blocked array, returns a map from axis index to list of other-axis indices.
/// - `group_by_axis=0`: groups by row, returns `row -> [cols with non-zero blocks]`
/// - `group_by_axis=1`: groups by col, returns `col -> [rows with non-zero blocks]`
fn build_nonzero_index<T: Scalar, A: BlockedArrayLike<T>>(
    arr: &A,
    group_by_axis: usize,
) -> HashMap<usize, Vec<usize>> {
    let mut index: HashMap<usize, Vec<usize>> = HashMap::new();

    for (block_idx, _) in arr.iter_nonzero_blocks() {
        let key = block_idx[group_by_axis];
        let value = block_idx[1 - group_by_axis];
        index.entry(key).or_default().push(value);
    }

    index
}

/// Blocked matrix multiplication: C = A @ B.
///
/// Uses sparse-matrix-style algorithm (outer product formulation):
/// only computes products of non-zero block pairs.
///
/// # Algorithm
/// ```text
/// // Build indices of non-zero blocks
/// A_by_k[k] = [i : A[i,k] is nonzero]
/// B_by_k[k] = [j : B[k,j] is nonzero]
///
/// // Only iterate over non-zero combinations
/// for k in (keys of A_by_k ∩ keys of B_by_k):
///     for i in A_by_k[k]:
///         for j in B_by_k[k]:
///             C[i,j] += A[i,k] @ B[k,j]
/// ```
///
/// # Errors
/// - Returns `BlockedArrayError::NotMatrix` if inputs are not 2D.
/// - Returns `BlockedArrayError::IncompatiblePartitions` if inner partitions don't match.
pub fn blocked_matmul<T, A, B>(a: &A, b: &B) -> Result<BlockedArray<T>>
where
    T: Scalar,
    A: BlockedArrayLike<T>,
    B: BlockedArrayLike<T>,
{
    // Check: must be 2D
    if a.rank() != 2 {
        return Err(BlockedArrayError::NotMatrix(a.rank()));
    }
    if b.rank() != 2 {
        return Err(BlockedArrayError::NotMatrix(b.rank()));
    }

    let a_parts = a.partitions();
    let b_parts = b.partitions();

    // Check compatibility: a.partitions[1] == b.partitions[0]
    if a_parts[1] != b_parts[0] {
        return Err(BlockedArrayError::IncompatiblePartitions);
    }

    let result_partitions = vec![a_parts[0].clone(), b_parts[1].clone()];
    let mut result = BlockedArray::new(result_partitions);

    // Build indices: group A blocks by column (k), B blocks by row (k)
    // A_by_k[k] = list of row indices i where A[i,k] is non-zero
    // B_by_k[k] = list of col indices j where B[k,j] is non-zero
    let a_by_k = build_nonzero_index(a, 1); // group by column
    let b_by_k = build_nonzero_index(b, 0); // group by row

    // Outer product formulation: iterate only over k values with non-zero blocks in both A and B
    for (k, a_rows) in &a_by_k {
        let b_cols = match b_by_k.get(k) {
            Some(cols) => cols,
            None => continue, // No B blocks for this k
        };

        for &i in a_rows {
            // A[i,k] is guaranteed to be non-zero
            let a_block = a.get_block(&vec![i, *k]).unwrap();

            for &j in b_cols {
                // B[k,j] is guaranteed to be non-zero
                let b_block = b.get_block(&vec![*k, j]).unwrap();

                // Compute A[i,k] @ B[k,j]
                let product = block_matmul(&a_block, &b_block)?;

                // Accumulate into C[i,j]
                result.accumulate_block(vec![i, j], product);
            }
        }
    }

    Ok(result)
}

/// Multiply two blocks (2D matrices).
///
/// Handles three cases for efficiency:
/// - Case 1: scalar x scalar (both 1x1)
/// - Case 2: scalar x dense or dense x scalar -> scaling operation
/// - Case 3: dense x dense -> delegates to mdarray-linalg (Faer backend)
fn block_matmul<T: Scalar>(a: &BlockData<T>, b: &BlockData<T>) -> Result<BlockData<T>> {
    let [m, k1] = a.shape();
    let [k2, n] = b.shape();

    if k1 != k2 {
        return Err(BlockedArrayError::ShapeMismatch {
            expected: vec![m, k1],
            actual: vec![k2, n],
        });
    }

    let k = k1;

    // Case 1: scalar x scalar (both 1x1)
    if m == 1 && k == 1 && n == 1 {
        let result = a.get([0, 0]) * b.get([0, 0]);
        return Ok(BlockData::scalar(result));
    }

    // Case 2: scalar x dense (a is 1x1)
    if m == 1 && k == 1 {
        let scalar = a.get([0, 0]);
        let result: BlockTensor2<T> =
            DTensor::<T, 2>::from_fn([1, n], |idx| scalar * b.get([0, idx[1]]));
        return Ok(BlockData::from_tensor(result));
    }

    // Case 2: dense x scalar (b is 1x1)
    if k == 1 && n == 1 {
        let scalar = b.get([0, 0]);
        let result: BlockTensor2<T> =
            DTensor::<T, 2>::from_fn([m, 1], |idx| a.get([idx[0], 0]) * scalar);
        return Ok(BlockData::from_tensor(result));
    }

    // Case 3: dense x dense - delegate to mdarray-linalg
    // Use mdarray slices directly with Faer backend
    let a_slice = a.as_slice();
    let b_slice = b.as_slice();

    let c_tensor = Faer.matmul(a_slice, b_slice).eval();

    Ok(BlockData::from_tensor(c_tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocked_array::BlockedArray;
    use crate::partition::BlockPartition;
    use num_complex::Complex64;

    fn test_blocked_matmul_identity_generic<T: Scalar>() {
        // Identity matrix @ vector
        // [1 0] @ [a] = [a]
        // [0 1]   [b]   [b]

        let a_parts = vec![
            BlockPartition::uniform(1, 2),
            BlockPartition::uniform(1, 2),
        ];
        let mut a = BlockedArray::<T>::new(a_parts);
        a.set_block(vec![0, 0], BlockData::scalar(T::from_f64(1.0)));
        a.set_block(vec![1, 1], BlockData::scalar(T::from_f64(1.0)));

        let b_parts = vec![
            BlockPartition::uniform(1, 2),
            BlockPartition::trivial(1),
        ];
        let mut b = BlockedArray::<T>::new(b_parts);
        b.set_block(vec![0, 0], BlockData::scalar(T::from_f64(3.0)));
        b.set_block(vec![1, 0], BlockData::scalar(T::from_f64(5.0)));

        let c = blocked_matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), vec![2, 1]);
        let c00 = c.get_block(&vec![0, 0]).unwrap();
        let c10 = c.get_block(&vec![1, 0]).unwrap();
        assert_eq!(c00.get([0, 0]), T::from_f64(3.0));
        assert_eq!(c10.get([0, 0]), T::from_f64(5.0));
    }

    #[test]
    fn test_blocked_matmul_identity_f64() {
        test_blocked_matmul_identity_generic::<f64>();
    }

    #[test]
    fn test_blocked_matmul_identity_c64() {
        test_blocked_matmul_identity_generic::<Complex64>();
    }

    fn test_blocked_matmul_simple_generic<T: Scalar>() {
        // [1 2] @ [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]

        let parts = vec![
            BlockPartition::trivial(2),
            BlockPartition::trivial(2),
        ];

        let mut a = BlockedArray::<T>::new(parts.clone());
        a.set_block(
            vec![0, 0],
            BlockData::new(
                vec![1.0, 2.0, 3.0, 4.0]
                    .into_iter()
                    .map(T::from_f64)
                    .collect(),
                [2, 2],
            ),
        );

        let mut b = BlockedArray::<T>::new(parts.clone());
        b.set_block(
            vec![0, 0],
            BlockData::new(
                vec![5.0, 6.0, 7.0, 8.0]
                    .into_iter()
                    .map(T::from_f64)
                    .collect(),
                [2, 2],
            ),
        );

        let c = blocked_matmul(&a, &b).unwrap();

        let c_block = c.get_block(&vec![0, 0]).unwrap();
        assert_eq!(c_block.get([0, 0]), T::from_f64(19.0));
        assert_eq!(c_block.get([0, 1]), T::from_f64(22.0));
        assert_eq!(c_block.get([1, 0]), T::from_f64(43.0));
        assert_eq!(c_block.get([1, 1]), T::from_f64(50.0));
    }

    #[test]
    fn test_blocked_matmul_simple_f64() {
        test_blocked_matmul_simple_generic::<f64>();
    }

    #[test]
    fn test_blocked_matmul_simple_c64() {
        test_blocked_matmul_simple_generic::<Complex64>();
    }

    fn test_blocked_matmul_sparse_generic<T: Scalar>() {
        // Block-diagonal matrix @ Block-diagonal matrix
        // [A 0] @ [C 0] = [AC  0]
        // [0 B]   [0 D]   [ 0 BD]

        let parts = vec![
            BlockPartition::uniform(2, 2),
            BlockPartition::uniform(2, 2),
        ];

        let mut a = BlockedArray::<T>::new(parts.clone());
        a.set_block(
            vec![0, 0],
            BlockData::new(
                vec![1.0, 2.0, 3.0, 4.0]
                    .into_iter()
                    .map(T::from_f64)
                    .collect(),
                [2, 2],
            ),
        ); // A
        a.set_block(
            vec![1, 1],
            BlockData::new(
                vec![5.0, 6.0, 7.0, 8.0]
                    .into_iter()
                    .map(T::from_f64)
                    .collect(),
                [2, 2],
            ),
        ); // B

        let mut b = BlockedArray::<T>::new(parts.clone());
        b.set_block(
            vec![0, 0],
            BlockData::new(
                vec![1.0, 0.0, 0.0, 1.0]
                    .into_iter()
                    .map(T::from_f64)
                    .collect(),
                [2, 2],
            ),
        ); // C = I
        b.set_block(
            vec![1, 1],
            BlockData::new(
                vec![2.0, 0.0, 0.0, 2.0]
                    .into_iter()
                    .map(T::from_f64)
                    .collect(),
                [2, 2],
            ),
        ); // D = 2I

        let c = blocked_matmul(&a, &b).unwrap();

        // Should have 2 non-zero blocks
        assert_eq!(c.num_nonzero_blocks(), 2);

        // AC = A * I = A
        let c00 = c.get_block(&vec![0, 0]).unwrap();
        assert_eq!(c00.get([0, 0]), T::from_f64(1.0));
        assert_eq!(c00.get([1, 1]), T::from_f64(4.0));

        // BD = B * 2I = 2B
        let c11 = c.get_block(&vec![1, 1]).unwrap();
        assert_eq!(c11.get([0, 0]), T::from_f64(10.0));
        assert_eq!(c11.get([1, 1]), T::from_f64(16.0));

        // Zero blocks
        assert!(c.get_block(&vec![0, 1]).is_none());
        assert!(c.get_block(&vec![1, 0]).is_none());
    }

    #[test]
    fn test_blocked_matmul_sparse_f64() {
        test_blocked_matmul_sparse_generic::<f64>();
    }

    #[test]
    fn test_blocked_matmul_sparse_c64() {
        test_blocked_matmul_sparse_generic::<Complex64>();
    }

    fn test_blocked_matmul_transposed_generic<T: Scalar>() {
        // Test with transposed input
        // A^T @ B where A is stored and we use transpose view

        let parts = vec![
            BlockPartition::trivial(2),
            BlockPartition::trivial(2),
        ];

        let mut a = BlockedArray::<T>::new(parts.clone());
        // A = [1 3]
        //     [2 4]
        // Stored as row-major: [1, 3, 2, 4]
        a.set_block(
            vec![0, 0],
            BlockData::new(
                vec![1.0, 3.0, 2.0, 4.0]
                    .into_iter()
                    .map(T::from_f64)
                    .collect(),
                [2, 2],
            ),
        );

        // A^T = [1 2]
        //       [3 4]
        let a_t = a.transpose();

        let mut b = BlockedArray::<T>::new(parts.clone());
        b.set_block(
            vec![0, 0],
            BlockData::new(
                vec![1.0, 0.0, 0.0, 1.0]
                    .into_iter()
                    .map(T::from_f64)
                    .collect(),
                [2, 2],
            ),
        ); // Identity

        let c = blocked_matmul(&a_t, &b).unwrap();

        let c_block = c.get_block(&vec![0, 0]).unwrap();
        // A^T @ I = A^T = [1 2]
        //                 [3 4]
        assert_eq!(c_block.get([0, 0]), T::from_f64(1.0));
        assert_eq!(c_block.get([0, 1]), T::from_f64(2.0));
        assert_eq!(c_block.get([1, 0]), T::from_f64(3.0));
        assert_eq!(c_block.get([1, 1]), T::from_f64(4.0));
    }

    #[test]
    fn test_blocked_matmul_transposed_f64() {
        test_blocked_matmul_transposed_generic::<f64>();
    }

    #[test]
    fn test_blocked_matmul_transposed_c64() {
        test_blocked_matmul_transposed_generic::<Complex64>();
    }

    #[test]
    fn test_blocked_matmul_incompatible() {
        let a_parts = vec![
            BlockPartition::uniform(2, 2),
            BlockPartition::uniform(3, 2), // Inner dim: 6
        ];
        let a = BlockedArray::<f64>::new(a_parts);

        let b_parts = vec![
            BlockPartition::uniform(4, 2), // Inner dim: 8 (mismatch!)
            BlockPartition::uniform(2, 2),
        ];
        let b = BlockedArray::<f64>::new(b_parts);

        let result = blocked_matmul(&a, &b);
        assert!(matches!(
            result,
            Err(BlockedArrayError::IncompatiblePartitions)
        ));
    }

    fn test_block_matmul_scalar_times_scalar_generic<T: Scalar>() {
        // Case 1: 1x1 x 1x1 = 1x1
        // [3] @ [4] = [12]
        let a_parts = vec![
            BlockPartition::trivial(1),
            BlockPartition::trivial(1),
        ];
        let mut a = BlockedArray::<T>::new(a_parts);
        a.set_block(vec![0, 0], BlockData::scalar(T::from_f64(3.0)));

        let b_parts = vec![
            BlockPartition::trivial(1),
            BlockPartition::trivial(1),
        ];
        let mut b = BlockedArray::<T>::new(b_parts);
        b.set_block(vec![0, 0], BlockData::scalar(T::from_f64(4.0)));

        let c = blocked_matmul(&a, &b).unwrap();
        let c_block = c.get_block(&vec![0, 0]).unwrap();
        assert_eq!(c_block.get([0, 0]), T::from_f64(12.0));
    }

    #[test]
    fn test_block_matmul_scalar_times_scalar_f64() {
        test_block_matmul_scalar_times_scalar_generic::<f64>();
    }

    #[test]
    fn test_block_matmul_scalar_times_scalar_c64() {
        test_block_matmul_scalar_times_scalar_generic::<Complex64>();
    }

    fn test_block_matmul_scalar_times_dense_generic<T: Scalar>() {
        // Case 2a: 1x1 x 1x3 = 1x3
        // [2] @ [1 2 3] = [2 4 6]
        let a_parts = vec![
            BlockPartition::trivial(1),
            BlockPartition::trivial(1),
        ];
        let mut a = BlockedArray::<T>::new(a_parts);
        a.set_block(vec![0, 0], BlockData::scalar(T::from_f64(2.0)));

        let b_parts = vec![
            BlockPartition::trivial(1),
            BlockPartition::trivial(3),
        ];
        let mut b = BlockedArray::<T>::new(b_parts);
        b.set_block(
            vec![0, 0],
            BlockData::new(
                vec![1.0, 2.0, 3.0]
                    .into_iter()
                    .map(T::from_f64)
                    .collect(),
                [1, 3],
            ),
        );

        let c = blocked_matmul(&a, &b).unwrap();
        let c_block = c.get_block(&vec![0, 0]).unwrap();
        assert_eq!(c_block.shape(), [1, 3]);
        assert_eq!(c_block.get([0, 0]), T::from_f64(2.0));
        assert_eq!(c_block.get([0, 1]), T::from_f64(4.0));
        assert_eq!(c_block.get([0, 2]), T::from_f64(6.0));
    }

    #[test]
    fn test_block_matmul_scalar_times_dense_f64() {
        test_block_matmul_scalar_times_dense_generic::<f64>();
    }

    #[test]
    fn test_block_matmul_scalar_times_dense_c64() {
        test_block_matmul_scalar_times_dense_generic::<Complex64>();
    }

    fn test_block_matmul_dense_times_scalar_generic<T: Scalar>() {
        // Case 2b: 3x1 x 1x1 = 3x1
        // [1]     [2]
        // [2] @ [2] = [4]
        // [3]     [6]
        let a_parts = vec![
            BlockPartition::trivial(3),
            BlockPartition::trivial(1),
        ];
        let mut a = BlockedArray::<T>::new(a_parts);
        a.set_block(
            vec![0, 0],
            BlockData::new(
                vec![1.0, 2.0, 3.0]
                    .into_iter()
                    .map(T::from_f64)
                    .collect(),
                [3, 1],
            ),
        );

        let b_parts = vec![
            BlockPartition::trivial(1),
            BlockPartition::trivial(1),
        ];
        let mut b = BlockedArray::<T>::new(b_parts);
        b.set_block(vec![0, 0], BlockData::scalar(T::from_f64(2.0)));

        let c = blocked_matmul(&a, &b).unwrap();
        let c_block = c.get_block(&vec![0, 0]).unwrap();
        assert_eq!(c_block.shape(), [3, 1]);
        assert_eq!(c_block.get([0, 0]), T::from_f64(2.0));
        assert_eq!(c_block.get([1, 0]), T::from_f64(4.0));
        assert_eq!(c_block.get([2, 0]), T::from_f64(6.0));
    }

    #[test]
    fn test_block_matmul_dense_times_scalar_f64() {
        test_block_matmul_dense_times_scalar_generic::<f64>();
    }

    #[test]
    fn test_block_matmul_dense_times_scalar_c64() {
        test_block_matmul_dense_times_scalar_generic::<Complex64>();
    }

    fn test_block_matmul_dense_times_dense_with_faer_generic<T: Scalar>() {
        // Case 3: 3x2 x 2x4 = 3x4 (uses mdarray-linalg/Faer)
        // [1 2]   [1 2 3 4]   [1+10 2+12 3+14 4+16]   [11 14 17 20]
        // [3 4] @ [5 6 7 8] = [3+20 6+24 9+28 12+32] = [23 30 37 44]
        // [5 6]               [5+30 10+36 15+42 20+48] [35 46 57 68]
        let a_parts = vec![
            BlockPartition::trivial(3),
            BlockPartition::trivial(2),
        ];
        let mut a = BlockedArray::<T>::new(a_parts);
        a.set_block(
            vec![0, 0],
            BlockData::new(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
                    .into_iter()
                    .map(T::from_f64)
                    .collect(),
                [3, 2],
            ),
        );

        let b_parts = vec![
            BlockPartition::trivial(2),
            BlockPartition::trivial(4),
        ];
        let mut b = BlockedArray::<T>::new(b_parts);
        b.set_block(
            vec![0, 0],
            BlockData::new(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                    .into_iter()
                    .map(T::from_f64)
                    .collect(),
                [2, 4],
            ),
        );

        let c = blocked_matmul(&a, &b).unwrap();
        let c_block = c.get_block(&vec![0, 0]).unwrap();
        assert_eq!(c_block.shape(), [3, 4]);

        // First row: [1,2] @ [[1,2,3,4],[5,6,7,8]] = [11, 14, 17, 20]
        assert_eq!(c_block.get([0, 0]), T::from_f64(11.0));
        assert_eq!(c_block.get([0, 1]), T::from_f64(14.0));
        assert_eq!(c_block.get([0, 2]), T::from_f64(17.0));
        assert_eq!(c_block.get([0, 3]), T::from_f64(20.0));

        // Second row: [3,4] @ [[1,2,3,4],[5,6,7,8]] = [23, 30, 37, 44]
        assert_eq!(c_block.get([1, 0]), T::from_f64(23.0));
        assert_eq!(c_block.get([1, 1]), T::from_f64(30.0));
        assert_eq!(c_block.get([1, 2]), T::from_f64(37.0));
        assert_eq!(c_block.get([1, 3]), T::from_f64(44.0));

        // Third row: [5,6] @ [[1,2,3,4],[5,6,7,8]] = [35, 46, 57, 68]
        assert_eq!(c_block.get([2, 0]), T::from_f64(35.0));
        assert_eq!(c_block.get([2, 1]), T::from_f64(46.0));
        assert_eq!(c_block.get([2, 2]), T::from_f64(57.0));
        assert_eq!(c_block.get([2, 3]), T::from_f64(68.0));
    }

    #[test]
    fn test_block_matmul_dense_times_dense_with_faer_f64() {
        test_block_matmul_dense_times_dense_with_faer_generic::<f64>();
    }

    #[test]
    fn test_block_matmul_dense_times_dense_with_faer_c64() {
        test_block_matmul_dense_times_dense_with_faer_generic::<Complex64>();
    }

    #[test]
    fn test_complex_matmul_with_imaginary() {
        // Test complex multiplication with non-zero imaginary parts
        // [1+i] @ [2-i] = (1+i)(2-i) = 2 - i + 2i + 1 = 3 + i
        let a_parts = vec![
            BlockPartition::trivial(1),
            BlockPartition::trivial(1),
        ];
        let mut a = BlockedArray::<Complex64>::new(a_parts);
        a.set_block(
            vec![0, 0],
            BlockData::scalar(Complex64::new(1.0, 1.0)),
        );

        let b_parts = vec![
            BlockPartition::trivial(1),
            BlockPartition::trivial(1),
        ];
        let mut b = BlockedArray::<Complex64>::new(b_parts);
        b.set_block(
            vec![0, 0],
            BlockData::scalar(Complex64::new(2.0, -1.0)),
        );

        let c = blocked_matmul(&a, &b).unwrap();
        let c_block = c.get_block(&vec![0, 0]).unwrap();
        let result = c_block.get([0, 0]);
        assert!((result.re - 3.0).abs() < 1e-10);
        assert!((result.im - 1.0).abs() < 1e-10);
    }
}
