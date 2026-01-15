//! Identity tensor construction for operator composition.
//!
//! When composing exclusive operators, gap positions (nodes not covered by any operator)
//! need identity tensors that pass information through unchanged.
//!
//! This module provides convenience wrappers around `TensorLike::delta()`.

use std::sync::Arc;

use anyhow::Result;
use num_complex::Complex64;

use tensor4all_core::storage::{DenseStorageC64, Storage};
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorLike};

/// Build an identity operator tensor for a gap node.
///
/// For a node with site indices `{s1, s2, ...}` and bond indices `{l1, l2, ...}`,
/// this creates an identity tensor where:
/// - Each site index `s` gets a primed version `s'` (output index)
/// - The tensor is diagonal: `T[s1, s1', s2, s2', ...] = δ_{s1,s1'} × δ_{s2,s2'} × ...`
///
/// This is a convenience wrapper around `TensorDynLen::delta()`.
///
/// # Arguments
///
/// * `site_indices` - The site indices at this node
/// * `output_site_indices` - The output (primed) site indices, must have same dimensions
///
/// # Returns
///
/// A tensor representing the identity operator on the given site space.
///
/// # Example
///
/// For a single site index of dimension 2:
/// ```text
/// T[s, s'] = δ_{s,s'} = [[1, 0], [0, 1]]
/// ```
pub fn build_identity_operator_tensor(
    site_indices: &[DynIndex],
    output_site_indices: &[DynIndex],
) -> Result<TensorDynLen> {
    TensorDynLen::delta(site_indices, output_site_indices)
}

/// Build an identity operator tensor with complex data type.
///
/// Same as [`build_identity_operator_tensor`] but returns a complex tensor.
pub fn build_identity_operator_tensor_c64(
    site_indices: &[DynIndex],
    output_site_indices: &[DynIndex],
) -> Result<TensorDynLen> {
    // Validate same number of input and output indices
    if site_indices.len() != output_site_indices.len() {
        return Err(anyhow::anyhow!(
            "Number of input indices ({}) must match output indices ({})",
            site_indices.len(),
            output_site_indices.len()
        ));
    }

    // Validate dimensions match
    for (inp, out) in site_indices.iter().zip(output_site_indices.iter()) {
        if inp.dim() != out.dim() {
            return Err(anyhow::anyhow!(
                "Dimension mismatch: input index has dim {}, output has dim {}",
                inp.dim(),
                out.dim(),
            ));
        }
    }

    if site_indices.is_empty() {
        let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(vec![
            Complex64::new(1.0, 0.0),
        ])));
        return Ok(TensorDynLen::new(vec![], vec![], storage));
    }

    // Build combined index list
    let mut all_indices: Vec<DynIndex> = Vec::with_capacity(site_indices.len() * 2);
    let mut dims: Vec<usize> = Vec::with_capacity(site_indices.len() * 2);

    for (inp, out) in site_indices.iter().zip(output_site_indices.iter()) {
        all_indices.push(inp.clone());
        all_indices.push(out.clone());
        let dim = inp.dim();
        dims.push(dim);
        dims.push(dim);
    }

    let total_size: usize = dims.iter().product();

    fn compute_strides(dims: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }

    let strides = compute_strides(&dims);
    let n_pairs = site_indices.len();
    let input_dims: Vec<usize> = site_indices.iter().map(|i| i.dim()).collect();
    let input_total: usize = input_dims.iter().product();

    let mut data = vec![Complex64::new(0.0, 0.0); total_size];

    for input_linear in 0..input_total {
        let mut input_multi = vec![0usize; n_pairs];
        let mut remaining = input_linear;
        for i in (0..n_pairs).rev() {
            input_multi[i] = remaining % input_dims[i];
            remaining /= input_dims[i];
        }

        let mut full_multi = vec![0usize; n_pairs * 2];
        for i in 0..n_pairs {
            full_multi[2 * i] = input_multi[i];
            full_multi[2 * i + 1] = input_multi[i];
        }

        let linear_idx: usize = full_multi.iter().zip(&strides).map(|(&m, &s)| m * s).sum();
        data[linear_idx] = Complex64::new(1.0, 0.0);
    }

    let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(data)));
    Ok(TensorDynLen::new(all_indices, dims, storage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::index::Index;

    fn make_index(dim: usize) -> DynIndex {
        Index::new_dyn(dim)
    }

    fn get_f64_data(tensor: &TensorDynLen) -> &[f64] {
        match tensor.storage().as_ref() {
            Storage::DenseF64(d) => d.as_slice(),
            _ => panic!("Expected DenseF64 storage"),
        }
    }

    fn get_c64_data(tensor: &TensorDynLen) -> &[Complex64] {
        match tensor.storage().as_ref() {
            Storage::DenseC64(d) => d.as_slice(),
            _ => panic!("Expected DenseC64 storage"),
        }
    }

    #[test]
    fn test_identity_single_site() {
        let s_in = make_index(2);
        let s_out = make_index(2);

        let tensor = build_identity_operator_tensor(
            std::slice::from_ref(&s_in),
            std::slice::from_ref(&s_out),
        )
        .unwrap();

        assert_eq!(tensor.indices.len(), 2);
        assert_eq!(tensor.dims, vec![2, 2]);

        // Check diagonal elements are 1
        let data = get_f64_data(&tensor);
        // Layout: [s_in, s_out] in row-major
        // (0,0) -> idx 0, (0,1) -> idx 1, (1,0) -> idx 2, (1,1) -> idx 3
        assert_eq!(data[0], 1.0); // (0,0)
        assert_eq!(data[1], 0.0); // (0,1)
        assert_eq!(data[2], 0.0); // (1,0)
        assert_eq!(data[3], 1.0); // (1,1)
    }

    #[test]
    fn test_identity_two_sites() {
        let s1_in = make_index(2);
        let s1_out = make_index(2);
        let s2_in = make_index(3);
        let s2_out = make_index(3);

        let tensor = build_identity_operator_tensor(
            &[s1_in.clone(), s2_in.clone()],
            &[s1_out.clone(), s2_out.clone()],
        )
        .unwrap();

        assert_eq!(tensor.indices.len(), 4);
        assert_eq!(tensor.dims, vec![2, 2, 3, 3]);

        let data = get_f64_data(&tensor);
        let total_size: usize = tensor.dims.iter().product();
        assert_eq!(data.len(), total_size);

        // Count non-zero elements: should be 2*3 = 6 (diagonal)
        let nonzero_count = data.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(nonzero_count, 6);

        // All non-zero should be 1.0
        for &val in data.iter() {
            assert!(val == 0.0 || val == 1.0);
        }
    }

    #[test]
    fn test_identity_dimension_mismatch() {
        let s_in = make_index(2);
        let s_out = make_index(3); // Different dimension

        let result = build_identity_operator_tensor(&[s_in], &[s_out]);
        assert!(result.is_err());
    }

    #[test]
    fn test_identity_empty() {
        let tensor = build_identity_operator_tensor(&[], &[]).unwrap();

        assert_eq!(tensor.indices.len(), 0);
        assert_eq!(tensor.dims.len(), 0);

        let data = get_f64_data(&tensor);
        assert_eq!(data.len(), 1);
        assert_eq!(data[0], 1.0);
    }

    #[test]
    fn test_identity_c64() {
        let s_in = make_index(2);
        let s_out = make_index(2);

        let tensor = build_identity_operator_tensor_c64(
            std::slice::from_ref(&s_in),
            std::slice::from_ref(&s_out),
        )
        .unwrap();

        let data = get_c64_data(&tensor);
        assert_eq!(data[0], Complex64::new(1.0, 0.0));
        assert_eq!(data[1], Complex64::new(0.0, 0.0));
        assert_eq!(data[2], Complex64::new(0.0, 0.0));
        assert_eq!(data[3], Complex64::new(1.0, 0.0));
    }
}
