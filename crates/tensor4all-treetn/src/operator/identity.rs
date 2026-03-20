//! Identity tensor construction for operator composition.
//!
//! When composing exclusive operators, gap positions (nodes not covered by any operator)
//! need identity tensors that pass information through unchanged.
//!
//! This module provides convenience wrappers around `TensorLike::delta()`.

use anyhow::Result;
use num_complex::Complex64;

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
        return Ok(TensorDynLen::scalar(Complex64::new(1.0, 0.0)).unwrap());
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

    let n_pairs = site_indices.len();
    let input_dims: Vec<usize> = site_indices.iter().map(|i| i.dim()).collect();
    let input_total: usize = input_dims.iter().product();

    let mut data = vec![Complex64::new(0.0, 0.0); total_size];

    for input_linear in 0..input_total {
        let mut input_multi = vec![0usize; n_pairs];
        let mut remaining = input_linear;
        for i in 0..n_pairs {
            input_multi[i] = remaining % input_dims[i];
            remaining /= input_dims[i];
        }

        let mut full_multi = vec![0usize; n_pairs * 2];
        for i in 0..n_pairs {
            full_multi[2 * i] = input_multi[i];
            full_multi[2 * i + 1] = input_multi[i];
        }

        let mut linear_idx = 0usize;
        for i in (0..full_multi.len()).rev() {
            linear_idx = linear_idx * dims[i] + full_multi[i];
        }
        data[linear_idx] = Complex64::new(1.0, 0.0);
    }

    Ok(TensorDynLen::from_dense(all_indices, data).unwrap())
}

#[cfg(test)]
mod tests;
