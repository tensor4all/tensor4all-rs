//! Common types and helper functions for quantics transformations.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use num_complex::Complex64;
use num_traits::One;
use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::storage::{DenseStorageC64, Storage};
use tensor4all_core::TensorDynLen;
use tensor4all_simplett::{
    types::tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain,
};
use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};

/// Type alias for the default index type.
pub type DynIndex = Index<DynId, TagSet>;

/// Boundary condition for quantics transformations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// Periodic boundary: operations wrap around mod 2^R
    #[default]
    Periodic,
    /// Open boundary: operations beyond boundaries return zero
    Open,
}

/// Direction for carry propagation in binary arithmetic operations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CarryDirection {
    /// Carry propagates from left (MSB) to right (LSB)
    #[default]
    LeftToRight,
    /// Carry propagates from right (LSB) to left (MSB)
    RightToLeft,
}

/// Type alias for the standard LinearOperator used in this crate.
/// Uses TensorDynLen as the tensor type and usize as the node name type.
pub type QuanticsOperator = LinearOperator<TensorDynLen, usize>;

/// Convert a TensorTrain (MPO form) to a LinearOperator.
///
/// The TensorTrain is assumed to be an MPO with site dimension 4 (2x2 for input/output).
/// Each site tensor has shape (left_bond, site_dim=4, right_bond) where site_dim
/// encodes (s_out, s_in) = (2, 2).
///
/// # Arguments
/// * `tt` - TensorTrain representing an MPO
/// * `site_dims` - Site dimensions for input/output (typically all 2s)
///
/// # Returns
/// LinearOperator wrapping the MPO as a TreeTN
pub fn tensortrain_to_linear_operator(
    tt: &TensorTrain<Complex64>,
    site_dims: &[usize],
) -> Result<QuanticsOperator> {
    let n = tt.len();
    if n == 0 {
        return Err(anyhow::anyhow!("Empty tensor train"));
    }

    // Create site indices for input and output
    let mut site_in_indices: Vec<DynIndex> = Vec::with_capacity(n);
    let mut site_out_indices: Vec<DynIndex> = Vec::with_capacity(n);
    let mut internal_in_indices: Vec<DynIndex> = Vec::with_capacity(n);
    let mut internal_out_indices: Vec<DynIndex> = Vec::with_capacity(n);

    for &dim in site_dims.iter() {
        // True site indices (for state)
        site_in_indices.push(Index::new_dyn(dim));
        site_out_indices.push(Index::new_dyn(dim));
        // Internal MPO indices
        internal_in_indices.push(Index::new_dyn(dim));
        internal_out_indices.push(Index::new_dyn(dim));
    }

    // Create bond indices
    let mut bond_indices: Vec<DynIndex> = Vec::with_capacity(n + 1);

    for i in 0..=n {
        let dim = if i == 0 {
            1
        } else {
            tt.site_tensor(i - 1).right_dim()
        };
        bond_indices.push(Index::new_dyn(dim));
    }

    // Build tensors for TreeTN
    let mut tensors: Vec<TensorDynLen> = Vec::with_capacity(n);
    let mut node_names: Vec<usize> = Vec::with_capacity(n);

    for i in 0..n {
        let tensor = tt.site_tensor(i);
        let left_dim = tensor.left_dim();
        let site_dim = tensor.site_dim();
        let right_dim = tensor.right_dim();

        // Expected site_dim is product of input and output dimensions
        let expected_site_dim = site_dims[i] * site_dims[i];
        if site_dim != expected_site_dim {
            return Err(anyhow::anyhow!(
                "Site {} has dimension {} but expected {} ({}x{})",
                i,
                site_dim,
                expected_site_dim,
                site_dims[i],
                site_dims[i]
            ));
        }

        // Create indices for this tensor: (left_bond, site_out, site_in, right_bond)
        // For first tensor: (site_out, site_in, right_bond)
        // For last tensor: (left_bond, site_out, site_in)
        // For middle: (left_bond, site_out, site_in, right_bond)
        let mut indices: Vec<DynIndex> = Vec::with_capacity(4);
        let mut dims_vec: Vec<usize> = Vec::with_capacity(4);

        if i > 0 {
            indices.push(bond_indices[i].clone());
            dims_vec.push(left_dim);
        }
        indices.push(internal_out_indices[i].clone());
        dims_vec.push(site_dims[i]);
        indices.push(internal_in_indices[i].clone());
        dims_vec.push(site_dims[i]);
        if i < n - 1 {
            indices.push(bond_indices[i + 1].clone());
            dims_vec.push(right_dim);
        }

        // Reshape tensor data: (left, site_out*site_in, right) -> (left, site_out, site_in, right)
        // or appropriate variant for boundary tensors
        let total_size: usize = dims_vec.iter().product();
        let mut data: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); total_size];

        // Map from TT format to TreeTN format
        if i == 0 && n == 1 {
            // Single tensor: (site_out, site_in)
            for s_out in 0..site_dims[i] {
                for s_in in 0..site_dims[i] {
                    let s = s_out * site_dims[i] + s_in;
                    let idx = s_out * site_dims[i] + s_in;
                    data[idx] = *tensor.get3(0, s, 0);
                }
            }
        } else if i == 0 {
            // First tensor: (site_out, site_in, right_bond)
            for s_out in 0..site_dims[i] {
                for s_in in 0..site_dims[i] {
                    for r in 0..right_dim {
                        let s = s_out * site_dims[i] + s_in;
                        let idx = (s_out * site_dims[i] + s_in) * right_dim + r;
                        data[idx] = *tensor.get3(0, s, r);
                    }
                }
            }
        } else if i == n - 1 {
            // Last tensor: (left_bond, site_out, site_in)
            for l in 0..left_dim {
                for s_out in 0..site_dims[i] {
                    for s_in in 0..site_dims[i] {
                        let s = s_out * site_dims[i] + s_in;
                        let idx = (l * site_dims[i] + s_out) * site_dims[i] + s_in;
                        data[idx] = *tensor.get3(l, s, 0);
                    }
                }
            }
        } else {
            // Middle tensor: (left_bond, site_out, site_in, right_bond)
            for l in 0..left_dim {
                for s_out in 0..site_dims[i] {
                    for s_in in 0..site_dims[i] {
                        for r in 0..right_dim {
                            let s = s_out * site_dims[i] + s_in;
                            let idx =
                                ((l * site_dims[i] + s_out) * site_dims[i] + s_in) * right_dim + r;
                            data[idx] = *tensor.get3(l, s, r);
                        }
                    }
                }
            }
        }

        let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(data)));
        let tensor_dyn = TensorDynLen::new(indices, dims_vec, storage);
        tensors.push(tensor_dyn);
        node_names.push(i);
    }

    // Build TreeTN from tensors
    let treetn = TreeTN::from_tensors(tensors, node_names)?;

    // Build index mappings
    let mut input_mapping: HashMap<usize, IndexMapping<DynIndex>> = HashMap::new();
    let mut output_mapping: HashMap<usize, IndexMapping<DynIndex>> = HashMap::new();

    for i in 0..n {
        input_mapping.insert(
            i,
            IndexMapping {
                true_index: site_in_indices[i].clone(),
                internal_index: internal_in_indices[i].clone(),
            },
        );
        output_mapping.insert(
            i,
            IndexMapping {
                true_index: site_out_indices[i].clone(),
                internal_index: internal_out_indices[i].clone(),
            },
        );
    }

    Ok(LinearOperator::new(treetn, input_mapping, output_mapping))
}

/// Convert a TensorTrain (MPO form) to a LinearOperator with asymmetric dimensions.
///
/// This variant supports different input and output dimensions, useful for
/// multi-variable transformations like affine transforms.
///
/// # Arguments
/// * `tt` - TensorTrain representing an MPO
/// * `input_dims` - Input dimensions per site
/// * `output_dims` - Output dimensions per site
///
/// # Returns
/// LinearOperator wrapping the MPO as a TreeTN
pub fn tensortrain_to_linear_operator_asymmetric(
    tt: &TensorTrain<Complex64>,
    input_dims: &[usize],
    output_dims: &[usize],
) -> Result<QuanticsOperator> {
    let n = tt.len();
    if n == 0 {
        return Err(anyhow::anyhow!("Empty tensor train"));
    }
    if input_dims.len() != n || output_dims.len() != n {
        return Err(anyhow::anyhow!(
            "Dimension arrays must have length {}",
            n
        ));
    }

    // Create site indices for input and output
    let mut site_in_indices: Vec<DynIndex> = Vec::with_capacity(n);
    let mut site_out_indices: Vec<DynIndex> = Vec::with_capacity(n);
    let mut internal_in_indices: Vec<DynIndex> = Vec::with_capacity(n);
    let mut internal_out_indices: Vec<DynIndex> = Vec::with_capacity(n);

    for i in 0..n {
        // True site indices (for state)
        site_in_indices.push(Index::new_dyn(input_dims[i]));
        site_out_indices.push(Index::new_dyn(output_dims[i]));
        // Internal MPO indices
        internal_in_indices.push(Index::new_dyn(input_dims[i]));
        internal_out_indices.push(Index::new_dyn(output_dims[i]));
    }

    // Create bond indices
    let mut bond_indices: Vec<DynIndex> = Vec::with_capacity(n + 1);

    for i in 0..=n {
        let dim = if i == 0 {
            1
        } else {
            tt.site_tensor(i - 1).right_dim()
        };
        bond_indices.push(Index::new_dyn(dim));
    }

    // Build tensors for TreeTN
    let mut tensors: Vec<TensorDynLen> = Vec::with_capacity(n);
    let mut node_names: Vec<usize> = Vec::with_capacity(n);

    for i in 0..n {
        let tensor = tt.site_tensor(i);
        let left_dim = tensor.left_dim();
        let site_dim = tensor.site_dim();
        let right_dim = tensor.right_dim();

        let in_dim = input_dims[i];
        let out_dim = output_dims[i];

        // Expected site_dim is product of input and output dimensions
        let expected_site_dim = in_dim * out_dim;
        if site_dim != expected_site_dim {
            return Err(anyhow::anyhow!(
                "Site {} has dimension {} but expected {} ({}x{})",
                i,
                site_dim,
                expected_site_dim,
                out_dim,
                in_dim
            ));
        }

        // Create indices for this tensor: (left_bond, site_out, site_in, right_bond)
        let mut indices: Vec<DynIndex> = Vec::with_capacity(4);
        let mut dims_vec: Vec<usize> = Vec::with_capacity(4);

        if i > 0 {
            indices.push(bond_indices[i].clone());
            dims_vec.push(left_dim);
        }
        indices.push(internal_out_indices[i].clone());
        dims_vec.push(out_dim);
        indices.push(internal_in_indices[i].clone());
        dims_vec.push(in_dim);
        if i < n - 1 {
            indices.push(bond_indices[i + 1].clone());
            dims_vec.push(right_dim);
        }

        // Reshape tensor data: (left, site_out*site_in, right) -> (left, site_out, site_in, right)
        let total_size: usize = dims_vec.iter().product();
        let mut data: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); total_size];

        // Map from TT format to TreeTN format
        // TT format has site index = s_out * in_dim + s_in (output major, input minor)
        if i == 0 && n == 1 {
            // Single tensor: (site_out, site_in)
            for s_out in 0..out_dim {
                for s_in in 0..in_dim {
                    let s = s_out * in_dim + s_in;
                    let idx = s_out * in_dim + s_in;
                    data[idx] = *tensor.get3(0, s, 0);
                }
            }
        } else if i == 0 {
            // First tensor: (site_out, site_in, right_bond)
            for s_out in 0..out_dim {
                for s_in in 0..in_dim {
                    for r in 0..right_dim {
                        let s = s_out * in_dim + s_in;
                        let idx = (s_out * in_dim + s_in) * right_dim + r;
                        data[idx] = *tensor.get3(0, s, r);
                    }
                }
            }
        } else if i == n - 1 {
            // Last tensor: (left_bond, site_out, site_in)
            for l in 0..left_dim {
                for s_out in 0..out_dim {
                    for s_in in 0..in_dim {
                        let s = s_out * in_dim + s_in;
                        let idx = (l * out_dim + s_out) * in_dim + s_in;
                        data[idx] = *tensor.get3(l, s, 0);
                    }
                }
            }
        } else {
            // Middle tensor: (left_bond, site_out, site_in, right_bond)
            for l in 0..left_dim {
                for s_out in 0..out_dim {
                    for s_in in 0..in_dim {
                        for r in 0..right_dim {
                            let s = s_out * in_dim + s_in;
                            let idx =
                                ((l * out_dim + s_out) * in_dim + s_in) * right_dim + r;
                            data[idx] = *tensor.get3(l, s, r);
                        }
                    }
                }
            }
        }

        let storage = Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(data)));
        let tensor_dyn = TensorDynLen::new(indices, dims_vec, storage);
        tensors.push(tensor_dyn);
        node_names.push(i);
    }

    // Build TreeTN from tensors
    let treetn = TreeTN::from_tensors(tensors, node_names)?;

    // Build index mappings
    let mut input_mapping: HashMap<usize, IndexMapping<DynIndex>> = HashMap::new();
    let mut output_mapping: HashMap<usize, IndexMapping<DynIndex>> = HashMap::new();

    for i in 0..n {
        input_mapping.insert(
            i,
            IndexMapping {
                true_index: site_in_indices[i].clone(),
                internal_index: internal_in_indices[i].clone(),
            },
        );
        output_mapping.insert(
            i,
            IndexMapping {
                true_index: site_out_indices[i].clone(),
                internal_index: internal_out_indices[i].clone(),
            },
        );
    }

    Ok(LinearOperator::new(treetn, input_mapping, output_mapping))
}

/// Create an identity MPO for r sites with dimension 2.
#[allow(dead_code)]
pub fn identity_mpo(r: usize) -> Result<TensorTrain<Complex64>> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }

    let mut tensors = Vec::with_capacity(r);

    for _ in 0..r {
        // Identity tensor: delta_{s_out, s_in}
        // Shape: (1, 4, 1) where 4 = 2*2 for (s_out, s_in)
        let mut t = tensor3_zeros(1, 4, 1);
        // s = s_out * 2 + s_in
        // Identity: s_out == s_in
        t.set3(0, 0, 0, Complex64::one()); // (0, 0)
        t.set3(0, 3, 0, Complex64::one()); // (1, 1)
        tensors.push(t);
    }

    TensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("Failed to create identity MPO: {}", e))
}

/// Create a scalar MPO (constant times identity).
#[allow(dead_code)]
pub fn scalar_mpo(r: usize, value: Complex64) -> Result<TensorTrain<Complex64>> {
    let mut mpo = identity_mpo(r)?;
    mpo.scale(value);
    Ok(mpo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_condition_default() {
        assert_eq!(BoundaryCondition::default(), BoundaryCondition::Periodic);
    }

    #[test]
    fn test_carry_direction_default() {
        assert_eq!(CarryDirection::default(), CarryDirection::LeftToRight);
    }

    #[test]
    fn test_identity_mpo() {
        let mpo = identity_mpo(4).unwrap();
        assert_eq!(mpo.len(), 4);

        // Check that it's an identity operator
        for i in 0..4 {
            let t = mpo.site_tensor(i);
            assert_eq!(t.left_dim(), 1);
            assert_eq!(t.site_dim(), 4);
            assert_eq!(t.right_dim(), 1);
        }
    }
}
