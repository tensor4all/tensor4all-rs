//! Common types and helper functions for quantics transformations.

use crate::error::QuanticsTransformError;
use std::collections::HashMap;
use std::mem::size_of;

use anyhow::Result;
use num_complex::Complex64;
#[cfg(feature = "simplett-transforms")]
use num_traits::One;
use tensor4all_core::index::{DynId, Index, TagSet};
#[cfg(feature = "simplett-transforms")]
use tensor4all_core::DenseElementBudget;
use tensor4all_core::{ColMajorArray, IdxTensor, IndexLike};
#[cfg(feature = "simplett-transforms")]
use tensor4all_simplett::{
    tensor3_from_data, types::tensor3_zeros, AbstractTensorTrain, SimpleTensorTrain, Tensor3Ops,
};
use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};

/// Type alias for the default index type.
pub type DynIndex = Index<DynId, TagSet>;

/// Boundary condition for quantics transformations.
/// Controls how operators handle values that exceed the representable range
/// `[0, 2^R)`.
/// # Variants
/// - **`Periodic`** (default): Results wrap around modulo 2^R.
///
///   Use when functions are periodic or when wraparound is acceptable.
/// - **`AntiPeriodic`**: Results wrap around modulo 2^R and receive a sign
///
///   `(-1)^q`, where `q` is the integer wrap quotient.
/// - **`Open`**: Out-of-range results produce zeros.
///
///   Use when the function has compact support or when boundary effects matter.
/// # Examples
/// ```
/// use tensor4all_quanticstransform::BoundaryCondition;
/// // Default is Periodic
/// let bc = BoundaryCondition::default();
/// assert_eq!(bc, BoundaryCondition::Periodic);
/// // Periodic: shift(7, 2) in 3-bit (mod 8) wraps to 1
/// // AntiPeriodic: the same wrap receives a -1 sign
/// // Open: shift(7, 2) in 3-bit goes to 9 >= 8, produces zero
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg(feature = "simplett-transforms")]
pub enum BoundaryCondition {
    /// Periodic boundary: operations wrap around mod 2^R.
    ///
    /// Use for periodic functions or when wraparound is desired.
    #[default]
    Periodic,
    /// Anti-periodic boundary: operations wrap around mod 2^R and receive a
    /// sign for each wrap.
    ///
    /// Use for anti-periodic functions where `f(x + q * 2^R) = (-1)^q f(x)`.
    AntiPeriodic,
    /// Open boundary: operations beyond `[0, 2^R)` return zero.
    ///
    /// Use when the function has compact support or boundary effects matter.
    Open,
}

/// Direction for carry propagation in binary arithmetic operations.
/// This is an internal detail of how binary arithmetic (addition, subtraction)
/// is implemented in the MPO construction. Most users do not need to set this
/// directly.
/// # Variants
/// - **`LeftToRight`** (default): Carry propagates from MSB to LSB.
/// - **`RightToLeft`**: Carry propagates from LSB to MSB.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg(feature = "simplett-transforms")]
pub enum CarryDirection {
    /// Carry propagates from left (MSB) to right (LSB).
    #[default]
    LeftToRight,
    /// Carry propagates from right (LSB) to left (MSB).
    RightToLeft,
}

/// Type alias for the standard LinearOperator used in this crate.
/// Uses IdxTensor as the tensor type and usize as the node name type.
/// # Examples
/// ```
/// use tensor4all_quanticstransform::{
///     quantics_fourier_operator, FourierOptions, QuanticsOperator,
/// };
/// let operator: QuanticsOperator = quantics_fourier_operator(3, FourierOptions::forward())?;
/// assert_eq!(operator.mpo().link_dims(), vec![2, 2]);
/// # Ok::<(), tensor4all_quanticstransform::QuanticsTransformError>(())
/// ```
pub type QuanticsOperator = LinearOperator<IdxTensor, usize>;

/// Convert a SimpleTensorTrain (MPO form) to a LinearOperator.
/// The SimpleTensorTrain is assumed to be an MPO with site dimension 4 (2x2 for input/output).
/// Each site tensor has shape (left_bond, site_dim=4, right_bond) where site_dim
/// encodes (s_out, s_in) = (2, 2).
/// # Arguments
/// * `tt` - SimpleTensorTrain representing an MPO
/// * `site_dims` - Site dimensions for input/output (typically all 2s)
/// # Returns
/// LinearOperator wrapping the MPO as a TreeTN.
/// # Errors
/// Returns [`QuanticsTransformError::Operation`] when `tt` is empty, when
/// `site_dims` has a different length than `tt`, when a site dimension does
/// not equal `site_dims[i]^2`, or when a site tensor construction or TreeTN
/// assembly fails.
/// # Examples
/// ```
/// use num_complex::Complex64;
/// use tensor4all_quanticstransform::tensortrain_to_linear_operator;
/// use tensor4all_simplett::{tensor3_zeros, AbstractTensorTrain, Tensor3Ops, SimpleTensorTrain};
/// let mut tensor = tensor3_zeros(1, 4, 1).unwrap();
/// tensor.set3(0, 0, 0, Complex64::new(1.0, 0.0));
/// tensor.set3(0, 3, 0, Complex64::new(1.0, 0.0));
/// let mpo = SimpleTensorTrain::new(vec![tensor]).unwrap();
/// let operator = tensortrain_to_linear_operator(&mpo, &[2]).unwrap();
/// assert_eq!(operator.mpo().node_count(), 1);
/// assert!(tensortrain_to_linear_operator(&mpo, &[2, 2]).is_err());
/// ```
#[cfg(feature = "simplett-transforms")]
pub fn tensortrain_to_linear_operator(
    tt: &SimpleTensorTrain<Complex64>,
    site_dims: &[usize],
) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
    tensortrain_to_linear_operator_asymmetric(tt, site_dims, site_dims)
}

/// Convert a SimpleTensorTrain (MPO form) to a LinearOperator with asymmetric dimensions.
/// This variant supports different input and output dimensions, useful for
/// multi-variable transformations like affine transforms.
/// # Arguments
/// * `tt` - SimpleTensorTrain representing an MPO
/// * `input_dims` - Input dimensions per site
/// * `output_dims` - Output dimensions per site
/// # Returns
/// LinearOperator wrapping the MPO as a TreeTN.
/// # Errors
/// Returns [`QuanticsTransformError::Operation`] when `tt` is empty, when
/// `input_dims`/`output_dims` have a different length than `tt`, when a site
/// dimension does not equal `output_dims[i] * input_dims[i]`, or when a site
/// tensor construction or TreeTN assembly fails.
/// # Examples
/// ```
/// use num_complex::Complex64;
/// use tensor4all_quanticstransform::tensortrain_to_linear_operator_asymmetric;
/// use tensor4all_simplett::{tensor3_zeros, AbstractTensorTrain, Tensor3Ops, SimpleTensorTrain};
/// let mut tensor = tensor3_zeros(1, 6, 1).unwrap();
/// tensor.set3(0, 0, 0, Complex64::new(1.0, 0.0));
/// tensor.set3(0, 5, 0, Complex64::new(1.0, 0.0));
/// let mpo = SimpleTensorTrain::new(vec![tensor]).unwrap();
/// let operator = tensortrain_to_linear_operator_asymmetric(&mpo, &[2], &[3]).unwrap();
/// assert_eq!(operator.mpo().node_count(), 1);
/// assert!(tensortrain_to_linear_operator_asymmetric(&mpo, &[2], &[2]).is_err());
/// ```
#[cfg(feature = "simplett-transforms")]
pub fn tensortrain_to_linear_operator_asymmetric(
    tt: &SimpleTensorTrain<Complex64>,
    input_dims: &[usize],
    output_dims: &[usize],
) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
    let n = tt.len();
    if input_dims.len() != n || output_dims.len() != n {
        return Err(anyhow::anyhow!("Dimension arrays must have length {n}").into());
    }
    chain_operator(
        tt.site_tensors(),
        input_dims.iter().copied().zip(output_dims.iter().copied()),
        |core| Ok(*core.dims()),
        |core, l, s, r| *core.get3(l, s, r),
    )
}

pub(crate) fn binary_cores_to_operator(
    cores: &[ColMajorArray<Complex64>],
) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
    chain_operator(
        cores,
        std::iter::repeat_n((2, 2), cores.len()),
        |core| {
            core.shape()
                .try_into()
                .map_err(|_| anyhow::anyhow!("Fourier core must have rank three").into())
        },
        |core, l, s, r| core.data()[l + core.shape()[0] * (s + core.shape()[1] * r)],
    )
}

fn chain_operator<C>(
    cores: &[C],
    site_dims: impl Iterator<Item = (usize, usize)>,
    shape: impl Fn(&C) -> std::result::Result<[usize; 3], QuanticsTransformError>,
    value: impl Fn(&C, usize, usize, usize) -> Complex64,
) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
    let n = cores.len();
    if n == 0 {
        return Err(anyhow::anyhow!("Empty tensor train").into());
    }
    let mut tensors = try_vec_with_capacity::<IdxTensor>("operator tensor list", n)?;
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    let mut left_bond: Option<DynIndex> = None;
    for (i, (core, (input, output))) in cores.iter().zip(site_dims).enumerate() {
        let [left, site, right] = shape(core)?;
        if left_bond.as_ref().map_or(1, IndexLike::dim) != left || (i + 1 == n && right != 1) {
            return Err(anyhow::anyhow!("Site {i} has incompatible bond dimensions").into());
        }
        let expected =
            checked_allocation_len::<Complex64>(&[output, input], "operator physical dimensions")?;
        if site != expected {
            return Err(anyhow::anyhow!(
                "Site {i} has dimension {site} but expected {expected} ({output}x{input})"
            )
            .into());
        }
        let total = checked_allocation_len::<Complex64>(
            &[left, output, input, right],
            "operator site data",
        )?;
        let input_index = Index::new_dyn(input);
        let output_index = Index::new_dyn(output);
        let right_bond = (i + 1 < n).then(|| Index::new_dyn(right));
        let mut indices = Vec::with_capacity(4);
        indices.extend(left_bond);
        indices.extend([output_index.clone(), input_index.clone()]);
        indices.extend(right_bond.iter().cloned());

        // The source fuses input fastest. Emit output fastest for named
        // column-major axes [left, output, input, right].
        let mut data = try_vec_with_capacity::<Complex64>("operator site data", total)?;
        if total != 0 {
            for r in 0..right {
                for s_in in 0..input {
                    for s_out in 0..output {
                        let s = s_out * input + s_in;
                        data.extend((0..left).map(|l| value(core, l, s, r)));
                    }
                }
            }
        }
        tensors.push(IdxTensor::from_dense(indices, data)?);
        input_mapping.insert(
            i,
            IndexMapping {
                true_index: Index::new_dyn(input),
                internal_index: input_index,
            },
        );
        output_mapping.insert(
            i,
            IndexMapping {
                true_index: Index::new_dyn(output),
                internal_index: output_index,
            },
        );
        left_bond = right_bond;
    }

    let mut node_names = try_vec_with_capacity::<usize>("operator node-name list", n)?;
    node_names.extend(0..n);
    let treetn = TreeTN::from_tensors(tensors, node_names)?;
    Ok(LinearOperator::new(treetn, input_mapping, output_mapping))
}

#[cfg(feature = "simplett-transforms")]
pub(crate) fn checked_pow2(width: usize, name: &str) -> Result<usize> {
    let shift = u32::try_from(width)
        .map_err(|_| anyhow::anyhow!("{name} {width} exceeds usize shift width"))?;
    1usize
        .checked_shl(shift)
        .ok_or_else(|| anyhow::anyhow!("{name} {width} exceeds usize shift width"))
}

pub(crate) fn checked_allocation_len<T>(dims: &[usize], name: &str) -> Result<usize> {
    let elements = dims.iter().try_fold(1usize, |product, &dim| {
        product.checked_mul(dim).ok_or_else(|| {
            anyhow::anyhow!("{name} element count overflows usize for dimensions {dims:?}")
        })
    })?;
    let element_size = size_of::<T>();
    if element_size != 0 {
        let bytes = elements.checked_mul(element_size).ok_or_else(|| {
            anyhow::anyhow!("{name} byte length overflows usize for dimensions {dims:?}")
        })?;
        if bytes > isize::MAX as usize {
            return Err(anyhow::anyhow!(
                "{name} byte length exceeds isize::MAX for dimensions {dims:?}"
            ));
        }
    }
    Ok(elements)
}

pub(crate) fn try_vec_with_capacity<T>(name: &str, capacity: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|err| {
        anyhow::anyhow!("{name} allocation failed for capacity {capacity}: {err}")
    })?;
    Ok(values)
}

#[cfg(feature = "simplett-transforms")]
pub(crate) fn checked_multivar_dims(nvariables: usize) -> Result<(usize, usize)> {
    if nvariables < 2 {
        anyhow::bail!("nvariables must be at least 2, got {nvariables}");
    }

    let local_dim = checked_pow2(nvariables, "nvariables")?;
    let site_dim = local_dim.checked_mul(local_dim).ok_or_else(|| {
        anyhow::anyhow!("multi-variable site dimension overflows usize for nvariables {nvariables}")
    })?;
    checked_allocation_len::<Complex64>(&[site_dim], "multi-variable site tensor")?;
    Ok((local_dim, site_dim))
}

/// Embed a single-variable MPO into a multi-variable context.
/// The original MPO acts on one variable (site_dim = d*d for d=2, i.e., in/out dim 2).
/// The embedded MPO acts on `nvariables` variables, applying the original
/// operator to `target_var` and identity on all others.
/// Site index encoding in the embedded MPO:
/// `s = s_out * (2^nvariables) + s_in` where
/// `s_out = var0_out + 2*var1_out + ...` and similarly for `s_in`.
/// # Arguments
/// * `mpo` - Single-variable MPO (R sites, site_dim = 4)
/// * `nvariables` - Total number of variables (must be >= 2)
/// * `target_var` - Which variable to apply the operator to (0-indexed)
/// * `budget` - Element-count budget for the embedded sites, checked before
///   any of them is allocated
///
/// # Errors
///
/// Returns [`QuanticsTransformError::Operation`] when `nvariables`,
/// `target_var` or a site dimension is invalid or a tensor allocation fails,
/// and [`QuanticsTransformError::DenseBudget`] when the embedded sites exceed
/// `budget`.
#[cfg(feature = "simplett-transforms")]
pub(crate) fn embed_single_var_mpo(
    mpo: &SimpleTensorTrain<Complex64>,
    nvariables: usize,
    target_var: usize,
    budget: DenseElementBudget,
) -> std::result::Result<SimpleTensorTrain<Complex64>, QuanticsTransformError> {
    if target_var >= nvariables {
        return Err(anyhow::anyhow!(
            "target_var {} must be less than nvariables {}",
            target_var,
            nvariables
        )
        .into());
    }
    let (dim_multi, site_dim_new) = checked_multivar_dims(nvariables)?;
    let r = mpo.len();
    let mut site_shapes = try_vec_with_capacity::<[usize; 3]>("embedded MPO site shapes", r)?;
    for i in 0..r {
        let tensor = mpo.site_tensor(i);
        let shape = [tensor.left_dim(), site_dim_new, tensor.right_dim()];
        // Keep the structural byte-length error reachable for shapes that cannot be
        // allocated at all; the budget is a policy limit, not an allocation limit.
        checked_allocation_len::<Complex64>(&shape, "embedded MPO tensor")?;
        site_shapes.push(shape);
    }
    budget.check_total(
        "multivariable quantics embedded MPO",
        site_shapes.iter().map(|shape| &shape[..]),
    )?;
    let mut new_tensors = try_vec_with_capacity::<tensor4all_simplett::Tensor3<Complex64>>(
        "embedded MPO tensor list",
        r,
    )?;

    for i in 0..r {
        let tensor = mpo.site_tensor(i);
        let left_dim = tensor.left_dim();
        let right_dim = tensor.right_dim();

        assert_eq!(
            tensor.site_dim(),
            4,
            "Input MPO must have site_dim=4 (single variable)"
        );

        let total_size = checked_allocation_len::<Complex64>(
            &[left_dim, site_dim_new, right_dim],
            "embedded MPO tensor",
        )?;
        let mut data = try_vec_with_capacity::<Complex64>("embedded MPO tensor", total_size)?;
        data.resize(total_size, Complex64::new(0.0, 0.0));
        let mut t = tensor3_from_data(data, left_dim, site_dim_new, right_dim)
            .map_err(|err| anyhow::anyhow!("Failed to allocate embedded MPO tensor: {err}"))?;

        for s_out_multi in 0..dim_multi {
            for s_in_multi in 0..dim_multi {
                // Check identity constraint on non-target variables
                let mut identity_ok = true;
                for v in 0..nvariables {
                    if v != target_var {
                        let out_bit = (s_out_multi >> v) & 1;
                        let in_bit = (s_in_multi >> v) & 1;
                        if out_bit != in_bit {
                            identity_ok = false;
                            break;
                        }
                    }
                }
                if !identity_ok {
                    continue;
                }

                // Extract target variable bits
                let target_out = (s_out_multi >> target_var) & 1;
                let target_in = (s_in_multi >> target_var) & 1;
                let s_orig = target_out * 2 + target_in;

                // New fused site index
                let s_new = s_out_multi * dim_multi + s_in_multi;

                for l in 0..left_dim {
                    for rr in 0..right_dim {
                        let val = *tensor.get3(l, s_orig, rr);
                        if val != Complex64::new(0.0, 0.0) {
                            t.set3(l, s_new, rr, val);
                        }
                    }
                }
            }
        }

        new_tensors.push(t);
    }

    SimpleTensorTrain::new(new_tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create embedded MPO: {}", e).into())
}

/// Create an identity MPO for `r` binary sites.
/// # Errors
///
/// Returns [`QuanticsTransformError::Operation`] when `r` is zero, when the
/// site-list allocation overflows `usize`, or when a site tensor or the MPO
/// assembly fails.
/// # Examples
/// ```
/// use tensor4all_quanticstransform::identity_mpo;
/// use tensor4all_simplett::AbstractTensorTrain;
/// let mpo = identity_mpo(2).unwrap();
/// assert_eq!(mpo.len(), 2);
/// assert_eq!(mpo.site_dims(), vec![4, 4]);
/// assert!(identity_mpo(0).is_err());
/// ```
#[cfg(feature = "simplett-transforms")]
pub fn identity_mpo(
    r: usize,
) -> std::result::Result<SimpleTensorTrain<Complex64>, QuanticsTransformError> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive").into());
    }
    let mut tensors = try_vec_with_capacity::<tensor4all_simplett::Tensor3<Complex64>>(
        "identity MPO site list",
        r,
    )?;

    for _ in 0..r {
        // Identity tensor: delta_{s_out, s_in}
        // Shape: (1, 4, 1) where 4 = 2*2 for (s_out, s_in)
        let mut t = tensor3_zeros(1, 4, 1).map_err(anyhow::Error::from)?;
        // s = s_out * 2 + s_in
        // Identity: s_out == s_in
        t.set3(0, 0, 0, Complex64::one()); // (0, 0)
        t.set3(0, 3, 0, Complex64::one()); // (1, 1)
        tensors.push(t);
    }

    SimpleTensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create identity MPO: {e}"))
        .map_err(QuanticsTransformError::from)
}

#[cfg(test)]
mod tests;
