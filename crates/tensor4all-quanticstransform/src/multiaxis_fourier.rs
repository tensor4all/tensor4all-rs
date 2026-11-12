//! Fourier products with an explicit logical axis/bit layout.

use std::collections::{BTreeMap, HashMap};

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, Tensor3, Tensor3Ops};
use tensor4all_treetn::{compose_exclusive_linear_operators, RestructureOptions, SiteIndexNetwork};

use crate::common::{
    checked_allocation_len, tensortrain_to_linear_operator, try_vec_with_capacity,
};
use crate::fourier::{quantics_fourier_mpo, zero_tensor3};
use crate::{quantics_fourier_operator, FourierOptions, QuanticsOperator, QuanticsTransformError};

/// One logical axis bit at an abstract operator node.
///
/// Unlike a state index, this describes significance independently of physical
/// placement: `bit = 0` denotes the most significant input bit of `axis`.
///
/// # Examples
/// ```
/// use tensor4all_quanticstransform::{AxisBit, FourierOptions, quantics_fourier_operator_multiaxis};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let layout = [AxisBit { axis: 0, bit: 0 }, AxisBit { axis: 0, bit: 1 }];
/// let operator = quantics_fourier_operator_multiaxis(&layout, FourierOptions::forward())?;
/// assert!((operator.mpo().clone().norm()? - 2.0).abs() < 1e-10);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AxisBit {
    /// Zero-based axis number; axis numbers must be contiguous from zero.
    pub axis: usize,
    /// Zero-based bit significance within the axis, most significant first.
    pub bit: usize,
}

pub(crate) fn invalid(message: impl Into<String>) -> QuanticsTransformError {
    QuanticsTransformError::InvalidConfiguration {
        message: message.into(),
    }
}

/// Validate before constructing tensors; return layout positions in bit order.
pub(crate) fn axis_positions(
    layout: &[AxisBit],
) -> Result<Vec<Vec<usize>>, QuanticsTransformError> {
    if layout.is_empty() {
        return Err(invalid(
            "Fourier layout must not be empty; provide at least one axis with two bits",
        ));
    }
    let mut axes: BTreeMap<usize, BTreeMap<usize, usize>> = BTreeMap::new();
    for (position, entry) in layout.iter().enumerate() {
        if axes
            .entry(entry.axis)
            .or_default()
            .insert(entry.bit, position)
            .is_some()
        {
            return Err(invalid(
                "Fourier layout repeats an (axis, bit) pair; list each bit exactly once",
            ));
        }
    }
    let mut result = Vec::with_capacity(axes.len());
    for (expected_axis, (axis, bits)) in axes.into_iter().enumerate() {
        if axis != expected_axis {
            return Err(invalid(
                "Fourier axis numbers must be contiguous from zero; renumber the axes",
            ));
        }
        if bits.len() < 2 {
            return Err(invalid(
                "each Fourier axis needs at least two bits; provide a longer axis",
            ));
        }
        let mut positions = Vec::with_capacity(bits.len());
        for (expected_bit, (bit, position)) in bits.into_iter().enumerate() {
            if bit != expected_bit {
                return Err(invalid(
                    "Fourier bit numbers must be contiguous from zero; include every bit",
                ));
            }
            positions.push(position);
        }
        result.push(positions);
    }
    Ok(result)
}

/// Construct a product of Fourier operators on abstract nodes in `layout` order.
///
/// Node `j` carries `layout[j]`, with one input/output mapping per node. Each
/// axis must contain every bit exactly once, numbered from zero, and have at
/// least two bits. Axis numbers are also contiguous from zero. Bits can be
/// interleaved, grouped, or permuted. Output significance is reversed **within
/// each axis**, as in [`quantics_fourier_operator`].
///
/// `options` controls each one-axis construction. When every axis's bits are
/// monotone along the layout (increasing or decreasing, as in grouped and
/// interleaved layouts), the per-axis trains are embedded directly: a site of
/// one axis passes the other axes' bonds through unchanged. This is exact, needs
/// no factorization, and the bond at each cut is the product of the per-axis
/// bonds there (for example `12 * 12 = 144` for two interleaved axes at the
/// default options), so it can exceed `options.max_bond_dim`. Other layouts are
/// composed in separated axis order and restructured exactly by site swaps,
/// whose intermediate bonds can grow much further. Normalization is
/// `1/sqrt(N)` per axis when enabled. Reuse the returned operator for repeated
/// applications.
///
/// # Errors
/// Returns [`QuanticsTransformError::InvalidConfiguration`] for an empty,
/// duplicate, missing-bit, missing-axis, or short-axis layout, invalid Fourier
/// options, or a product bond that overflows `usize`. Underlying construction,
/// composition and restructure errors retain their source chains in
/// [`QuanticsTransformError::Operation`].
///
/// # Examples
/// ```
/// use tensor4all_quanticstransform::{quantics_fourier_operator_multiaxis, AxisBit, FourierOptions};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let layout = [AxisBit { axis: 0, bit: 0 }, AxisBit { axis: 1, bit: 0 },
///               AxisBit { axis: 0, bit: 1 }, AxisBit { axis: 1, bit: 1 }];
/// let operator = quantics_fourier_operator_multiaxis(&layout, FourierOptions::forward())?;
/// // A normalized 16-by-16 Fourier product has Frobenius norm sqrt(16).
/// assert!((operator.mpo().clone().norm()? - 4.0).abs() < 1e-10);
/// // Interleaved bonds are products of the per-axis bonds: 2, 2*2, 2.
/// assert_eq!(operator.mpo().link_dims(), vec![2, 4, 2]);
/// # Ok(())
/// # }
/// ```
pub fn quantics_fourier_operator_multiaxis(
    layout: &[AxisBit],
    options: FourierOptions,
) -> Result<QuanticsOperator, QuanticsTransformError> {
    options.validate()?;
    let axes = axis_positions(layout)?;
    if axes.iter().all(|positions| is_monotone(positions)) {
        return interleaved_fourier_operator(layout, &axes, &options);
    }
    let mut product: Option<QuanticsOperator> = None;
    let mut separated = SiteIndexNetwork::<usize, DynIndex>::new();
    let mut requested = SiteIndexNetwork::<usize, DynIndex>::new();
    let mut offset = 0;
    for positions in axes {
        let operator = quantics_fourier_operator(positions.len(), options.clone())?;
        let renames: Vec<_> = (0..positions.len()).map(|j| (j, offset + j)).collect();
        let operator = operator.rename_nodes(&renames)?;
        for (bit, position) in positions.into_iter().enumerate() {
            let node = offset + bit;
            let space = operator
                .mpo()
                .site_index_network()
                .site_space(&node)
                .ok_or_else(|| invalid("Fourier operator is missing its internal site space"))?;
            separated.add_node(node, space.clone())?;
            requested.add_node(position, space.clone())?;
        }
        let next_offset = offset + operator.mpo().node_count();
        for j in offset.max(1)..next_offset {
            separated.add_edge(&(j - 1), &j)?;
        }
        // Compose adjacent subtrees pairwise: composing three separated axes at
        // once would fail path exclusivity because the middle axis lies between
        // the outer two. The accumulated prefix is one connected subtree.
        product = Some(match product {
            None => operator,
            Some(prefix) => compose_exclusive_linear_operators::<IdxTensor, usize>(
                &separated,
                &[&prefix, &operator],
                &HashMap::new(),
            )?,
        });
        offset = next_offset;
    }
    for j in 1..layout.len() {
        requested.add_edge(&(j - 1), &j)?;
    }
    let product = product.ok_or_else(|| invalid("Fourier layout must contain an axis"))?;
    Ok(product.restructure_to(&requested, &RestructureOptions::default())?)
}

/// Whether an axis's bits appear in strictly increasing or decreasing order.
fn is_monotone(positions: &[usize]) -> bool {
    positions.windows(2).all(|w| w[0] < w[1]) || positions.windows(2).all(|w| w[0] > w[1])
}

/// Reverse a train's site order; core `(l, s, r)` becomes `(r, s, l)`.
fn reversed_train(
    train: &SimpleTensorTrain<Complex64>,
) -> Result<Vec<Tensor3<Complex64>>, QuanticsTransformError> {
    let mut cores = try_vec_with_capacity("reversed Fourier train", train.len())?;
    for j in (0..train.len()).rev() {
        let core = train.site_tensor(j);
        let (left, site, right) = (core.left_dim(), core.site_dim(), core.right_dim());
        let mut reversed = zero_tensor3(right, site, left, "reversed Fourier core")?;
        for l in 0..left {
            for s in 0..site {
                for r in 0..right {
                    reversed.set3(r, s, l, *core.get3(l, s, r));
                }
            }
        }
        cores.push(reversed);
    }
    Ok(cores)
}

/// Build a product of per-axis Fourier trains directly in `layout` order.
///
/// Each axis's bits must be monotone along the chain. A site of axis `a`
/// applies that axis's core and passes every other axis's current bond through
/// unchanged, so the result is exact and the bond at each cut is the product of
/// the per-axis bonds there; no factorization or site swap is needed.
fn interleaved_fourier_operator(
    layout: &[AxisBit],
    axes: &[Vec<usize>],
    options: &FourierOptions,
) -> Result<QuanticsOperator, QuanticsTransformError> {
    let mut trains =
        try_vec_with_capacity::<Vec<Tensor3<Complex64>>>("Fourier axis trains", axes.len())?;
    for positions in axes {
        let train = quantics_fourier_mpo(positions.len(), options)?;
        // A decreasing axis meets its least significant bit first.
        let cores = if positions[0] < positions[positions.len() - 1] {
            train.site_tensors().to_vec()
        } else {
            reversed_train(&train)?
        };
        trains.push(cores);
    }
    let mut consumed = vec![0usize; axes.len()];
    let mut bonds = vec![1usize; axes.len()];
    let mut cores =
        try_vec_with_capacity::<Tensor3<Complex64>>("Fourier product cores", layout.len())?;
    for entry in layout {
        let axis = entry.axis;
        let core = &trains[axis][consumed[axis]];
        let mut right_bonds = bonds.clone();
        right_bonds[axis] = core.right_dim();
        let left_dim = checked_product(&bonds)?;
        let right_dim = checked_product(&right_bonds)?;
        checked_allocation_len::<Complex64>(
            &[left_dim, core.site_dim(), right_dim],
            "Fourier product core",
        )?;
        // Mixed radix with axis 0 least significant on both sides.
        let left_stride: usize = bonds[..axis].iter().product();
        let right_stride: usize = right_bonds[..axis].iter().product();
        let mut product =
            zero_tensor3(left_dim, core.site_dim(), right_dim, "Fourier product core")?;
        for left in 0..left_dim {
            let low = left % left_stride;
            let digit = (left / left_stride) % bonds[axis];
            let high = left / (left_stride * bonds[axis]);
            let base = low + high * right_stride * right_bonds[axis];
            for s in 0..core.site_dim() {
                for r in 0..right_bonds[axis] {
                    let value = *core.get3(digit, s, r);
                    product.set3(left, s, base + r * right_stride, value);
                }
            }
        }
        cores.push(product);
        bonds = right_bonds;
        consumed[axis] += 1;
    }
    let train = SimpleTensorTrain::new(cores).map_err(|e| {
        QuanticsTransformError::from(
            anyhow::Error::new(e).context("Failed to assemble the multi-axis Fourier train"),
        )
    })?;
    tensortrain_to_linear_operator(&train, &vec![2; layout.len()])
}

fn checked_product(dims: &[usize]) -> Result<usize, QuanticsTransformError> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| invalid("multi-axis Fourier bond dimension overflows usize; use fewer axes"))
}
