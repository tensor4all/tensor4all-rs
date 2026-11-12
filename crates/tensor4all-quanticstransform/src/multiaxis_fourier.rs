//! Fourier products with an explicit logical axis/bit layout.

use std::collections::{BTreeMap, HashMap};

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::{compose_exclusive_linear_operators, RestructureOptions, SiteIndexNetwork};

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
/// `options` controls each one-axis construction. The product is composed in
/// separated axis order and restructured exactly to the requested layout.
/// Restructuring adds no intentional truncation, but its final and intermediate
/// bonds can exceed `options.max_bond_dim`. Normalization is `1/sqrt(N)` per
/// axis when enabled. Reuse the returned operator for repeated applications.
///
/// # Errors
/// Returns [`QuanticsTransformError::InvalidConfiguration`] for an empty,
/// duplicate, missing-bit, missing-axis, or short-axis layout, or invalid Fourier
/// options. Underlying construction, composition and restructure errors retain
/// their source chains in [`QuanticsTransformError::Operation`].
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
/// # Ok(())
/// # }
/// ```
pub fn quantics_fourier_operator_multiaxis(
    layout: &[AxisBit],
    options: FourierOptions,
) -> Result<QuanticsOperator, QuanticsTransformError> {
    options.validate()?;
    let axes = axis_positions(layout)?;
    if axes.len() == 1 && layout.iter().enumerate().all(|(j, bit)| bit.bit == j) {
        return quantics_fourier_operator(layout.len(), options);
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
