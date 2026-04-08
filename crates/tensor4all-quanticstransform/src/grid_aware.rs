//! Grid-aware shift operators for quantics transformations.
//!
//! This module provides shift operators that work directly with
//! [`DiscretizedGrid`] objects, automatically handling different
//! unfolding schemes (Grouped, Fused, Interleaved).

use anyhow::Result;
use quanticsgrids::{DiscretizedGrid, UnfoldingScheme};
use tensor4all_core::TensorIndex;
use tensor4all_simplett::AbstractTensorTrain;

use crate::common::{
    tensortrain_to_linear_operator, tensortrain_to_linear_operator_asymmetric, BoundaryCondition,
    QuanticsOperator,
};
use crate::shift::{shift_mpo, shift_operator_multivar};

/// Detect the unfolding scheme of a grid by inspecting its index table.
///
/// # Arguments
///
/// * `grid` - The discretized grid to inspect
///
/// # Returns
///
/// The detected [`UnfoldingScheme`].
///
/// # Examples
///
/// ```ignore
/// use quanticsgrids::{DiscretizedGrid, UnfoldingScheme};
/// use tensor4all_quanticstransform::detect_unfolding_scheme;
///
/// let grid = DiscretizedGrid::builder(&[3, 2])
///     .with_variable_names(&["x", "y"])
///     .with_unfolding_scheme(UnfoldingScheme::Grouped)
///     .build()
///     .unwrap();
/// assert_eq!(detect_unfolding_scheme(&grid), UnfoldingScheme::Grouped);
/// ```
pub fn detect_unfolding_scheme(grid: &DiscretizedGrid) -> UnfoldingScheme {
    let index_table = grid.index_table();
    let ndims = grid.ndims();

    if ndims <= 1 {
        // For 1D grids, all schemes are equivalent; return Grouped as default.
        return UnfoldingScheme::Grouped;
    }

    // Check if Fused: at least one site has entries for multiple variables
    if index_table.iter().any(|site| site.len() >= 2) {
        return UnfoldingScheme::Fused;
    }

    // All sites have exactly 1 entry. Distinguish Grouped vs Interleaved.
    // Grouped: all bits of variable 0 come first, then variable 1, etc.
    // Interleaved: bits alternate between variables.
    let var_names = grid.variable_names();
    let rs = grid.rs();

    // Build expected grouped pattern: [var0_bit1, var0_bit2, ..., var1_bit1, ...]
    let mut expected_grouped = Vec::new();
    for (d, name) in var_names.iter().enumerate() {
        for bit in 1..=rs[d] {
            expected_grouped.push((name.clone(), bit));
        }
    }

    let actual: Vec<(String, usize)> = index_table
        .iter()
        .filter_map(|site| {
            if site.len() == 1 {
                Some(site[0].clone())
            } else {
                None
            }
        })
        .collect();

    if actual == expected_grouped {
        return UnfoldingScheme::Grouped;
    }

    UnfoldingScheme::Interleaved
}

/// Create a shift operator on a grid with index-based variable selection.
///
/// For each variable in the grid, apply a shift by the corresponding offset.
/// Variables with offset 0 are treated as identity.
///
/// # Arguments
///
/// * `grid` - The discretized grid
/// * `offsets` - Shift offset per variable (one per dimension)
/// * `bc` - Boundary condition per variable (one per dimension)
///
/// # Returns
///
/// A `QuanticsOperator` (LinearOperator) representing the composed shift.
///
/// # Errors
///
/// Returns an error if:
/// - `offsets.len() != grid.ndims()`
/// - `bc.len() != grid.ndims()`
/// - The grid uses an Interleaved unfolding scheme with multiple variables
///   (not yet implemented)
///
/// # Examples
///
/// ```ignore
/// use quanticsgrids::{DiscretizedGrid, UnfoldingScheme};
/// use tensor4all_quanticstransform::{shift_operator_on_grid, BoundaryCondition};
///
/// let grid = DiscretizedGrid::builder(&[4])
///     .with_variable_names(&["x"])
///     .with_unfolding_scheme(UnfoldingScheme::Grouped)
///     .build()
///     .unwrap();
/// let op = shift_operator_on_grid(&grid, &[3], &[BoundaryCondition::Periodic]).unwrap();
/// assert_eq!(op.mpo.node_count(), 4);
/// ```
pub fn shift_operator_on_grid(
    grid: &DiscretizedGrid,
    offsets: &[i64],
    bc: &[BoundaryCondition],
) -> Result<QuanticsOperator> {
    let ndims = grid.ndims();

    if offsets.len() != ndims {
        return Err(anyhow::anyhow!(
            "offsets length {} does not match grid dimensions {}",
            offsets.len(),
            ndims
        ));
    }
    if bc.len() != ndims {
        return Err(anyhow::anyhow!(
            "bc length {} does not match grid dimensions {}",
            bc.len(),
            ndims
        ));
    }

    let scheme = detect_unfolding_scheme(grid);

    match scheme {
        UnfoldingScheme::Grouped => shift_on_grid_grouped(grid, offsets, bc),
        UnfoldingScheme::Fused => shift_on_grid_fused(grid, offsets, bc),
        UnfoldingScheme::Interleaved => {
            if ndims > 1 {
                Err(anyhow::anyhow!(
                    "Interleaved unfolding scheme with multiple variables is not yet implemented"
                ))
            } else {
                // 1D interleaved is same as grouped
                shift_on_grid_grouped(grid, offsets, bc)
            }
        }
    }
}

/// Create a shift operator on a grid using variable names (tag-based).
///
/// Only specified variables are shifted; unspecified variables get identity (offset 0).
///
/// # Arguments
///
/// * `grid` - The discretized grid
/// * `var_offsets` - Slice of (variable_name, offset, boundary_condition) tuples
///
/// # Returns
///
/// A `QuanticsOperator` representing the composed shift.
///
/// # Errors
///
/// Returns an error if a variable name is not found in the grid.
///
/// # Examples
///
/// ```ignore
/// use quanticsgrids::{DiscretizedGrid, UnfoldingScheme};
/// use tensor4all_quanticstransform::{shift_operator_on_grid_by_tag, BoundaryCondition};
///
/// let grid = DiscretizedGrid::builder(&[3, 2])
///     .with_variable_names(&["x", "y"])
///     .with_unfolding_scheme(UnfoldingScheme::Grouped)
///     .build()
///     .unwrap();
/// let op = shift_operator_on_grid_by_tag(
///     &grid,
///     &[("x", 1, BoundaryCondition::Periodic)],
/// ).unwrap();
/// assert_eq!(op.mpo.node_count(), 5);
/// ```
pub fn shift_operator_on_grid_by_tag(
    grid: &DiscretizedGrid,
    var_offsets: &[(&str, i64, BoundaryCondition)],
) -> Result<QuanticsOperator> {
    let ndims = grid.ndims();
    let var_names = grid.variable_names();

    let mut offsets = vec![0i64; ndims];
    let mut bcs = vec![BoundaryCondition::Periodic; ndims];

    for &(name, offset, bc) in var_offsets {
        let idx = var_names.iter().position(|n| n == name).ok_or_else(|| {
            anyhow::anyhow!(
                "Unknown variable name '{}'. Available: {:?}",
                name,
                var_names
            )
        })?;
        offsets[idx] = offset;
        bcs[idx] = bc;
    }

    shift_operator_on_grid(grid, &offsets, &bcs)
}

/// Build shift operator for Grouped unfolding scheme.
///
/// Strategy: Build independent shift TensorTrain per variable, then concatenate
/// them into a single chain. Each per-variable TT has left_dim=1 and right_dim=1
/// at its boundaries, so they naturally connect.
fn shift_on_grid_grouped(
    grid: &DiscretizedGrid,
    offsets: &[i64],
    bc: &[BoundaryCondition],
) -> Result<QuanticsOperator> {
    let rs = grid.rs();
    let ndims = grid.ndims();

    // Build per-variable TensorTrains and collect all site tensors
    let mut all_tensors = Vec::new();

    for d in 0..ndims {
        let r = rs[d];
        if r == 0 {
            continue;
        }

        let mpo = shift_mpo(r, offsets[d], bc[d])?;

        // Collect tensors from this variable's TT
        for i in 0..mpo.len() {
            all_tensors.push(mpo.site_tensor(i).clone());
        }
    }

    if all_tensors.is_empty() {
        return Err(anyhow::anyhow!("Grid has no sites"));
    }

    let combined_tt = tensor4all_simplett::TensorTrain::new(all_tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create concatenated TensorTrain: {}", e))?;

    // Site dimensions: each site in grouped layout has dim 2 (binary)
    let site_dims: Vec<usize> = rs.iter().flat_map(|&r| vec![2; r]).collect();

    tensortrain_to_linear_operator(&combined_tt, &site_dims)
}

/// Build shift operator for Fused unfolding scheme.
///
/// Strategy: For each non-zero offset, build a multivar shift operator using
/// `shift_operator_multivar`. When multiple variables need shifting, compose
/// them by contracting the MPOs.
fn shift_on_grid_fused(
    grid: &DiscretizedGrid,
    offsets: &[i64],
    bc: &[BoundaryCondition],
) -> Result<QuanticsOperator> {
    let rs = grid.rs();
    let ndims = grid.ndims();

    // All variables must have the same resolution for fused layout
    let r = rs[0];
    for &ri in &rs[1..] {
        if ri != r {
            return Err(anyhow::anyhow!(
                "Fused layout requires equal resolution for all variables, got {:?}",
                rs
            ));
        }
    }

    // Collect non-zero shifts
    let non_zero_shifts: Vec<usize> = (0..ndims).filter(|&d| offsets[d] != 0).collect();

    if non_zero_shifts.is_empty() {
        // All zero offsets: return identity operator
        let mpo = shift_mpo(r, 0, BoundaryCondition::Periodic)?;
        let embedded = crate::common::embed_single_var_mpo(&mpo, ndims, 0)?;
        let dim_multi = 1 << ndims;
        let dims = vec![dim_multi; r];
        return tensortrain_to_linear_operator_asymmetric(&embedded, &dims, &dims);
    }

    // Build the first non-zero shift operator
    let first_var = non_zero_shifts[0];
    let mut result =
        shift_operator_multivar(r, offsets[first_var], bc[first_var], ndims, first_var)?;

    // Compose remaining shifts via MPO-MPO contraction
    for &var in &non_zero_shifts[1..] {
        let next_op = shift_operator_multivar(r, offsets[var], bc[var], ndims, var)?;
        result = compose_operators(&result, &next_op)?;
    }

    Ok(result)
}

/// Compose two operators A and B into A*B (apply A after B).
///
/// This connects B's output indices to A's input indices and contracts the MPOs.
fn compose_operators(op_a: &QuanticsOperator, op_b: &QuanticsOperator) -> Result<QuanticsOperator> {
    use std::collections::HashMap;
    use tensor4all_treetn::treetn::contraction::{contract, ContractionOptions};

    let node_names: Vec<usize> = op_a.mpo.node_names();

    // We want result = A * B:
    // input_B -> MPO_B -> [connect B_output = A_input] -> MPO_A -> output_A
    //
    // Replace B's output internal indices with A's input internal indices,
    // so that when we contract the two TreeTNs, the shared indices are contracted.

    let mut old_indices = Vec::new();
    let mut new_indices = Vec::new();

    for &name in &node_names {
        let a_input = op_a
            .input_mapping
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("Missing input mapping for node {}", name))?;
        let b_output = op_b
            .output_mapping
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("Missing output mapping for node {}", name))?;

        old_indices.push(b_output.internal_index.clone());
        new_indices.push(a_input.internal_index.clone());
    }

    let mpo_b_adjusted = op_b.mpo.replaceinds(&old_indices, &new_indices)?;

    // Contract the two MPOs
    let center = node_names
        .first()
        .ok_or_else(|| anyhow::anyhow!("Empty operator"))?;

    let contracted = contract(
        &op_a.mpo,
        &mpo_b_adjusted,
        center,
        ContractionOptions::default(),
    )?;

    // Build result mappings: input from B, output from A
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for &name in &node_names {
        input_mapping.insert(name, op_b.input_mapping[&name].clone());
        output_mapping.insert(name, op_a.output_mapping[&name].clone());
    }

    Ok(QuanticsOperator {
        mpo: contracted,
        input_mapping,
        output_mapping,
    })
}

#[cfg(test)]
mod tests;
