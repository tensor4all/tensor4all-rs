//! Grid-aware shift operators for quantics transformations.
//!
//! This module provides shift operators that work directly with
//! [`DiscretizedGrid`] objects, automatically handling different
//! unfolding schemes (Grouped, Fused, Interleaved).

use std::collections::HashMap;

use anyhow::Result;
use num_complex::Complex64;
use quanticsgrids::{DiscretizedGrid, IndexTable};
use tensor4all_core::{index::Index, IndexLike, TensorDynLen, TensorIndex};
use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_treetn::{
    factorize_tensor_to_treetn, IndexMapping, SwapOptions, TreeTN, TreeTopology,
};

use crate::common::{
    tensortrain_to_linear_operator, tensortrain_to_linear_operator_asymmetric, BoundaryCondition,
    DynIndex, QuanticsOperator,
};
use crate::shift::{shift_mpo, shift_operator_multivar};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GridLayoutKind {
    Grouped,
    Interleaved,
    Fused,
    Custom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AffineOperatorSemantics {
    Forward,
    Pullback,
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
/// - The grid layout is custom
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

    // In 1D, grouped/interleaved/fused layouts are physically equivalent.
    // Avoid the fused multivariable path, which requires nvariables >= 2.
    if ndims == 1 {
        return shift_on_grid_grouped(grid, offsets, bc);
    }

    match detect_layout_kind(grid)? {
        GridLayoutKind::Grouped => shift_on_grid_grouped(grid, offsets, bc),
        GridLayoutKind::Fused => shift_on_grid_fused(grid, offsets, bc),
        GridLayoutKind::Interleaved => {
            // Multi-variable interleaved uses the grouped binary operator and
            // relocates site pairs.
            shift_on_grid_interleaved(grid, offsets, bc)
        }
        GridLayoutKind::Custom => Err(anyhow::anyhow!(
            "shift_operator_on_grid does not support custom grid layouts"
        )),
    }
}

/// Create a shift operator on a grid using variable names.
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
/// use tensor4all_quanticstransform::{
///     shift_operator_on_grid_by_variable_name, BoundaryCondition,
/// };
///
/// let grid = DiscretizedGrid::builder(&[3, 2])
///     .with_variable_names(&["x", "y"])
///     .with_unfolding_scheme(UnfoldingScheme::Grouped)
///     .build()
///     .unwrap();
/// let op = shift_operator_on_grid_by_variable_name(
///     &grid,
///     &[("x", 1, BoundaryCondition::Periodic)],
/// ).unwrap();
/// assert_eq!(op.mpo.node_count(), 5);
/// ```
pub fn shift_operator_on_grid_by_variable_name(
    grid: &DiscretizedGrid,
    var_offsets: &[(&str, i64, BoundaryCondition)],
) -> Result<QuanticsOperator> {
    let ndims = grid.ndims();

    let mut offsets = vec![0i64; ndims];
    let mut bcs = vec![BoundaryCondition::Periodic; ndims];

    for &(name, offset, bc) in var_offsets {
        let idx = variable_id_from_names(grid.variable_names(), name).map_err(|_| {
            anyhow::anyhow!(
                "Unknown variable name '{}'. Available: {:?}",
                name,
                grid.variable_names()
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

/// Build shift operator for Interleaved unfolding scheme.
///
/// Strategy: build the grouped binary operator and relocate each input/output
/// site pair to the target node order described by the grid.
fn shift_on_grid_interleaved(
    grid: &DiscretizedGrid,
    offsets: &[i64],
    bc: &[BoundaryCondition],
) -> Result<QuanticsOperator> {
    let rs = grid.rs();
    let r = rs[0];
    if rs.iter().any(|&ri| ri != r) {
        return Err(anyhow::anyhow!(
            "Interleaved layout requires equal resolution for all variables, got {:?}",
            rs
        ));
    }

    let grouped = shift_on_grid_grouped(grid, offsets, bc)?;
    let source = grouped_sequence(rs);
    let target = grid_binary_sequence(grid)?;
    relayout_binary_operator(grouped, &source, &target)
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

fn grouped_sequence(rs: &[usize]) -> Vec<(usize, usize)> {
    let total_sites: usize = rs.iter().sum();
    let mut sequence = Vec::with_capacity(total_sites);
    for (var, &r) in rs.iter().enumerate() {
        for level in 0..r {
            sequence.push((var, level));
        }
    }
    sequence
}

fn interleaved_sequence(ndims: usize, r: usize) -> Vec<(usize, usize)> {
    let mut sequence = Vec::with_capacity(ndims * r);
    for level in 0..r {
        for var in 0..ndims {
            sequence.push((var, level));
        }
    }
    sequence
}

fn grid_binary_sequence(grid: &DiscretizedGrid) -> Result<Vec<(usize, usize)>> {
    let mut by_position = Vec::new();
    for (var, name) in grid.variable_names().iter().enumerate() {
        let sites = variable_sites(grid, &[name.as_str()], true)?;
        for (level, position) in sites.into_iter().enumerate() {
            by_position.push((position, var, level));
        }
    }
    by_position.sort_by_key(|(position, _, _)| *position);
    Ok(by_position
        .into_iter()
        .map(|(_, var, level)| (var, level))
        .collect())
}

fn relayout_binary_operator(
    mut op: QuanticsOperator,
    source_sequence: &[(usize, usize)],
    target_sequence: &[(usize, usize)],
) -> Result<QuanticsOperator> {
    if source_sequence.len() != target_sequence.len() {
        return Err(anyhow::anyhow!(
            "source/target binary layout length mismatch: {} vs {}",
            source_sequence.len(),
            target_sequence.len()
        ));
    }

    let mut current_sequence = source_sequence.to_vec();
    let mut input_mappings: Vec<IndexMapping<DynIndex>> = (0..source_sequence.len())
        .map(|node| {
            op.get_input_mapping(&node)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing input mapping for node {}", node))
        })
        .collect::<Result<_>>()?;
    let mut output_mappings: Vec<IndexMapping<DynIndex>> = (0..source_sequence.len())
        .map(|node| {
            op.get_output_mapping(&node)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing output mapping for node {}", node))
        })
        .collect::<Result<_>>()?;

    for (target_node, desired) in target_sequence.iter().copied().enumerate() {
        let mut current_node = current_sequence
            .iter()
            .position(|&physical| physical == desired)
            .ok_or_else(|| anyhow::anyhow!("missing target binary site {:?}", desired))?;

        while current_node > target_node {
            let left = current_node - 1;
            let right = current_node;
            let local_assignment = HashMap::from([
                (input_mappings[left].internal_index.clone(), right),
                (output_mappings[left].internal_index.clone(), right),
                (input_mappings[right].internal_index.clone(), left),
                (output_mappings[right].internal_index.clone(), left),
            ]);
            op.mpo
                .swap_site_indices_by_index(&local_assignment, &SwapOptions::default())?;
            current_sequence.swap(left, right);
            input_mappings.swap(left, right);
            output_mappings.swap(left, right);
            current_node -= 1;
        }
    }

    op.input_mapping = input_mappings.into_iter().enumerate().collect();
    op.output_mapping = output_mappings.into_iter().enumerate().collect();
    Ok(op)
}

fn col_major_offset(dims: &[usize], coords: &[usize]) -> usize {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&coord, &dim) in coords.iter().zip(dims.iter()) {
        offset += coord * stride;
        stride *= dim;
    }
    offset
}

fn build_affine_binary_interleaved_operator(
    r: usize,
    params: &crate::affine::AffineParams,
    bc: &[BoundaryCondition],
    semantics: AffineOperatorSemantics,
) -> Result<QuanticsOperator> {
    let ndims = params.m;
    let site_tensors = crate::affine::affine_transform_tensors_unfused(r, params, bc)?;
    let info = crate::affine::UnfusedTensorInfo::new(params);

    let first_group_indices: Vec<Vec<DynIndex>> = (0..r)
        .map(|_| (0..ndims).map(|_| Index::new_dyn(2)).collect())
        .collect();
    let second_group_indices: Vec<Vec<DynIndex>> = (0..r)
        .map(|_| (0..ndims).map(|_| Index::new_dyn(2)).collect())
        .collect();
    let bonds: Vec<DynIndex> = site_tensors
        .iter()
        .take(r.saturating_sub(1))
        .map(|tensor| Index::new_dyn(tensor.dims()[2]))
        .collect();

    let mut tensors = Vec::with_capacity(r);
    for level in 0..r {
        let tensor = &site_tensors[level];
        let left_dim = tensor.dims()[0];
        let site_dim = tensor.dims()[1];
        let right_dim = tensor.dims()[2];

        let mut indices = Vec::new();
        let mut dims = Vec::new();

        if level > 0 {
            indices.push(bonds[level - 1].clone());
            dims.push(left_dim);
        }
        for idx in &first_group_indices[level] {
            indices.push(idx.clone());
            dims.push(2);
        }
        for idx in &second_group_indices[level] {
            indices.push(idx.clone());
            dims.push(2);
        }
        if level + 1 < r {
            indices.push(bonds[level].clone());
            dims.push(right_dim);
        }

        let mut data = vec![Complex64::new(0.0, 0.0); dims.iter().product()];
        for left in 0..left_dim {
            for fused_idx in 0..site_dim {
                let (first_group_bits, second_group_bits) = info.decode_fused_index(fused_idx);
                for right in 0..right_dim {
                    let value = tensor[[left, fused_idx, right]];
                    if value == Complex64::new(0.0, 0.0) {
                        continue;
                    }

                    let mut coords = Vec::with_capacity(dims.len());
                    if level > 0 {
                        coords.push(left);
                    }
                    coords.extend(first_group_bits.iter().copied());
                    coords.extend(second_group_bits.iter().copied());
                    if level + 1 < r {
                        coords.push(right);
                    }

                    let flat = col_major_offset(&dims, &coords);
                    data[flat] = value;
                }
            }
        }

        tensors.push(
            TensorDynLen::from_dense(indices, data)
                .map_err(|err| anyhow::anyhow!("failed to build affine site tensor: {}", err))?,
        );
    }

    let mut split_tensors = Vec::with_capacity(r * ndims);
    let mut split_node_names = Vec::with_capacity(r * ndims);
    for (level, fused_tensor) in tensors.into_iter().enumerate() {
        let local = factorize_affine_site_to_interleaved_chain(
            level,
            ndims,
            &first_group_indices[level],
            &second_group_indices[level],
            level.checked_sub(1).map(|left| bonds[left].clone()),
            (level + 1 < r).then(|| bonds[level].clone()),
            fused_tensor,
        )?;
        for node_name in local.node_names() {
            let node_idx = local
                .node_index(&node_name)
                .ok_or_else(|| anyhow::anyhow!("missing local affine node {}", node_name))?;
            let tensor = local
                .tensor(node_idx)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing local affine tensor {}", node_name))?;
            split_node_names.push(node_name);
            split_tensors.push(tensor);
        }
    }

    let mpo = TreeTN::from_tensors(split_tensors, split_node_names)?;
    let sequence = interleaved_sequence(ndims, r);
    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();
    for (node_name, (var, level)) in sequence.into_iter().enumerate() {
        let (input_internal, output_internal) = match semantics {
            AffineOperatorSemantics::Forward => (
                second_group_indices[level][var].clone(),
                first_group_indices[level][var].clone(),
            ),
            AffineOperatorSemantics::Pullback => (
                first_group_indices[level][var].clone(),
                second_group_indices[level][var].clone(),
            ),
        };
        input_mapping.insert(
            node_name,
            IndexMapping {
                true_index: Index::new_dyn(2),
                internal_index: input_internal,
            },
        );
        output_mapping.insert(
            node_name,
            IndexMapping {
                true_index: Index::new_dyn(2),
                internal_index: output_internal,
            },
        );
    }

    Ok(QuanticsOperator {
        mpo,
        input_mapping,
        output_mapping,
    })
}

fn factorize_affine_site_to_interleaved_chain(
    level: usize,
    ndims: usize,
    output_indices: &[DynIndex],
    input_indices: &[DynIndex],
    left_bond: Option<DynIndex>,
    right_bond: Option<DynIndex>,
    fused_tensor: TensorDynLen,
) -> Result<TreeTN<TensorDynLen, usize>> {
    let mut nodes = HashMap::new();
    let mut edges = Vec::new();

    for var in 0..ndims {
        let node_name = level * ndims + var;
        let mut node_indices = Vec::with_capacity(4);
        if var == 0 {
            if let Some(left) = &left_bond {
                node_indices.push(*left.id());
            }
        }
        node_indices.push(*output_indices[var].id());
        node_indices.push(*input_indices[var].id());
        if var + 1 == ndims {
            if let Some(right) = &right_bond {
                node_indices.push(*right.id());
            }
        }
        nodes.insert(node_name, node_indices);
        if var > 0 {
            edges.push((node_name - 1, node_name));
        }
    }

    let topology = TreeTopology::new(nodes, edges);
    let root = level * ndims + (ndims - 1);
    factorize_tensor_to_treetn(&fused_tensor, &topology, &root)
        .map_err(|err| anyhow::anyhow!("failed to factorize affine site {}: {}", level, err))
}

/// Build an affine operator that is aware of the grid's unfolding scheme and
/// resolution.
///
/// This is a convenience wrapper around [`crate::affine_operator`] that
/// extracts the number of quantics bits from the grid and validates layout
/// compatibility.
///
/// # Supported layouts
///
/// | Layout | Behaviour |
/// |---|---|
/// | 1D (single variable) | Delegates directly to `affine_operator(r, params, bc)`. |
/// | Fused | All Rs must be equal. Delegates to `affine_operator(r, params, bc)`. |
/// | Grouped | All Rs must be equal. Builds a binary operator and reorders it to grouped layout. |
/// | Interleaved | All Rs must be equal. Builds a binary operator in interleaved layout. |
///
/// # Examples
///
/// ```ignore
/// use quanticsgrids::{DiscretizedGrid, UnfoldingScheme};
/// use tensor4all_quanticstransform::{
///     affine_operator_on_grid, AffineParams, BoundaryCondition,
/// };
///
/// let grid = DiscretizedGrid::builder(&[4])
///     .with_variable_names(&["x"])
///     .with_lower_bound(&[0.0])
///     .with_upper_bound(&[1.0])
///     .build()
///     .unwrap();
/// let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
/// let bc = vec![BoundaryCondition::Periodic];
/// let op = affine_operator_on_grid(&grid, &params, &bc).unwrap();
/// assert_eq!(op.mpo.node_count(), 4);
/// ```
pub fn affine_operator_on_grid(
    grid: &DiscretizedGrid,
    params: &crate::affine::AffineParams,
    bc: &[BoundaryCondition],
) -> Result<QuanticsOperator> {
    let rs = grid.rs();
    let ndims = grid.ndims();

    if params.m != ndims || params.n != ndims {
        return Err(anyhow::anyhow!(
            "affine_operator_on_grid requires params.m == params.n == grid.ndims(), got m={}, n={}, ndims={}",
            params.m,
            params.n,
            ndims
        ));
    }

    if ndims == 1 {
        return crate::affine_operator(rs[0], params, bc);
    }

    let r = rs[0];
    if rs.iter().any(|&ri| ri != r) {
        return Err(anyhow::anyhow!(
            "affine_operator_on_grid requires all resolutions to be equal, got {:?}",
            rs
        ));
    }

    match detect_layout_kind(grid)? {
        GridLayoutKind::Fused => crate::affine_operator(r, params, bc),
        GridLayoutKind::Grouped => {
            let interleaved = build_affine_binary_interleaved_operator(
                r,
                params,
                bc,
                AffineOperatorSemantics::Forward,
            )?;
            let source = interleaved_sequence(ndims, r);
            let target = grid_binary_sequence(grid)?;
            relayout_binary_operator(interleaved, &source, &target)
        }
        GridLayoutKind::Interleaved => build_affine_binary_interleaved_operator(
            r,
            params,
            bc,
            AffineOperatorSemantics::Forward,
        ),
        GridLayoutKind::Custom => Err(anyhow::anyhow!(
            "affine_operator_on_grid does not support custom grid layouts"
        )),
    }
}

/// Build an affine pullback operator that is aware of the grid's unfolding
/// scheme and resolution.
///
/// This is the grid-aware counterpart of
/// [`crate::affine_pullback_operator`]. The operator maps a source function
/// `g(x)` on `grid` to the pullback `f(y) = g(A * y + b)`, while preserving
/// the grid's physical site layout.
///
/// # Supported layouts
///
/// | Layout | Behaviour |
/// |---|---|
/// | 1D (single variable) | Delegates directly to `affine_pullback_operator(r, params, bc)`. |
/// | Fused | All Rs must be equal. Delegates to `affine_pullback_operator(r, params, bc)`. |
/// | Grouped | All Rs must be equal. Builds a binary pullback operator and reorders it to grouped layout. |
/// | Interleaved | All Rs must be equal. Builds a binary pullback operator in interleaved layout. |
///
/// # Examples
///
/// ```ignore
/// use quanticsgrids::{DiscretizedGrid, UnfoldingScheme};
/// use tensor4all_quanticstransform::{
///     affine_pullback_operator_on_grid, AffineParams, BoundaryCondition,
/// };
///
/// let grid = DiscretizedGrid::builder(&[3, 3])
///     .with_variable_names(&["x", "y"])
///     .with_unfolding_scheme(UnfoldingScheme::Grouped)
///     .build()
///     .unwrap();
/// let params = AffineParams::from_integers(vec![1, 0, 1, 1], vec![0, 0], 2, 2).unwrap();
/// let op = affine_pullback_operator_on_grid(
///     &grid,
///     &params,
///     &[BoundaryCondition::Open, BoundaryCondition::Open],
/// )
/// .unwrap();
/// assert_eq!(op.mpo.node_count(), 6);
/// ```
pub fn affine_pullback_operator_on_grid(
    grid: &DiscretizedGrid,
    params: &crate::affine::AffineParams,
    bc: &[BoundaryCondition],
) -> Result<QuanticsOperator> {
    let rs = grid.rs();
    let ndims = grid.ndims();

    if params.m != ndims || params.n != ndims {
        return Err(anyhow::anyhow!(
            "affine_pullback_operator_on_grid requires params.m == params.n == grid.ndims(), got m={}, n={}, ndims={}",
            params.m,
            params.n,
            ndims
        ));
    }

    if ndims == 1 {
        return crate::affine_pullback_operator(rs[0], params, bc);
    }

    let r = rs[0];
    if rs.iter().any(|&ri| ri != r) {
        return Err(anyhow::anyhow!(
            "affine_pullback_operator_on_grid requires all resolutions to be equal, got {:?}",
            rs
        ));
    }

    match detect_layout_kind(grid)? {
        GridLayoutKind::Fused => crate::affine_pullback_operator(r, params, bc),
        GridLayoutKind::Grouped => {
            let interleaved = build_affine_binary_interleaved_operator(
                r,
                params,
                bc,
                AffineOperatorSemantics::Pullback,
            )?;
            let source = interleaved_sequence(ndims, r);
            let target = grid_binary_sequence(grid)?;
            relayout_binary_operator(interleaved, &source, &target)
        }
        GridLayoutKind::Interleaved => build_affine_binary_interleaved_operator(
            r,
            params,
            bc,
            AffineOperatorSemantics::Pullback,
        ),
        GridLayoutKind::Custom => Err(anyhow::anyhow!(
            "affine_pullback_operator_on_grid does not support custom grid layouts"
        )),
    }
}

fn variable_id_from_names(variable_names: &[String], name: &str) -> Result<usize> {
    variable_names
        .iter()
        .position(|candidate| candidate == name)
        .ok_or_else(|| anyhow::anyhow!("unknown variable {}", name))
}

fn variable_sites(grid: &DiscretizedGrid, names: &[&str], do_sort: bool) -> Result<Vec<usize>> {
    let mut sites = Vec::new();
    let variable_names = grid.variable_names();
    let index_table = grid.index_table();
    for &name in names {
        let var_id = variable_id_from_names(variable_names, name)?;
        let variable_name = &variable_names[var_id];
        for (site, indices) in index_table.iter().enumerate() {
            if indices
                .iter()
                .any(|(index_name, _)| index_name == variable_name)
            {
                sites.push(site);
            }
        }
    }
    if do_sort {
        sites.sort_unstable();
    }
    Ok(sites)
}

fn detect_layout_kind(grid: &DiscretizedGrid) -> Result<GridLayoutKind> {
    let signature = layout_signature(grid.index_table(), grid.variable_names())?;
    let rs = grid.rs();

    if signature == canonical_layout_signature(rs, GridLayoutKind::Fused) {
        Ok(GridLayoutKind::Fused)
    } else if signature == canonical_layout_signature(rs, GridLayoutKind::Interleaved) {
        Ok(GridLayoutKind::Interleaved)
    } else if signature == canonical_layout_signature(rs, GridLayoutKind::Grouped) {
        Ok(GridLayoutKind::Grouped)
    } else {
        Ok(GridLayoutKind::Custom)
    }
}

fn layout_signature(
    index_table: &IndexTable,
    variable_names: &[String],
) -> Result<Vec<Vec<(usize, usize)>>> {
    index_table
        .iter()
        .map(|site| {
            site.iter()
                .map(|(var_name, bitnumber)| {
                    let var_idx = variable_id_from_names(variable_names, var_name)?;
                    Ok((var_idx, *bitnumber))
                })
                .collect()
        })
        .collect()
}

fn canonical_layout_signature(rs: &[usize], kind: GridLayoutKind) -> Vec<Vec<(usize, usize)>> {
    match kind {
        GridLayoutKind::Grouped => canonical_grouped_signature(rs),
        GridLayoutKind::Interleaved => canonical_interleaved_signature(rs),
        GridLayoutKind::Fused => canonical_fused_signature(rs),
        GridLayoutKind::Custom => Vec::new(),
    }
}

fn canonical_grouped_signature(rs: &[usize]) -> Vec<Vec<(usize, usize)>> {
    let mut signature = Vec::new();
    for (var_idx, &r) in rs.iter().enumerate() {
        for bitnumber in 1..=r {
            signature.push(vec![(var_idx, bitnumber)]);
        }
    }
    signature
}

fn canonical_interleaved_signature(rs: &[usize]) -> Vec<Vec<(usize, usize)>> {
    let mut signature = Vec::new();
    let max_r = *rs.iter().max().unwrap_or(&0);
    for bitnumber in 1..=max_r {
        for (var_idx, &r) in rs.iter().enumerate() {
            if bitnumber <= r {
                signature.push(vec![(var_idx, bitnumber)]);
            }
        }
    }
    signature
}

fn canonical_fused_signature(rs: &[usize]) -> Vec<Vec<(usize, usize)>> {
    let mut signature = Vec::new();
    let max_r = *rs.iter().max().unwrap_or(&0);
    for bitnumber in 1..=max_r {
        let mut site = Vec::new();
        for (var_idx, &r) in rs.iter().enumerate().rev() {
            if bitnumber <= r {
                site.push((var_idx, bitnumber));
            }
        }
        if !site.is_empty() {
            signature.push(site);
        }
    }
    signature
}

#[cfg(test)]
mod tests;
