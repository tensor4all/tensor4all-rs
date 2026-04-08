#![allow(clippy::needless_range_loop)]

use super::*;
use num_complex::Complex64;
use num_traits::{One, Zero};
use std::collections::HashMap;
use tensor4all_core::index::DynId;
use tensor4all_core::IndexLike;

// ============================================================================
// detect_unfolding_scheme tests
// ============================================================================

#[test]
fn test_detect_grouped() {
    let grid = DiscretizedGrid::builder(&[3, 2])
        .with_variable_names(&["x", "y"])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();
    assert_eq!(detect_unfolding_scheme(&grid), UnfoldingScheme::Grouped);
}

#[test]
fn test_detect_fused() {
    let grid = DiscretizedGrid::builder(&[3, 3])
        .with_variable_names(&["x", "y"])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .unwrap();
    assert_eq!(detect_unfolding_scheme(&grid), UnfoldingScheme::Fused);
}

#[test]
fn test_detect_interleaved() {
    let grid = DiscretizedGrid::builder(&[3, 3])
        .with_variable_names(&["x", "y"])
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .build()
        .unwrap();
    assert_eq!(detect_unfolding_scheme(&grid), UnfoldingScheme::Interleaved);
}

#[test]
fn test_detect_1d_returns_grouped() {
    let grid = DiscretizedGrid::builder(&[4])
        .with_variable_names(&["x"])
        .build()
        .unwrap();
    assert_eq!(detect_unfolding_scheme(&grid), UnfoldingScheme::Grouped);
}

// ============================================================================
// shift_operator_on_grid tests - structural
// ============================================================================

#[test]
fn test_shift_on_grid_grouped_1d() {
    let grid = DiscretizedGrid::builder(&[4])
        .with_variable_names(&["x"])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();
    let op = shift_operator_on_grid(&grid, &[3], &[BoundaryCondition::Periodic]).unwrap();
    assert_eq!(op.mpo.node_count(), 4);
}

#[test]
fn test_shift_on_grid_grouped_2d_structure() {
    let grid = DiscretizedGrid::builder(&[3, 2])
        .with_variable_names(&["x", "y"])
        .with_lower_bound(&[0.0, 0.0])
        .with_upper_bound(&[1.0, 1.0])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();
    let op = shift_operator_on_grid(
        &grid,
        &[1, 0],
        &[BoundaryCondition::Periodic, BoundaryCondition::Periodic],
    )
    .unwrap();
    assert_eq!(op.mpo.node_count(), 5); // 3 + 2 sites
}

#[test]
fn test_shift_on_grid_fused_2d() {
    let grid = DiscretizedGrid::builder(&[3, 3])
        .with_variable_names(&["x", "y"])
        .with_lower_bound(&[0.0, 0.0])
        .with_upper_bound(&[1.0, 1.0])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .unwrap();
    let op = shift_operator_on_grid(
        &grid,
        &[1, 0],
        &[BoundaryCondition::Periodic, BoundaryCondition::Periodic],
    )
    .unwrap();
    assert_eq!(op.mpo.node_count(), 3); // fused: 3 sites
}

// ============================================================================
// shift_operator_on_grid_by_tag tests
// ============================================================================

#[test]
fn test_shift_on_grid_by_tag() {
    let grid = DiscretizedGrid::builder(&[3, 2])
        .with_variable_names(&["x", "y"])
        .with_lower_bound(&[0.0, 0.0])
        .with_upper_bound(&[1.0, 1.0])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();
    let op =
        shift_operator_on_grid_by_tag(&grid, &[("x", 1, BoundaryCondition::Periodic)]).unwrap();
    assert_eq!(op.mpo.node_count(), 5);
}

// ============================================================================
// Error tests
// ============================================================================

#[test]
fn test_shift_on_grid_wrong_offsets_length() {
    let grid = DiscretizedGrid::builder(&[4])
        .with_variable_names(&["x"])
        .build()
        .unwrap();
    assert!(shift_operator_on_grid(&grid, &[1, 2], &[BoundaryCondition::Periodic]).is_err());
}

#[test]
fn test_shift_on_grid_wrong_bc_length() {
    let grid = DiscretizedGrid::builder(&[4])
        .with_variable_names(&["x"])
        .build()
        .unwrap();
    assert!(shift_operator_on_grid(
        &grid,
        &[1],
        &[BoundaryCondition::Periodic, BoundaryCondition::Open]
    )
    .is_err());
}

#[test]
fn test_shift_on_grid_by_tag_unknown_var() {
    let grid = DiscretizedGrid::builder(&[4])
        .with_variable_names(&["x"])
        .build()
        .unwrap();
    assert!(
        shift_operator_on_grid_by_tag(&grid, &[("z", 1, BoundaryCondition::Periodic)]).is_err()
    );
}

// ============================================================================
// Correctness tests
// ============================================================================

/// Contract an operator's MPO to a dense matrix.
///
/// For n_sites sites each with site_dim, produces a (dim x dim) matrix
/// where dim = site_dim^n_sites.
fn contract_operator_to_dense_matrix(
    op: &QuanticsOperator,
    n_sites: usize,
    site_dim: usize,
) -> Vec<Vec<Complex64>> {
    let dim: usize = site_dim.pow(n_sites as u32);

    let dense_tensor = op.mpo.contract_to_tensor().expect("Failed to contract MPO");

    let ext_indices = &dense_tensor.indices;
    let data = dense_tensor
        .to_vec::<Complex64>()
        .expect("Expected DenseC64 storage");
    let tensor_dims = dense_tensor.dims();
    let ndims = tensor_dims.len();

    let mut id_to_pos: HashMap<DynId, usize> = HashMap::new();
    for (pos, idx) in ext_indices.iter().enumerate() {
        id_to_pos.insert(*idx.id(), pos);
    }

    let input_positions: Vec<usize> = (0..n_sites)
        .map(|i| {
            let internal_id = *op
                .get_input_mapping(&i)
                .expect("Missing input mapping")
                .internal_index
                .id();
            *id_to_pos.get(&internal_id).expect("Input index not found")
        })
        .collect();

    let output_positions: Vec<usize> = (0..n_sites)
        .map(|i| {
            let internal_id = *op
                .get_output_mapping(&i)
                .expect("Missing output mapping")
                .internal_index
                .id();
            *id_to_pos.get(&internal_id).expect("Output index not found")
        })
        .collect();

    let mut matrix = vec![vec![Complex64::zero(); dim]; dim];

    for out_idx in 0..dim {
        for in_idx in 0..dim {
            let mut multi_idx = vec![0usize; ndims];

            let mut out_remainder = out_idx;
            let mut in_remainder = in_idx;
            for i in 0..n_sites {
                let stride = site_dim.pow((n_sites - 1 - i) as u32);
                multi_idx[output_positions[i]] = out_remainder / stride;
                out_remainder %= stride;
                multi_idx[input_positions[i]] = in_remainder / stride;
                in_remainder %= stride;
            }

            let mut flat_idx = 0;
            let mut stride = 1;
            for i in 0..ndims {
                flat_idx += multi_idx[i] * stride;
                stride *= tensor_dims[i];
            }

            matrix[out_idx][in_idx] = data[flat_idx];
        }
    }

    matrix
}

/// Convert grouped flat index (x_val, y_val, ...) to the flat index
/// used in the dense matrix representation.
///
/// For grouped layout with variable resolutions [R_x, R_y, ...],
/// the total sites are R_x + R_y + ..., each with site_dim=2.
/// The flat index encodes the binary digits in site order:
/// first R_x bits for x (big-endian), then R_y bits for y, etc.
fn grouped_flat_index(values: &[usize], rs: &[usize]) -> usize {
    let total_sites: usize = rs.iter().sum();
    let mut idx = 0;

    let mut site = 0;
    for (d, &r) in rs.iter().enumerate() {
        for bit_pos in 0..r {
            let bit = (values[d] >> (r - 1 - bit_pos)) & 1;
            idx |= bit << (total_sites - 1 - site);
            site += 1;
        }
    }

    idx
}

#[test]
fn test_shift_on_grid_grouped_1d_correctness() {
    let r = 4;
    let n: usize = 1 << r;

    let grid = DiscretizedGrid::builder(&[r])
        .with_variable_names(&["x"])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();

    for offset in [-3i64, -1, 0, 1, 3, 7] {
        let op = shift_operator_on_grid(&grid, &[offset], &[BoundaryCondition::Periodic]).unwrap();
        let matrix = contract_operator_to_dense_matrix(&op, r, 2);

        for x in 0..n {
            let input_idx = grouped_flat_index(&[x], &[r]);
            let expected_y = ((x as i64 + offset).rem_euclid(n as i64)) as usize;
            let expected_idx = grouped_flat_index(&[expected_y], &[r]);

            for y_idx in 0..n {
                let expected_val = if y_idx == expected_idx {
                    Complex64::one()
                } else {
                    Complex64::zero()
                };
                let actual = matrix[y_idx][input_idx];
                assert!(
                    (actual - expected_val).norm() < 1e-10,
                    "1D grouped shift offset={}: x={} y_idx={} got ({:.6},{:.6}) expected ({:.6},{:.6})",
                    offset, x, y_idx, actual.re, actual.im, expected_val.re, expected_val.im
                );
            }
        }
    }
}

#[test]
fn test_shift_on_grid_grouped_2d_correctness() {
    let r_x = 3;
    let r_y = 2;
    let n_x: usize = 1 << r_x;
    let n_y: usize = 1 << r_y;
    let total_sites = r_x + r_y;
    let total_dim: usize = 1 << total_sites;

    let grid = DiscretizedGrid::builder(&[r_x, r_y])
        .with_variable_names(&["x", "y"])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();

    // Test shift only x by +1
    let op = shift_operator_on_grid(
        &grid,
        &[1, 0],
        &[BoundaryCondition::Periodic, BoundaryCondition::Periodic],
    )
    .unwrap();

    let matrix = contract_operator_to_dense_matrix(&op, total_sites, 2);

    for x in 0..n_x {
        for y in 0..n_y {
            let input_idx = grouped_flat_index(&[x, y], &[r_x, r_y]);
            let expected_x = ((x as i64 + 1) % n_x as i64) as usize;
            let expected_idx = grouped_flat_index(&[expected_x, y], &[r_x, r_y]);

            for out_idx in 0..total_dim {
                let expected_val = if out_idx == expected_idx {
                    Complex64::one()
                } else {
                    Complex64::zero()
                };
                let actual = matrix[out_idx][input_idx];
                assert!(
                    (actual - expected_val).norm() < 1e-10,
                    "2D grouped shift x+1: x={} y={} out_idx={} got ({:.6},{:.6}) expected ({:.6},{:.6})",
                    x, y, out_idx, actual.re, actual.im, expected_val.re, expected_val.im
                );
            }
        }
    }

    // Test shift only y by +2
    let op2 = shift_operator_on_grid(
        &grid,
        &[0, 2],
        &[BoundaryCondition::Periodic, BoundaryCondition::Periodic],
    )
    .unwrap();

    let matrix2 = contract_operator_to_dense_matrix(&op2, total_sites, 2);

    for x in 0..n_x {
        for y in 0..n_y {
            let input_idx = grouped_flat_index(&[x, y], &[r_x, r_y]);
            let expected_y = ((y as i64 + 2) % n_y as i64) as usize;
            let expected_idx = grouped_flat_index(&[x, expected_y], &[r_x, r_y]);

            for out_idx in 0..total_dim {
                let expected_val = if out_idx == expected_idx {
                    Complex64::one()
                } else {
                    Complex64::zero()
                };
                let actual = matrix2[out_idx][input_idx];
                assert!(
                    (actual - expected_val).norm() < 1e-10,
                    "2D grouped shift y+2: x={} y={} out_idx={} got ({:.6},{:.6}) expected ({:.6},{:.6})",
                    x, y, out_idx, actual.re, actual.im, expected_val.re, expected_val.im
                );
            }
        }
    }

    // Test simultaneous shift: x+1, y-1
    let op3 = shift_operator_on_grid(
        &grid,
        &[1, -1],
        &[BoundaryCondition::Periodic, BoundaryCondition::Periodic],
    )
    .unwrap();

    let matrix3 = contract_operator_to_dense_matrix(&op3, total_sites, 2);

    for x in 0..n_x {
        for y in 0..n_y {
            let input_idx = grouped_flat_index(&[x, y], &[r_x, r_y]);
            let expected_x = ((x as i64 + 1).rem_euclid(n_x as i64)) as usize;
            let expected_y = ((y as i64 - 1).rem_euclid(n_y as i64)) as usize;
            let expected_idx = grouped_flat_index(&[expected_x, expected_y], &[r_x, r_y]);

            for out_idx in 0..total_dim {
                let expected_val = if out_idx == expected_idx {
                    Complex64::one()
                } else {
                    Complex64::zero()
                };
                let actual = matrix3[out_idx][input_idx];
                assert!(
                    (actual - expected_val).norm() < 1e-10,
                    "2D grouped shift x+1,y-1: x={} y={} out_idx={} got ({:.6},{:.6}) expected ({:.6},{:.6})",
                    x, y, out_idx, actual.re, actual.im, expected_val.re, expected_val.im
                );
            }
        }
    }
}

/// Convert fused flat index (x_val, y_val, ...) to the flat index
/// used in the dense matrix representation.
///
/// For fused layout with nvariables all at resolution r,
/// each site has dim = 2^nvariables, and there are r sites.
/// The flat index encodes site-by-site with variable bits interleaved
/// within each site: idx = sum_n s_n * (2^nvars)^(R-1-n)
/// where s_n = sum_v bit_v(n) * 2^v
fn fused_flat_index(values: &[usize], nvariables: usize, r: usize) -> usize {
    let site_dim: usize = 1 << nvariables;
    let mut idx = 0;
    for n in 0..r {
        let mut s = 0usize;
        for v in 0..nvariables {
            let bit = (values[v] >> (r - 1 - n)) & 1;
            s |= bit << v;
        }
        idx = idx * site_dim + s;
    }
    idx
}

#[test]
fn test_shift_on_grid_fused_2d_correctness() {
    let r = 3;
    let n: usize = 1 << r;
    let nvariables = 2;

    let grid = DiscretizedGrid::builder(&[r, r])
        .with_variable_names(&["x", "y"])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .unwrap();

    // Test shift only x by +1
    let op = shift_operator_on_grid(
        &grid,
        &[1, 0],
        &[BoundaryCondition::Periodic, BoundaryCondition::Periodic],
    )
    .unwrap();

    let site_dim: usize = 1 << nvariables;
    let total_dim = site_dim.pow(r as u32);
    let matrix = contract_operator_to_dense_matrix(&op, r, site_dim);

    for x in 0..n {
        for y in 0..n {
            let input_idx = fused_flat_index(&[x, y], nvariables, r);
            let expected_x = ((x as i64 + 1) % n as i64) as usize;
            let expected_idx = fused_flat_index(&[expected_x, y], nvariables, r);

            for out_idx in 0..total_dim {
                let expected_val = if out_idx == expected_idx {
                    Complex64::one()
                } else {
                    Complex64::zero()
                };
                let actual = matrix[out_idx][input_idx];
                assert!(
                    (actual - expected_val).norm() < 1e-10,
                    "Fused 2D shift x+1: x={} y={} out_idx={} got ({:.6},{:.6}) expected ({:.6},{:.6})",
                    x, y, out_idx, actual.re, actual.im, expected_val.re, expected_val.im
                );
            }
        }
    }
}

#[test]
fn test_shift_on_grid_fused_2d_both_vars_correctness() {
    let r = 3;
    let n: usize = 1 << r;
    let nvariables = 2;

    let grid = DiscretizedGrid::builder(&[r, r])
        .with_variable_names(&["x", "y"])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .unwrap();

    // Test shift both variables: x+2, y-1
    let op = shift_operator_on_grid(
        &grid,
        &[2, -1],
        &[BoundaryCondition::Periodic, BoundaryCondition::Periodic],
    )
    .unwrap();

    let site_dim: usize = 1 << nvariables;
    let total_dim = site_dim.pow(r as u32);
    let matrix = contract_operator_to_dense_matrix(&op, r, site_dim);

    for x in 0..n {
        for y in 0..n {
            let input_idx = fused_flat_index(&[x, y], nvariables, r);
            let expected_x = ((x as i64 + 2).rem_euclid(n as i64)) as usize;
            let expected_y = ((y as i64 - 1).rem_euclid(n as i64)) as usize;
            let expected_idx = fused_flat_index(&[expected_x, expected_y], nvariables, r);

            for out_idx in 0..total_dim {
                let expected_val = if out_idx == expected_idx {
                    Complex64::one()
                } else {
                    Complex64::zero()
                };
                let actual = matrix[out_idx][input_idx];
                assert!(
                    (actual - expected_val).norm() < 1e-10,
                    "Fused 2D shift x+2,y-1: x={} y={} out_idx={} got ({:.6},{:.6}) expected ({:.6},{:.6})",
                    x, y, out_idx, actual.re, actual.im, expected_val.re, expected_val.im
                );
            }
        }
    }
}

#[test]
fn test_shift_on_grid_by_tag_correctness() {
    let r_x = 3;
    let r_y = 2;
    let n_x: usize = 1 << r_x;
    let n_y: usize = 1 << r_y;
    let total_sites = r_x + r_y;
    let total_dim: usize = 1 << total_sites;

    let grid = DiscretizedGrid::builder(&[r_x, r_y])
        .with_variable_names(&["x", "y"])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();

    // Shift only y by +1 using tag
    let op =
        shift_operator_on_grid_by_tag(&grid, &[("y", 1, BoundaryCondition::Periodic)]).unwrap();

    let matrix = contract_operator_to_dense_matrix(&op, total_sites, 2);

    for x in 0..n_x {
        for y in 0..n_y {
            let input_idx = grouped_flat_index(&[x, y], &[r_x, r_y]);
            let expected_y = ((y as i64 + 1) % n_y as i64) as usize;
            let expected_idx = grouped_flat_index(&[x, expected_y], &[r_x, r_y]);

            for out_idx in 0..total_dim {
                let expected_val = if out_idx == expected_idx {
                    Complex64::one()
                } else {
                    Complex64::zero()
                };
                let actual = matrix[out_idx][input_idx];
                assert!(
                    (actual - expected_val).norm() < 1e-10,
                    "By-tag shift y+1: x={} y={} out_idx={} got ({:.6},{:.6}) expected ({:.6},{:.6})",
                    x, y, out_idx, actual.re, actual.im, expected_val.re, expected_val.im
                );
            }
        }
    }
}

// ============================================================================
// affine_operator_on_grid tests
// ============================================================================

#[test]
fn test_affine_on_grid_1d() {
    let grid = DiscretizedGrid::builder(&[4])
        .with_variable_names(&["x"])
        .with_lower_bound(&[0.0])
        .with_upper_bound(&[1.0])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();
    let params = crate::AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
    let op = affine_operator_on_grid(&grid, &params, &[BoundaryCondition::Periodic]).unwrap();
    assert_eq!(op.mpo.node_count(), 4);
}

#[test]
fn test_affine_on_grid_fused_2d() {
    let grid = DiscretizedGrid::builder(&[3, 3])
        .with_variable_names(&["x", "y"])
        .with_lower_bound(&[0.0, 0.0])
        .with_upper_bound(&[1.0, 1.0])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .unwrap();
    let params = crate::AffineParams::from_integers(vec![1, 0, 0, 1], vec![0, 0], 2, 2).unwrap();
    let op = affine_operator_on_grid(
        &grid,
        &params,
        &[BoundaryCondition::Periodic, BoundaryCondition::Periodic],
    )
    .unwrap();
    assert_eq!(op.mpo.node_count(), 3);
}

#[test]
fn test_affine_on_grid_grouped_2d() {
    let grid = DiscretizedGrid::builder(&[3, 3])
        .with_variable_names(&["x", "y"])
        .with_lower_bound(&[0.0, 0.0])
        .with_upper_bound(&[1.0, 1.0])
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .build()
        .unwrap();
    let params = crate::AffineParams::from_integers(vec![1, 0, 0, 1], vec![0, 0], 2, 2).unwrap();
    let op = affine_operator_on_grid(
        &grid,
        &params,
        &[BoundaryCondition::Periodic, BoundaryCondition::Periodic],
    )
    .unwrap();
    // Grouped with equal Rs delegates to affine_operator(r=3, ...)
    // which produces fused-form with 3 sites.
    assert_eq!(op.mpo.node_count(), 3);
}

#[test]
fn test_affine_on_grid_interleaved_returns_error() {
    let grid = DiscretizedGrid::builder(&[3, 3])
        .with_variable_names(&["x", "y"])
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .build()
        .unwrap();
    let params = crate::AffineParams::from_integers(vec![1, 0, 0, 1], vec![0, 0], 2, 2).unwrap();
    let result = affine_operator_on_grid(
        &grid,
        &params,
        &[BoundaryCondition::Periodic, BoundaryCondition::Periodic],
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("not yet implemented"));
}
