//! Regressions for the pivot-basis invariant of TreeACI commits (#1449).
//!
//! A commit writes its two cores on the existing incoming bonds and assumes
//! that each bond is in the pivot basis: one basis shared by both adjacent
//! cores, with bond axis `k` denoting the `k`-th retained candidate. A CI
//! canonicalization of the whole output between passes kept the dense tensor
//! but re-selected its own pivots, which moved bonds out of that basis; the
//! spine-only return pass then committed against them and never rewrote the
//! off-spine bond at the junction. The run stopped `Converged` at the true
//! ranks with a recorded error of ~3e-16 and a dense relative error of 1.31.

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treeaci::{tree_elementwise, TreeAciOptions, TreeAciTermination};
use tensor4all_treetn::TreeTN;

/// Binary sites per branch.
const LEVELS: usize = 3;
/// Grid points per branch coordinate.
const N: usize = 1 << LEVELS;
/// Rank of the matrix that couples the x and z branches.
const INNER_RANK: usize = 3;
/// Name of the junction node, after the `3 * LEVELS` branch sites.
const CENTRE: usize = 3 * LEVELS;

/// Rank-3 `N x N` matrix `M = A B`, column-major, with the entries of `A` and
/// `B` in `[-1, 1)` from a fixed 64-bit LCG.
fn rank_three_matrix() -> Vec<f64> {
    let mut state: u64 = 4u64.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(12345);
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let a: Vec<f64> = (0..N * INNER_RANK).map(|_| next()).collect();
    let b: Vec<f64> = (0..INNER_RANK * N).map(|_| next()).collect();
    let mut m = vec![0.0; N * N];
    for i in 0..N {
        for j in 0..N {
            m[i + N * j] = (0..INNER_RANK)
                .map(|l| a[i + N * l] * b[l + INNER_RANK * j])
                .sum();
        }
    }
    m
}

/// Exact TreeTN of `g(x, y, z) = M[x, z]` if `y == 0`, else `0`.
///
/// Three branches (x, y, z) of `LEVELS` binary sites join at a centre whose
/// only site has dimension 1, so the centre has degree 3. Site
/// `3 * level + branch` carries bit `LEVELS - 1 - level` of its branch's
/// coordinate, and each branch bond carries the integer prefix read so far.
fn three_branch_input() -> TreeTN<IdxTensor, usize> {
    let matrix = rank_three_matrix();
    let sites: Vec<DynIndex> = (0..CENTRE).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<Vec<DynIndex>> = (0..3)
        .map(|_| {
            (0..LEVELS)
                .map(|level| DynIndex::new_dyn(1 << (level + 1)))
                .collect()
        })
        .collect();
    let mut tensors = Vec::new();
    let mut names = Vec::new();
    for level in 0..LEVELS {
        for branch in 0..3 {
            let in_dim = 1usize << level;
            let out_dim = 1usize << (level + 1);
            // T[in, s, out] = [out == 2 in + s], column-major.
            let mut data = vec![0.0f64; in_dim * 2 * out_dim];
            for i in 0..in_dim {
                for s in 0..2 {
                    data[i + in_dim * (s + 2 * (2 * i + s))] = 1.0;
                }
            }
            let site = sites[3 * level + branch].clone();
            let out = bonds[branch][level].clone();
            let indices = if level == 0 {
                vec![site, out]
            } else {
                vec![bonds[branch][level - 1].clone(), site, out]
            };
            tensors.push(IdxTensor::from_dense(indices, data).unwrap());
            names.push(3 * level + branch);
        }
    }
    // C[x, y, z, 0] = M[x, z] if y == 0, column-major.
    let mut centre = vec![0.0f64; N * N * N];
    for x in 0..N {
        for z in 0..N {
            centre[x + N * N * z] = matrix[x + N * z];
        }
    }
    tensors.push(
        IdxTensor::from_dense(
            vec![
                bonds[0][LEVELS - 1].clone(),
                bonds[1][LEVELS - 1].clone(),
                bonds[2][LEVELS - 1].clone(),
                DynIndex::new_dyn(1),
            ],
            centre,
        )
        .unwrap(),
    );
    names.push(CENTRE);
    TreeTN::from_tensors(tensors, names).unwrap()
}

/// Exact cut rank of the edge whose lower endpoint is `lower`.
///
/// The x and z branches have ranks 2 and 4 inside the branch and rank(M) = 3
/// at the centre; the y branch only selects `y == 0`, a product state of rank 1.
fn exact_cut_rank(lower: usize) -> usize {
    match (lower % 3, lower / 3) {
        (1, _) => 1,
        (_, 0) => 2,
        (_, 1) => 4,
        _ => INNER_RANK,
    }
}

/// With the global Guard off, the local passes alone must reach the exact
/// tree: every edge at its cut rank and the dense output within `tolerance`.
/// Before #1449 this run stopped `Converged` after two passes at the true
/// ranks with a dense relative error of 1.31.
#[test]
fn guard_off_three_branch_run_converges_to_the_exact_tree() {
    let input = three_branch_input();
    let options = TreeAciOptions {
        tolerance: 1e-10,
        rng_seed: 1,
        enable_global_guard: false,
        ..TreeAciOptions::default()
    };
    let result =
        tree_elementwise::<f64, _, _>(|v| v[0], std::slice::from_ref(&input), &options).unwrap();

    assert_eq!(result.termination, TreeAciTermination::Converged);
    assert_eq!(result.diagnostics.edge_ranks.len(), 3 * LEVELS);
    for &(lower, upper, rank) in &result.diagnostics.edge_ranks {
        assert_eq!(rank, exact_cut_rank(lower), "edge {lower}-{upper}");
    }
    let reference = input.to_dense().unwrap();
    let scale = reference.maxabs().unwrap();
    let error = result
        .tree
        .to_dense()
        .unwrap()
        .sub(&reference)
        .unwrap()
        .maxabs()
        .unwrap()
        / scale;
    assert!(
        error <= options.tolerance,
        "dense relative error {error:e} above the tolerance at Converged (recorded {:?})",
        result.max_errors
    );
}
