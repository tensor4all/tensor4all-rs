use super::*;
use crate::matrix::{from_vec2d, mat_mul};

#[test]
fn test_matrixluci_from_matrix() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
    ]);

    let luci = MatrixLUCI::from_matrix(&m, None).unwrap();
    assert_eq!(luci.nrows(), 3);
    assert_eq!(luci.ncols(), 3);
    assert_eq!(luci.rank(), 3);
}

#[test]
fn test_matrixluci_reconstruct() {
    let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let luci = MatrixLUCI::from_matrix(&m, None).unwrap();
    let approx = luci.to_matrix();

    for i in 0..2 {
        for j in 0..2 {
            let diff = (m[[i, j]] - approx[[i, j]]).abs();
            assert!(
                diff < 1e-10,
                "Reconstruction error at ({}, {}): {}",
                i,
                j,
                diff
            );
        }
    }
}

#[test]
fn test_matrixluci_rank2_iplusj_left_orthogonal() {
    // Pi matrix for f(i,j) = i + j on 4x4 grid
    let m = from_vec2d(vec![
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![3.0, 4.0, 5.0, 6.0],
    ]);

    let opts = RrLUOptions {
        left_orthogonal: true,
        ..Default::default()
    };
    let luci = MatrixLUCI::from_matrix(&m, Some(opts)).unwrap();
    assert_eq!(luci.rank(), 2);

    // Check left() * right() = Pi
    let left = luci.left();
    let right = luci.right();
    let reconstructed = mat_mul(&left, &right);
    for i in 0..4 {
        for j in 0..4 {
            let diff = (m[[i, j]] - reconstructed[[i, j]]).abs();
            assert!(
                diff < 1e-10,
                "Reconstruction error at ({}, {}): expected {} got {} (diff {})",
                i,
                j,
                m[[i, j]],
                reconstructed[[i, j]],
                diff
            );
        }
    }
}

#[test]
fn test_matrixluci_rank_deficient() {
    // Rank-1 matrix
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 6.0],
        vec![3.0, 6.0, 9.0],
    ]);

    let luci = MatrixLUCI::from_matrix(&m, None).unwrap();
    assert_eq!(luci.rank(), 1);
}
