
use super::*;

#[test]
fn test_matrix_basic() {
    let mut m = zeros::<f64>(3, 3);
    m[[0, 0]] = 1.0;
    m[[1, 1]] = 2.0;
    m[[2, 2]] = 3.0;

    assert_eq!(m[[0, 0]], 1.0);
    assert_eq!(m[[1, 1]], 2.0);
    assert_eq!(m[[2, 2]], 3.0);
}

#[test]
fn test_matrix_transpose() {
    let m = from_vec2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let mt = transpose(&m);

    assert_eq!(nrows(&mt), 3);
    assert_eq!(ncols(&mt), 2);
    assert_eq!(mt[[0, 0]], 1.0);
    assert_eq!(mt[[0, 1]], 4.0);
    assert_eq!(mt[[2, 0]], 3.0);
}

#[test]
fn test_submatrix_argmax() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let (r, c, _) = submatrix_argmax(&m, 0..3, 0..3);
    assert_eq!((r, c), (2, 2));
}

#[test]
fn test_set_diff() {
    let set = vec![1, 2, 3, 4, 5];
    let exclude = vec![2, 4];
    let diff = set_diff(&set, &exclude);
    assert_eq!(diff, vec![1, 3, 5]);
}

#[test]
fn test_mat_mul() {
    let a = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let c = mat_mul(&a, &b);

    assert_eq!(c[[0, 0]], 19.0);
    assert_eq!(c[[0, 1]], 22.0);
    assert_eq!(c[[1, 0]], 43.0);
    assert_eq!(c[[1, 1]], 50.0);
}
