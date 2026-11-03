use super::*;

#[test]
fn test_opt_first_pivot_finds_nonzero() {
    // Function that is zero at origin but nonzero elsewhere
    let f = |idx: &MultiIndex| (idx[0] as f64 + idx[1] as f64 + 1.0).powi(2);
    let local_dims = vec![4, 4];
    let first_pivot = vec![0, 0]; // f(0,0) = 1.0

    let pivot = opt_first_pivot::<f64, _>(&f, &local_dims, &first_pivot, 1000);
    // Should find the maximum: f(3,3) = 49.0
    assert_eq!(pivot, vec![3, 3]);
}

#[test]
fn test_opt_first_pivot_already_optimal() {
    let f = |idx: &MultiIndex| idx[0] as f64 * idx[1] as f64;
    let local_dims = vec![4, 4];
    let first_pivot = vec![3, 3]; // Already at maximum

    let pivot = opt_first_pivot::<f64, _>(&f, &local_dims, &first_pivot, 1000);
    assert_eq!(pivot, vec![3, 3]);
}

#[test]
fn test_opt_first_pivot_complex() {
    use num_complex::Complex64;
    let f = |idx: &MultiIndex| Complex64::new(idx[0] as f64 * 2.0, idx[1] as f64 * 3.0);
    let local_dims = vec![4, 4];
    let first_pivot = vec![0, 0];

    let pivot = opt_first_pivot::<Complex64, _>(&f, &local_dims, &first_pivot, 1000);
    // Maximum |f| at (3, 3) = |6 + 9i| = sqrt(117)
    assert_eq!(pivot, vec![3, 3]);
}
