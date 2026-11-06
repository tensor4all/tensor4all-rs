use super::*;
use num_complex::Complex64;

/// `t[l, s, r] = 1 + l + 2 s + 4 r`, column-major (r slowest, l fastest).
fn real_site_tensor() -> tensor4all_simplett::Tensor3<f64> {
    tensor3_from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 2, 2)
        .expect("valid 2x2x2 tensor")
}

/// The same values as [`real_site_tensor`], embedded in `Complex64`.
fn complex_site_tensor() -> tensor4all_simplett::Tensor3<Complex64> {
    tensor3_from_data(
        (1..=8)
            .map(|value| Complex64::new(f64::from(value), 0.0))
            .collect(),
        2,
        2,
        2,
    )
    .expect("valid complex 2x2x2 tensor")
}

#[test]
fn weighted_site_matrix_matches_hand_computed_reduction() {
    // matrix[l, r] = sum_s t[l, s, r] * w[s] with w = [2, 3]:
    //   r = 0: l = 0 -> 1*2 + 3*3 = 11, l = 1 -> 2*2 + 4*3 = 16
    //   r = 1: l = 0 -> 5*2 + 7*3 = 31, l = 1 -> 6*2 + 8*3 = 36
    let matrix = weighted_site_matrix(&real_site_tensor(), &[2.0, 3.0]).unwrap();
    assert_eq!(matrix, vec![11.0, 16.0, 31.0, 36.0]);
}

#[test]
fn weighted_site_matrix_matches_hand_computed_complex_reduction() {
    // Same reduction with w = [i, 1]:
    //   r = 0: l = 0 -> 1*i + 3 = 3 + i,  l = 1 -> 2*i + 4 = 4 + 2i
    //   r = 1: l = 0 -> 5*i + 7 = 7 + 5i, l = 1 -> 6*i + 8 = 8 + 6i
    let matrix = weighted_site_matrix(
        &complex_site_tensor(),
        &[Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0)],
    )
    .unwrap();
    assert_eq!(
        matrix,
        vec![
            Complex64::new(3.0, 1.0),
            Complex64::new(4.0, 2.0),
            Complex64::new(7.0, 5.0),
            Complex64::new(8.0, 6.0),
        ]
    );
}

#[test]
fn absorb_reduced_site_into_right_matches_hand_computed_contraction() {
    // reduced = [[1, 3], [2, 4]] column-major, so
    // result[l, s, r] = sum_k reduced[l, k] * t[k, s, r]:
    //   s = 0, r = 0: l = 0 -> 1*1 + 3*2 = 7,  l = 1 -> 2*1 + 4*2 = 10
    //   s = 1, r = 0: l = 0 -> 1*3 + 3*4 = 15, l = 1 -> 2*3 + 4*4 = 22
    //   s = 0, r = 1: l = 0 -> 1*5 + 3*6 = 23, l = 1 -> 2*5 + 4*6 = 34
    //   s = 1, r = 1: l = 0 -> 1*7 + 3*8 = 31, l = 1 -> 2*7 + 4*8 = 46
    let removed = tensor3_from_data(vec![0.0; 4], 2, 1, 2).expect("valid removed site");
    let absorbed =
        absorb_reduced_site_into_right(&[1.0, 2.0, 3.0, 4.0], &removed, &real_site_tensor())
            .unwrap();

    assert_eq!((absorbed.left_dim(), absorbed.site_dim()), (2, 2));
    assert_eq!(
        absorbed.to_col_major_vec(),
        vec![7.0, 10.0, 15.0, 22.0, 23.0, 34.0, 31.0, 46.0]
    );
}

#[test]
fn absorb_reduced_site_into_left_matches_hand_computed_contraction() {
    // result[l, s] = sum_k t[l, s, k] * w[k] with w = [2, 3], so
    //   (l = 0, s = 0) -> 1*2 + 5*3 = 17, (l = 1, s = 0) -> 2*2 + 6*3 = 22
    //   (l = 0, s = 1) -> 3*2 + 7*3 = 27, (l = 1, s = 1) -> 4*2 + 8*3 = 32
    let absorbed = absorb_reduced_site_into_left(&real_site_tensor(), &[2.0, 3.0]).unwrap();
    assert_eq!(absorbed.left_dim(), 2);
    assert_eq!(absorbed.site_dim(), 2);
    assert_eq!(absorbed.right_dim(), 1);
    assert_eq!(absorbed.to_col_major_vec(), vec![17.0, 22.0, 27.0, 32.0]);
}
