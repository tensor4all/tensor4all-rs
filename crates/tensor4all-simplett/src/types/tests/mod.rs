use super::*;

#[test]
fn test_tensor3_zeros() {
    let t: Tensor3<f64> = tensor3_zeros(2, 3, 4);
    assert_eq!(t.left_dim(), 2);
    assert_eq!(t.site_dim(), 3);
    assert_eq!(t.right_dim(), 4);

    for l in 0..2 {
        for s in 0..3 {
            for r in 0..4 {
                assert_eq!(*t.get3(l, s, r), 0.0);
            }
        }
    }
}

#[test]
fn test_try_tensor3_zeros_rejects_shape_overflow() {
    let error = try_tensor3_zeros::<f64>(usize::MAX, 2, 1).unwrap_err();
    assert!(error.to_string().contains("overflows usize"));
}

#[test]
fn test_tensor3_from_data() {
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let t = tensor3_from_data(data, 2, 3, 4).unwrap();

    assert_eq!(t.left_dim(), 2);
    assert_eq!(t.site_dim(), 3);
    assert_eq!(t.right_dim(), 4);

    assert_eq!(*t.get3(0, 0, 0), 0.0);
    assert_eq!(*t.get3(1, 0, 0), 1.0);
    assert_eq!(*t.get3(0, 1, 0), 2.0);
    assert_eq!(*t.get3(0, 0, 1), 6.0);
    assert_eq!(*t.get3(0, 0, 3), 18.0);
    assert_eq!(*t.get3(1, 2, 3), 23.0);
}

#[test]
fn tensor3_from_data_rejects_length_mismatch() {
    let err = tensor3_from_data(vec![1.0, 2.0], 2, 2, 1).unwrap_err();

    assert!(err.to_string().contains("expected 4 elements"));
    assert!(err.to_string().contains("got 2"));
}

#[test]
fn test_get3_set3_get3_mut() {
    let mut t: Tensor3<f64> = tensor3_zeros(2, 3, 4);

    t.set3(1, 2, 3, 42.0);
    assert_eq!(*t.get3(1, 2, 3), 42.0);
    assert_eq!(*t.get3(0, 0, 0), 0.0);

    *t.get3_mut(0, 1, 2) = 7.5;
    assert_eq!(*t.get3(0, 1, 2), 7.5);
}

#[test]
fn test_slice_site() {
    let mut t: Tensor3<f64> = tensor3_zeros(2, 3, 4);
    for l in 0..2 {
        for r in 0..4 {
            t.set3(l, 1, r, (l * 4 + r) as f64);
        }
    }

    let slice = t.slice_site(1);
    assert_eq!(slice.len(), 8); // 2 * 4
    assert_eq!(slice[0], 0.0); // l=0, r=0
    assert_eq!(slice[1], 4.0); // l=1, r=0
    assert_eq!(slice[2], 1.0); // l=0, r=1
    assert_eq!(slice[3], 5.0); // l=1, r=1
    assert_eq!(slice[4], 2.0); // l=0, r=2
    assert_eq!(slice[5], 6.0); // l=1, r=2

    let slice_zero = t.slice_site(0);
    assert!(slice_zero.iter().all(|&v| v == 0.0));

    let fallible_slice = t.try_slice_site(1).unwrap();
    assert_eq!(fallible_slice, slice);
    let error = t.try_slice_site(3).unwrap_err();
    assert!(matches!(
        error,
        SimpleTensorTrainError::IndexOutOfBounds { .. }
    ));
}

#[test]
fn test_as_left_matrix() {
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let t = tensor3_from_data(data, 2, 3, 4).unwrap();
    let (left_dim, site_dim) = (2usize, 3usize);

    let (mat, rows, cols) = t.as_left_matrix();
    assert_eq!(rows, left_dim * site_dim);
    assert_eq!(cols, 4);
    assert_eq!(mat.len(), 24);

    // The fused row index is site-major (`row = site + site_dim * left`), so
    // `(left, site, right)` sits at `row + rows * right`, while the tensor's own
    // flat order is `left + left_dim * (site + site_dim * right)`.
    let flat = |l: usize, site: usize, r: usize| (l + 2 * (site + 3 * r)) as f64;
    let at = |l: usize, site: usize, r: usize| (site + site_dim * l) + rows * r;
    for (l, site, r) in [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, 2, 3),
        (0, 2, 3),
    ] {
        assert_eq!(mat[at(l, site, r)], flat(l, site, r), "({l},{site},{r})");
    }
    // A column-major reshape of `(left, site)` would put `(1, 0, 0)` at row 1;
    // this convention puts `(0, 1, 0)` there instead.
    assert_eq!(at(0, 1, 0), 1);
    assert_ne!(at(1, 0, 0), 1);

    let (fallible_mat, fallible_rows, fallible_cols) = t.try_as_left_matrix().unwrap();
    assert_eq!(
        (fallible_mat, fallible_rows, fallible_cols),
        (mat, rows, cols)
    );
}

#[test]
fn test_as_right_matrix() {
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let t = tensor3_from_data(data, 2, 3, 4).unwrap();
    let (left_dim, site_dim, right_dim) = (2usize, 3usize, 4usize);

    let (mat, rows, cols) = t.as_right_matrix();
    assert_eq!(rows, left_dim);
    assert_eq!(cols, site_dim * right_dim);
    assert_eq!(mat.len(), 24);

    // The fused column index is right-major (`column = right + right_dim * site`),
    // so `(left, site, right)` sits at `left + rows * column`.
    let flat = |l: usize, site: usize, r: usize| (l + left_dim * (site + site_dim * r)) as f64;
    let column = |site: usize, r: usize| r + right_dim * site;
    let at = |l: usize, site: usize, r: usize| l + rows * column(site, r);
    for (l, site, r) in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 2, 3), (0, 2, 3)] {
        assert_eq!(mat[at(l, site, r)], flat(l, site, r), "({l},{site},{r})");
    }
    // A column-major reshape of `(site, right)` would put `(0, 1, 0)` at column
    // 1 (site fastest); this convention puts `(0, 0, 1)` there instead.
    assert_eq!(column(0, 1), 1);
    assert_ne!(column(1, 0), 1);

    let (fallible_mat, fallible_rows, fallible_cols) = t.try_as_right_matrix().unwrap();
    assert_eq!(
        (fallible_mat, fallible_rows, fallible_cols),
        (mat, rows, cols)
    );
}
