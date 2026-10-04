use super::*;

#[test]
fn test_tensor4_zeros() {
    let t: Tensor4<f64> = tensor4_zeros(2, 3, 4, 5);
    assert_eq!(t.left_dim(), 2);
    assert_eq!(t.site_dim_1(), 3);
    assert_eq!(t.site_dim_2(), 4);
    assert_eq!(t.right_dim(), 5);

    for l in 0..2 {
        for s1 in 0..3 {
            for s2 in 0..4 {
                for r in 0..5 {
                    assert_eq!(*t.get4(l, s1, s2, r), 0.0);
                }
            }
        }
    }
}

#[test]
#[allow(clippy::approx_constant)]
fn test_tensor4_get_set() {
    let mut t: Tensor4<f64> = tensor4_zeros(2, 2, 2, 2);
    t.set4(0, 1, 0, 1, 3.14);
    assert_eq!(*t.get4(0, 1, 0, 1), 3.14);
    assert_eq!(*t.get4(0, 0, 0, 0), 0.0);
}

#[test]
fn test_tensor4_from_data() {
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let t = tensor4_from_data(data, 2, 3, 2, 2).unwrap();

    assert_eq!(t.left_dim(), 2);
    assert_eq!(t.site_dim_1(), 3);
    assert_eq!(t.site_dim_2(), 2);
    assert_eq!(t.right_dim(), 2);

    // Check some values
    assert_eq!(*t.get4(0, 0, 0, 0), 0.0);
    assert_eq!(*t.get4(1, 0, 0, 0), 1.0);
    assert_eq!(*t.get4(0, 1, 0, 0), 2.0);
    assert_eq!(*t.get4(0, 0, 1, 0), 6.0);
    assert_eq!(*t.get4(0, 0, 0, 1), 12.0);
    assert_eq!(*t.get4(1, 2, 1, 1), 23.0);
}

#[test]
fn tensor4_from_data_rejects_length_mismatch() {
    let err = tensor4_from_data(vec![1.0, 2.0], 1, 2, 2, 1).unwrap_err();

    assert!(err.to_string().contains("expected 4 elements"));
    assert!(err.to_string().contains("got 2"));
}

#[test]
fn test_slice_site() {
    let mut t: Tensor4<f64> = tensor4_zeros(2, 2, 2, 3);
    for l in 0..2 {
        for r in 0..3 {
            t.set4(l, 1, 0, r, (l * 3 + r) as f64);
        }
    }

    let slice = t.slice_site(1, 0);
    assert_eq!(slice.len(), 6); // 2 * 3
    assert_eq!(slice[0], 0.0); // l=0, r=0
    assert_eq!(slice[1], 3.0); // l=1, r=0
    assert_eq!(slice[2], 1.0); // l=0, r=1
    assert_eq!(slice[3], 4.0); // l=1, r=1
}

#[test]
fn test_as_left_matrix() {
    let t: Tensor4<f64> =
        tensor4_from_data((0..24).map(|x| x as f64).collect(), 2, 3, 2, 2).unwrap();
    let (left_dim, site_dim_1, site_dim_2) = (2usize, 3usize, 2usize);

    let (mat, rows, cols) = t.as_left_matrix();
    assert_eq!(rows, left_dim * site_dim_1 * site_dim_2);
    assert_eq!(cols, 2);
    assert_eq!(mat.len(), 24);

    // Fused row index `s2 + site_dim_2 * (s1 + site_dim_1 * left)`, so
    // `(left, s1, s2, right)` sits at `row + rows * right`; the tensor's own
    // flat order is `left + left_dim * (s1 + site_dim_1 * (s2 + site_dim_2 * right))`.
    let flat = |l: usize, s1: usize, s2: usize, r: usize| {
        (l + left_dim * (s1 + site_dim_1 * (s2 + site_dim_2 * r))) as f64
    };
    let at = |l: usize, s1: usize, s2: usize, r: usize| {
        (s2 + site_dim_2 * (s1 + site_dim_1 * l)) + rows * r
    };
    for (l, s1, s2, r) in [(0, 0, 0, 0), (1, 0, 0, 0), (1, 2, 1, 1), (0, 1, 0, 1)] {
        assert_eq!(
            mat[at(l, s1, s2, r)],
            flat(l, s1, s2, r),
            "({l},{s1},{s2},{r})"
        );
    }
    // A column-major reshape of `(left, s1, s2)` would put `(1, 0, 0)` at row 1.
    assert_eq!(at(0, 0, 1, 0), 1);
    assert_ne!(at(1, 0, 0, 0), 1);
}

#[test]
fn test_as_right_matrix() {
    let t: Tensor4<f64> =
        tensor4_from_data((0..24).map(|x| x as f64).collect(), 2, 3, 2, 2).unwrap();
    let (left_dim, site_dim_1, site_dim_2, right_dim) = (2usize, 3usize, 2usize, 2usize);

    let (mat, rows, cols) = t.as_right_matrix();
    assert_eq!(rows, left_dim);
    assert_eq!(cols, site_dim_1 * site_dim_2 * right_dim);
    assert_eq!(mat.len(), 24);

    // Fused column index `right + right_dim * (s2 + site_dim_2 * s1)`.
    let flat = |l: usize, s1: usize, s2: usize, r: usize| {
        (l + left_dim * (s1 + site_dim_1 * (s2 + site_dim_2 * r))) as f64
    };
    let column = |s1: usize, s2: usize, r: usize| r + right_dim * (s2 + site_dim_2 * s1);
    let at = |l: usize, s1: usize, s2: usize, r: usize| l + rows * column(s1, s2, r);
    for (l, s1, s2, r) in [(0, 0, 0, 0), (1, 0, 0, 0), (1, 2, 1, 1), (0, 1, 0, 1)] {
        assert_eq!(
            mat[at(l, s1, s2, r)],
            flat(l, s1, s2, r),
            "({l},{s1},{s2},{r})"
        );
    }
    assert_eq!(column(0, 0, 1), 1);
    assert_ne!(column(1, 0, 0), 1);
}

#[test]
fn test_as_center_matrix() {
    let t: Tensor4<f64> =
        tensor4_from_data((0..24).map(|x| x as f64).collect(), 2, 3, 2, 2).unwrap();
    let (left_dim, site_dim_1, site_dim_2, right_dim) = (2usize, 3usize, 2usize, 2usize);

    let (mat, rows, cols) = t.as_center_matrix();
    assert_eq!(rows, left_dim * site_dim_1);
    assert_eq!(cols, site_dim_2 * right_dim);
    assert_eq!(mat.len(), 24);

    // Row `s1 + site_dim_1 * left`, column `right + right_dim * s2`.
    let flat = |l: usize, s1: usize, s2: usize, r: usize| {
        (l + left_dim * (s1 + site_dim_1 * (s2 + site_dim_2 * r))) as f64
    };
    let at = |l: usize, s1: usize, s2: usize, r: usize| {
        (s1 + site_dim_1 * l) + rows * (r + right_dim * s2)
    };
    for (l, s1, s2, r) in [(0, 0, 0, 0), (1, 0, 0, 0), (1, 2, 1, 1), (0, 1, 0, 1)] {
        assert_eq!(
            mat[at(l, s1, s2, r)],
            flat(l, s1, s2, r),
            "({l},{s1},{s2},{r})"
        );
    }
    assert_ne!(at(1, 0, 0, 0), 1);
    assert_eq!(at(0, 1, 0, 0), 1);
}
