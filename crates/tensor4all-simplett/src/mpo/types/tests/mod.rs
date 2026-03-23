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
    let t = tensor4_from_data(data, 2, 3, 2, 2);

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
    assert_eq!(slice[1], 1.0); // l=0, r=1
    assert_eq!(slice[2], 2.0); // l=0, r=2
    assert_eq!(slice[3], 3.0); // l=1, r=0
}

#[test]
fn test_as_left_matrix() {
    let t: Tensor4<f64> = tensor4_from_data((0..24).map(|x| x as f64).collect(), 2, 3, 2, 2);

    let (mat, rows, cols) = t.as_left_matrix();
    assert_eq!(rows, 12); // 2 * 3 * 2
    assert_eq!(cols, 2);
    assert_eq!(mat.len(), 24);
}

#[test]
fn test_as_right_matrix() {
    let t: Tensor4<f64> = tensor4_from_data((0..24).map(|x| x as f64).collect(), 2, 3, 2, 2);

    let (mat, rows, cols) = t.as_right_matrix();
    assert_eq!(rows, 2);
    assert_eq!(cols, 12); // 3 * 2 * 2
    assert_eq!(mat.len(), 24);
}

#[test]
fn test_as_center_matrix() {
    let t: Tensor4<f64> = tensor4_from_data((0..24).map(|x| x as f64).collect(), 2, 3, 2, 2);

    let (mat, rows, cols) = t.as_center_matrix();
    assert_eq!(rows, 6); // 2 * 3
    assert_eq!(cols, 4); // 2 * 2
    assert_eq!(mat.len(), 24);
}
