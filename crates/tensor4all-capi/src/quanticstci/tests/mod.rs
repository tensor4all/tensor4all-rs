use super::*;

// Discrete callback: f(i, j) = i + j (1-indexed)
extern "C" fn discrete_sum_callback(
    indices: *const i64,
    ndims: libc::size_t,
    result: *mut f64,
    _user_data: *mut c_void,
) -> i32 {
    unsafe {
        let sum: i64 = (0..ndims).map(|i| *indices.add(i)).sum();
        *result = sum as f64;
    }
    0
}

// Continuous callback: f(x, y) = x + y
extern "C" fn continuous_sum_callback(
    coords: *const libc::c_double,
    ndims: libc::size_t,
    result: *mut libc::c_double,
    _user_data: *mut c_void,
) -> i32 {
    unsafe {
        let sum: f64 = (0..ndims).map(|i| *coords.add(i)).sum();
        *result = sum;
    }
    0
}

#[test]
fn test_discrete_qtci() {
    let sizes: [libc::size_t; 2] = [4, 4];
    let mut qtci: *mut t4a_qtci_f64 = std::ptr::null_mut();

    let status = t4a_quanticscrossinterpolate_discrete_f64(
        sizes.as_ptr(),
        2,
        discrete_sum_callback,
        std::ptr::null_mut(),
        1e-10,
        0, // unlimited
        100,
        t4a_unfolding_scheme::Fused,
        &mut qtci,
    );

    assert_eq!(status, T4A_SUCCESS);
    assert!(!qtci.is_null());

    // Check rank
    let mut rank: libc::size_t = 0;
    let status = t4a_qtci_f64_rank(qtci, &mut rank);
    assert_eq!(status, T4A_SUCCESS);
    assert!(rank > 0);

    // Evaluate at (3, 4) -> should be 7
    let indices: [i64; 2] = [3, 4];
    let mut value: f64 = 0.0;
    let status = t4a_qtci_f64_evaluate(qtci, indices.as_ptr(), 2, &mut value);
    assert_eq!(status, T4A_SUCCESS);
    assert!((value - 7.0).abs() < 1e-6, "Expected 7.0, got {}", value);

    // Evaluate at (1, 1) -> should be 2
    let indices: [i64; 2] = [1, 1];
    let status = t4a_qtci_f64_evaluate(qtci, indices.as_ptr(), 2, &mut value);
    assert_eq!(status, T4A_SUCCESS);
    assert!((value - 2.0).abs() < 1e-6, "Expected 2.0, got {}", value);

    // Sum should return a finite positive value
    let mut sum: f64 = 0.0;
    let status = t4a_qtci_f64_sum(qtci, &mut sum);
    assert_eq!(status, T4A_SUCCESS);
    assert!(sum.is_finite());
    assert!(sum > 0.0);

    t4a_qtci_f64_release(qtci);
}

#[test]
fn test_continuous_qtci() {
    // Create a 1D discretized grid via the C API
    let rs: [libc::size_t; 1] = [4]; // 2^4 = 16 grid points
    let lower: [f64; 1] = [0.0];
    let upper: [f64; 1] = [1.0];
    let mut grid: *mut t4a_qgrid_disc = std::ptr::null_mut();

    let status = crate::quanticsgrids::t4a_qgrid_disc_new(
        1,
        rs.as_ptr(),
        lower.as_ptr(),
        upper.as_ptr(),
        t4a_unfolding_scheme::Fused,
        &mut grid,
    );
    assert_eq!(status, T4A_SUCCESS);

    let mut qtci: *mut t4a_qtci_f64 = std::ptr::null_mut();
    let status = t4a_quanticscrossinterpolate_f64(
        grid,
        continuous_sum_callback,
        std::ptr::null_mut(),
        1e-10,
        0,
        100,
        &mut qtci,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert!(!qtci.is_null());

    // Check rank
    let mut rank: libc::size_t = 0;
    t4a_qtci_f64_rank(qtci, &mut rank);
    assert!(rank > 0);

    // Integral of f(x) = x from 0 to 1 should be ~0.5
    let mut integral: f64 = 0.0;
    let status = t4a_qtci_f64_integral(qtci, &mut integral);
    assert_eq!(status, T4A_SUCCESS);
    // Discretization introduces some error
    assert!(
        (integral - 0.5).abs() < 0.1,
        "Expected ~0.5, got {}",
        integral
    );

    t4a_qtci_f64_release(qtci);
    crate::quanticsgrids::t4a_qgrid_disc_release(grid);
}

#[test]
fn test_qtci_to_tensor_train() {
    let sizes: [libc::size_t; 2] = [4, 4];
    let mut qtci: *mut t4a_qtci_f64 = std::ptr::null_mut();

    let status = t4a_quanticscrossinterpolate_discrete_f64(
        sizes.as_ptr(),
        2,
        discrete_sum_callback,
        std::ptr::null_mut(),
        1e-10,
        0,
        100,
        t4a_unfolding_scheme::Fused,
        &mut qtci,
    );
    assert_eq!(status, T4A_SUCCESS);

    let tt = t4a_qtci_f64_to_tensor_train(qtci);
    assert!(!tt.is_null());

    // Check that the tensor train has correct number of sites
    let mut len: libc::size_t = 0;
    crate::simplett::t4a_simplett_f64_len(tt, &mut len);
    assert!(len > 0);

    crate::simplett::t4a_simplett_f64_release(tt);
    t4a_qtci_f64_release(qtci);
}

#[test]
fn test_null_pointer_guards() {
    let mut rank: libc::size_t = 0;
    assert_eq!(
        t4a_qtci_f64_rank(std::ptr::null(), &mut rank),
        T4A_NULL_POINTER
    );

    let mut value: f64 = 0.0;
    assert_eq!(
        t4a_qtci_f64_sum(std::ptr::null(), &mut value),
        T4A_NULL_POINTER
    );

    assert_eq!(
        t4a_qtci_f64_integral(std::ptr::null(), &mut value),
        T4A_NULL_POINTER
    );

    let indices: [i64; 2] = [1, 1];
    assert_eq!(
        t4a_qtci_f64_evaluate(std::ptr::null(), indices.as_ptr(), 2, &mut value),
        T4A_NULL_POINTER
    );

    let tt = t4a_qtci_f64_to_tensor_train(std::ptr::null());
    assert!(tt.is_null());

    // Release null should not crash
    t4a_qtci_f64_release(std::ptr::null_mut());

    // Constructor null checks
    let mut qtci: *mut t4a_qtci_f64 = std::ptr::null_mut();
    assert_eq!(
        t4a_quanticscrossinterpolate_discrete_f64(
            std::ptr::null(),
            2,
            discrete_sum_callback,
            std::ptr::null_mut(),
            1e-10,
            0,
            100,
            t4a_unfolding_scheme::Fused,
            &mut qtci,
        ),
        T4A_NULL_POINTER
    );
}
