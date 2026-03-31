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

// Complex discrete callback: f(i, j) = (i + j) + (i - j)*i
extern "C" fn discrete_complex_callback(
    indices: *const i64,
    ndims: libc::size_t,
    result: *mut libc::c_double, // [re, im]
    _user_data: *mut c_void,
) -> i32 {
    unsafe {
        let sum: i64 = (0..ndims).map(|i| *indices.add(i)).sum();
        let diff: i64 = if ndims >= 2 {
            *indices.add(0) - *indices.add(1)
        } else {
            *indices.add(0)
        };
        *result = sum as f64;
        *result.add(1) = diff as f64;
    }
    0
}

// Complex continuous callback: f(x, y) = (x+y) + (x-y)*i
extern "C" fn continuous_complex_callback(
    coords: *const libc::c_double,
    ndims: libc::size_t,
    result: *mut libc::c_double,
    _user_data: *mut c_void,
) -> i32 {
    unsafe {
        let sum: f64 = (0..ndims).map(|i| *coords.add(i)).sum();
        let x = *coords.add(0);
        let y = if ndims >= 2 { *coords.add(1) } else { 0.0 };
        *result = sum;
        *result.add(1) = x - y;
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
        std::ptr::null(), // no options
        1e-10,
        0, // unlimited
        100,
        t4a_unfolding_scheme::Fused,
        std::ptr::null(),
        0,
        &mut qtci,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
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
    let rs: [libc::size_t; 1] = [4];
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
        std::ptr::null(),
        1e-10,
        0,
        100,
        std::ptr::null(),
        0,
        &mut qtci,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
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
        std::ptr::null(),
        1e-10,
        0,
        100,
        t4a_unfolding_scheme::Fused,
        std::ptr::null(),
        0,
        &mut qtci,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    assert_eq!(status, T4A_SUCCESS);

    let tt = t4a_qtci_f64_to_tensor_train(qtci);
    assert!(!tt.is_null());

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

    t4a_qtci_f64_release(std::ptr::null_mut());

    let mut qtci: *mut t4a_qtci_f64 = std::ptr::null_mut();
    assert_eq!(
        t4a_quanticscrossinterpolate_discrete_f64(
            std::ptr::null(),
            2,
            discrete_sum_callback,
            std::ptr::null_mut(),
            std::ptr::null(),
            1e-10,
            0,
            100,
            t4a_unfolding_scheme::Fused,
            std::ptr::null(),
            0,
            &mut qtci,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ),
        T4A_NULL_POINTER
    );
}

// ============================================================================
// QtciOptions tests
// ============================================================================

#[test]
fn test_qtci_options_default_and_setters() {
    let opts = t4a_qtci_options_default();
    assert!(!opts.is_null());
    assert_eq!(t4a_qtci_options_is_assigned(opts), 1);

    // Set various options
    assert_eq!(t4a_qtci_options_set_tolerance(opts, 1e-6), T4A_SUCCESS);
    assert_eq!(t4a_qtci_options_set_maxbonddim(opts, 50), T4A_SUCCESS);
    assert_eq!(t4a_qtci_options_set_maxiter(opts, 300), T4A_SUCCESS);
    assert_eq!(t4a_qtci_options_set_nrandominitpivot(opts, 10), T4A_SUCCESS);
    assert_eq!(
        t4a_qtci_options_set_unfoldingscheme(opts, t4a_unfolding_scheme::Fused),
        T4A_SUCCESS
    );
    assert_eq!(t4a_qtci_options_set_normalize_error(opts, 0), T4A_SUCCESS);
    assert_eq!(t4a_qtci_options_set_verbosity(opts, 2), T4A_SUCCESS);
    assert_eq!(
        t4a_qtci_options_set_nsearchglobalpivot(opts, 10),
        T4A_SUCCESS
    );
    assert_eq!(t4a_qtci_options_set_nsearch(opts, 200), T4A_SUCCESS);

    // Clone and verify
    let cloned = t4a_qtci_options_clone(opts);
    assert!(!cloned.is_null());

    t4a_qtci_options_release(opts);
    t4a_qtci_options_release(cloned);
}

#[test]
fn test_qtci_options_null_guards() {
    assert_eq!(
        t4a_qtci_options_set_tolerance(std::ptr::null_mut(), 1.0),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_qtci_options_set_maxbonddim(std::ptr::null_mut(), 10),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_qtci_options_set_maxiter(std::ptr::null_mut(), 10),
        T4A_NULL_POINTER
    );
    t4a_qtci_options_release(std::ptr::null_mut());
    assert!(t4a_qtci_options_clone(std::ptr::null()).is_null());
    assert_eq!(t4a_qtci_options_is_assigned(std::ptr::null()), 0);
}

#[test]
fn test_qtci_with_options_handle() {
    let opts = t4a_qtci_options_default();
    assert_eq!(t4a_qtci_options_set_tolerance(opts, 1e-10), T4A_SUCCESS);
    assert_eq!(t4a_qtci_options_set_maxiter(opts, 100), T4A_SUCCESS);
    assert_eq!(
        t4a_qtci_options_set_unfoldingscheme(opts, t4a_unfolding_scheme::Fused),
        T4A_SUCCESS
    );

    let sizes: [libc::size_t; 2] = [4, 4];
    let mut qtci: *mut t4a_qtci_f64 = std::ptr::null_mut();
    let mut n_iters: libc::size_t = 0;

    let status = t4a_quanticscrossinterpolate_discrete_f64(
        sizes.as_ptr(),
        2,
        discrete_sum_callback,
        std::ptr::null_mut(),
        opts,
        0.0, // ignored when options is non-null
        0,
        0,
        t4a_unfolding_scheme::Fused, // ignored when options is non-null
        std::ptr::null(),
        0,
        &mut qtci,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        &mut n_iters,
    );

    assert_eq!(status, T4A_SUCCESS);
    assert!(!qtci.is_null());
    assert!(n_iters > 0);

    t4a_qtci_f64_release(qtci);
    t4a_qtci_options_release(opts);
}

#[test]
fn test_convergence_info_output() {
    let sizes: [libc::size_t; 2] = [4, 4];
    let mut qtci: *mut t4a_qtci_f64 = std::ptr::null_mut();
    let mut ranks = [0usize; 200];
    let mut errors = [0.0f64; 200];
    let mut n_iters: libc::size_t = 0;

    let status = t4a_quanticscrossinterpolate_discrete_f64(
        sizes.as_ptr(),
        2,
        discrete_sum_callback,
        std::ptr::null_mut(),
        std::ptr::null(),
        1e-10,
        0,
        100,
        t4a_unfolding_scheme::Fused,
        std::ptr::null(),
        0,
        &mut qtci,
        ranks.as_mut_ptr(),
        errors.as_mut_ptr(),
        &mut n_iters,
    );

    assert_eq!(status, T4A_SUCCESS);
    assert!(n_iters > 0);
    // First rank should be > 0
    assert!(ranks[0] > 0);

    t4a_qtci_f64_release(qtci);
}

// ============================================================================
// f64 clone test
// ============================================================================

#[test]
fn test_qtci_f64_clone() {
    let sizes: [libc::size_t; 2] = [4, 4];
    let mut qtci: *mut t4a_qtci_f64 = std::ptr::null_mut();

    let status = t4a_quanticscrossinterpolate_discrete_f64(
        sizes.as_ptr(),
        2,
        discrete_sum_callback,
        std::ptr::null_mut(),
        std::ptr::null(),
        1e-10,
        0,
        100,
        t4a_unfolding_scheme::Fused,
        std::ptr::null(),
        0,
        &mut qtci,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    assert_eq!(status, T4A_SUCCESS);

    let cloned = t4a_qtci_f64_clone(qtci);
    assert!(!cloned.is_null());

    // Both should evaluate identically
    let indices: [i64; 2] = [2, 3];
    let mut val1 = 0.0;
    let mut val2 = 0.0;
    assert_eq!(
        t4a_qtci_f64_evaluate(qtci, indices.as_ptr(), 2, &mut val1),
        T4A_SUCCESS
    );
    assert_eq!(
        t4a_qtci_f64_evaluate(cloned, indices.as_ptr(), 2, &mut val2),
        T4A_SUCCESS
    );
    assert!((val1 - val2).abs() < 1e-12);

    t4a_qtci_f64_release(qtci);
    t4a_qtci_f64_release(cloned);
}

// ============================================================================
// c64 discrete QTCI tests
// ============================================================================

#[test]
fn test_discrete_qtci_c64() {
    let sizes: [libc::size_t; 2] = [4, 4];
    let mut qtci: *mut t4a_qtci_c64 = std::ptr::null_mut();

    let status = t4a_quanticscrossinterpolate_discrete_c64(
        sizes.as_ptr(),
        2,
        discrete_complex_callback,
        std::ptr::null_mut(),
        std::ptr::null(),
        1e-10,
        0,
        100,
        t4a_unfolding_scheme::Fused,
        std::ptr::null(),
        0,
        &mut qtci,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );

    assert_eq!(status, T4A_SUCCESS);
    assert!(!qtci.is_null());

    // Check rank
    let mut rank: libc::size_t = 0;
    assert_eq!(t4a_qtci_c64_rank(qtci, &mut rank), T4A_SUCCESS);
    assert!(rank > 0);

    // Evaluate at (3, 4) -> re = 7, im = -1
    let indices: [i64; 2] = [3, 4];
    let (mut re, mut im) = (0.0, 0.0);
    let status = t4a_qtci_c64_evaluate(qtci, indices.as_ptr(), 2, &mut re, &mut im);
    assert_eq!(status, T4A_SUCCESS);
    assert!((re - 7.0).abs() < 1e-6, "Expected re=7.0, got {}", re);
    assert!((im - (-1.0)).abs() < 1e-6, "Expected im=-1.0, got {}", im);

    // Sum should be finite
    let (mut sum_re, mut sum_im) = (0.0, 0.0);
    let status = t4a_qtci_c64_sum(qtci, &mut sum_re, &mut sum_im);
    assert_eq!(status, T4A_SUCCESS);
    assert!(sum_re.is_finite());

    // to_tensor_train
    let tt = t4a_qtci_c64_to_tensor_train(qtci);
    assert!(!tt.is_null());
    crate::simplett::t4a_simplett_c64_release(tt);

    // clone
    let cloned = t4a_qtci_c64_clone(qtci);
    assert!(!cloned.is_null());

    // cloned should evaluate the same
    let (mut re2, mut im2) = (0.0, 0.0);
    assert_eq!(
        t4a_qtci_c64_evaluate(cloned, indices.as_ptr(), 2, &mut re2, &mut im2),
        T4A_SUCCESS
    );
    assert!((re - re2).abs() < 1e-12);
    assert!((im - im2).abs() < 1e-12);

    t4a_qtci_c64_release(cloned);
    t4a_qtci_c64_release(qtci);
}

#[test]
fn test_continuous_qtci_c64() {
    let rs: [libc::size_t; 1] = [4];
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

    let mut qtci: *mut t4a_qtci_c64 = std::ptr::null_mut();
    let status = t4a_quanticscrossinterpolate_c64(
        grid,
        continuous_complex_callback,
        std::ptr::null_mut(),
        std::ptr::null(),
        1e-10,
        0,
        100,
        std::ptr::null(),
        0,
        &mut qtci,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    assert_eq!(status, T4A_SUCCESS);
    assert!(!qtci.is_null());

    let mut rank: libc::size_t = 0;
    t4a_qtci_c64_rank(qtci, &mut rank);
    assert!(rank > 0);

    t4a_qtci_c64_release(qtci);
    crate::quanticsgrids::t4a_qgrid_disc_release(grid);
}

// ============================================================================
// c64 null pointer guards
// ============================================================================

#[test]
fn test_c64_null_pointer_guards() {
    let mut rank: libc::size_t = 0;
    assert_eq!(
        t4a_qtci_c64_rank(std::ptr::null(), &mut rank),
        T4A_NULL_POINTER
    );

    let (mut re, mut im) = (0.0, 0.0);
    assert_eq!(
        t4a_qtci_c64_sum(std::ptr::null(), &mut re, &mut im),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_qtci_c64_integral(std::ptr::null(), &mut re, &mut im),
        T4A_NULL_POINTER
    );

    let indices: [i64; 2] = [1, 1];
    assert_eq!(
        t4a_qtci_c64_evaluate(std::ptr::null(), indices.as_ptr(), 2, &mut re, &mut im),
        T4A_NULL_POINTER
    );

    let tt = t4a_qtci_c64_to_tensor_train(std::ptr::null());
    assert!(tt.is_null());

    t4a_qtci_c64_release(std::ptr::null_mut());
    assert!(t4a_qtci_c64_clone(std::ptr::null()).is_null());
    assert_eq!(t4a_qtci_c64_is_assigned(std::ptr::null()), 0);
}

// ============================================================================
// Category 4: TreeTCI2 state accessor tests
// ============================================================================

#[test]
fn test_qtci_f64_max_bond_error_and_max_rank() {
    let sizes: [libc::size_t; 2] = [4, 4];
    let mut qtci: *mut t4a_qtci_f64 = std::ptr::null_mut();

    let status = t4a_quanticscrossinterpolate_discrete_f64(
        sizes.as_ptr(),
        2,
        discrete_sum_callback,
        std::ptr::null_mut(),
        std::ptr::null(),
        1e-10,
        0,
        100,
        t4a_unfolding_scheme::Fused,
        std::ptr::null(),
        0,
        &mut qtci,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    assert_eq!(status, T4A_SUCCESS);

    let mut max_error = -1.0;
    assert_eq!(
        t4a_qtci_f64_max_bond_error(qtci, &mut max_error),
        T4A_SUCCESS
    );
    assert!(max_error >= 0.0);

    let mut max_rank: libc::size_t = 0;
    assert_eq!(t4a_qtci_f64_max_rank(qtci, &mut max_rank), T4A_SUCCESS);
    assert!(max_rank > 0);

    t4a_qtci_f64_release(qtci);
}

#[test]
fn test_qtci_c64_max_bond_error_and_max_rank() {
    let sizes: [libc::size_t; 2] = [4, 4];
    let mut qtci: *mut t4a_qtci_c64 = std::ptr::null_mut();

    let status = t4a_quanticscrossinterpolate_discrete_c64(
        sizes.as_ptr(),
        2,
        discrete_complex_callback,
        std::ptr::null_mut(),
        std::ptr::null(),
        1e-10,
        0,
        100,
        t4a_unfolding_scheme::Fused,
        std::ptr::null(),
        0,
        &mut qtci,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    assert_eq!(status, T4A_SUCCESS);

    let mut max_error = -1.0;
    assert_eq!(
        t4a_qtci_c64_max_bond_error(qtci, &mut max_error),
        T4A_SUCCESS
    );
    assert!(max_error >= 0.0);

    let mut max_rank: libc::size_t = 0;
    assert_eq!(t4a_qtci_c64_max_rank(qtci, &mut max_rank), T4A_SUCCESS);
    assert!(max_rank > 0);

    t4a_qtci_c64_release(qtci);
}

#[test]
fn test_max_bond_error_null_guards() {
    let mut val = 0.0;
    assert_eq!(
        t4a_qtci_f64_max_bond_error(std::ptr::null(), &mut val),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_qtci_c64_max_bond_error(std::ptr::null(), &mut val),
        T4A_NULL_POINTER
    );
    let mut rank = 0usize;
    assert_eq!(
        t4a_qtci_f64_max_rank(std::ptr::null(), &mut rank),
        T4A_NULL_POINTER
    );
    assert_eq!(
        t4a_qtci_c64_max_rank(std::ptr::null(), &mut rank),
        T4A_NULL_POINTER
    );
}
