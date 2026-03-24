use super::*;

// Helper to create a 1D DiscretizedGrid via C API
fn create_disc_grid_1d(r: usize) -> *mut t4a_qgrid_disc {
    let rs = [r];
    let lower = [0.0f64];
    let upper = [1.0f64];
    let mut out: *mut t4a_qgrid_disc = std::ptr::null_mut();
    let status = t4a_qgrid_disc_new(
        1,
        rs.as_ptr(),
        lower.as_ptr(),
        upper.as_ptr(),
        t4a_unfolding_scheme::Fused,
        &mut out,
    );
    assert_eq!(status, T4A_SUCCESS);
    assert!(!out.is_null());
    out
}

#[test]
fn test_disc_grid_1d_properties() {
    let grid = create_disc_grid_1d(3);

    // ndims
    let mut ndims: libc::size_t = 0;
    assert_eq!(t4a_qgrid_disc_ndims(grid, &mut ndims), T4A_SUCCESS);
    assert_eq!(ndims, 1);

    // rs
    let mut rs = [0usize; 1];
    assert_eq!(t4a_qgrid_disc_rs(grid, rs.as_mut_ptr(), 1), T4A_SUCCESS);
    assert_eq!(rs[0], 3);

    // lower_bound
    let mut lb = [0.0f64; 1];
    assert_eq!(
        t4a_qgrid_disc_lower_bound(grid, lb.as_mut_ptr(), 1),
        T4A_SUCCESS
    );
    assert!((lb[0] - 0.0).abs() < 1e-10);

    // upper_bound
    let mut ub = [0.0f64; 1];
    assert_eq!(
        t4a_qgrid_disc_upper_bound(grid, ub.as_mut_ptr(), 1),
        T4A_SUCCESS
    );
    assert!((ub[0] - 1.0).abs() < 1e-10);

    // grid_step
    let mut step = [0.0f64; 1];
    assert_eq!(
        t4a_qgrid_disc_grid_step(grid, step.as_mut_ptr(), 1),
        T4A_SUCCESS
    );
    assert!((step[0] - 0.125).abs() < 1e-10); // 1/8

    // local_dims
    let mut dims = [0usize; 8];
    let mut n_out: libc::size_t = 0;
    assert_eq!(
        t4a_qgrid_disc_local_dims(grid, dims.as_mut_ptr(), 8, &mut n_out),
        T4A_SUCCESS
    );
    assert_eq!(n_out, 3);
    assert_eq!(&dims[..3], &[2, 2, 2]);

    t4a_qgrid_disc_release(grid);
}

#[test]
fn test_disc_grid_1d_roundtrip() {
    let grid = create_disc_grid_1d(3);
    let ndims = 1;

    // Test origcoord -> quantics -> origcoord roundtrip
    let coord = [0.5f64];
    let mut quantics = [0i64; 8];
    let mut n_out: libc::size_t = 0;
    assert_eq!(
        t4a_qgrid_disc_origcoord_to_quantics(
            grid,
            coord.as_ptr(),
            ndims,
            quantics.as_mut_ptr(),
            8,
            &mut n_out
        ),
        T4A_SUCCESS
    );

    let mut coord_back = [0.0f64; 1];
    assert_eq!(
        t4a_qgrid_disc_quantics_to_origcoord(
            grid,
            quantics.as_ptr(),
            n_out,
            coord_back.as_mut_ptr(),
            1
        ),
        T4A_SUCCESS
    );
    assert!((coord_back[0] - 0.5).abs() < 1e-10);

    // Test origcoord -> grididx -> origcoord roundtrip
    let mut grididx = [0i64; 1];
    assert_eq!(
        t4a_qgrid_disc_origcoord_to_grididx(grid, coord.as_ptr(), ndims, grididx.as_mut_ptr(), 1),
        T4A_SUCCESS
    );
    assert_eq!(grididx[0], 5); // 1-indexed, 0.5 maps to grid index 5

    let mut coord_back2 = [0.0f64; 1];
    assert_eq!(
        t4a_qgrid_disc_grididx_to_origcoord(
            grid,
            grididx.as_ptr(),
            ndims,
            coord_back2.as_mut_ptr(),
            1
        ),
        T4A_SUCCESS
    );
    assert!((coord_back2[0] - 0.5).abs() < 1e-10);

    t4a_qgrid_disc_release(grid);
}

#[test]
fn test_disc_grid_2d_interleaved() {
    let rs = [3usize, 2];
    let lower = [0.0f64, 0.0];
    let upper = [1.0f64, 1.0];
    let mut out: *mut t4a_qgrid_disc = std::ptr::null_mut();

    let status = t4a_qgrid_disc_new(
        2,
        rs.as_ptr(),
        lower.as_ptr(),
        upper.as_ptr(),
        t4a_unfolding_scheme::Interleaved,
        &mut out,
    );
    assert_eq!(status, T4A_SUCCESS);

    let mut ndims: libc::size_t = 0;
    assert_eq!(t4a_qgrid_disc_ndims(out, &mut ndims), T4A_SUCCESS);
    assert_eq!(ndims, 2);

    // Test roundtrip for a 2D coordinate
    let coord = [0.25f64, 0.5];
    let mut quantics = [0i64; 16];
    let mut n_out: libc::size_t = 0;
    assert_eq!(
        t4a_qgrid_disc_origcoord_to_quantics(
            out,
            coord.as_ptr(),
            2,
            quantics.as_mut_ptr(),
            16,
            &mut n_out
        ),
        T4A_SUCCESS
    );

    let mut coord_back = [0.0f64; 2];
    assert_eq!(
        t4a_qgrid_disc_quantics_to_origcoord(
            out,
            quantics.as_ptr(),
            n_out,
            coord_back.as_mut_ptr(),
            2
        ),
        T4A_SUCCESS
    );
    assert!((coord_back[0] - 0.25).abs() < 1e-10);
    assert!((coord_back[1] - 0.5).abs() < 1e-10);

    t4a_qgrid_disc_release(out);
}

#[test]
fn test_disc_grid_grididx_quantics_roundtrip() {
    let grid = create_disc_grid_1d(3);

    // Test all grid indices roundtrip through quantics
    for idx in 1..=8i64 {
        let grididx = [idx];
        let mut quantics = [0i64; 8];
        let mut n_q: libc::size_t = 0;
        assert_eq!(
            t4a_qgrid_disc_grididx_to_quantics(
                grid,
                grididx.as_ptr(),
                1,
                quantics.as_mut_ptr(),
                8,
                &mut n_q
            ),
            T4A_SUCCESS
        );

        let mut grididx_back = [0i64; 1];
        let mut n_g: libc::size_t = 0;
        assert_eq!(
            t4a_qgrid_disc_quantics_to_grididx(
                grid,
                quantics.as_ptr(),
                n_q,
                grididx_back.as_mut_ptr(),
                1,
                &mut n_g
            ),
            T4A_SUCCESS
        );
        assert_eq!(grididx_back[0], idx);
    }

    t4a_qgrid_disc_release(grid);
}

#[test]
fn test_int_grid_1d_properties() {
    let rs = [3usize];
    let mut out: *mut t4a_qgrid_int = std::ptr::null_mut();
    let status = t4a_qgrid_int_new(
        1,
        rs.as_ptr(),
        std::ptr::null(),
        t4a_unfolding_scheme::Fused,
        &mut out,
    );
    assert_eq!(status, T4A_SUCCESS);

    let mut ndims: libc::size_t = 0;
    assert_eq!(t4a_qgrid_int_ndims(out, &mut ndims), T4A_SUCCESS);
    assert_eq!(ndims, 1);

    let mut rs_out = [0usize; 1];
    assert_eq!(t4a_qgrid_int_rs(out, rs_out.as_mut_ptr(), 1), T4A_SUCCESS);
    assert_eq!(rs_out[0], 3);

    let mut origin = [0i64; 1];
    assert_eq!(
        t4a_qgrid_int_origin(out, origin.as_mut_ptr(), 1),
        T4A_SUCCESS
    );
    assert_eq!(origin[0], 1); // default origin

    t4a_qgrid_int_release(out);
}

#[test]
fn test_int_grid_1d_roundtrip() {
    let rs = [3usize];
    let origin = [0i64];
    let mut out: *mut t4a_qgrid_int = std::ptr::null_mut();
    let status = t4a_qgrid_int_new(
        1,
        rs.as_ptr(),
        origin.as_ptr(),
        t4a_unfolding_scheme::Fused,
        &mut out,
    );
    assert_eq!(status, T4A_SUCCESS);

    // origcoord -> quantics -> origcoord
    let coord = [3i64];
    let mut quantics = [0i64; 8];
    let mut n_out: libc::size_t = 0;
    assert_eq!(
        t4a_qgrid_int_origcoord_to_quantics(
            out,
            coord.as_ptr(),
            1,
            quantics.as_mut_ptr(),
            8,
            &mut n_out
        ),
        T4A_SUCCESS
    );

    let mut coord_back = [0i64; 1];
    assert_eq!(
        t4a_qgrid_int_quantics_to_origcoord(
            out,
            quantics.as_ptr(),
            n_out,
            coord_back.as_mut_ptr(),
            1
        ),
        T4A_SUCCESS
    );
    assert_eq!(coord_back[0], 3);

    // grididx -> quantics -> grididx
    for idx in 1..=8i64 {
        let grididx = [idx];
        let mut q = [0i64; 8];
        let mut nq: libc::size_t = 0;
        assert_eq!(
            t4a_qgrid_int_grididx_to_quantics(out, grididx.as_ptr(), 1, q.as_mut_ptr(), 8, &mut nq),
            T4A_SUCCESS
        );

        let mut g_back = [0i64; 1];
        let mut ng: libc::size_t = 0;
        assert_eq!(
            t4a_qgrid_int_quantics_to_grididx(out, q.as_ptr(), nq, g_back.as_mut_ptr(), 1, &mut ng),
            T4A_SUCCESS
        );
        assert_eq!(g_back[0], idx);
    }

    t4a_qgrid_int_release(out);
}

#[test]
fn test_null_pointer_guards() {
    let mut out: *mut t4a_qgrid_disc = std::ptr::null_mut();
    assert_eq!(
        t4a_qgrid_disc_new(
            1,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            t4a_unfolding_scheme::Fused,
            &mut out
        ),
        T4A_NULL_POINTER
    );

    let mut ndims: libc::size_t = 0;
    assert_eq!(
        t4a_qgrid_disc_ndims(std::ptr::null(), &mut ndims),
        T4A_NULL_POINTER
    );

    let mut out_int: *mut t4a_qgrid_int = std::ptr::null_mut();
    assert_eq!(
        t4a_qgrid_int_new(
            1,
            std::ptr::null(),
            std::ptr::null(),
            t4a_unfolding_scheme::Fused,
            &mut out_int
        ),
        T4A_NULL_POINTER
    );
}

#[test]
fn test_disc_grid_clone() {
    let grid = create_disc_grid_1d(3);
    let cloned = t4a_qgrid_disc_clone(grid);
    assert!(!cloned.is_null());

    let mut ndims: libc::size_t = 0;
    assert_eq!(t4a_qgrid_disc_ndims(cloned, &mut ndims), T4A_SUCCESS);
    assert_eq!(ndims, 1);

    t4a_qgrid_disc_release(grid);
    t4a_qgrid_disc_release(cloned);
}
