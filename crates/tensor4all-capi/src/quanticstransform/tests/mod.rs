use super::*;
use crate::{t4a_index, t4a_treetn_release, T4A_INVALID_ARGUMENT, T4A_SUCCESS};
use num_complex::Complex64;
use num_rational::Rational64;
use tensor4all_core::{ColMajorArrayRef, TensorDynLen};
use tensor4all_quanticstransform::{
    affine_operator, affine_pullback_operator, binaryop_operator, cumsum_operator, flip_operator,
    phase_rotation_operator, quantics_fourier_operator, shift_operator, AffineParams, BinaryCoeffs,
    BoundaryCondition, FourierOptions,
};
use tensor4all_treetn::LinearOperator;

fn new_layout(kind: t4a_qtt_layout_kind, resolutions: &[usize]) -> *mut t4a_qtt_layout {
    let mut out = std::ptr::null_mut();
    assert_eq!(
        t4a_qtt_layout_new(kind, resolutions.len(), resolutions.as_ptr(), &mut out),
        T4A_SUCCESS
    );
    assert!(!out.is_null());
    out
}

fn last_error() -> String {
    let mut len = 0usize;
    assert_eq!(
        crate::t4a_last_error_message(std::ptr::null_mut(), 0, &mut len),
        T4A_SUCCESS
    );
    let mut buf = vec![0u8; len];
    assert_eq!(
        crate::t4a_last_error_message(buf.as_mut_ptr(), buf.len(), &mut len),
        T4A_SUCCESS
    );
    std::ffi::CStr::from_bytes_until_nul(&buf)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
}

fn decode_mixed_radix(mut flat: usize, dims: &[usize]) -> Vec<usize> {
    let mut values = Vec::with_capacity(dims.len());
    for &dim in dims {
        values.push(flat % dim);
        flat /= dim;
    }
    values
}

fn encode_mixed_radix(values: &[usize], dims: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for (&value, &dim) in values.iter().zip(dims.iter()) {
        flat += value * stride;
        stride *= dim;
    }
    flat
}

fn rust_operator_matrix(
    op: &LinearOperator<TensorDynLen, usize>,
    out_dims: &[usize],
    in_dims: &[usize],
) -> Vec<Complex64> {
    let nsites = out_dims.len();
    assert_eq!(in_dims.len(), nsites);
    let nrows: usize = out_dims.iter().product();
    let ncols: usize = in_dims.iter().product();

    let mut indices = Vec::with_capacity(2 * nsites);
    for site in 0..nsites {
        indices.push(op.output_mapping.get(&site).unwrap().internal_index.clone());
        indices.push(op.input_mapping.get(&site).unwrap().internal_index.clone());
    }

    let mut matrix = vec![Complex64::new(0.0, 0.0); nrows * ncols];
    for x in 0..ncols {
        let in_values = decode_mixed_radix(x, in_dims);
        for y in 0..nrows {
            let out_values = decode_mixed_radix(y, out_dims);
            let mut values = Vec::with_capacity(2 * nsites);
            for site in 0..nsites {
                values.push(out_values[site]);
                values.push(in_values[site]);
            }
            let shape = [values.len(), 1];
            let result = op
                .mpo
                .evaluate_at(&indices, ColMajorArrayRef::new(&values, &shape))
                .unwrap();
            matrix[y + nrows * x] = result[0].clone().into();
        }
    }
    matrix
}

fn c_operator_matrix(
    op: *const t4a_treetn,
    out_dims: &[usize],
    in_dims: &[usize],
) -> Vec<Complex64> {
    let nsites = out_dims.len();
    assert_eq!(in_dims.len(), nsites);

    let mut n_vertices = 0usize;
    assert_eq!(
        crate::treetn::t4a_treetn_num_vertices(op, &mut n_vertices),
        T4A_SUCCESS
    );
    assert_eq!(n_vertices, nsites);

    let mut out_indices = Vec::with_capacity(nsites);
    let mut in_indices = Vec::with_capacity(nsites);
    for site in 0..nsites {
        let mut len = 0usize;
        assert_eq!(
            crate::treetn::t4a_treetn_siteinds(op, site, std::ptr::null_mut(), 0, &mut len),
            T4A_SUCCESS
        );
        assert_eq!(len, 2);
        let mut pair = vec![std::ptr::null_mut(); len];
        assert_eq!(
            crate::treetn::t4a_treetn_siteinds(op, site, pair.as_mut_ptr(), pair.len(), &mut len),
            T4A_SUCCESS
        );

        let mut dim_out = 0usize;
        let mut dim_in = 0usize;
        assert_eq!(
            crate::index::t4a_index_dim(pair[0], &mut dim_out),
            T4A_SUCCESS
        );
        assert_eq!(
            crate::index::t4a_index_dim(pair[1], &mut dim_in),
            T4A_SUCCESS
        );
        assert_eq!(dim_out, out_dims[site]);
        assert_eq!(dim_in, in_dims[site]);

        out_indices.push(pair[0]);
        in_indices.push(pair[1]);
    }

    let nrows: usize = out_dims.iter().product();
    let ncols: usize = in_dims.iter().product();
    let mut matrix = vec![Complex64::new(0.0, 0.0); nrows * ncols];
    let ordered_indices: Vec<*const t4a_index> = (0..nsites)
        .flat_map(|site| {
            [
                out_indices[site] as *const t4a_index,
                in_indices[site] as *const t4a_index,
            ]
        })
        .collect();

    for x in 0..ncols {
        let in_values = decode_mixed_radix(x, in_dims);
        for y in 0..nrows {
            let out_values = decode_mixed_radix(y, out_dims);
            let mut values = Vec::with_capacity(2 * nsites);
            for site in 0..nsites {
                values.push(out_values[site]);
                values.push(in_values[site]);
            }
            let mut re = 0.0;
            let mut im = 0.0;
            assert_eq!(
                crate::treetn::t4a_treetn_evaluate(
                    op,
                    ordered_indices.as_ptr(),
                    ordered_indices.len(),
                    values.as_ptr(),
                    1,
                    &mut re,
                    &mut im
                ),
                T4A_SUCCESS
            );
            matrix[y + nrows * x] = Complex64::new(re, im);
        }
    }

    for index in out_indices.into_iter().chain(in_indices) {
        crate::index::t4a_index_release(index);
    }
    matrix
}

fn assert_matrix_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (slot, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let err = (*a - *e).norm();
        assert!(
            err < 1e-10,
            "matrix mismatch at slot {slot}: actual={a:?}, expected={e:?}, err={err}"
        );
    }
}

#[test]
fn test_shift_grouped_materialization_matches_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Grouped, &[2, 1]);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_shift_materialize(layout, 0, 1, t4a_boundary_condition::Periodic, &mut op),
        T4A_SUCCESS
    );

    let actual = c_operator_matrix(op, &[2, 2, 2], &[2, 2, 2]);
    let shift = shift_operator(2, 1, BoundaryCondition::Periodic).unwrap();
    let shift_mat = rust_operator_matrix(&shift, &[2, 2], &[2, 2]);
    let mut expected = vec![Complex64::new(0.0, 0.0); 8 * 8];
    for x in 0..8 {
        let in_values = decode_mixed_radix(x, &[2, 2, 2]);
        let in_shift = encode_mixed_radix(&in_values[..2], &[2, 2]);
        for y in 0..8 {
            let out_values = decode_mixed_radix(y, &[2, 2, 2]);
            if out_values[2] == in_values[2] {
                let out_shift = encode_mixed_radix(&out_values[..2], &[2, 2]);
                expected[y + 8 * x] = shift_mat[out_shift + 4 * in_shift];
            }
        }
    }

    assert_matrix_close(&actual, &expected);
    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_flip_materialization_matches_rust_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2]);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_flip_materialize(layout, 0, t4a_boundary_condition::Periodic, &mut op),
        T4A_SUCCESS
    );
    let actual = c_operator_matrix(op, &[2, 2], &[2, 2]);
    let expected = rust_operator_matrix(
        &flip_operator(2, BoundaryCondition::Periodic).unwrap(),
        &[2, 2],
        &[2, 2],
    );
    assert_matrix_close(&actual, &expected);
    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_phase_rotation_materialization_matches_rust_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2]);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_phase_rotation_materialize(layout, 0, std::f64::consts::PI / 4.0, &mut op),
        T4A_SUCCESS
    );
    let actual = c_operator_matrix(op, &[2, 2], &[2, 2]);
    let expected = rust_operator_matrix(
        &phase_rotation_operator(2, std::f64::consts::PI / 4.0).unwrap(),
        &[2, 2],
        &[2, 2],
    );
    assert_matrix_close(&actual, &expected);
    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_cumsum_materialization_matches_rust_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2]);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_cumsum_materialize(layout, 0, &mut op),
        T4A_SUCCESS
    );
    let actual = c_operator_matrix(op, &[2, 2], &[2, 2]);
    let expected = rust_operator_matrix(&cumsum_operator(2).unwrap(), &[2, 2], &[2, 2]);
    assert_matrix_close(&actual, &expected);
    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_fourier_materialization_matches_rust_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2]);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_fourier_materialize(layout, 0, 1, 8, 1e-12, &mut op),
        T4A_SUCCESS
    );
    let actual = c_operator_matrix(op, &[2, 2], &[2, 2]);
    let mut options = FourierOptions::forward();
    options.maxbonddim = 8;
    options.tolerance = 1e-12;
    let expected = rust_operator_matrix(
        &quantics_fourier_operator(2, options).unwrap(),
        &[2, 2],
        &[2, 2],
    );
    assert_matrix_close(&actual, &expected);
    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_qtt_layout_clone_assignment_and_inverse_fourier_match_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2]);
    assert_eq!(t4a_qtt_layout_is_assigned(layout), 1);
    assert_eq!(t4a_qtt_layout_is_assigned(std::ptr::null()), 0);

    let mut clone = std::ptr::null_mut();
    assert_eq!(t4a_qtt_layout_clone(layout, &mut clone), T4A_SUCCESS);
    assert!(!clone.is_null());

    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_fourier_materialize(clone, 0, 0, 0, 0.0, &mut op),
        T4A_SUCCESS
    );

    let actual = c_operator_matrix(op, &[2, 2], &[2, 2]);
    let expected = rust_operator_matrix(
        &quantics_fourier_operator(2, FourierOptions::inverse()).unwrap(),
        &[2, 2],
        &[2, 2],
    );
    assert_matrix_close(&actual, &expected);

    t4a_treetn_release(op);
    t4a_qtt_layout_release(clone);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_binaryop_interleaved_materialization_matches_rust_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Interleaved, &[2, 2]);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_binaryop_materialize(
            layout,
            0,
            1,
            1,
            1,
            0,
            1,
            t4a_boundary_condition::Periodic,
            t4a_boundary_condition::Periodic,
            &mut op
        ),
        T4A_SUCCESS
    );
    let actual = c_operator_matrix(op, &[2, 2, 2, 2], &[2, 2, 2, 2]);
    let expected = rust_operator_matrix(
        &binaryop_operator(
            2,
            BinaryCoeffs::sum(),
            BinaryCoeffs::select_y(),
            [BoundaryCondition::Periodic, BoundaryCondition::Periodic],
        )
        .unwrap(),
        &[2, 2, 2, 2],
        &[2, 2, 2, 2],
    );
    assert_matrix_close(&actual, &expected);
    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_binaryop_fused_materialization_matches_reindexed_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2, 2]);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_binaryop_materialize(
            layout,
            0,
            1,
            1,
            1,
            0,
            1,
            t4a_boundary_condition::Periodic,
            t4a_boundary_condition::Periodic,
            &mut op
        ),
        T4A_SUCCESS
    );

    let actual = c_operator_matrix(op, &[4, 4], &[4, 4]);
    let interleaved = rust_operator_matrix(
        &binaryop_operator(
            2,
            BinaryCoeffs::sum(),
            BinaryCoeffs::select_y(),
            [BoundaryCondition::Periodic, BoundaryCondition::Periodic],
        )
        .unwrap(),
        &[2, 2, 2, 2],
        &[2, 2, 2, 2],
    );

    let mut expected = vec![Complex64::new(0.0, 0.0); 16 * 16];
    for x in 0..16 {
        let in_sites = decode_mixed_radix(x, &[4, 4]);
        let in_bits = [
            in_sites[0] & 1,
            (in_sites[0] >> 1) & 1,
            in_sites[1] & 1,
            (in_sites[1] >> 1) & 1,
        ];
        let x_ref = encode_mixed_radix(&in_bits, &[2, 2, 2, 2]);
        for y in 0..16 {
            let out_sites = decode_mixed_radix(y, &[4, 4]);
            let out_bits = [
                out_sites[0] & 1,
                (out_sites[0] >> 1) & 1,
                out_sites[1] & 1,
                (out_sites[1] >> 1) & 1,
            ];
            let y_ref = encode_mixed_radix(&out_bits, &[2, 2, 2, 2]);
            expected[y + 16 * x] = interleaved[y_ref + 16 * x_ref];
        }
    }

    assert_matrix_close(&actual, &expected);
    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_affine_fused_materialization_matches_rust_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2]);
    let mut op = std::ptr::null_mut();
    let a_num = [1i64];
    let a_den = [1i64];
    let b_num = [1i64];
    let b_den = [1i64];
    let bc = [t4a_boundary_condition::Periodic];
    assert_eq!(
        t4a_qtransform_affine_materialize(
            layout,
            a_num.as_ptr(),
            a_den.as_ptr(),
            b_num.as_ptr(),
            b_den.as_ptr(),
            1,
            1,
            bc.as_ptr(),
            &mut op
        ),
        T4A_SUCCESS
    );

    let actual = c_operator_matrix(op, &[2, 2], &[2, 2]);
    let params = AffineParams::new(
        vec![Rational64::from_integer(1)],
        vec![Rational64::from_integer(1)],
        1,
        1,
    )
    .unwrap();
    let expected = rust_operator_matrix(
        &affine_operator(2, &params, &[BoundaryCondition::Periodic]).unwrap(),
        &[2, 2],
        &[2, 2],
    );
    assert_matrix_close(&actual, &expected);

    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_affine_pullback_fused_materialization_matches_rust_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[2]);
    let mut op = std::ptr::null_mut();
    let a_num = [1i64];
    let a_den = [1i64];
    let b_num = [0i64];
    let b_den = [1i64];
    let bc = [t4a_boundary_condition::Periodic];
    assert_eq!(
        t4a_qtransform_affine_pullback_materialize(
            layout,
            a_num.as_ptr(),
            a_den.as_ptr(),
            b_num.as_ptr(),
            b_den.as_ptr(),
            1,
            1,
            bc.as_ptr(),
            &mut op
        ),
        T4A_SUCCESS
    );

    let actual = c_operator_matrix(op, &[2, 2], &[2, 2]);
    let params = AffineParams::new(
        vec![Rational64::from_integer(1)],
        vec![Rational64::from_integer(0)],
        1,
        1,
    )
    .unwrap();
    let expected = rust_operator_matrix(
        &affine_pullback_operator(2, &params, &[BoundaryCondition::Periodic]).unwrap(),
        &[2, 2],
        &[2, 2],
    );
    assert_matrix_close(&actual, &expected);

    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_affine_pullback_fused_swap_matches_rust_reference() {
    let layout = new_layout(t4a_qtt_layout_kind::Fused, &[1, 1]);
    let mut op = std::ptr::null_mut();
    let a_num = [0i64, 1i64, 1i64, 0i64];
    let a_den = [1i64; 4];
    let b_num = [0i64, 0i64];
    let b_den = [1i64; 2];
    let bc = [
        t4a_boundary_condition::Periodic,
        t4a_boundary_condition::Periodic,
    ];
    assert_eq!(
        t4a_qtransform_affine_pullback_materialize(
            layout,
            a_num.as_ptr(),
            a_den.as_ptr(),
            b_num.as_ptr(),
            b_den.as_ptr(),
            2,
            2,
            bc.as_ptr(),
            &mut op
        ),
        T4A_SUCCESS
    );

    let actual = c_operator_matrix(op, &[4], &[4]);
    let params = AffineParams::new(
        vec![
            Rational64::from_integer(0),
            Rational64::from_integer(1),
            Rational64::from_integer(1),
            Rational64::from_integer(0),
        ],
        vec![Rational64::from_integer(0), Rational64::from_integer(0)],
        2,
        2,
    )
    .unwrap();
    let expected = rust_operator_matrix(
        &affine_pullback_operator(
            1,
            &params,
            &[BoundaryCondition::Periodic, BoundaryCondition::Periodic],
        )
        .unwrap(),
        &[4],
        &[4],
    );
    assert_matrix_close(&actual, &expected);

    t4a_treetn_release(op);
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_grouped_binaryop_is_rejected_with_explicit_message() {
    let layout = new_layout(t4a_qtt_layout_kind::Grouped, &[2, 2]);
    let mut op = std::ptr::null_mut();
    let status = t4a_qtransform_binaryop_materialize(
        layout,
        0,
        1,
        1,
        1,
        0,
        1,
        t4a_boundary_condition::Periodic,
        t4a_boundary_condition::Periodic,
        &mut op,
    );
    assert_eq!(status, T4A_INVALID_ARGUMENT);
    assert!(last_error().contains("grouped layouts"));
    assert!(op.is_null());
    t4a_qtt_layout_release(layout);
}

#[test]
fn test_layout_and_affine_validation_errors_are_reported() {
    let mut layout = std::ptr::null_mut();
    assert_eq!(
        t4a_qtt_layout_new(t4a_qtt_layout_kind::Fused, 0, std::ptr::null(), &mut layout),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("nvariables must be greater than zero"));

    let resolutions = [2usize];
    assert_eq!(
        t4a_qtt_layout_new(
            t4a_qtt_layout_kind::Fused,
            resolutions.len(),
            std::ptr::null(),
            &mut layout
        ),
        T4A_NULL_POINTER
    );
    assert!(last_error().contains("variable_resolutions is null"));

    let fused = new_layout(t4a_qtt_layout_kind::Fused, &resolutions);
    let mut op = std::ptr::null_mut();
    assert_eq!(
        t4a_qtransform_shift_materialize(fused, 1, 0, t4a_boundary_condition::Periodic, &mut op),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("target_var must be smaller than nvariables"));

    let grouped = new_layout(t4a_qtt_layout_kind::Grouped, &resolutions);
    let a_num = [1i64];
    let a_den = [1i64];
    let b_num = [0i64];
    let b_den = [1i64];
    let bc = [t4a_boundary_condition::Periodic];
    assert_eq!(
        t4a_qtransform_affine_materialize(
            grouped,
            a_num.as_ptr(),
            a_den.as_ptr(),
            b_num.as_ptr(),
            b_den.as_ptr(),
            1,
            1,
            bc.as_ptr(),
            &mut op
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("fused layouts only"));
    assert_eq!(
        t4a_qtransform_affine_pullback_materialize(
            grouped,
            a_num.as_ptr(),
            a_den.as_ptr(),
            b_num.as_ptr(),
            b_den.as_ptr(),
            1,
            1,
            bc.as_ptr(),
            &mut op
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("fused layouts only"));

    let zero_den = [0i64];
    assert_eq!(
        t4a_qtransform_affine_materialize(
            fused,
            a_num.as_ptr(),
            zero_den.as_ptr(),
            b_num.as_ptr(),
            b_den.as_ptr(),
            1,
            1,
            bc.as_ptr(),
            &mut op
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(last_error().contains("zero denominator"));

    t4a_qtt_layout_release(grouped);
    t4a_qtt_layout_release(fused);
}
