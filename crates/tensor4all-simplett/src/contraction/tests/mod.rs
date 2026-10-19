use super::*;
use crate::einsum_helper::{
    einsum_tensors, matrix_times_col_vector, row_vector_times_matrix, tensor_to_col_major_vec,
    typed_tensor_from_col_major_slice, typed_tensor_reshape, EinsumScalar,
};
use crate::types::{tensor3_zeros, Tensor3};

#[test]
fn test_dot_constant() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3], 3.0);

    let result = tt1.inner_product(&tt2).unwrap();

    // Each element product is 2.0 * 3.0 = 6.0
    // Sum over 2*3=6 elements: 6.0 * 6 = 36.0
    assert!((result - 36.0).abs() < 1e-10);
}

#[test]
fn test_dot_different_tensors() {
    let mut t0_a: Tensor3<f64> = tensor3_zeros(1, 3, 2).unwrap();
    for s in 0..3 {
        for r in 0..2 {
            t0_a.set3(0, s, r, (s + r + 1) as f64);
        }
    }

    let mut t1_a: Tensor3<f64> = tensor3_zeros(2, 2, 1).unwrap();
    for l in 0..2 {
        for s in 0..2 {
            t1_a.set3(l, s, 0, (l + s + 1) as f64);
        }
    }

    let tt_a = SimpleTensorTrain::new(vec![t0_a.clone(), t1_a.clone()]).unwrap();

    let mut t0_b: Tensor3<f64> = tensor3_zeros(1, 3, 1).unwrap();
    for s in 0..3 {
        t0_b.set3(0, s, 0, (s + 1) as f64 * 0.5);
    }

    let mut t1_b: Tensor3<f64> = tensor3_zeros(1, 2, 1).unwrap();
    for s in 0..2 {
        t1_b.set3(0, s, 0, (s + 2) as f64 * 0.3);
    }

    let tt_b = SimpleTensorTrain::new(vec![t0_b, t1_b]).unwrap();

    let dot_result = tt_a.inner_product(&tt_b).unwrap();

    // Compute expected value by brute force
    let mut expected = 0.0;
    for i0 in 0..3 {
        for i1 in 0..2 {
            let val_a = tt_a.evaluate(&[i0, i1]).unwrap();
            let val_b = tt_b.evaluate(&[i0, i1]).unwrap();
            expected += val_a * val_b;
        }
    }

    assert!(
        (dot_result - expected).abs() < 1e-10,
        "inner_product = {}, expected = {}",
        dot_result,
        expected
    );
}

#[test]
fn test_dot_length_mismatch() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    let err = tt1.inner_product(&tt2).unwrap_err();
    assert!(err.to_string().contains("different lengths"));
    assert!(err.to_string().contains("2 vs 3"));
}

#[test]
fn test_dot_site_dim_mismatch() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 4], 1.0);
    let err = tt1.inner_product(&tt2).unwrap_err();
    assert!(err
        .to_string()
        .contains("Site dimensions mismatch at site 1"));
    assert!(err.to_string().contains("3 vs 4"));
}

#[test]
fn test_dot_site_dim_mismatch_middle() {
    // Mismatch at site 1 (not site 0) to cover the inner loop error path
    let mut t0: Tensor3<f64> = tensor3_zeros(1, 2, 1).unwrap();
    t0.set3(0, 0, 0, 1.0);
    t0.set3(0, 1, 0, 1.0);

    let mut t1_a: Tensor3<f64> = tensor3_zeros(1, 3, 1).unwrap();
    for s in 0..3 {
        t1_a.set3(0, s, 0, 1.0);
    }
    let mut t1_b: Tensor3<f64> = tensor3_zeros(1, 4, 1).unwrap();
    for s in 0..4 {
        t1_b.set3(0, s, 0, 1.0);
    }

    let tt_a = SimpleTensorTrain::new(vec![t0.clone(), t1_a]).unwrap();
    let tt_b = SimpleTensorTrain::new(vec![t0, t1_b]).unwrap();
    let err = tt_a.inner_product(&tt_b).unwrap_err();
    assert!(err
        .to_string()
        .contains("Site dimensions mismatch at site 1"));
    assert!(err.to_string().contains("3 vs 4"));
}

#[test]
fn test_dot_empty() {
    let tt1 = SimpleTensorTrain::<f64>::from_tensors_unchecked(Vec::new());
    let tt2 = SimpleTensorTrain::<f64>::from_tensors_unchecked(Vec::new());
    let result = tt1.inner_product(&tt2).unwrap();
    assert!((result - 0.0).abs() < 1e-15);
}

fn test_dot_generic<T: TTScalar + tensor4all_core::Scalar + Default + EinsumScalar>() {
    let tt1 = SimpleTensorTrain::<T>::constant(&[2, 3], T::from_f64(2.0));
    let tt2 = SimpleTensorTrain::<T>::constant(&[2, 3], T::from_f64(3.0));

    let result = tt1.inner_product(&tt2).unwrap();
    // Each element: 2*3 = 6, sum over 2*3=6 elements = 36
    assert!(
        (result - T::from_f64(36.0)).abs_sq().sqrt() < 1e-10,
        "inner_product product mismatch"
    );
}

#[test]
fn test_dot_f64() {
    test_dot_generic::<f64>();
}

#[test]
fn test_dot_c64() {
    test_dot_generic::<num_complex::Complex64>();
}

#[test]
fn test_dot_convenience_function() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3], 3.0);
    let result = inner_product(&tt1, &tt2).unwrap();
    assert!((result - 36.0).abs() < 1e-10);
}

#[test]
fn test_contraction_options_default_values() {
    let opts = ContractionOptions::default();
    assert!((opts.tolerance - 1e-12).abs() < 1e-15);
    assert_eq!(opts.max_bond_dim, None);
    assert!(matches!(opts.method, CompressionMethod::LU));
}

#[test]
fn test_contraction_helper_error_formats_context() {
    let err = contraction_helper_error("while contracting", "backend rejected labels");
    let message = err.to_string();
    assert!(message.contains("while contracting"));
    assert!(message.contains("backend rejected labels"));
}

#[test]
fn test_inner_product_read_error_formats_context() {
    let err = inner_product_read_error("host data unavailable");
    assert!(err
        .to_string()
        .contains("Failed to read inner_product result: host data unavailable"));
}

#[test]
fn test_dot_three_sites() {
    // Test inner_product product with 3 sites to exercise the inner loop more fully
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let result = tt1.inner_product(&tt2).unwrap();
    // Each element: 1*2 = 2, total elements = 2*3*2 = 12, sum = 24
    assert!((result - 24.0).abs() < 1e-10);
}

#[test]
fn test_einsum_tensors_matmul() {
    let a = typed_tensor_from_col_major_slice(&[1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    let b = typed_tensor_from_col_major_slice(&[5.0, 7.0, 6.0, 8.0], &[2, 2]).unwrap();

    let c = einsum_tensors("ij,jk->ik", &[&a, &b]).unwrap();
    assert_eq!(
        tensor_to_col_major_vec(&c).unwrap(),
        vec![19.0, 43.0, 22.0, 50.0]
    );
}

#[test]
fn test_einsum_tensors_reports_backend_error() {
    let a = typed_tensor_from_col_major_slice(&[1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();

    let err = einsum_tensors("ij,jk->ik", &[&a]).unwrap_err();
    assert!(err.to_string().contains("einsum failed"));
    assert!(err.to_string().contains("ij,jk->ik"));
    // The shared entry rejects a missing operand before any contraction.
    assert!(err.to_string().contains("2 input term(s)"));
    assert!(err.to_string().contains("1 operand(s)"));
}

#[test]
fn test_einsum_tensors_rejects_surplus_operand() {
    let a = typed_tensor_from_col_major_slice(&[1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();

    let err = einsum_tensors("ij->ji", &[&a, &a]).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("einsum failed for 'ij->ji'"), "{message}");
    assert!(message.contains("1 input term(s)"), "{message}");
    assert!(message.contains("2 operand(s)"), "{message}");
}

#[test]
fn test_einsum_tensors_complex_matmul_and_trace() {
    use num_complex::Complex64;

    let a = typed_tensor_from_col_major_slice(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, -1.0),
        ],
        &[2, 2],
    )
    .unwrap();
    let identity = typed_tensor_from_col_major_slice(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        &[2, 2],
    )
    .unwrap();

    let product = einsum_tensors("ij,jk->ik", &[&a, &identity]).unwrap();
    assert_eq!(
        tensor_to_col_major_vec(&product).unwrap(),
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, -1.0),
        ]
    );

    let trace = einsum_tensors("ii->", &[&a]).unwrap();
    assert_eq!(
        tensor_to_col_major_vec(&trace).unwrap(),
        vec![Complex64::new(2.0, 0.0)]
    );
}

#[test]
fn test_row_vector_times_matrix_matches_mat_vec() {
    // Column-major (2 x 3): M[r, c] at r + 2*c.
    let v = [1.0, -2.0];
    let m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out = row_vector_times_matrix::<f64>(&v, &m, 2, 3).unwrap();
    // out[c] = v0*m[0,c] + v1*m[1,c]
    assert_eq!(
        out,
        vec![
            1.0 * 1.0 + (-2.0) * 2.0,
            1.0 * 3.0 + (-2.0) * 4.0,
            1.0 * 5.0 + (-2.0) * 6.0
        ]
    );
}

#[test]
fn test_matrix_times_col_vector_matches_mat_vec() {
    // Column-major (2 x 3): M[r, c] at r + 2*c.
    let m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let v = [1.0, -2.0, 3.0];
    let out = matrix_times_col_vector::<f64>(&m, 2, 3, &v).unwrap();
    // out[r] = sum_c M[r,c]*v[c]
    assert_eq!(
        out,
        vec![
            1.0 * 1.0 + 3.0 * (-2.0) + 5.0 * 3.0,
            2.0 * 1.0 + 4.0 * (-2.0) + 6.0 * 3.0
        ]
    );
}

#[test]
fn test_row_vector_times_matrix_reports_shape_error() {
    let err = row_vector_times_matrix::<f64>(&[1.0], &[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap_err();

    assert!(err.to_string().contains("row vector length mismatch"));
    assert!(err.to_string().contains("expected 2"));
    assert!(err.to_string().contains("got 1"));
}

#[test]
fn test_einsum_helper_error_paths_report_context() {
    let err = typed_tensor_from_col_major_slice::<f64>(&[], &[usize::MAX, 2]).unwrap_err();
    assert!(err.to_string().contains("shape element count overflow"));

    let err = typed_tensor_from_col_major_slice::<f64>(&[1.0, 2.0], &[3]).unwrap_err();
    assert!(err.to_string().contains("tensor data length mismatch"));
    assert!(err.to_string().contains("expected 3"));
    assert!(err.to_string().contains("got 2"));

    let tensor = typed_tensor_from_col_major_slice::<f64>(&[1.0, 2.0], &[2]).unwrap();
    let err = typed_tensor_reshape(&tensor, &[3]).unwrap_err();
    assert!(err
        .to_string()
        .contains("tensor reshape element count mismatch"));

    let err = row_vector_times_matrix::<f64>(&[1.0, 2.0], &[1.0, 2.0, 3.0], 2, 2).unwrap_err();
    assert!(err
        .to_string()
        .contains("row vector times matrix length mismatch"));

    let err = matrix_times_col_vector::<f64>(&[1.0, 2.0, 3.0], 2, 2, &[1.0, 2.0]).unwrap_err();
    assert!(err
        .to_string()
        .contains("matrix times column vector length mismatch"));

    let err = matrix_times_col_vector::<f64>(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[1.0]).unwrap_err();
    assert!(err.to_string().contains("column vector length mismatch"));
}

/// An independent dense expansion of a train, computed from its raw site tensors.
///
/// It walks the bond paths itself - it never calls `evaluate`, `inner_product` or any einsum - so it cannot
/// inherit a defect from the code the tests check.
fn tt_dense_from_sites<T: TTScalar>(tt: &SimpleTensorTrain<T>) -> Vec<T> {
    let site_dims: Vec<usize> = (0..tt.len())
        .map(|site| tt.site_tensor(site).site_dim())
        .collect();
    let total: usize = site_dims.iter().product();
    let mut dense = Vec::with_capacity(total);
    for linear in 0..total {
        let mut site = vec![0usize; site_dims.len()];
        let mut rest = linear;
        for (axis, dim) in site_dims.iter().enumerate() {
            site[axis] = rest % dim;
            rest /= dim;
        }
        dense.push(bond_sum(tt, &site, 0, 0));
    }
    dense
}

fn bond_sum<T: TTScalar>(
    tt: &SimpleTensorTrain<T>,
    site: &[usize],
    position: usize,
    left_bond: usize,
) -> T {
    let tensor = tt.site_tensor(position);
    let mut total = T::zero();
    for right_bond in 0..tensor.right_dim() {
        let factor = *tensor.get3(left_bond, site[position], right_bond);
        let tail = if position + 1 == tt.len() {
            T::one()
        } else {
            bond_sum(tt, site, position + 1, right_bond)
        };
        total = total + factor * tail;
    }
    total
}

/// The bilinear product of two dense expansions: simplett conjugates neither operand.
fn dense_bilinear<T: Scalar + Copy>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .fold(T::zero(), |total, (left, right)| total + *left * *right)
}

fn session_context() -> tensor4all_tensorbackend::CpuExecutionContext {
    let backend = tenferro_cpu::CpuBackend::with_threads(1).expect("1T CPU backend");
    tensor4all_tensorbackend::CpuExecutionContext::from_backend(backend)
}

/// A train with the given site dimensions and bond dimensions, filled with asymmetric data.
fn asymmetric_train_f64(
    site_dims: &[usize],
    bonds: &[usize],
    offset: f64,
) -> SimpleTensorTrain<f64> {
    let tensors: Vec<Tensor3<f64>> = site_dims
        .iter()
        .enumerate()
        .map(|(site, &site_dim)| {
            let left = bonds[site];
            let right = bonds[site + 1];
            let mut tensor = tensor3_zeros(left, site_dim, right).unwrap();
            for l in 0..left {
                for s in 0..site_dim {
                    for r in 0..right {
                        let value =
                            offset + (site * 7 + l * 3 + s * 5 + r * 11) as f64 * 0.125 - 1.0;
                        tensor.set3(l, s, r, value);
                    }
                }
            }
            tensor
        })
        .collect();
    SimpleTensorTrain::new(tensors).expect("a well-formed train")
}

#[test]
fn session_inner_product_matches_an_independent_dense_oracle() {
    let context = session_context();
    // One site (the first-site contraction is the whole computation), two sites, three sites, and a pair
    // with unequal bond dimensions on both sides.
    let cases: Vec<(SimpleTensorTrain<f64>, SimpleTensorTrain<f64>)> = vec![
        (
            asymmetric_train_f64(&[3], &[1, 1], 0.5),
            asymmetric_train_f64(&[3], &[1, 1], -0.25),
        ),
        (
            asymmetric_train_f64(&[2, 3], &[1, 2, 1], 0.75),
            asymmetric_train_f64(&[2, 3], &[1, 3, 1], 1.25),
        ),
        (
            asymmetric_train_f64(&[2, 3, 2], &[1, 2, 3, 1], -0.5),
            asymmetric_train_f64(&[2, 3, 2], &[1, 3, 2, 1], 0.25),
        ),
    ];

    for (index, (a, b)) in cases.into_iter().enumerate() {
        let expected = dense_bilinear(&tt_dense_from_sites(&a), &tt_dense_from_sites(&b));
        let compatibility = a.inner_product(&b).expect("compatibility inner product");
        assert!(
            (compatibility - expected).abs() < 1e-12,
            "case {index}: the oracle disagrees with the compatibility route: {compatibility} != {expected}"
        );
        let actual = context
            .with_concrete_session(|session| a.inner_product_with_session(&b, session))
            .expect("session entry")
            .expect("session inner product");
        assert!(
            (actual - expected).abs() < 1e-12,
            "case {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn one_site_complex_product_is_bilinear_and_not_hermitian() {
    use num_complex::Complex64;

    let context = session_context();
    let mut a_site: Tensor3<Complex64> = tensor3_zeros(1, 2, 1).unwrap();
    a_site.set3(0, 0, 0, Complex64::new(1.0, 1.0));
    a_site.set3(0, 1, 0, Complex64::new(2.0, -1.0));
    let mut b_site: Tensor3<Complex64> = tensor3_zeros(1, 2, 1).unwrap();
    b_site.set3(0, 0, 0, Complex64::new(3.0, 2.0));
    b_site.set3(0, 1, 0, Complex64::new(-1.0, 1.0));

    let a = SimpleTensorTrain::new(vec![a_site]).expect("train a");
    let b = SimpleTensorTrain::new(vec![b_site]).expect("train b");

    // (1+i)(3+2i) + (2-i)(-1+i) = 8i, while conjugating the left operand would give 2.
    let expected = Complex64::new(0.0, 8.0);
    let actual = context
        .with_concrete_session(|session| a.inner_product_with_session(&b, session))
        .expect("session entry")
        .expect("session inner product");
    assert!(
        (actual - expected).norm() < 1e-12,
        "the product must be bilinear, got {actual}"
    );
    assert!(
        (a.inner_product(&b).expect("compatibility") - expected).norm() < 1e-12,
        "the compatibility route must agree"
    );
}

#[test]
fn single_precision_and_refusals_survive_the_session_route() {
    let context = session_context();

    // f32 keeps its dtype through extraction.
    let a = SimpleTensorTrain::<f32>::constant(&[2, 3], 1.5);
    let b = SimpleTensorTrain::<f32>::constant(&[2, 3], 2.0);
    let actual = context
        .with_concrete_session(|session| a.inner_product_with_session(&b, session))
        .expect("session entry")
        .expect("session inner product");
    assert!((actual - 18.0).abs() < 1e-4, "got {actual}");

    // An empty train is zero, as in the compatibility route.
    let empty = SimpleTensorTrain::<f64>::constant(&[], 1.0);
    let zero = context
        .with_concrete_session(|session| empty.inner_product_with_session(&empty, session))
        .expect("session entry")
        .expect("session inner product");
    assert_eq!(zero, 0.0);

    let short = asymmetric_train_f64(&[2], &[1, 1], 0.0);
    let long = asymmetric_train_f64(&[2, 3], &[1, 2, 1], 0.0);
    let mismatched = asymmetric_train_f64(&[4, 3], &[1, 2, 1], 0.0);

    context
        .with_concrete_session(|session| {
            // Length mismatch, then a site-dimension mismatch at site 0, then one at a later site.
            let error = long
                .inner_product_with_session(&short, session)
                .expect_err("length mismatch");
            assert!(error.to_string().contains("different lengths"), "{error}");
            let error = long
                .inner_product_with_session(&mismatched, session)
                .expect_err("site dimension mismatch at site 0");
            assert!(error.to_string().contains("site 0"), "{error}");
            let later = asymmetric_train_f64(&[2, 5], &[1, 2, 1], 0.0);
            let error = long
                .inner_product_with_session(&later, session)
                .expect_err("site dimension mismatch at a later site");
            assert!(error.to_string().contains("site 1"), "{error}");

            // The held session stays usable after each refusal.
            let value = long
                .inner_product_with_session(&long, session)
                .expect("session inner product after refusals");
            let expected = dense_bilinear(&tt_dense_from_sites(&long), &tt_dense_from_sites(&long));
            assert!((value - expected).abs() < 1e-12);
            Ok::<(), SimpleTensorTrainError>(())
        })
        .expect("session entry")
        .expect("session work after refusals");
}

/// A two-site train over `Complex64` with genuinely non-zero imaginary parts in every core.
fn complex_two_site(offset: f64) -> SimpleTensorTrain<num_complex::Complex64> {
    use num_complex::Complex64;
    let mut first: Tensor3<Complex64> = tensor3_zeros(1, 3, 2).unwrap();
    for s in 0..3 {
        for r in 0..2 {
            first.set3(
                0,
                s,
                r,
                Complex64::new(
                    (s * 2 + r) as f64 - 2.0 + offset,
                    (s + r * 3) as f64 * 0.5 - 1.0 - offset,
                ),
            );
        }
    }
    let mut second: Tensor3<Complex64> = tensor3_zeros(2, 2, 1).unwrap();
    for l in 0..2 {
        for s in 0..2 {
            second.set3(
                l,
                s,
                0,
                Complex64::new(
                    (l + s) as f64 * 0.5 - offset,
                    1.5 - (l * 2 + s) as f64 * 0.25 + offset,
                ),
            );
        }
    }
    SimpleTensorTrain::new(vec![first, second]).expect("complex train")
}

#[test]
fn session_inner_product_keeps_bilinear_semantics_at_every_site() {
    let context = session_context();
    // Two sites, so a conjugation introduced only in the later-site loop is observable.
    let a = complex_two_site(0.0);
    let b = complex_two_site(0.25);
    let expected = dense_bilinear(&tt_dense_from_sites(&a), &tt_dense_from_sites(&b));
    let actual = context
        .with_concrete_session(|session| a.inner_product_with_session(&b, session))
        .expect("session entry")
        .expect("session inner product");
    assert!((actual - expected).norm() < 1e-12, "{actual} != {expected}");
    let compatibility = a.inner_product(&b).expect("compatibility");
    assert!(
        (compatibility - expected).norm() < 1e-12,
        "compatibility disagrees with the oracle: {compatibility} != {expected}"
    );
    // A Hermitian product would differ; check that this is the bilinear value.
    assert!(
        (actual - expected).norm() < (actual - expected.conj()).norm(),
        "the result must not look conjugated"
    );
}

#[test]
fn single_precision_agrees_with_the_compatibility_route() {
    let context = session_context();
    let a = asymmetric_train_f64(&[2, 3], &[1, 2, 1], 0.5);
    let b = asymmetric_train_f64(&[2, 3], &[1, 2, 1], -0.5);

    // The same numbers in f32, so both routes see the same values.
    let a_f32 = SimpleTensorTrain::<f32>::new(
        (0..a.len())
            .map(|site| {
                let source = a.site_tensor(site);
                let mut target =
                    tensor3_zeros(source.left_dim(), source.site_dim(), source.right_dim())
                        .unwrap();
                for l in 0..source.left_dim() {
                    for s in 0..source.site_dim() {
                        for r in 0..source.right_dim() {
                            target.set3(l, s, r, *source.get3(l, s, r) as f32);
                        }
                    }
                }
                target
            })
            .collect(),
    )
    .expect("f32 train a");
    let b_f32 = SimpleTensorTrain::<f32>::new(
        (0..b.len())
            .map(|site| {
                let source = b.site_tensor(site);
                let mut target =
                    tensor3_zeros(source.left_dim(), source.site_dim(), source.right_dim())
                        .unwrap();
                for l in 0..source.left_dim() {
                    for s in 0..source.site_dim() {
                        for r in 0..source.right_dim() {
                            target.set3(l, s, r, *source.get3(l, s, r) as f32);
                        }
                    }
                }
                target
            })
            .collect(),
    )
    .expect("f32 train b");

    let compatibility = a_f32.inner_product(&b_f32).expect("compatibility");
    let actual = context
        .with_concrete_session(|session| a_f32.inner_product_with_session(&b_f32, session))
        .expect("session entry")
        .expect("session inner product");
    assert!(
        (actual - compatibility).abs() <= 1e-4 * compatibility.abs().max(1.0),
        "{actual} != {compatibility}"
    );
}

#[test]
fn complex32_is_served() {
    use num_complex::Complex32;

    let context = session_context();
    let mut site: Tensor3<Complex32> = tensor3_zeros(1, 2, 1).unwrap();
    site.set3(0, 0, 0, Complex32::new(1.5, -0.5));
    site.set3(0, 1, 0, Complex32::new(2.0, 1.0));
    let a = SimpleTensorTrain::new(vec![site.clone()]).expect("Complex32 train");
    let mut other: Tensor3<Complex32> = tensor3_zeros(1, 2, 1).unwrap();
    other.set3(0, 0, 0, Complex32::new(0.5, 0.25));
    other.set3(0, 1, 0, Complex32::new(-1.0, 2.0));
    let b = SimpleTensorTrain::new(vec![other]).expect("Complex32 train");

    let expected = dense_bilinear(&tt_dense_from_sites(&a), &tt_dense_from_sites(&b));
    let actual = context
        .with_concrete_session(|session| a.inner_product_with_session(&b, session))
        .expect("session entry")
        .expect("session inner product");
    assert!((actual - expected).norm() < 1e-5, "{actual} != {expected}");
}

#[test]
fn frontend_refusals_match_the_compatibility_route_exactly_and_keep_the_session() {
    let context = session_context();
    let short = asymmetric_train_f64(&[2], &[1, 1], 0.0);
    let long = asymmetric_train_f64(&[2, 3], &[1, 2, 1], 0.0);
    let mismatched = asymmetric_train_f64(&[4, 3], &[1, 2, 1], 0.0);
    let later = asymmetric_train_f64(&[2, 5], &[1, 2, 1], 0.0);

    // Compatibility runs with no session held: entering it from inside one is rejected by design.
    let compatibility_length = long.inner_product(&short).unwrap_err();
    let compatibility_site = long.inner_product(&mismatched).unwrap_err();
    let compatibility_later = long.inner_product(&later).unwrap_err();
    let compatibility_self = long.inner_product(&long).expect("compatibility");

    context
        .with_concrete_session(|session| {
            let cases = [
                (&short, compatibility_length),
                (&mismatched, compatibility_site),
                (&later, compatibility_later),
            ];
            for (other, expected) in cases {
                let actual = long.inner_product_with_session(other, session).unwrap_err();
                match (actual, expected) {
                    (
                        SimpleTensorTrainError::InvalidOperation { message: actual },
                        SimpleTensorTrainError::InvalidOperation { message: expected },
                    ) => assert_eq!(
                        actual, expected,
                        "the refusal must be the compatibility route's"
                    ),
                    (actual, expected) => panic!("variant mismatch: {actual:?} vs {expected:?}"),
                }
            }

            // Several successes inside one held region equal the separate calls.
            let first = long
                .inner_product_with_session(&long, session)
                .expect("first");
            let second = long
                .inner_product_with_session(&long, session)
                .expect("second");
            assert_eq!(first, second);
            assert_eq!(first, compatibility_self);
            Ok::<(), SimpleTensorTrainError>(())
        })
        .expect("session entry")
        .expect("session work");
}

#[test]
fn a_malformed_final_environment_is_refused_instead_of_silently_differing() {
    let context = session_context();

    // A core replaced through the mutable accessor can leave a two-element environment. The compatibility
    // route selects its first element; this method refuses rather than answering something different.
    let mut a = asymmetric_train_f64(&[2], &[1, 1], 0.0);
    let mut wide: Tensor3<f64> = tensor3_zeros(1, 2, 2).unwrap();
    for s in 0..2 {
        wide.set3(0, s, 0, 1.0);
        wide.set3(0, s, 1, 2.0);
    }
    a.site_tensors_mut()[0] = wide;
    let b = asymmetric_train_f64(&[2], &[1, 1], 0.0);

    let error = inner_product_in_fresh_session(&a, &b, &context)
        .expect_err("a two-element environment is refused");
    assert!(
        error.to_string().contains("one element"),
        "the refusal names the malformed finish: {error}"
    );

    // A core whose left bond no longer matches the running environment is a backend shape failure, and the
    // held session stays usable afterwards.
    let mut broken = asymmetric_train_f64(&[2, 3], &[1, 2, 1], 0.0);
    let mut mismatched_bond: Tensor3<f64> = tensor3_zeros(3, 3, 1).unwrap();
    for l in 0..3 {
        for s in 0..3 {
            mismatched_bond.set3(l, s, 0, 1.0);
        }
    }
    broken.site_tensors_mut()[1] = mismatched_bond;
    let other = asymmetric_train_f64(&[2, 3], &[1, 2, 1], 0.0);
    // Compatibility runs outside any held session.
    let compatibility_other = other.inner_product(&other).expect("compatibility");

    context
        .with_concrete_session(|session| {
            let error = broken
                .inner_product_with_session(&other, session)
                .expect_err("a mismatched bond is a backend failure");
            assert!(matches!(
                error,
                SimpleTensorTrainError::InvalidOperation { .. }
            ));
            let value = other
                .inner_product_with_session(&other, session)
                .expect("session inner product after a backend failure");
            assert_eq!(value, compatibility_other);
            Ok::<(), SimpleTensorTrainError>(())
        })
        .expect("session entry")
        .expect("session work after a backend failure");
}

/// Run one inner product on a fresh explicit session, for tests that need their own context.
fn inner_product_in_fresh_session(
    lhs: &SimpleTensorTrain<f64>,
    rhs: &SimpleTensorTrain<f64>,
    context: &tensor4all_tensorbackend::CpuExecutionContext,
) -> Result<f64> {
    context
        .with_concrete_session(|session| lhs.inner_product_with_session(rhs, session))
        .expect("session entry")
}
