use super::*;
use num_complex::Complex64;
use tenferro_tensor::TensorScalar;

// Generic test functions for f64 and Complex64

fn test_compress_constant_generic<T>()
where
    T: TTScalar + Scalar + Default + tensor4all_core::MatrixLuciScalar,
    f64: From<<T as TensorScalar>::Real>,
{
    let tt = SimpleTensorTrain::<T>::constant(&[2, 3, 2], <T as Scalar>::from_f64(1.0));
    let original_sum = tt.sum().unwrap();

    let mut tt_compressed = tt.clone();
    tt_compressed
        .compress(&CompressionOptions::default())
        .unwrap();

    let compressed_sum = tt_compressed.sum().unwrap();
    assert!(<T as Scalar>::abs_sq(original_sum - compressed_sum).sqrt() < 1e-10);
}

fn test_compress_preserves_values_generic<T>()
where
    T: TTScalar + Scalar + Default + tensor4all_core::MatrixLuciScalar,
    f64: From<<T as TensorScalar>::Real>,
{
    // Create a simple tensor train
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 2).unwrap();
    t0.set3(0, 0, 0, <T as Scalar>::from_f64(1.0));
    t0.set3(0, 0, 1, <T as Scalar>::from_f64(0.5));
    t0.set3(0, 1, 0, <T as Scalar>::from_f64(0.0));
    t0.set3(0, 1, 1, <T as Scalar>::from_f64(1.0));

    let mut t1: Tensor3<T> = tensor3_zeros(2, 3, 2).unwrap();
    for l in 0..2 {
        for s in 0..3 {
            for r in 0..2 {
                t1.set3(
                    l,
                    s,
                    r,
                    <T as Scalar>::from_f64(((l + s + r) as f64) * 0.1 + 0.1),
                );
            }
        }
    }

    let mut t2: Tensor3<T> = tensor3_zeros(2, 2, 1).unwrap();
    t2.set3(0, 0, 0, <T as Scalar>::from_f64(1.0));
    t2.set3(0, 1, 0, <T as Scalar>::from_f64(0.5));
    t2.set3(1, 0, 0, <T as Scalar>::from_f64(0.5));
    t2.set3(1, 1, 0, <T as Scalar>::from_f64(1.0));

    let tt = SimpleTensorTrain::new(vec![t0, t1, t2]).unwrap();
    let original_sum = tt.sum().unwrap();

    let mut tt_compressed = tt.clone();
    tt_compressed
        .compress(&CompressionOptions::default())
        .unwrap();

    let compressed_sum = tt_compressed.sum().unwrap();
    assert!(<T as Scalar>::abs_sq(original_sum - compressed_sum).sqrt() < 1e-8);
}

fn test_compress_with_max_bond_dim_generic<T>()
where
    T: TTScalar + Scalar + Default + tensor4all_core::MatrixLuciScalar,
    f64: From<<T as TensorScalar>::Real>,
{
    // Create a tensor train with higher bond dimension
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 3).unwrap();
    for s in 0..2 {
        for r in 0..3 {
            t0.set3(0, s, r, <T as Scalar>::from_f64((s + r + 1) as f64));
        }
    }

    let mut t1: Tensor3<T> = tensor3_zeros(3, 2, 1).unwrap();
    for l in 0..3 {
        for s in 0..2 {
            t1.set3(l, s, 0, <T as Scalar>::from_f64((l + s + 1) as f64));
        }
    }

    let tt = SimpleTensorTrain::new(vec![t0, t1]).unwrap();
    let original_norm = tt.norm().unwrap();

    let options = CompressionOptions {
        max_bond_dim: Some(2),
        tolerance: 1e-12,
        ..Default::default()
    };

    let mut tt_compressed = tt.clone();
    tt_compressed.compress(&options).unwrap();

    // Norm should be approximately preserved (with some truncation error)
    let compressed_norm = tt_compressed.norm().unwrap();
    assert!((original_norm - compressed_norm).abs() < original_norm * 0.1);
}

// f64 tests
#[test]
fn test_compress_constant_f64() {
    test_compress_constant_generic::<f64>();
}

#[test]
fn test_compress_preserves_values_f64() {
    test_compress_preserves_values_generic::<f64>();
}

#[test]
fn test_compress_with_max_bond_dim_f64() {
    test_compress_with_max_bond_dim_generic::<f64>();
}

// Complex64 tests
#[test]
fn test_compress_constant_c64() {
    test_compress_constant_generic::<Complex64>();
}

#[test]
fn test_compress_preserves_values_c64() {
    test_compress_preserves_values_generic::<Complex64>();
}

#[test]
fn test_compress_with_max_bond_dim_c64() {
    test_compress_with_max_bond_dim_generic::<Complex64>();
}

fn test_compress_svd_constant_generic<T>()
where
    T: TTScalar + Scalar + Default + tensor4all_core::MatrixLuciScalar,
    f64: From<<T as TensorScalar>::Real>,
{
    let tt = SimpleTensorTrain::<T>::constant(&[2, 3, 2], <T as Scalar>::from_f64(1.0));
    let original_sum = tt.sum().unwrap();

    let mut tt_compressed = tt.clone();
    let options = CompressionOptions {
        method: CompressionMethod::SVD,
        ..Default::default()
    };
    tt_compressed.compress(&options).unwrap();

    let compressed_sum = tt_compressed.sum().unwrap();
    assert!(<T as Scalar>::abs_sq(original_sum - compressed_sum).sqrt() < 1e-10);
}

fn test_compress_svd_with_truncation_generic<T>()
where
    T: TTScalar + Scalar + Default + tensor4all_core::MatrixLuciScalar,
    f64: From<<T as TensorScalar>::Real>,
{
    // Create a rank-3 TT and compress with SVD to max_bond_dim=2
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 3).unwrap();
    for s in 0..2 {
        for r in 0..3 {
            t0.set3(0, s, r, <T as Scalar>::from_f64((s + r + 1) as f64));
        }
    }
    let mut t1: Tensor3<T> = tensor3_zeros(3, 2, 1).unwrap();
    for l in 0..3 {
        for s in 0..2 {
            t1.set3(l, s, 0, <T as Scalar>::from_f64((l + s + 1) as f64));
        }
    }
    let tt = SimpleTensorTrain::new(vec![t0, t1]).unwrap();
    let original_norm = tt.norm().unwrap();

    let options = CompressionOptions {
        method: CompressionMethod::SVD,
        max_bond_dim: Some(2),
        tolerance: 1e-12,
        ..Default::default()
    };
    let mut tt_compressed = tt.clone();
    tt_compressed.compress(&options).unwrap();

    let compressed_norm = tt_compressed.norm().unwrap();
    assert!((original_norm - compressed_norm).abs() < original_norm * 0.1);
}

#[test]
fn test_compress_svd_constant_f64() {
    test_compress_svd_constant_generic::<f64>();
}

#[test]
fn test_compress_svd_constant_c64() {
    test_compress_svd_constant_generic::<Complex64>();
}

#[test]
fn test_compress_svd_with_truncation_f64() {
    test_compress_svd_with_truncation_generic::<f64>();
}

#[test]
fn test_compress_svd_with_truncation_c64() {
    test_compress_svd_with_truncation_generic::<Complex64>();
}

/// Two-site train whose bond carries exactly the singular values `1e6` and
/// `1e-3`, built from orthonormal site vectors so the Schmidt values are the
/// coefficients themselves.
fn two_scale_tt<T>() -> SimpleTensorTrain<T>
where
    T: TTScalar + Scalar + Default,
{
    let half = std::f64::consts::FRAC_1_SQRT_2;
    let left_vectors = [[half, half], [half, -half]];
    let right_vectors = [[half, half], [half, -half]];
    let coefficients = [1.0e6, 1.0e-3];

    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 2).unwrap();
    let mut t1: Tensor3<T> = tensor3_zeros(2, 2, 1).unwrap();
    for (component, &coefficient) in coefficients.iter().enumerate() {
        for site in 0..2 {
            t0.set3(
                0,
                site,
                component,
                <T as Scalar>::from_f64(coefficient * left_vectors[component][site]),
            );
            t1.set3(
                component,
                site,
                0,
                <T as Scalar>::from_f64(right_vectors[component][site]),
            );
        }
    }
    SimpleTensorTrain::new(vec![t0, t1]).unwrap()
}

fn test_normalize_error_selects_relative_or_absolute_threshold_generic<T>()
where
    T: TTScalar + Scalar + Default + tensor4all_core::MatrixLuciScalar,
    f64: From<<T as TensorScalar>::Real>,
{
    let tt = two_scale_tt::<T>();
    assert_eq!(tt.rank(), 2);

    // Relative: the threshold is 1e-6 * 1e6 = 1, so the 1e-3 component goes.
    let relative = tt
        .compressed(&CompressionOptions {
            method: CompressionMethod::SVD,
            tolerance: 1e-6,
            max_bond_dim: None,
            normalize_error: true,
        })
        .unwrap();
    assert_eq!(relative.rank(), 1);

    // Absolute: the threshold is 1e-6 itself, well below 1e-3, so both
    // components are kept and the train is reproduced.
    let absolute = tt
        .compressed(&CompressionOptions {
            method: CompressionMethod::SVD,
            tolerance: 1e-6,
            max_bond_dim: None,
            normalize_error: false,
        })
        .unwrap();
    assert_eq!(absolute.rank(), 2);
    for left in 0..2 {
        for right in 0..2 {
            let want = tt.evaluate(&[left, right]).unwrap();
            let got = absolute.evaluate(&[left, right]).unwrap();
            let deviation = <T as Scalar>::abs_val(got - want);
            assert!(deviation < 1e-6, "deviation {deviation:.3e} above 1e-6");
        }
    }
}

#[test]
fn normalize_error_selects_relative_or_absolute_threshold_f64() {
    test_normalize_error_selects_relative_or_absolute_threshold_generic::<f64>();
}

#[test]
fn normalize_error_selects_relative_or_absolute_threshold_c64() {
    test_normalize_error_selects_relative_or_absolute_threshold_generic::<Complex64>();
}

/// The #1028 input: a 128 x 128 complex matrix whose only nonzero singular values are 0.5968 and
/// 6.3e-4. t4a-faer 0.25.0/0.25.1 return 6.3e-4 after a run of zeros (tensor4all/faer#82).
pub(crate) fn unsorted_spectrum_entries() -> [(usize, usize, Complex64); 3] {
    [
        (
            41,
            80,
            Complex64::new(-1.3368711811580607e-64, 3.48452997681476e-64),
        ),
        (
            67,
            80,
            Complex64::new(-5.871970588424443e-1, -1.065196368760547e-1),
        ),
        (
            5,
            112,
            Complex64::new(-2.8153454475366503e-4, 5.652970997533713e-4),
        ),
    ]
}

#[test]
fn descending_order_is_none_for_ordered_values_and_sorts_the_rest() {
    assert_eq!(descending_order::<f64>(&[], |value| value), None);
    assert_eq!(descending_order(&[3.0, 3.0, 1.0, 0.0], |value| value), None);
    assert_eq!(
        descending_order(&[0.5, 0.0, 1.0e-3], |value| value),
        Some(vec![0, 2, 1])
    );
    assert_eq!(
        descending_order(&[0.0, 2.0, 1.0, 2.0], |value| value),
        Some(vec![1, 3, 2, 0])
    );
}

/// Regression for #1028: SVD compression keeps a larger singular value that the backend returns
/// after smaller ones, together with its vectors.
#[test]
fn factorize_svd_keeps_a_larger_value_that_follows_a_smaller_one() {
    let n = 128;
    let mut matrix = Matrix::<Complex64>::zeros(n, n);
    for (row, col, value) in unsorted_spectrum_entries() {
        matrix[[row, col]] = value;
    }
    for left_orthogonal in [true, false] {
        let (left, right, rank) =
            factorize_svd(&matrix, 1e-12, true, None, left_orthogonal).unwrap();
        assert_eq!(rank, 2, "left_orthogonal = {left_orthogonal}");
        let mut error = 0.0;
        for i in 0..n {
            for j in 0..n {
                let mut value = Complex64::new(0.0, 0.0);
                for k in 0..rank {
                    value += left[[i, k]] * right[[k, j]];
                }
                error += (value - matrix[[i, j]]).norm_sqr();
            }
        }
        assert!(
            error.sqrt() <= 1e-12 * 0.6,
            "left_orthogonal = {left_orthogonal}: reconstruction error {:e}",
            error.sqrt()
        );
    }
}

fn test_svd_session_generic<T>()
where
    T: TTScalar<Real = f64>
        + Scalar
        + Default
        + tensor4all_core::MatrixLuciScalar
        + Copy
        + PartialEq
        + std::fmt::Debug,
{
    use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut session = Session::new(backend.engine_mut());
    let tt = SimpleTensorTrain::new(vec![
        Tensor3::from_fn([1, 2, 2], |[_, s, r]| {
            <T as Scalar>::from_f64((1 + s + 2 * r) as f64)
        })
        .unwrap(),
        Tensor3::from_fn([2, 2, 2], |[l, s, r]| {
            <T as Scalar>::from_f64((1 + 3 * l + s + r * r) as f64)
        })
        .unwrap(),
        Tensor3::from_fn([2, 2, 1], |[l, s, _]| {
            <T as Scalar>::from_f64((2 + l + s * s) as f64)
        })
        .unwrap(),
    ])
    .unwrap();
    let (reference, _) = tt.full_tensor().unwrap();
    for (tol, cap) in [(1e-12, None), (0.0, Some(1)), (0.5, None)] {
        let mut actual = tt.clone();
        actual
            .compress_svd_with_session(&mut session, tol, cap)
            .unwrap();
        if tol < 1e-10 && cap.is_none() {
            // The near-lossless cutoff reproduces the dense tensor.
            let (dense, _) = actual.full_tensor().unwrap();
            let max_error = reference
                .iter()
                .zip(&dense)
                .map(|(&a, &b)| (a - b).abs_val())
                .fold(0.0_f64, f64::max);
            assert!(max_error < 1e-10, "max error {max_error}");
        }
        if let Some(cap) = cap {
            assert!(actual.rank() <= cap);
        }
        let expected = tt
            .compressed(&CompressionOptions {
                method: CompressionMethod::SVD,
                tolerance: tol,
                max_bond_dim: cap,
                normalize_error: true,
            })
            .unwrap();
        assert_eq!(actual.site_tensors(), expected.site_tensors());
    }
}

#[test]
fn svd_session_f64() {
    test_svd_session_generic::<f64>();
}

#[test]
fn svd_session_c64() {
    test_svd_session_generic::<Complex64>();
}

#[test]
fn svd_session_validates_before_single_site_shortcut() {
    use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut session = Session::new(backend.engine_mut());
    let mut tt = SimpleTensorTrain::<f64>::constant(&[2], 3.0);
    for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(tt
            .compress_svd_with_session(&mut session, tolerance, None)
            .is_err());
    }
    assert!(tt
        .compress_svd_with_session(&mut session, 0.0, Some(0))
        .is_err());
    tt.compress_svd_with_session(&mut session, 0.0, None)
        .unwrap();
    assert_eq!(tt.evaluate(&[1]).unwrap(), 3.0);
}

#[test]
fn compression_revalidates_replaced_site_tensors() {
    use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut session = Session::new(backend.engine_mut());
    for (dims, tensor) in [
        (vec![2], Tensor3::from_elem([2, 2, 1], 1.0_f64).unwrap()),
        (vec![2], Tensor3::from_elem([1, 2, 2], 1.0_f64).unwrap()),
        (vec![2, 2], Tensor3::from_elem([1, 2, 2], 1.0_f64).unwrap()),
    ] {
        let mut tt = SimpleTensorTrain::constant(&dims, 1.0_f64);
        tt.site_tensors_mut()[0] = tensor;
        assert!(tt
            .compress_svd_with_session(&mut session, 1e-12, None)
            .is_err());
        assert!(tt.compress(&CompressionOptions::default()).is_err());
    }
}

// This helper deliberately has no MatrixLuciScalar bound: compiling it checks
// the public session API's SVD-only scalar contract.
fn compress_svd_only<T>(tt: &mut SimpleTensorTrain<T>, session: &mut Session<'_>)
where
    T: TTScalar<Real = f64> + Scalar + Default + Copy + BackendLinalgScalar,
{
    tt.compress_svd_with_session(session, 1e-12, None).unwrap();
}

#[test]
fn session_svd_requires_no_lu_pivoting_bound() {
    fn check<T>()
    where
        T: TTScalar<Real = f64> + Scalar + Default + Copy + BackendLinalgScalar,
    {
        let mut backend = tensor4all_tensorbackend::CpuBackend::with_threads(1).unwrap();
        let mut session = Session::new(backend.engine_mut());
        let mut tt = SimpleTensorTrain::constant(&[2, 2, 2], <T as Scalar>::from_f64(3.0));
        compress_svd_only(&mut tt, &mut session);
        let (dense, _) = tt.full_tensor().unwrap();
        let residual = dense
            .into_iter()
            .map(|x| (x - <T as Scalar>::from_f64(3.0)).abs_val())
            .fold(0.0_f64, f64::max);
        assert!(residual < 1e-12, "residual {residual}");
    }
    check::<f64>();
    check::<Complex64>();
}

#[test]
fn session_compression_rejects_unsupported_methods_before_mutation_or_shortcuts() {
    let mut backend = tensor4all_tensorbackend::CpuBackend::with_threads(1).unwrap();
    let mut session = Session::new(backend.engine_mut());
    for method in [CompressionMethod::LU, CompressionMethod::CI] {
        for dims in [&[2][..], &[2, 2, 2][..]] {
            let mut tt = SimpleTensorTrain::<f64>::constant(dims, 3.0);
            let original = tt.site_tensors().to_vec();
            let error = tt
                .compress_impl(
                    &CompressionOptions {
                        method,
                        ..Default::default()
                    },
                    &mut Some(&mut session),
                    |_, _, _, _| panic!("unsupported session method reached factorization"),
                )
                .unwrap_err();
            assert!(
                matches!(error, SimpleTensorTrainError::UnsupportedCompressionMethod { method: actual } if actual == method)
            );
            assert_eq!(tt.site_tensors(), original);
            assert!(error.to_string().contains("use CompressionMethod::SVD"));
        }
    }
}
