use super::*;
use num_complex::Complex64;
use std::sync::Arc;

use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};

/// Deterministic fixture values for the generic comparison below, so the test
/// does not depend on a seeded RNG. Complex fixtures keep a non-zero imaginary
/// part so the conjugation-free einsum is exercised on genuinely complex data.
trait Fixture: TreeRsiScalar {
    fn fixture(index: usize) -> Self;
}

fn fixture_components(index: usize) -> (f64, f64) {
    let real = ((index * 37 + 11) % 19) as f64 / 7.0 - 1.0;
    let imag = ((index * 53 + 5) % 23) as f64 / 11.0 - 1.0;
    (real, imag)
}

impl Fixture for f64 {
    fn fixture(index: usize) -> Self {
        fixture_components(index).0
    }
}

impl Fixture for Complex64 {
    fn fixture(index: usize) -> Self {
        let (real, imag) = fixture_components(index);
        Complex64::new(real, imag)
    }
}

/// Largest absolute value, refusing a non-finite entry (`f64::max` drops NaN).
fn largest_absolute<T: TreeRsiScalar>(values: &[T]) -> f64 {
    let mut worst = 0.0f64;
    for (column, value) in values.iter().enumerate() {
        let magnitude = tensor4all_core::Scalar::abs_val(*value);
        assert!(
            magnitude.is_finite(),
            "non-finite value at column {column}: magnitude {magnitude}"
        );
        worst = worst.max(magnitude);
    }
    worst
}

/// Largest absolute componentwise difference.
///
/// `f64::max` drops a NaN operand and `zip` stops at the shorter side, so a
/// non-finite or missing entry would otherwise read as a zero residual.
fn largest_absolute_difference<T: TreeRsiScalar>(left: &[T], right: &[T]) -> f64 {
    assert_eq!(
        left.len(),
        right.len(),
        "compared vectors must have equal length: {} and {}",
        left.len(),
        right.len()
    );
    let mut worst = 0.0f64;
    for (column, (a, b)) in left.iter().zip(right).enumerate() {
        let error = tensor4all_core::Scalar::abs_val(*a - *b);
        assert!(
            error.is_finite(),
            "non-finite error at column {column}: magnitudes {} and {}",
            tensor4all_core::Scalar::abs_val(*a),
            tensor4all_core::Scalar::abs_val(*b)
        );
        worst = worst.max(error);
    }
    worst
}

/// Naive reference: `out(r.., l) = sum_a input(r.., a, l) weights(a, l)` on the
/// already permuted column-major `rest.. x n x k` buffer.
fn naive_contract_axis_diag_probe<T: TreeRsiScalar>(
    rows: usize,
    n: usize,
    k: usize,
    data: &[T],
    weights: &[T],
) -> Vec<T> {
    let mut out = vec![T::default(); rows * k];
    for probe in 0..k {
        for row in 0..rows {
            let mut sum = T::default();
            for alpha in 0..n {
                sum = sum + data[row + rows * (alpha + n * probe)] * weights[alpha + n * probe];
            }
            out[row + rows * probe] = sum;
        }
    }
    out
}

fn cpu_context() -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    )))
}

fn assert_contract_axis_diag_probe_matches_naive<T: Fixture>(context: &ExecutionContext) {
    // Every case is `rest.. x n x k`: the probe axis is last and `axis` is the
    // contracted one, so each case exercises a multi-probe (`k > 1`) call. The
    // rank-2 case has an empty rest prefix (`rows == 1`, one scalar per probe),
    // the rank-3 case a single retained axis of extent 1, and the rank-4 case two
    // non-trivial retained axes of unequal extent.
    let cases: [&[usize]; 3] = [&[4, 5], &[1, 4, 5], &[2, 3, 4, 5]];
    for dims in cases {
        let k = dims[dims.len() - 1];
        for axis in 0..dims.len() - 1 {
            let n = dims[axis];
            let total: usize = dims.iter().product();
            let data = (0..total).map(T::fixture).collect::<Vec<_>>();
            let weights = (0..n * k)
                .map(|index| T::fixture(10_000 + index))
                .collect::<Vec<_>>();
            let mut expected_dims = dims[..dims.len() - 1]
                .iter()
                .enumerate()
                .filter(|&(position, _)| position != axis)
                .map(|(_, &dim)| dim)
                .collect::<Vec<_>>();
            expected_dims.push(k);
            let rows: usize = expected_dims[..expected_dims.len() - 1].iter().product();

            let actual = contract_axis_diag_probe(
                context,
                Dense {
                    dims: dims.to_vec(),
                    data: data.clone(),
                },
                axis,
                &weights,
            )
            .unwrap();

            // Rebuild the permuted `rest.. x n x k` reference buffer with the
            // same axis order the kernel applies.
            let mut order = (0..dims.len() - 1)
                .filter(|&position| position != axis)
                .collect::<Vec<_>>();
            order.extend([axis, dims.len() - 1]);
            let permuted = permute(
                &Dense {
                    dims: dims.to_vec(),
                    data,
                },
                &order,
            );
            let expected = naive_contract_axis_diag_probe(rows, n, k, &permuted.data, &weights);

            assert_eq!(actual.dims, expected_dims);
            let scale = largest_absolute(&expected);
            assert!(scale > 1.0e-3, "fixture must not be degenerate");
            let residual = largest_absolute_difference(&actual.data, &expected);
            assert!(
                residual <= 1.0e-13 * scale,
                "diag-probe einsum disagrees with the naive loop: \
                 residual {residual:.3e}, scale {scale:.3e}, dims {dims:?}, axis {axis}"
            );
        }
    }
}

#[test]
fn contract_axis_diag_probe_matches_the_naive_probe_loop_f64() {
    assert_contract_axis_diag_probe_matches_naive::<f64>(&cpu_context());
}

#[test]
fn contract_axis_diag_probe_matches_the_naive_probe_loop_c64() {
    assert_contract_axis_diag_probe_matches_naive::<Complex64>(&cpu_context());
}

/// The probe label of each weight column must reach its own probe column of the
/// output: a transposed or shifted probe axis is invisible to a whole-product
/// comparison, and these values are small enough to check by hand.
#[test]
fn contract_axis_diag_probe_keeps_each_probe_column_separate() {
    // Input `2 x 2 x 3` column-major (`rest x n x k`), contract `axis = 1`.
    let input = Dense {
        dims: vec![2, 2, 3],
        data: (1..=12).map(|value| value as f64).collect(),
    };
    let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out = contract_axis_diag_probe(&cpu_context(), input, 1, &weights).unwrap();
    assert_eq!(out.dims, vec![2, 3]);
    assert_eq!(out.data, vec![7.0, 10.0, 43.0, 50.0, 111.0, 122.0]);
}

#[cfg(feature = "tenferro-cuda")]
mod cuda {
    use super::*;

    fn cuda_context() -> ExecutionContext {
        ExecutionContext::Cuda(cuda_execution())
    }

    fn cuda_execution() -> Arc<tensor4all_tensorbackend::CudaExecutionContext> {
        cuda_execution_or_skip("a CUDA primitive")
            .expect("this caller requires a visible CUDA ordinal 0")
    }

    /// A CUDA context, or `None` with a printed SKIP on a host without one.
    ///
    /// The routing tests below are not `#[ignore]`d, so they run in CI: on a
    /// GPU-less host they must say that their CUDA assertions did not run
    /// instead of passing as if they had.
    fn cuda_execution_or_skip(
        name: &str,
    ) -> Option<Arc<tensor4all_tensorbackend::CudaExecutionContext>> {
        match tensor4all_tensorbackend::CudaExecutionContext::new() {
            Ok(cuda) => Some(Arc::new(cuda)),
            Err(error) => {
                println!(
                    "SKIP {name}: no usable CUDA ordinal 0 ({error}); CUDA assertions did not run"
                );
                None
            }
        }
    }

    fn cuda_einsums() -> usize {
        super::CUDA_EINSUMS.with(std::cell::Cell::get)
    }

    #[test]
    #[ignore = "requires visible CUDA ordinal 0"]
    fn contract_axis_diag_probe_matches_the_naive_probe_loop_f64_on_cuda() {
        assert_contract_axis_diag_probe_matches_naive::<f64>(&cuda_context());
    }

    #[test]
    #[ignore = "requires visible CUDA ordinal 0"]
    fn contract_axis_diag_probe_matches_the_naive_probe_loop_c64_on_cuda() {
        assert_contract_axis_diag_probe_matches_naive::<Complex64>(&cuda_context());
    }

    #[test]
    #[ignore = "requires visible CUDA ordinal 0"]
    fn contract_axis_diag_probe_keeps_each_probe_column_separate_on_cuda() {
        let input = Dense {
            dims: vec![2, 2, 3],
            data: (1..=12).map(|value| value as f64).collect(),
        };
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = contract_axis_diag_probe(&cuda_context(), input, 1, &weights).unwrap();
        assert_eq!(out.dims, vec![2, 3]);
        assert_eq!(out.data, vec![7.0, 10.0, 43.0, 50.0, 111.0, 122.0]);
    }

    /// The primitives must dispatch to the caller's CUDA engine, not to a host
    /// route that happens to return the same numbers.
    ///
    /// Holding this context's session makes the primitive's own entry fail with
    /// the typed re-entry before any transfer. A CPU fallback would not: the
    /// process-global CPU context is not entered here, so it would return a
    /// result and the assertion below would fail.
    #[test]
    fn cuda_primitives_enter_the_callers_context_engine() {
        let Some(cuda) = cuda_execution_or_skip("cuda_primitives_enter_the_callers_context_engine")
        else {
            return;
        };
        let context = ExecutionContext::Cuda(Arc::clone(&cuda));
        let (probe, product) = cuda
            .with_session(|_engine| {
                let probe = contract_axis_diag_probe(
                    &context,
                    Dense {
                        dims: vec![2, 2, 3],
                        data: (1..=12).map(|value| value as f64).collect(),
                    },
                    1,
                    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                );
                let product = mat_mul_cols(
                    &context,
                    vec![1.0, 3.0, 2.0, 4.0],
                    2,
                    2,
                    vec![5.0, 7.0, 6.0, 8.0],
                    2,
                );
                (probe, product)
            })
            .expect("holding one CUDA session succeeds");
        for (name, error) in [
            ("contract_axis_diag_probe", probe.expect_err("nested entry")),
            ("mat_mul_cols", product.expect_err("nested entry")),
        ] {
            let TreeRsiError::Numerical { source } = error else {
                panic!("{name} must fail on the CUDA session entry: {error}");
            };
            assert!(
                matches!(
                    source.downcast_ref::<tensor4all_tensorbackend::CudaExecutionContextError>(),
                    Some(tensor4all_tensorbackend::CudaExecutionContextError::Reentered)
                ),
                "{name} must reach this context's engine and be rejected on the \
                 nested entry, not run somewhere else: {source}"
            );
        }
    }

    /// Every CUDA dispatch is counted, so "the primitive ran on the device" is
    /// observed instead of inferred from matching values.
    ///
    /// Swapping the CUDA arm in `einsum_host` for a host call leaves the numbers
    /// correct and this counter unchanged, which is what makes the routing
    /// testable at all. The CPU arm must stay uncounted, or the counter would
    /// not separate the two routes.
    #[test]
    fn cuda_primitives_dispatch_to_the_device_arm() {
        let Some(cuda) = cuda_execution_or_skip("cuda_primitives_dispatch_to_the_device_arm")
        else {
            return;
        };
        let context = ExecutionContext::Cuda(Arc::clone(&cuda));
        let probe_input = || Dense {
            dims: vec![2, 2, 3],
            data: (1..=12).map(|value| value as f64).collect(),
        };

        let before = cuda_einsums();
        contract_axis_diag_probe(&context, probe_input(), 1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        mat_mul_cols(
            &context,
            vec![1.0, 3.0, 2.0, 4.0],
            2,
            2,
            vec![5.0, 7.0, 6.0, 8.0],
            2,
        )
        .unwrap();
        let dispatched = cuda_einsums() - before;
        assert_eq!(
            dispatched, 2,
            "each primitive must dispatch exactly one einsum through the CUDA engine, saw \
             {dispatched}"
        );

        contract_axis_diag_probe(
            &cpu_context(),
            probe_input(),
            1,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        assert_eq!(
            cuda_einsums(),
            before + dispatched,
            "a CPU-context call must not be counted as a device dispatch"
        );
    }

    #[test]
    #[ignore = "requires visible CUDA ordinal 0"]
    fn mat_mul_cols_matches_the_cpu_route_on_cuda() {
        let a = vec![1.0_f64, 3.0, 2.0, 4.0];
        let b = vec![5.0_f64, 7.0, 6.0, 8.0];
        let cpu = mat_mul_cols(&cpu_context(), a.clone(), 2, 2, b.clone(), 2).unwrap();
        let cuda = mat_mul_cols(&cuda_context(), a, 2, 2, b, 2).unwrap();
        // 2x2 column-major `[1 2; 3 4] [5 6; 7 8] = [19 22; 43 50]`.
        assert_eq!(cpu, vec![19.0, 43.0, 22.0, 50.0]);
        assert_eq!(cuda, cpu);
    }
}
