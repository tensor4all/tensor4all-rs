use super::*;
use num_complex::Complex64;
use tensor4all_tensorbackend::CpuBackend;

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

fn assert_contract_axis_diag_probe_matches_naive<T: Fixture>() {
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap());
    // Rank-4 input `[a, b, n, k]`; the probe axis is last, so every `axis`
    // below yields a multi-probe (`k > 1`) call with a non-trivial rest shape
    // except for `axis == 0`, which covers the `rows == 1` rank-2 corner.
    let dims = [2usize, 3, 4, 5];
    let k = dims[3];
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
            &context,
            Dense {
                dims: dims.to_vec(),
                data: data.clone(),
            },
            axis,
            &weights,
        )
        .unwrap();

        // Rebuild the permuted `rest.. x n x k` reference buffer with the same
        // axis order the kernel applies.
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
        let scale = expected
            .iter()
            .map(|value| tensor4all_core::Scalar::abs_val(*value))
            .fold(0.0f64, f64::max);
        assert!(scale > 1.0e-3, "fixture must not be degenerate");
        let residual = actual
            .data
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).abs_val())
            .fold(0.0f64, f64::max);
        assert!(
            residual <= 1.0e-13 * scale,
            "diag-probe einsum disagrees with the naive loop: \
             residual {residual:.3e}, scale {scale:.3e}, axis {axis}"
        );
    }
}

#[test]
fn contract_axis_diag_probe_matches_the_naive_probe_loop_f64() {
    assert_contract_axis_diag_probe_matches_naive::<f64>();
}

#[test]
fn contract_axis_diag_probe_matches_the_naive_probe_loop_c64() {
    assert_contract_axis_diag_probe_matches_naive::<Complex64>();
}
