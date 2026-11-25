use num_complex::Complex64;
use tensor4all_core::{MatrixLuciScalar, MultiIndex, Scalar};
use tensor4all_simplett::{AbstractTensorTrain, TTScalar};
use tensor4all_tensorci::{crossinterpolate2, PivotSearchStrategy, TCI2Options};

fn options(pivot_search: PivotSearchStrategy) -> TCI2Options {
    TCI2Options {
        pivot_search,
        tolerance: 1e-12,
        nsearch: 0,
        max_nglobal_pivot: 0,
        strictly_nested: true,
        seed: Some(0),
        ..Default::default()
    }
}

fn assert_reconstruction<T: Scalar + TTScalar + Default + MatrixLuciScalar>(
    dims: &[usize],
    actual: &[T],
    oracle: impl Fn(&MultiIndex) -> T,
) {
    assert_eq!(actual.len(), dims.iter().product::<usize>());
    // Build the reference in the column-major order of full_tensor().
    let max_error = actual
        .iter()
        .enumerate()
        .map(|(mut offset, &actual)| {
            let index = dims
                .iter()
                .map(|&dim| {
                    let coordinate = offset % dim;
                    offset /= dim;
                    coordinate
                })
                .collect();
            Scalar::abs_sq(actual - oracle(&index)).sqrt()
        })
        .fold(0.0_f64, f64::max);
    assert!(max_error < 1e-10, "max reconstruction error = {max_error}");
}

fn rank_two_fixture<T>(mix: impl Fn(f64, f64) -> T)
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
{
    for dims in [vec![3, 4, 2], vec![3, 1, 4, 1, 2]] {
        let f = |index: &MultiIndex| {
            assert_eq!(index.len(), dims.len());
            // Opposite slopes make the initial paths independent at every cut.
            let (left, right) =
                index
                    .iter()
                    .zip(&dims)
                    .fold((1.0, 1.0), |(left, right), (&coordinate, &dim)| {
                        assert!(coordinate < dim);
                        (
                            left * (1.0 + coordinate as f64 / 4.0),
                            right * (1.0 + (dim - 1 - coordinate) as f64 / 8.0),
                        )
                    });
            mix(left, right)
        };
        for strategy in [PivotSearchStrategy::Full, PivotSearchStrategy::Rook] {
            let result = crossinterpolate2::<_, _, fn(&[MultiIndex]) -> Vec<T>>(
                f,
                None,
                dims.clone(),
                vec![vec![0; dims.len()], dims.iter().map(|d| d - 1).collect()],
                options(strategy),
            )
            .unwrap();
            let tt = result.tci.to_tensor_train().unwrap();
            assert_eq!(tt.link_dims(), vec![2; dims.len() - 1]);
            let (actual, shape) = tt.full_tensor().unwrap();
            assert_eq!(shape, dims);
            assert_reconstruction(&dims, &actual, f);
        }
    }
}

#[test]
fn scalar_sampling_reconstructs_real_rank_two() {
    rank_two_fixture(|left, right| left + 0.5 * right);
}

#[test]
fn scalar_sampling_reconstructs_complex_rank_two() {
    rank_two_fixture(Complex64::new);
}

#[test]
fn full_batch_requests_have_owned_storage_and_row_then_column_order() {
    let f = |index: &MultiIndex| {
        let row = index[0] as f64;
        let col = index[1] as f64;
        1.0 + 4.0 * row + col + row * col
    };
    let expected: Vec<_> = (0..2)
        .flat_map(|row| (0..3).map(move |col| vec![row, col]))
        .collect();
    let batches = std::cell::Cell::new(0);
    let batch = |indices: &[MultiIndex]| {
        batches.set(batches.get() + 1);
        assert_eq!(indices, expected);
        let pointers: std::collections::HashSet<_> =
            indices.iter().map(|index| index.as_ptr()).collect();
        assert_eq!(pointers.len(), indices.len());
        indices.iter().map(f).collect()
    };
    let result = crossinterpolate2(
        f,
        Some(batch),
        vec![2, 3],
        vec![vec![0, 0], vec![1, 2]],
        options(PivotSearchStrategy::Full),
    )
    .unwrap();
    assert!(batches.get() > 0);
    let (actual, shape) = result.tci.to_tensor_train().unwrap().full_tensor().unwrap();
    assert_eq!(shape, [2, 3]);
    assert_reconstruction(&shape, &actual, f);
}
