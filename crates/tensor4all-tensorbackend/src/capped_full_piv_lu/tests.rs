use super::capped_full_piv_lu;
use crate::{full_piv_lu_matrix_owned, Matrix};
use num_complex::{Complex32, Complex64};

fn lcg(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

/// Row index of the single unit entry in each row of a permutation matrix.
fn permutation_of(matrix: &Matrix<f64>) -> Vec<usize> {
    let n = matrix.nrows();
    (0..n)
        .map(|row| (0..n).find(|&col| matrix[[row, col]] > 0.5).unwrap())
        .collect()
}

/// Uncapped, the in-place factorization is the backend's complete-pivoting LU: the same row and
/// column permutations and the same `U` diagonal, bit for bit, as `full_piv_lu_matrix_owned`.
#[test]
fn an_uncapped_run_is_the_backend_full_piv_lu() {
    for n in [1usize, 2, 8, 37] {
        let data = lcg(n * n, 3 + n as u64);
        let capped =
            capped_full_piv_lu(Matrix::from_col_major_vec(n, n, data.clone()), None).unwrap();
        let full = full_piv_lu_matrix_owned(Matrix::from_col_major_vec(n, n, data)).unwrap();

        assert_eq!(capped.rank(), n, "n={n}: every pivot is taken");
        assert_eq!(capped.row_perm(), permutation_of(&full.p), "n={n}: rows");
        assert_eq!(capped.col_perm(), permutation_of(&full.q), "n={n}: cols");
        for (i, value) in capped.diagonal().iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                full.u[[i, i]].to_bits(),
                "n={n}: U diagonal entry {i}"
            );
        }
    }
}

/// A rank cap stops the elimination there: on a full-rank input the count is the cap, not `n`, and
/// the pivots it did take are the uncapped run's leading pivots.
#[test]
fn a_rank_cap_stops_the_elimination_at_the_cap() {
    let n = 64usize;
    let data = lcg(n * n, 29);
    let full = capped_full_piv_lu(Matrix::from_col_major_vec(n, n, data.clone()), None).unwrap();
    assert_eq!(full.rank(), n);

    for cap in [1usize, 7, 16, 63] {
        let capped =
            capped_full_piv_lu(Matrix::from_col_major_vec(n, n, data.clone()), Some(cap)).unwrap();
        assert_eq!(capped.rank(), cap, "cap={cap}: stops at the cap");
        assert_eq!(
            capped.row_perm()[..cap],
            full.row_perm()[..cap],
            "cap={cap}"
        );
        assert_eq!(
            capped.col_perm()[..cap],
            full.col_perm()[..cap],
            "cap={cap}"
        );
        for i in 0..cap {
            assert_eq!(
                capped.diagonal()[i].to_bits(),
                full.diagonal()[i].to_bits(),
                "cap={cap}: pivot {i}"
            );
        }
    }
}

/// A subnormal Schur complement stops the elimination before the cap does, and the count says so;
/// the trailing diagonal is left as it was, un-eliminated.
#[test]
fn a_subnormal_block_stops_before_the_cap() {
    let data = vec![
        1e-308f64, 0.0, 0.0, // column 0
        0.0, 9e-309, 0.0, // column 1
        0.0, 0.0, 8e-309, // column 2
    ];
    let lu = capped_full_piv_lu(Matrix::from_col_major_vec(3, 3, data), Some(2)).unwrap();
    assert_eq!(lu.rank(), 0, "faer stops before taking the cap");
    assert_eq!(lu.diagonal(), &[1e-308, 9e-309, 8e-309]);
}

/// The score faer's pivot search maximises over a complex trailing block. Which one runs is decided
/// by the runtime SIMD dispatch (`pulp::Arch::new`), so both are part of the contract and every
/// complex pivot-order test states its expectation under each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComplexPivotScore {
    /// Vector dispatch: `max(|re|, |im|)`, the real and imaginary parts compared lane by lane.
    LargerPart,
    /// Scalar dispatch: `|re| + |im|`.
    PartSum,
}

/// The score this host runs, read from the CPU independently of faer. It mirrors pulp 0.22's
/// `Arch::new`: x86-64-v3 (or v4) on x86, NEON on aarch64, simd128 on wasm32 dispatch to vectors;
/// anything else to the scalar path.
fn host_complex_pivot_score() -> ComplexPivotScore {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let v3 = std::arch::is_x86_feature_detected!("sse")
            && std::arch::is_x86_feature_detected!("sse2")
            && std::arch::is_x86_feature_detected!("fxsr")
            && std::arch::is_x86_feature_detected!("sse3")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1")
            && std::arch::is_x86_feature_detected!("sse4.2")
            && std::arch::is_x86_feature_detected!("popcnt")
            && std::arch::is_x86_feature_detected!("avx")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("lzcnt");
        if v3 {
            return ComplexPivotScore::LargerPart;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return ComplexPivotScore::LargerPart;
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return ComplexPivotScore::LargerPart;
    }
    #[allow(unreachable_code)]
    ComplexPivotScore::PartSum
}

/// The expectation under the score this host runs; both arms are written out by the caller.
fn under<T>(score: ComplexPivotScore, larger_part: T, part_sum: T) -> T {
    match score {
        ComplexPivotScore::LargerPart => larger_part,
        ComplexPivotScore::PartSum => part_sum,
    }
}

/// `diag(10, 3, 2.5 + 2.5i)` capped one past the leading pivot: the larger part ranks `3` (3 > 2.5)
/// second, the part sum ranks `2.5 + 2.5i` (5 > 3) second. The modulus (`3.54 > 3`) agrees with the
/// part sum only, so a caller cannot re-derive the order from the diagonal by hand.
#[test]
fn a_complex_pivot_follows_the_dispatched_score() {
    let score = host_complex_pivot_score();
    let zero = Complex64::new(0.0, 0.0);
    let data = vec![
        Complex64::new(10.0, 0.0),
        zero,
        zero,
        zero,
        Complex64::new(3.0, 0.0),
        zero,
        zero,
        zero,
        Complex64::new(2.5, 2.5),
    ];
    let lu = capped_full_piv_lu(Matrix::from_col_major_vec(3, 3, data), Some(2)).unwrap();
    assert_eq!(lu.rank(), 2);
    assert_eq!(
        lu.diagonal()[..2],
        under(
            score,
            [Complex64::new(10.0, 0.0), Complex64::new(3.0, 0.0)],
            [Complex64::new(10.0, 0.0), Complex64::new(2.5, 2.5)],
        ),
        "{score:?}"
    );
    assert_eq!(
        lu.row_perm()[..2],
        under(score, [0, 1], [0, 2]),
        "{score:?}"
    );
}

/// The 16x16 case measured in tensor4all/faer#342: a zero matrix with `A[(0,1)] = 3 + 4i` and
/// `A[(2,1)] = 5` pivots on `(2,1)` under the larger part (5 > 4) and on `(0,1)` under the part sum
/// (7 > 5), for both complex widths.
#[test]
fn the_faer_342_case_follows_the_dispatched_score_in_both_widths() {
    let score = host_complex_pivot_score();
    let n = 16usize;

    let mut c64 = vec![Complex64::new(0.0, 0.0); n * n];
    c64[n] = Complex64::new(3.0, 4.0);
    c64[2 + n] = Complex64::new(5.0, 0.0);
    let lu = capped_full_piv_lu(Matrix::from_col_major_vec(n, n, c64), Some(1)).unwrap();
    assert_eq!(lu.rank(), 1);
    assert_eq!(lu.col_perm()[0], 1, "{score:?}");
    assert_eq!(lu.row_perm()[0], under(score, 2, 0), "{score:?}");
    assert_eq!(
        lu.diagonal()[0],
        under(score, Complex64::new(5.0, 0.0), Complex64::new(3.0, 4.0)),
        "{score:?}"
    );

    let mut c32 = vec![Complex32::new(0.0, 0.0); n * n];
    c32[n] = Complex32::new(3.0, 4.0);
    c32[2 + n] = Complex32::new(5.0, 0.0);
    let lu = capped_full_piv_lu(Matrix::from_col_major_vec(n, n, c32), Some(1)).unwrap();
    assert_eq!(lu.col_perm()[0], 1, "{score:?}");
    assert_eq!(lu.row_perm()[0], under(score, 2, 0), "{score:?}");
}

/// What `rank()` reports for a zero matrix is the intended contract, not an elimination count:
/// uncapped, no rank revelation is requested and it is the bound `n`; capped, the elimination's
/// zero stop is revealed and it is `0`. Both runs stop before the first pivot.
#[test]
fn rank_reports_the_bound_when_no_revelation_is_requested() {
    let zeros = || Matrix::from_col_major_vec(2, 2, vec![0.0f64; 4]);

    let uncapped = capped_full_piv_lu(zeros(), None).unwrap();
    assert_eq!(uncapped.rank(), 2, "None: the bound n, not a revealed rank");

    let capped = capped_full_piv_lu(zeros(), Some(2)).unwrap();
    assert_eq!(capped.rank(), 0, "Some(cap): the revealed zero stop");

    for lu in [&uncapped, &capped] {
        assert_eq!(lu.row_perm(), &[0, 1]);
        assert_eq!(lu.col_perm(), &[0, 1]);
        assert_eq!(lu.diagonal(), &[0.0, 0.0]);
    }
}

/// A non-finite entry switches the revelation off even under a cap: the count is `min(cap, n)`
/// although the elimination took no pivot.
#[test]
fn rank_reports_the_cap_for_non_finite_input() {
    let lu = capped_full_piv_lu(
        Matrix::from_col_major_vec(2, 2, vec![f64::NAN, 0.0, 0.0, 0.0]),
        Some(2),
    )
    .unwrap();
    assert_eq!(lu.rank(), 2);
}

/// The narrow scalars factor too, so every dense LUCI scalar has a backend kernel.
#[test]
fn narrow_scalars_factor() {
    let f32_lu = capped_full_piv_lu(
        Matrix::from_col_major_vec(2, 2, vec![1.0f32, 0.0, 0.0, 4.0]),
        Some(1),
    )
    .unwrap();
    assert_eq!(f32_lu.rank(), 1);
    assert_eq!(f32_lu.row_perm()[0], 1);
    assert_eq!(f32_lu.col_perm()[0], 1);
    assert_eq!(f32_lu.diagonal()[0], 4.0);

    let c32_lu = capped_full_piv_lu(
        Matrix::from_col_major_vec(
            2,
            2,
            vec![
                Complex32::new(0.0, 2.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(1.0, 0.0),
            ],
        ),
        None,
    )
    .unwrap();
    assert_eq!(c32_lu.rank(), 2);
    assert_eq!(c32_lu.diagonal()[0], Complex32::new(0.0, 2.0));
}

/// An empty matrix factors to nothing; a rectangular one is refused typed rather than panicking.
#[test]
fn empty_is_trivial_and_rectangular_is_refused() {
    let empty =
        capped_full_piv_lu(Matrix::<f64>::from_col_major_vec(0, 0, vec![]), Some(3)).unwrap();
    assert_eq!(empty.rank(), 0);
    assert!(empty.row_perm().is_empty() && empty.col_perm().is_empty());
    assert!(empty.diagonal().is_empty());

    let error = capped_full_piv_lu(Matrix::from_col_major_vec(2, 3, vec![0.0f64; 6]), None)
        .expect_err("a rectangular input is refused");
    assert!(
        error.to_string().contains("square"),
        "the refusal names the shape requirement: {error}"
    );
}
