//! Differential tests of the fused update-and-argmax kernel (#1164) against the
//! two-pass code it replaced: the real `submatrix_argmax_col_major`, and the
//! pre-#1164 `update_trailing_submatrix` and factorization loop, restored
//! verbatim below as test oracles.
//!
//! Values are compared bit for bit, except that any two NaNs compare equal: Rust
//! does not specify NaN payloads, so they are not part of the bitwise contract.

use super::super::*;
use num_complex::Complex64;
use tensor4all_tensorbackend::Matrix;

/// The pre-#1164 rank-1 update of pivot step `pivot`, verbatim.
fn two_pass_update<T: Scalar>(data: &mut [T], nrows: usize, ncols: usize, pivot: usize) {
    let tail_row_start = pivot + 1;
    let tail_col_start = pivot + 1;
    if tail_row_start >= nrows || tail_col_start >= ncols {
        return;
    }

    let tail_len = nrows - tail_row_start;
    let pivot_col_tail_start = col_major_offset(nrows, tail_row_start, pivot);
    for col in tail_col_start..ncols {
        let y = col_major_get(data, nrows, pivot, col);
        let target_start = col_major_offset(nrows, tail_row_start, col);
        let (before_target, target_and_after) = data.split_at_mut(target_start);
        let pivot_col_tail = &before_target[pivot_col_tail_start..pivot_col_tail_start + tail_len];
        let target_tail = &mut target_and_after[..tail_len];
        for (target, &x) in target_tail.iter_mut().zip(pivot_col_tail.iter()) {
            *target = *target - x * y;
        }
    }
}

/// The pre-#1164 factorization loop, verbatim but for the two-pass update above.
fn two_pass_factorize<T: Scalar>(
    a: &mut Matrix<T>,
    opts: RrLUOptions,
) -> Result<RrLUFactorization> {
    let nr = a.nrows();
    let nc = a.ncols();
    let data = a.as_col_major_mut_slice();
    validate_col_major_matrix_len(nr, nc, data.len())?;

    let max_bond_dim = opts.max_bond_dim.min(nr).min(nc);
    let mut row_permutation = (0..nr).collect::<Vec<_>>();
    let mut col_permutation = (0..nc).collect::<Vec<_>>();
    let mut n_pivot = 0;
    let mut error = f64::NAN;
    let mut max_error = 0.0f64;

    while n_pivot < max_bond_dim {
        let k = n_pivot;
        if k >= nr || k >= nc {
            break;
        }

        let (pivot_row, pivot_col, pivot_val) =
            submatrix_argmax_col_major(data, nr, nc, k, nr, k, nc);
        let pivot_abs = pivot_val.abs_val();
        error = pivot_abs;
        if !pivot_abs.is_finite() {
            return Err(MatrixCIError::NaNEncountered {
                matrix: "residual".to_string(),
            });
        }

        if n_pivot > 0 && (pivot_abs < opts.rel_tol * max_error || pivot_abs < opts.abs_tol) {
            break;
        }

        if pivot_is_undividable::<T>(pivot_abs) {
            break;
        }

        max_error = max_error.max(pivot_abs);

        if pivot_row != k {
            swap_rows_col_major(data, nr, nc, k, pivot_row);
            row_permutation.swap(k, pivot_row);
        }
        if pivot_col != k {
            swap_cols_col_major(data, nr, nc, k, pivot_col);
            col_permutation.swap(k, pivot_col);
        }

        let pivot = col_major_get(data, nr, k, k);

        if opts.left_orthogonal {
            scale_column_tail(data, nr, k, k + 1, pivot);
        } else {
            scale_row_tail(data, nr, nc, k, k + 1, pivot);
        }

        two_pass_update(data, nr, nc, k);
        n_pivot += 1;
    }

    if n_pivot >= nr.min(nc) {
        error = 0.0;
    }

    Ok(RrLUFactorization {
        row_permutation,
        col_permutation,
        n_pivot,
        error,
    })
}

/// The real component types the kernel is generic over, with the magnitudes
/// that decide its fallback.
trait TestReal: Scalar + std::fmt::LowerExp {
    /// A value whose square is a non-zero sub-normal of this type.
    const SQUARE_SUBNORMAL: Self;
    /// A finite value whose square overflows to infinity in this type.
    const SQUARE_OVERFLOW: Self;
    /// A sub-normal value (its square underflows to zero).
    const SUBNORMAL: Self;
    const NAN: Self;
    const INFINITY: Self;
    fn of(value: f64) -> Self;
    fn up(self) -> Self;
    fn is_nan_value(self) -> bool;
    fn bits(self) -> u64;
}

impl TestReal for f64 {
    const SQUARE_SUBNORMAL: Self = 1e-160;
    const SQUARE_OVERFLOW: Self = 1e160;
    const SUBNORMAL: Self = 1e-310;
    const NAN: Self = f64::NAN;
    const INFINITY: Self = f64::INFINITY;
    fn of(value: f64) -> Self {
        value
    }
    fn up(self) -> Self {
        self.next_up()
    }
    fn is_nan_value(self) -> bool {
        self.is_nan()
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

impl TestReal for f32 {
    const SQUARE_SUBNORMAL: Self = 1e-20;
    const SQUARE_OVERFLOW: Self = 1e20;
    const SUBNORMAL: Self = 1e-42;
    const NAN: Self = f32::NAN;
    const INFINITY: Self = f32::INFINITY;
    fn of(value: f64) -> Self {
        value as f32
    }
    fn up(self) -> Self {
        self.next_up()
    }
    fn is_nan_value(self) -> bool {
        self.is_nan()
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

/// xorshift64: deterministic inputs without a dependency on the rand stream.
struct Gen(u64);

impl Gen {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// A small integer in -3..=3: ties are frequent.
    fn small(&mut self) -> f64 {
        self.below(7) as f64 - 3.0
    }
}

/// The value families that decide the scan's semantics.
const FAMILIES: [&str; 12] = [
    "small integers",
    "wide exponents",
    "signed zeros",
    "NaN anywhere",
    "NaN at the largest entry",
    "NaN and inf",
    "inf only",
    "sub-normals (squares underflow to zero)",
    "equal maxima in two columns (positions may coincide)",
    "all-zero column",
    "all zero",
    "sub-normal squares (non-zero, equal for neighbouring values)",
];

fn matrix_of<T: TestReal>(family: usize, nrows: usize, ncols: usize, g: &mut Gen) -> Vec<T> {
    let len = nrows * ncols;
    let mut data: Vec<T> = (0..len).map(|_| T::of(g.small())).collect();
    let at = |g: &mut Gen| g.below(len);
    match family {
        1 => {
            for v in &mut data {
                let exponent = g.below(61) as i32 - 30;
                *v = T::of((g.small() + 0.5) * 2f64.powi(exponent));
            }
        }
        2 => {
            for v in &mut data {
                *v = T::of([0.0, -0.0, 1.0, -1.0][g.below(4)]);
            }
        }
        3 => {
            for _ in 0..=g.below(2) {
                let i = at(g);
                data[i] = T::NAN;
            }
        }
        4 => {
            // NaN where the largest magnitude was (its first position, column-major).
            let largest = data.iter().map(|v| v.abs_val()).fold(0.0, f64::max);
            let i = data
                .iter()
                .position(|v| v.abs_val() == largest)
                .unwrap_or(0);
            data[i] = T::NAN;
            for v in data.iter_mut().filter(|v| !v.is_nan_value()) {
                *v = *v * T::of(0.5);
            }
        }
        5 => {
            let (i, j) = (at(g), at(g));
            data[i] = T::NAN;
            data[j] = if g.below(2) == 0 {
                T::INFINITY
            } else {
                T::of(-1.0) * T::INFINITY
            };
        }
        6 => {
            let i = at(g);
            data[i] = T::INFINITY;
        }
        7 => {
            for v in &mut data {
                *v = *v * T::SUBNORMAL;
            }
        }
        8 if ncols >= 2 => {
            let (c1, c2) = (g.below(ncols), g.below(ncols));
            let (r1, r2) = (g.below(nrows), g.below(nrows));
            let value = T::of(if g.below(2) == 0 { 9.0 } else { -9.0 });
            data[r1 + nrows * c1] = value;
            data[r2 + nrows * c2] = T::of(-1.0) * value;
        }
        9 => {
            let c = g.below(ncols);
            data[c * nrows..(c + 1) * nrows].fill(T::of(0.0));
        }
        10 => data.fill(T::of(0.0)),
        11 => {
            let a = T::SQUARE_SUBNORMAL;
            for v in &mut data {
                *v = [T::of(0.0), a, a.up(), T::of(-1.0) * a, T::of(-1.0) * a.up()][g.below(5)];
            }
        }
        _ => {}
    }
    data
}

fn same_bits<T: TestReal>(a: T, b: T) -> bool {
    (a.is_nan_value() && b.is_nan_value()) || a.bits() == b.bits()
}

fn assert_same_data<T: TestReal>(fused: &[T], reference: &[T], what: &str) {
    for (i, (&a, &b)) in fused.iter().zip(reference).enumerate() {
        assert!(
            same_bits(a, b),
            "{what}: entry {i} is {a:e}, reference {b:e}"
        );
    }
}

/// One shape list for every comparison: 1 × 1, one row, one column, fewer rows
/// than the 8 lanes and around them, and 150 random shapes up to 40 × 12.
fn shapes() -> Vec<(usize, usize)> {
    let mut g = Gen(0x2545_f491_4f6c_dd1d);
    let fixed = [
        (1, 1),
        (1, 9),
        (9, 1),
        (7, 3),
        (8, 3),
        (9, 3),
        (17, 2),
        (2, 17),
    ];
    fixed
        .into_iter()
        .chain((0..150).map(|_| (1 + g.below(40), 1 + g.below(12))))
        .collect()
}

/// The kernel at every pivot of every generated matrix: position, value and the
/// updated block equal the two-pass code's; a fallback (`None`) only where the
/// sequential scan's fast path leaves the normal range. Returns, per family, how
/// many cases fell back.
fn compare_kernel<T: TestReal>(seed: u64) -> [usize; FAMILIES.len()] {
    let mut g = Gen(seed);
    let mut fallbacks = [0usize; FAMILIES.len()];
    let mut cases = 0usize;
    for (nrows, ncols) in shapes() {
        for (family, name) in FAMILIES.iter().enumerate() {
            let original = matrix_of::<T>(family, nrows, ncols, &mut g);
            for pivot in 0..nrows.min(ncols) {
                let what = format!("{name} {nrows}x{ncols} pivot {pivot}");
                let mut reference = original.clone();
                two_pass_update(&mut reference, nrows, ncols, pivot);
                let mut fused = original.clone();
                let found = update_trailing_submatrix_with_argmax(&mut fused, nrows, ncols, pivot);
                assert_same_data(&fused, &reference, &what);
                cases += 1;

                let tail = pivot + 1;
                if tail >= nrows || tail >= ncols {
                    assert!(found.is_none(), "{what}: empty block");
                    continue;
                }
                let (row, col, value) =
                    submatrix_argmax_col_major(&reference, nrows, ncols, tail, nrows, tail, ncols);
                match found {
                    Some((found_row, found_col, found_value)) => {
                        assert_eq!((found_row, found_col), (row, col), "{what}");
                        assert!(same_bits(found_value, value), "{what}");
                    }
                    None => {
                        let max = value.abs_sq();
                        assert!(
                            !max.is_finite() || max < T::min_positive(),
                            "{what}: fused fell back with a normal maximum {value:e}"
                        );
                        fallbacks[family] += 1;
                    }
                }
            }
        }
    }
    assert!(cases > 5_000, "{cases} cases");
    fallbacks
}

#[test]
fn fused_kernel_matches_two_pass_update_and_sequential_scan() {
    for fallbacks in [
        compare_kernel::<f64>(0x9e37_79b9_7f4a_7c15),
        compare_kernel::<f32>(0x6a09_e667_f3bc_c908),
    ] {
        // Every fallback family is exercised, so the fallback is not the untested half.
        for family in [3, 4, 5, 6, 7, 10, 11] {
            assert!(
                fallbacks[family] > 0,
                "no fallback for {}",
                FAMILIES[family]
            );
        }
    }
}

/// #1178 review: squares that round to the same non-zero sub-normal. The robust
/// scan tells `a` from `a.next_up()`; the fast path's squares cannot, so the
/// kernel must fall back below the component type's normal range.
fn subnormal_square_tie<T: TestReal>() {
    let a = T::SQUARE_SUBNORMAL;
    let b = a.up();
    assert!(a.abs_sq() == b.abs_sq() && a.abs_sq() > 0.0);
    assert!(a.abs_sq() < T::min_positive());
    let zero = T::of(0.0);
    // The first pivot is small enough that the default rel_tol (1e-14 × it) lets
    // the factorization take a second pivot from the sub-normal-square block.
    let first = a * T::of(1e10);
    let original = vec![first, zero, zero, a, zero, b];
    let mut fused = original.clone();
    let found = update_trailing_submatrix_with_argmax(&mut fused, 2, 3, 0);
    assert!(found.is_none(), "a sub-normal maximum falls back");
    let mut reference = original.clone();
    two_pass_update(&mut reference, 2, 3, 0);
    assert_same_data(&fused, &reference, "sub-normal squares");
    let (row, col, _) = submatrix_argmax_col_major(&reference, 2, 3, 1, 2, 1, 3);
    assert_eq!((row, col), (1, 2), "the robust scan picks b");

    // And through the whole factorization, where the fallback is the caller's scan.
    let mut fused = Matrix::from_col_major_vec(2, 3, original.clone());
    let mut reference = Matrix::from_col_major_vec(2, 3, original);
    let f = factorize_validated_in_place(&mut fused, RrLUOptions::default()).unwrap();
    let r = two_pass_factorize(&mut reference, RrLUOptions::default()).unwrap();
    assert_eq!((f.n_pivot, &f.col_permutation), (2, &r.col_permutation));
    assert_eq!(r.col_permutation, [0, 2, 1]);
}

/// #1185: finite entries whose squares overflow to infinity. The fast path's
/// squares are all `inf`, so the kernel must fall back; the robust scan compares
/// magnitudes and picks `2a` over `a`.
fn overflowing_square_tie<T: TestReal>() {
    let a = T::SQUARE_OVERFLOW;
    let b = a * T::of(2.0);
    assert!(a.abs_sq() == f64::INFINITY && b.abs_val().is_finite());
    let zero = T::of(0.0);
    let original = vec![a * T::of(1e10), zero, zero, a, zero, b];
    let mut fused = original.clone();
    let found = update_trailing_submatrix_with_argmax(&mut fused, 2, 3, 0);
    assert!(found.is_none(), "an infinite squared maximum falls back");
    let mut reference = original.clone();
    two_pass_update(&mut reference, 2, 3, 0);
    assert_same_data(&fused, &reference, "overflowing squares");
    let (row, col, _) = submatrix_argmax_col_major(&reference, 2, 3, 1, 2, 1, 3);
    assert_eq!((row, col), (1, 2), "the robust scan picks 2a");

    let mut fused = Matrix::from_col_major_vec(2, 3, original.clone());
    let mut reference = Matrix::from_col_major_vec(2, 3, original);
    let f = factorize_validated_in_place(&mut fused, RrLUOptions::default()).unwrap();
    let r = two_pass_factorize(&mut reference, RrLUOptions::default()).unwrap();
    assert_eq!((f.n_pivot, &f.col_permutation), (2, &r.col_permutation));
    assert_eq!(r.col_permutation, [0, 2, 1]);
}

#[test]
fn overflowing_square_ties_fall_back_for_f64() {
    overflowing_square_tie::<f64>();
}

#[test]
fn overflowing_square_ties_fall_back_for_f32() {
    overflowing_square_tie::<f32>();
}

/// #1185: the "NaN at the largest entry" family puts the NaN where the largest
/// magnitude was.
#[test]
fn nan_family_replaces_the_largest_entry() {
    let mut g = Gen(2);
    let data = matrix_of::<f64>(4, 2, 2, &mut g);
    let mut g = Gen(2);
    let before = matrix_of::<f64>(0, 2, 2, &mut g);
    let largest = before.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let i = before.iter().position(|v| v.abs() == largest).unwrap();
    assert!(data[i].is_nan(), "{before:?} -> {data:?}");
    assert_eq!(data.iter().filter(|v| v.is_nan()).count(), 1);
}

#[test]
fn subnormal_square_ties_fall_back_for_f64() {
    subnormal_square_tie::<f64>();
}

#[test]
fn subnormal_square_ties_fall_back_for_f32() {
    subnormal_square_tie::<f32>();
}

/// Equal maxima in two distinct columns, the later column holding the earlier
/// row, in a block the update leaves unchanged (the pivot column's tail is zero):
/// the first column wins, as in the column-major sequential scan. Signed values
/// and signed zeros around them.
#[test]
fn equal_maxima_in_distinct_columns_keep_the_first_column() {
    let mut g = Gen(0x3c6e_f372_fe94_f82b);
    let mut checked = 0;
    for (nrows, ncols) in shapes().into_iter().filter(|&(r, c)| r >= 3 && c >= 3) {
        let mut data: Vec<f64> = (0..nrows * ncols).map(|_| g.small()).collect();
        data[1..nrows].fill(if g.below(2) == 0 { 0.0 } else { -0.0 });
        let (c1, c2) = (1 + g.below(ncols - 2), 0);
        let c2 = c2.max(c1 + 1 + g.below(ncols - 1 - c1));
        let r1 = 2 + g.below(nrows - 2);
        let r2 = 1 + g.below(r1 - 1);
        let value = if g.below(2) == 0 { 9.0 } else { -9.0 };
        data[r1 + nrows * c1] = value;
        data[r2 + nrows * c2] = -value;
        let mut fused = data.clone();
        let found = update_trailing_submatrix_with_argmax(&mut fused, nrows, ncols, 0);
        assert_eq!(fused, data, "the block is unchanged");
        assert_eq!(found, Some((r1, c1, value)), "{nrows}x{ncols}");
        assert_eq!(
            found,
            Some(submatrix_argmax_col_major(
                &data, nrows, ncols, 1, nrows, 1, ncols
            ))
        );
        checked += 1;
    }
    assert!(checked > 100, "{checked} shapes");
}

fn options() -> [RrLUOptions; 4] {
    let base = RrLUOptions::default();
    [
        base.clone(),
        RrLUOptions {
            left_orthogonal: false,
            ..base.clone()
        },
        RrLUOptions {
            max_bond_dim: 3,
            ..base.clone()
        },
        RrLUOptions {
            abs_tol: 0.5,
            rel_tol: 1e-3,
            ..base
        },
    ]
}

/// The whole factorization: permutations, rank, error and the factored data equal
/// those of the two-pass loop, over the same shapes and families as the kernel.
fn compare_factorization<T: TestReal>(seed: u64) {
    let mut g = Gen(seed);
    for (nrows, ncols) in shapes() {
        for (family, name) in FAMILIES.iter().enumerate() {
            let values = matrix_of::<T>(family, nrows, ncols, &mut g);
            for opts in options() {
                let what = format!("{name} {nrows}x{ncols} {opts:?}");
                let mut fused = Matrix::from_col_major_vec(nrows, ncols, values.clone());
                let mut reference = Matrix::from_col_major_vec(nrows, ncols, values.clone());
                let fused_state = factorize_validated_in_place(&mut fused, opts.clone());
                let reference_state = two_pass_factorize(&mut reference, opts);
                match (fused_state, reference_state) {
                    (Ok(f), Ok(r)) => {
                        assert_eq!(f.row_permutation, r.row_permutation, "{what}");
                        assert_eq!(f.col_permutation, r.col_permutation, "{what}");
                        assert_eq!(f.n_pivot, r.n_pivot, "{what}");
                        assert!(same_bits(f.error, r.error), "{what}");
                    }
                    (Err(_), Err(_)) => {}
                    (f, r) => panic!("{what}: fused {:?}, reference {:?}", f.is_ok(), r.is_ok()),
                }
                assert_same_data(
                    fused.as_col_major_slice(),
                    reference.as_col_major_slice(),
                    &what,
                );
            }
        }
    }
}

#[test]
fn fused_factorization_matches_two_pass_factorization() {
    compare_factorization::<f64>(0xd1b5_4a32_d192_ed03);
    compare_factorization::<f32>(0x510e_527f_ade6_82d1);
}

/// The kernel is generic: complex entries tie on |z|² with different phases.
#[test]
fn fused_factorization_matches_two_pass_on_complex_ties() {
    let mut g = Gen(0x5851_f42d_4c95_7f2d);
    for (nrows, ncols) in shapes() {
        let values: Vec<Complex64> = (0..nrows * ncols)
            .map(|_| Complex64::new(g.small(), g.small()))
            .collect();
        for opts in options() {
            let mut fused = Matrix::from_col_major_vec(nrows, ncols, values.clone());
            let mut reference = Matrix::from_col_major_vec(nrows, ncols, values.clone());
            let f = factorize_validated_in_place(&mut fused, opts.clone()).unwrap();
            let r = two_pass_factorize(&mut reference, opts).unwrap();
            assert_eq!(
                (
                    f.row_permutation,
                    f.col_permutation,
                    f.n_pivot,
                    f.error.to_bits()
                ),
                (
                    r.row_permutation,
                    r.col_permutation,
                    r.n_pivot,
                    r.error.to_bits()
                ),
                "{nrows}x{ncols}"
            );
            let bits = |m: &Matrix<Complex64>| -> Vec<(u64, u64)> {
                m.as_col_major_slice()
                    .iter()
                    .map(|z| (z.re.to_bits(), z.im.to_bits()))
                    .collect()
            };
            assert_eq!(bits(&fused), bits(&reference), "{nrows}x{ncols}");
        }
    }
}
