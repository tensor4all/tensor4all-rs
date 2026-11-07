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
const FAMILIES: [&str; 11] = [
    "small integers",
    "wide exponents",
    "signed zeros",
    "NaN anywhere",
    "NaN at the largest entry",
    "NaN and inf",
    "inf only",
    "sub-normals",
    "equal maxima in two columns",
    "all-zero column",
    "all zero",
];

fn matrix_of(family: usize, nrows: usize, ncols: usize, g: &mut Gen) -> Vec<f64> {
    let len = nrows * ncols;
    let mut data: Vec<f64> = (0..len).map(|_| g.small()).collect();
    let at = |g: &mut Gen| g.below(len);
    match family {
        0 => {}
        1 => {
            for v in &mut data {
                let exponent = g.below(81) as i32 - 40;
                *v = (g.small() + 0.5) * 2f64.powi(exponent);
            }
        }
        2 => {
            for v in &mut data {
                *v = [0.0, -0.0, 1.0, -1.0][g.below(4)];
            }
        }
        3 => {
            for _ in 0..=g.below(2) {
                let i = at(g);
                data[i] = f64::NAN;
            }
        }
        4 => {
            let i = at(g);
            data[i] = f64::NAN;
            data.iter_mut()
                .filter(|v| !v.is_nan())
                .for_each(|v| *v *= 0.5);
        }
        5 => {
            let (i, j) = (at(g), at(g));
            data[i] = f64::NAN;
            data[j] = if g.below(2) == 0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
        }
        6 => {
            let i = at(g);
            data[i] = f64::INFINITY;
        }
        7 => {
            for v in &mut data {
                *v *= 1e-310;
            }
        }
        8 if ncols >= 2 => {
            let (c1, c2) = (g.below(ncols), g.below(ncols));
            let (r1, r2) = (g.below(nrows), g.below(nrows));
            let value = if g.below(2) == 0 { 9.0 } else { -9.0 };
            data[r1 + nrows * c1] = value;
            data[r2 + nrows * c2] = -value;
        }
        9 => {
            let c = g.below(ncols);
            data[c * nrows..(c + 1) * nrows].fill(0.0);
        }
        10 => data.fill(0.0),
        _ => {}
    }
    data
}

fn same_bits(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn assert_same_data(fused: &[f64], reference: &[f64], what: &str) {
    for (i, (&a, &b)) in fused.iter().zip(reference).enumerate() {
        assert!(
            same_bits(a, b),
            "{what}: entry {i} is {a:e}, reference {b:e}"
        );
    }
}

/// Random shapes, including one row, one column and fewer rows than the 8 lanes.
fn shapes(g: &mut Gen) -> impl Iterator<Item = (usize, usize)> + '_ {
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
}

#[test]
fn fused_kernel_matches_two_pass_update_and_sequential_scan() {
    let mut g = Gen(0x9e37_79b9_7f4a_7c15);
    let mut fallbacks = [0usize; FAMILIES.len()];
    let mut cases = 0usize;
    for (nrows, ncols) in shapes(&mut Gen(0x2545_f491_4f6c_dd1d)) {
        for (family, name) in FAMILIES.iter().enumerate() {
            let original = matrix_of(family, nrows, ncols, &mut g);
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
                    // The fallback runs exactly when the sequential scan's fast path
                    // leaves its normal range; the caller then runs that scan itself.
                    None => {
                        let max = value.abs_sq();
                        assert!(
                            !max.is_finite() || max < f64::MIN_POSITIVE,
                            "{what}: fused fell back with a normal maximum {value:e}"
                        );
                        fallbacks[family] += 1;
                    }
                }
            }
        }
    }
    assert!(cases > 5_000, "{cases} cases");
    // Every fallback family is exercised, so the fallback is not the untested half.
    for family in [3, 4, 5, 6, 7, 10] {
        assert!(
            fallbacks[family] > 0,
            "no fallback for {}",
            FAMILIES[family]
        );
    }
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

/// The whole factorization, fallbacks included: permutations, rank, error and
/// the factored data equal those of the two-pass loop.
#[test]
fn fused_factorization_matches_two_pass_factorization() {
    let mut g = Gen(0xd1b5_4a32_d192_ed03);
    for (nrows, ncols) in shapes(&mut Gen(0x94d0_49bb_1331_11eb)) {
        for (family, name) in FAMILIES.iter().enumerate() {
            let values = matrix_of(family, nrows, ncols, &mut g);
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

/// The kernel is generic: complex entries tie on |z|² with different phases.
#[test]
fn fused_factorization_matches_two_pass_on_complex_ties() {
    let mut g = Gen(0x5851_f42d_4c95_7f2d);
    for (nrows, ncols) in shapes(&mut Gen(0x1405_7b7e_f767_814f)) {
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
