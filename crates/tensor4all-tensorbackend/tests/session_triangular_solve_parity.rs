//! `Session::triangular_solve_matrix` must equal the process-global
//! `triangular_solve_matrix` bit for bit, for every scalar type and every
//! combination of `left_side`, `lower` and `transpose_a` (tensor4all-rs#1244).
//!
//! The global route solves `f32` and `Complex32` in double precision and narrows the
//! result; the session route has to apply the same conversion, otherwise the two
//! routes differ in the last bit for the narrow scalar types.
#![cfg(feature = "global-defaults")]

use num_complex::{Complex32, Complex64};
use tensor4all_tensorbackend::explicit::Session;
use tensor4all_tensorbackend::{
    triangular_solve_matrix, CpuBackend, Matrix, MatrixTriangularSolveScalar,
};

/// Order of the triangular factor and number of right-hand sides in the fixture.
const N: usize = 8;
const NRHS: usize = 3;

fn lcg(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

/// Scalar under test: built from two `f64` parts, compared through its raw bits.
trait Scalar: MatrixTriangularSolveScalar + Copy + std::fmt::Debug {
    fn from_parts(re: f64, im: f64) -> Self;
    fn bits(self) -> (u64, u64);
}

impl Scalar for f64 {
    fn from_parts(re: f64, _: f64) -> Self {
        re
    }
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
}

impl Scalar for f32 {
    fn from_parts(re: f64, _: f64) -> Self {
        re as f32
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.to_bits()), 0)
    }
}

impl Scalar for Complex64 {
    fn from_parts(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}

impl Scalar for Complex32 {
    fn from_parts(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.re.to_bits()), u64::from(self.im.to_bits()))
    }
}

/// An `N x N` triangular factor (the other half is zero) with a dominant diagonal.
fn triangular<T: Scalar>(lower: bool) -> Matrix<T> {
    let re = lcg(N * N, 11);
    let im = lcg(N * N, 29);
    let data = (0..N * N)
        .map(|k| {
            let (row, col) = (k % N, k / N);
            let kept = if lower { row >= col } else { row <= col };
            if !kept {
                T::from_parts(0.0, 0.0)
            } else if row == col {
                T::from_parts(2.0 + re[k].abs(), 0.5 * im[k])
            } else {
                T::from_parts(re[k], im[k])
            }
        })
        .collect();
    Matrix::from_col_major_vec(N, N, data)
}

/// Right-hand side: `N x NRHS` for `op(A) X = B`, `NRHS x N` for `X op(A) = B`.
fn rhs<T: Scalar>(left_side: bool) -> Matrix<T> {
    let re = lcg(N * NRHS, 47);
    let im = lcg(N * NRHS, 83);
    let data = (0..N * NRHS).map(|k| T::from_parts(re[k], im[k])).collect();
    if left_side {
        Matrix::from_col_major_vec(N, NRHS, data)
    } else {
        Matrix::from_col_major_vec(NRHS, N, data)
    }
}

/// Returns the flag combinations whose session result is not bitwise equal to the
/// global result, each with its number of differing entries.
fn mismatches<T: Scalar>() -> Vec<String> {
    let mut backend = CpuBackend::with_threads(1).expect("single-thread backend");
    let mut session = Session::new(backend.engine_mut());
    let mut out = Vec::new();
    let mut cases = 0;
    for left_side in [true, false] {
        for lower in [true, false] {
            for transpose_a in [false, true] {
                let a = triangular::<T>(lower);
                let b = rhs::<T>(left_side);
                let global = triangular_solve_matrix(&a, &b, left_side, lower, transpose_a, false)
                    .expect("global triangular solve");
                let local = session
                    .triangular_solve_matrix(&a, &b, left_side, lower, transpose_a, false)
                    .expect("session triangular solve");
                assert_eq!(
                    (local.nrows(), local.ncols()),
                    (global.nrows(), global.ncols())
                );
                let differing = global
                    .as_col_major_slice()
                    .iter()
                    .zip(local.as_col_major_slice())
                    .filter(|(g, l)| g.bits() != l.bits())
                    .count();
                cases += 1;
                if differing > 0 {
                    out.push(format!(
                        "left_side={left_side} lower={lower} transpose_a={transpose_a}: \
                         {differing}/{} entries differ",
                        global.as_col_major_slice().len()
                    ));
                }
            }
        }
    }
    assert_eq!(cases, 8);
    out
}

#[test]
fn f64_session_triangular_solve_is_bitwise_global() {
    let bad = mismatches::<f64>();
    assert!(
        bad.is_empty(),
        "{} of 8 cases differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

#[test]
fn f32_session_triangular_solve_is_bitwise_global() {
    let bad = mismatches::<f32>();
    assert!(
        bad.is_empty(),
        "{} of 8 cases differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

#[test]
fn complex64_session_triangular_solve_is_bitwise_global() {
    let bad = mismatches::<Complex64>();
    assert!(
        bad.is_empty(),
        "{} of 8 cases differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

#[test]
fn complex32_session_triangular_solve_is_bitwise_global() {
    let bad = mismatches::<Complex32>();
    assert!(
        bad.is_empty(),
        "{} of 8 cases differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
}
