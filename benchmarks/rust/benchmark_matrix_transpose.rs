// Matrix transpose size ladder and public rrLU orientation conversion.
use num_complex::{Complex32, Complex64};
use std::hint::black_box;
use std::time::Instant;
use tensor4all_core::{rrlu, RrLUOptions, Scalar};
use tensor4all_tensorbackend::{transpose, Matrix};

fn ladder<T: Scalar>(name: &str) {
    for (rows, cols) in [
        (1, 4096),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (1024, 64),
        (64, 1024),
        (4096, 32),
        (32, 4096),
    ] {
        let input = Matrix::from_col_major_vec(
            rows,
            cols,
            (0..rows * cols)
                .map(|i| T::from_f64((i % 127) as f64 - 63.0))
                .collect(),
        );
        let check = transpose(&input);
        for j in 0..cols {
            for i in 0..rows {
                assert_eq!((check[[j, i]] - input[[i, j]]).abs_sq(), 0.0);
            }
        }
        let repeats = (5_000_000 / (rows * cols)).clamp(4, 10_000);
        let start = Instant::now();
        for _ in 0..repeats {
            black_box(transpose(black_box(&input)));
        }
        println!(
            "micro,{name},{rows},{cols},{repeats},{:.9}",
            start.elapsed().as_secs_f64() / repeats as f64
        );
    }
}

fn orientation() {
    for (rows, rank) in [(1024, 8), (1024, 32), (4096, 32), (4096, 128)] {
        let matrix = Matrix::from_col_major_vec(
            rows,
            rank,
            (0..rows * rank)
                .map(|i| {
                    let row = i % rows;
                    let col = i / rows;
                    if row % rank == col {
                        1.0 + row as f64 / rows as f64
                    } else {
                        0.0
                    }
                })
                .collect(),
        );
        let lu = rrlu(
            &matrix,
            Some(RrLUOptions {
                max_bond_dim: rank,
                rel_tol: 0.0,
                abs_tol: 0.0,
                left_orthogonal: true,
            }),
        )
        .unwrap();
        assert_eq!(lu.npivots(), rank);
        let check = lu.transpose().transpose();
        assert_eq!(
            lu.left(false).as_col_major_slice(),
            check.left(false).as_col_major_slice()
        );
        assert_eq!(
            lu.right(false).as_col_major_slice(),
            check.right(false).as_col_major_slice()
        );
        let repeats = (5_000_000 / (rows * rank)).clamp(4, 1000);
        let left = lu.left(false);
        let right = lu.right(false);
        let start = Instant::now();
        for _ in 0..repeats {
            black_box(transpose(black_box(&left)));
            black_box(transpose(black_box(&right)));
        }
        println!(
            "rrlu_phase,f64,{rows},{rank},{repeats},{:.9}",
            start.elapsed().as_secs_f64() / repeats as f64
        );
        let start = Instant::now();
        for _ in 0..repeats {
            black_box(black_box(&lu).transpose());
        }
        println!(
            "rrlu,f64,{rows},{rank},{repeats},{:.9}",
            start.elapsed().as_secs_f64() / repeats as f64
        );
    }
}

fn tuning() {
    fn blocked<const TILE: usize>(input: &Matrix<f64>) -> Matrix<f64> {
        let mut out = Matrix::zeros(input.ncols(), input.nrows());
        for i0 in (0..input.nrows()).step_by(TILE) {
            for j0 in (0..input.ncols()).step_by(TILE) {
                for i in i0..(i0 + TILE).min(input.nrows()) {
                    for j in j0..(j0 + TILE).min(input.ncols()) {
                        out[[j, i]] = input[[i, j]];
                    }
                }
            }
        }
        out
    }
    for (rows, cols) in [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (1024, 64),
        (64, 1024),
        (4096, 32),
        (32, 4096),
    ] {
        let input = Matrix::from_col_major_vec(
            rows,
            cols,
            (0..rows * cols).map(|i| (i % 127) as f64).collect(),
        );
        for (tile, kernel) in [
            (0_usize, transpose::<f64> as fn(&Matrix<f64>) -> Matrix<f64>),
            (16, blocked::<16>),
            (32, blocked::<32>),
            (64, blocked::<64>),
        ] {
            let check = kernel(&input);
            let original = transpose(&input);
            assert_eq!(check.as_col_major_slice(), original.as_col_major_slice());
            let repeats = (5_000_000 / (rows * cols)).clamp(4, 1000);
            let start = Instant::now();
            for _ in 0..repeats {
                black_box(kernel(black_box(&input)));
            }
            println!(
                "tuning,f64,{rows},{cols},{tile},{:.9}",
                start.elapsed().as_secs_f64() / repeats as f64
            );
        }
    }
}

fn main() {
    if std::env::args().any(|arg| arg == "--tune") {
        tuning();
        return;
    }
    println!("kind,dtype,rows,cols,repeats,seconds_per_call");
    ladder::<f32>("f32");
    ladder::<f64>("f64");
    ladder::<Complex32>("c32");
    ladder::<Complex64>("c64");
    orientation();
}
