//! Issue #1513 step 2: matrix readout and slab micro-benchmark.
//!
//! One process, one case. Builds only on APIs that exist at every compared
//! revision (pre-#1516, #1516, step 2), except the regions between
//! `BORROW-BEGIN`/`BORROW-END` (needs `Tensor::as_col_major_slice`) and
//! `FOLDED-BEGIN`/`FOLDED-END` (needs `to_folded`); `run.py` strips a region
//! from the revisions that lack the API.
//!
//! Usage:
//!   readout <corr|unrel> <rows> <cols> <rank> <reps>   readout of 1024 points, L = 12
//!   readout micro <reps>                               slice_site / borrow on pilot core shapes
//!
//! Readout row: `READOUT,case,max_bond,median_ms,hash[,folded_ms]`. `hash` is
//! FNV-1a over the `to_bits()` of every output entry of all 1024 points, so two
//! revisions agree bit for bit exactly when the hashes agree. Micro row:
//! `MICRO,left x site x right,slice_site_ns,slice_site_then_sum_ns[,borrow_then_sum_ns]`,
//! ns per slab; the last two columns run the same sum over the slab, on a copy
//! and on the borrowed buffer (strided, in place).

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::Instant;
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, Tensor3Ops};
use tensor4all_tensorci::{
    crossinterpolate2_matrix, MatrixShape, PivotSearchStrategy, TCI2Options,
};

const SEED: u64 = 1463;
const L: usize = 12;

/// Same synthetic targets as `docs/experiments/issue-1463/matrix_tci2.rs`
/// (cheap callback): corr = rank-`rank` sum, unrel = one function per entry.
struct Target {
    corr: bool,
    rows: usize,
    cols: usize,
    rank: usize,
    env: Vec<[f64; 4]>,
    u: Vec<f64>,
    v: Vec<f64>,
    comp: Vec<[f64; 5]>,
}

impl Target {
    fn new(corr: bool, rows: usize, cols: usize, rank: usize) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0xABCD);
        let (mut env, mut u, mut v, mut comp) = (vec![], vec![], vec![], vec![]);
        if corr {
            for k in 0..rank {
                env.push(if k % 2 == 0 {
                    [0.0, rng.gen_range(0.2..0.8), rng.gen_range(0.1..0.3), rng.gen_range(0.0..6.0)]
                } else {
                    [1.0, rng.gen_range(1.0..6.0), rng.gen_range(0.0..6.28), rng.gen_range(0.0..6.0)]
                });
                u.extend((0..rows).map(|_| rng.gen_range(-1.0..1.0)));
                v.extend((0..cols).map(|_| rng.gen_range(-1.0..1.0)));
            }
        } else {
            for _ in 0..rows * cols {
                comp.push([
                    rng.gen_range(1.0..8.0),
                    rng.gen_range(0.0..6.28),
                    rng.gen_range(0.2..0.8),
                    rng.gen_range(0.2..0.6),
                    rng.gen_range(0.0..6.0),
                ]);
            }
        }
        Self { corr, rows, cols, rank, env, u, v, comp }
    }

    fn matrix(&self, x: &[usize], out: &mut [f64]) {
        let t = (x.iter().fold(0usize, |a, &b| 2 * a + b) as f64 + 0.5) / (1usize << L) as f64;
        let tau = std::f64::consts::TAU;
        let a: Vec<f64> = self
            .env
            .iter()
            .map(|&[kind, p, q, th]| {
                let e = if kind == 0.0 { (-((t - p) / q).powi(2)).exp() } else { (tau * p * t + q).cos() };
                e * (th * t).cos()
            })
            .collect();
        for j in 0..self.cols {
            for i in 0..self.rows {
                out[i + self.rows * j] = if self.corr {
                    (0..self.rank).map(|k| a[k] * self.u[k * self.rows + i] * self.v[k * self.cols + j]).sum()
                } else {
                    let [f, ph, c, w, th] = self.comp[i + self.rows * j];
                    (tau * f * t + ph).sin() * (-((t - c) / w).powi(2)).exp() * (th * t).cos()
                };
            }
        }
    }
}

fn bits(mut idx: usize) -> Vec<usize> {
    let mut x = vec![0; L];
    for k in (0..L).rev() {
        x[k] = idx & 1;
        idx >>= 1;
    }
    x
}

fn fnv(hash: &mut u64, word: u64) {
    for byte in word.to_le_bytes() {
        *hash = (*hash ^ byte as u64).wrapping_mul(0x100000001b3);
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn readout(args: &[String]) {
    let (rows, cols, rank, reps): (usize, usize, usize, usize) =
        (args[1].parse().unwrap(), args[2].parse().unwrap(), args[3].parse().unwrap(), args[4].parse().unwrap());
    let t = Target::new(args[0] == "corr", rows, cols, rank);
    // Scale max|F| and the largest entry as initial pivot, outside timing (as in #1463).
    let (mut max_f, mut best, mut x0_idx, mut ij) = (0.0f64, -1.0f64, 0, (0, 0));
    let mut buf = vec![0.0; rows * cols];
    for idx in 0..1usize << L {
        t.matrix(&bits(idx), &mut buf);
        for (k, v) in buf.iter().enumerate() {
            max_f = max_f.max(v.abs());
            if v.abs() > best {
                (best, x0_idx, ij) = (v.abs(), idx, (k % rows, k / rows));
            }
        }
    }
    let mut pivot = vec![ij.0];
    pivot.extend(bits(x0_idx));
    pivot.push(ij.1);
    let opts = TCI2Options {
        tolerance: 1e-8 * max_f,
        normalize_error: false,
        max_iter: 20,
        pivot_search: PivotSearchStrategy::Full,
        seed: Some(SEED),
        ..TCI2Options::default()
    };
    let fit = crossinterpolate2_matrix::<f64, _>(
        |pts: &[Vec<usize>], out: &mut [f64]| {
            let len = rows * cols;
            for (p, x) in pts.iter().enumerate() {
                t.matrix(x, &mut out[p * len..(p + 1) * len]);
            }
        },
        MatrixShape { rows, cols },
        vec![2; L],
        vec![pivot],
        opts,
    )
    .unwrap();
    let points: Vec<Vec<usize>> = (0..1usize << L).step_by(4).map(bits).collect();
    assert_eq!(points.len(), 1024);
    let mut hash = 0xcbf29ce484222325u64;
    for x in &points {
        for v in fit.evaluate(x).unwrap().as_col_major_slice() {
            fnv(&mut hash, v.to_bits());
        }
    }
    // FOLDED-BEGIN
    let folded = fit.to_folded().unwrap();
    for x in &points {
        let (u, f) = (fit.evaluate(x).unwrap(), folded.evaluate(x).unwrap());
        for (p, q) in u.as_col_major_slice().iter().zip(f.as_col_major_slice()) {
            assert!((p - q).abs() <= 1e-10 * max_f, "layouts differ");
        }
    }
    let mut folded_ms = vec![];
    // FOLDED-END
    let mut ms = vec![];
    for round in 0..=reps {
        let t0 = Instant::now();
        for x in &points {
            black_box(fit.evaluate(black_box(x)).unwrap());
        }
        let d = t0.elapsed().as_secs_f64() * 1e3;
        // FOLDED-BEGIN
        let t0 = Instant::now();
        for x in &points {
            black_box(folded.evaluate(black_box(x)).unwrap());
        }
        let df = t0.elapsed().as_secs_f64() * 1e3;
        // FOLDED-END
        if round > 0 {
            ms.push(d);
            // FOLDED-BEGIN
            folded_ms.push(df);
            // FOLDED-END
        }
    }
    let bond = fit.to_tensor_train().unwrap().link_dims().into_iter().max().unwrap();
    let case = format!("{},{}x{},{}", args[0], rows, cols, rank);
    let mut row = format!("READOUT,{case},{bond},{:.2},{hash:016x}", median(ms));
    // FOLDED-BEGIN
    row += &format!(",{:.2}", median(folded_ms));
    // FOLDED-END
    println!("{row}");
}

/// Sum of one slab given as `right` runs of `left` elements at `stride`.
fn run_sum(buffer: &[f64], start: usize, left: usize, stride: usize, right: usize) -> f64 {
    let mut sum = 0.0;
    for r in 0..right {
        let from = start + stride * r;
        for &v in &buffer[from..from + left] {
            sum += v;
        }
    }
    sum
}

fn micro(reps: usize) {
    // Core shapes of the #1463 pilot at site dimension 2: bonds 12 (corr), 25 and 78 (unrel).
    for (left, site, right) in [(1, 2, 12), (12, 2, 12), (25, 2, 25), (78, 2, 78), (78, 2, 1)] {
        let data: Vec<f64> = (0..left * site * right).map(|i| (i as f64 * 0.37).sin()).collect();
        let tensor = tensor3_from_data(data, left, site, right).unwrap();
        let calls = (2_000_000 / (left * right)).max(200);
        let (mut copy_ns, mut copy_sum_ns) = (vec![], vec![]);
        // BORROW-BEGIN
        let mut borrow_sum_ns = vec![];
        // BORROW-END
        for round in 0..=reps {
            let t0 = Instant::now();
            for c in 0..calls {
                black_box(tensor.slice_site(black_box(c % site)));
            }
            let a = t0.elapsed().as_secs_f64() * 1e9 / calls as f64;
            // The same consumer (a sum over the slab) on a copy and on the borrow.
            let t0 = Instant::now();
            for c in 0..calls {
                let slab = tensor.slice_site(black_box(c % site));
                black_box(run_sum(&slab, 0, left, left, right));
            }
            let a_sum = t0.elapsed().as_secs_f64() * 1e9 / calls as f64;
            // BORROW-BEGIN
            let buffer = tensor.as_col_major_slice().unwrap();
            let t0 = Instant::now();
            for c in 0..calls {
                let s = black_box(c % site);
                black_box(run_sum(buffer, left * s, left, left * site, right));
            }
            let b_sum = t0.elapsed().as_secs_f64() * 1e9 / calls as f64;
            // BORROW-END
            if round > 0 {
                copy_ns.push(a);
                copy_sum_ns.push(a_sum);
                // BORROW-BEGIN
                borrow_sum_ns.push(b_sum);
                // BORROW-END
            }
        }
        let mut row = format!("MICRO,{left}x{site}x{right},{:.1},{:.1}", median(copy_ns), median(copy_sum_ns));
        // BORROW-BEGIN
        row += &format!(",{:.1}", median(borrow_sum_ns));
        // BORROW-END
        println!("{row}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args[0] == "micro" {
        micro(args[1].parse().unwrap());
    } else {
        assert_eq!(args.len(), 5, "usage: readout <corr|unrel> rows cols rank reps | micro reps");
        readout(&args);
    }
}
