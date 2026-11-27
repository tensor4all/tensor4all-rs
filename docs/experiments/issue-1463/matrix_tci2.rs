//! Issue #1463 pilot: matrix-valued chain TCI2 (A) against a cached endpoint
//! adapter (B), separate fits with own caches (C) and with one shared cache (D).
//!
//! One run = one case. Checks A == B and C == D exactly, prints one CSV row per
//! variant: `case,variant,source_calls,max_bond,err,tt_bytes,median_ms`.
//!
//! Usage: matrix_tci2 <corr|unrel> <rows> <cols> <rank> <full|rook> <cheap|burn> <reps> [fold]
//!
//! With the optional ninth argument `fold`, only fit A is run and the folded
//! layout (`to_folded`) is measured against the `L + 2` layout: core bytes of
//! both and the readout time of the whole matrix at 1024 points (every fourth
//! point of the 4096), median of `reps` rounds after one warm-up round. One
//! row: `FOLD,case,L,max_bond,bytes_L+2,bytes_folded,ms_L+2,ms_folded`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, Tensor3Ops};
use tensor4all_core::MultiIndex;
use tensor4all_tensorci::{
    crossinterpolate2, crossinterpolate2_matrix, MatrixShape, PivotSearchStrategy, TCI2OptimizationResult,
    TCI2Options,
};

const SEED: u64 = 1463;
const L: usize = 12;
const BURN_US: f64 = 50.0;

/// Synthetic F(t)[i,j], t = (idx + 1/2) / 2^L. Corr: sum_k a_k(t) U_k[i] V_k[j]
/// (Gaussian envelope for even k, cosine for odd k, times cos(th t)).
/// Unrel: sin(2 pi f t + ph) exp(-((t-c)/w)^2) cos(th t) with parameters per entry.
struct Target {
    corr: bool,
    rows: usize,
    cols: usize,
    rank: usize,
    burn: usize,        // busy-loop iterations per whole-matrix evaluation, 0 = cheap
    env: Vec<[f64; 4]>, // corr: (kind, p, q, th) per k
    u: Vec<f64>,        // rank * rows
    v: Vec<f64>,        // rank * cols
    comp: Vec<[f64; 5]>, // unrel: f, ph, c, w, th per (i + rows*j)
}

fn busy(n: usize) {
    let mut acc = 1.0f64;
    for _ in 0..n {
        acc = acc.mul_add(0.999999, 1e-7);
    }
    black_box(acc);
}

impl Target {
    fn new(corr: bool, rows: usize, cols: usize, rank: usize, burn: usize) -> Self {
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
        Self { corr, rows, cols, rank, burn, env, u, v, comp }
    }

    fn t_of(x: &[usize]) -> f64 {
        (x.iter().fold(0usize, |a, &b| 2 * a + b) as f64 + 0.5) / (1usize << L) as f64
    }

    /// Whole matrix at `x`, column-major, without the busy loop.
    fn matrix(&self, x: &[usize], out: &mut [f64]) {
        let t = Self::t_of(x);
        let tau = std::f64::consts::TAU;
        let a: Vec<f64> = self.env.iter().map(|&[kind, p, q, th]| {
            let e = if kind == 0.0 { (-((t - p) / q).powi(2)).exp() } else { (tau * p * t + q).cos() };
            e * (th * t).cos()
        }).collect();
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

    /// The source callback: busy work, then the whole matrix.
    fn source(&self, x: &[usize], out: &mut [f64]) {
        busy(self.burn);
        self.matrix(x, out);
    }
}

/// Whole-matrix memo keyed on `x` (adapter cache of B, shared cache of D).
struct Memo<'a> {
    t: &'a Target,
    map: RefCell<HashMap<Vec<usize>, Box<[f64]>>>,
    calls: Cell<usize>,
}
impl<'a> Memo<'a> {
    fn new(t: &'a Target) -> Self {
        Self { t, map: RefCell::new(HashMap::new()), calls: Cell::new(0) }
    }
    fn get(&self, x: &[usize], i: usize, j: usize) -> f64 {
        let mut map = self.map.borrow_mut();
        let block = map.entry(x.to_vec()).or_insert_with_key(|x| {
            self.calls.set(self.calls.get() + 1);
            let mut out = vec![0.0; self.t.rows * self.t.cols];
            self.t.source(x, &mut out);
            out.into_boxed_slice()
        });
        block[i + self.t.rows * j]
    }
}

struct Fit {
    ranks: Vec<usize>,
    errors: Vec<f64>,
    tt: SimpleTensorTrain<f64>,
}
struct Out {
    fits: Vec<Fit>, // A, B: one joint fit; C, D: rows*cols component fits
    calls: usize,
}

fn mk(r: TCI2OptimizationResult<f64>) -> Fit {
    Fit { tt: r.tci.to_tensor_train().unwrap(), ranks: r.ranks, errors: r.errors }
}

/// One scalar fit on `dims` with entry function `f`.
fn scalar_fit(f: &dyn Fn(&MultiIndex) -> f64, dims: Vec<usize>, pivot: &MultiIndex, o: &TCI2Options) -> Fit {
    let batch = |xs: &[MultiIndex]| xs.iter().map(|x| f(x)).collect::<Vec<f64>>();
    mk(crossinterpolate2::<f64, _, _>(|x| f(x), Some(batch), dims, vec![pivot.clone()], o.clone()).unwrap())
}

struct Setup {
    pivot_joint: MultiIndex, // [i0, x0, j0]
    x0: Vec<usize>,
    opts: TCI2Options,
}

fn run(variant: char, t: &Target, s: &Setup) -> Out {
    let (rows, cols) = (t.rows, t.cols);
    match variant {
        'A' => {
            let fit = crossinterpolate2_matrix::<f64, _>(
                |pts: &[Vec<usize>], out: &mut [f64]| {
                    let len = rows * cols;
                    for (p, x) in pts.iter().enumerate() {
                        t.source(x, &mut out[p * len..(p + 1) * len]);
                    }
                },
                MatrixShape { rows, cols },
                vec![2; L],
                vec![s.pivot_joint.clone()],
                s.opts.clone(),
            )
            .unwrap();
            let calls = fit.sampling_stats().source_points;
            let tt = fit.to_tensor_train().unwrap();
            Out { fits: vec![Fit { ranks: fit.result.ranks, errors: fit.result.errors, tt }], calls }
        }
        'B' => {
            let memo = Memo::new(t);
            let f = |i: &MultiIndex| memo.get(&i[1..i.len() - 1], i[0], i[i.len() - 1]);
            let mut dims = vec![rows];
            dims.extend(vec![2; L]);
            dims.push(cols);
            Out { fits: vec![scalar_fit(&f, dims, &s.pivot_joint, &s.opts)], calls: memo.calls.get() }
        }
        _ => {
            let shared = Memo::new(t); // D
            let (mut fits, mut calls) = (vec![], 0);
            for j in 0..cols {
                for i in 0..rows {
                    let own = Memo::new(t); // C: fresh cache per component
                    let m = if variant == 'C' { &own } else { &shared };
                    fits.push(scalar_fit(&|x| m.get(x, i, j), vec![2; L], &s.x0, &s.opts));
                    calls += own.calls.get();
                }
            }
            Out { fits, calls: if variant == 'C' { calls } else { shared.calls.get() } }
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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(a.len() == 8 || a.len() == 9 && a[8] == "fold", "usage: matrix_tci2 <corr|unrel> rows cols rank <full|rook> <cheap|burn> reps");
    let (rows, cols, rank, reps): (usize, usize, usize, usize) =
        (a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap(), a[7].parse().unwrap());
    let burn = if a[6] == "burn" {
        let probe = 4_000_000;
        let t0 = Instant::now();
        busy(probe);
        (BURN_US * 1000.0 / (t0.elapsed().as_nanos() as f64 / probe as f64)).round() as usize
    } else {
        0
    };
    let t = Target::new(a[1] == "corr", rows, cols, rank, burn);

    // Setup outside timing: scale max|F|, largest entry as initial pivot, error samples.
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
    let x0 = bits(x0_idx);
    t.matrix(&x0, &mut buf);
    assert!(buf.iter().all(|v| v.abs() > 1e-12 * max_f), "an entry vanishes at x0");
    let mut pivot_joint = vec![ij.0];
    pivot_joint.extend(&x0);
    pivot_joint.push(ij.1);
    let opts = TCI2Options {
        tolerance: 1e-8 * max_f, // absolute, same for every fit of every variant
        normalize_error: false,
        max_iter: 20,
        pivot_search: if a[5] == "full" { PivotSearchStrategy::Full } else { PivotSearchStrategy::Rook },
        seed: Some(SEED),
        ..TCI2Options::default()
    };
    let s = Setup { pivot_joint, x0, opts };
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let sample: Vec<Vec<usize>> = (0..2048).map(|_| bits(rng.gen_range(0..1usize << L))).collect();

    if a.len() == 9 {
        let fit = crossinterpolate2_matrix::<f64, _>(
            |pts: &[Vec<usize>], out: &mut [f64]| {
                let len = rows * cols;
                for (p, x) in pts.iter().enumerate() {
                    t.source(x, &mut out[p * len..(p + 1) * len]);
                }
            },
            MatrixShape { rows, cols },
            vec![2; L],
            vec![s.pivot_joint.clone()],
            s.opts.clone(),
        )
        .unwrap();
        let folded = fit.to_folded().unwrap();
        let points: Vec<Vec<usize>> = (0..1usize << L).step_by(4).map(bits).collect();
        assert_eq!(points.len(), 1024);
        // The two layouts agree on every readout point (checked outside timing).
        for x in &points {
            let (u, f) = (fit.evaluate(x).unwrap(), folded.evaluate(x).unwrap());
            for (p, q) in u.as_col_major_slice().iter().zip(f.as_col_major_slice()) {
                assert!((p - q).abs() <= 1e-10 * max_f, "layouts differ");
            }
        }
        let (mut tu, mut tf) = (vec![], vec![]);
        for round in 0..=reps {
            let t0 = Instant::now();
            for x in &points {
                black_box(fit.evaluate(black_box(x)).unwrap());
            }
            let du = t0.elapsed().as_secs_f64() * 1e3;
            let t0 = Instant::now();
            for x in &points {
                black_box(folded.evaluate(black_box(x)).unwrap());
            }
            let df = t0.elapsed().as_secs_f64() * 1e3;
            if round > 0 {
                tu.push(du);
                tf.push(df);
            }
        }
        for v in [&mut tu, &mut tf] {
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        }
        let bond = fit.to_tensor_train().unwrap().link_dims().into_iter().max().unwrap();
        let case = format!("{},{}x{},{},{},{}", a[1], rows, cols, if t.corr { rank } else { 0 }, a[5], a[6]);
        println!("FOLD,{case},{L},{bond},{},{},{:.2},{:.2}", fit.retained_bytes(), folded.retained_bytes(), tu[reps / 2], tf[reps / 2]);
        return;
    }

    // Warm-up round, also the source of counts and errors.
    let outs: Vec<Out> = "ABCD".chars().map(|v| run(v, &t, &s)).collect();
    let (oa, ob, oc, od) = (&outs[0], &outs[1], &outs[2], &outs[3]);
    assert_eq!(oa.fits[0].ranks, ob.fits[0].ranks, "A != B ranks");
    assert_eq!(oa.fits[0].errors, ob.fits[0].errors, "A != B errors");
    assert_eq!(oa.calls, ob.calls, "A != B source calls");
    for (c, d) in oc.fits.iter().zip(&od.fits) {
        assert_eq!((&c.ranks, &c.errors), (&d.ranks, &d.errors), "C != D");
    }

    // Independent max error over the sample, all entries, relative to max|F|.
    let mut err = |o: &Out, joint: bool| {
        let mut e = 0.0f64;
        for x in &sample {
            t.matrix(x, &mut buf);
            for j in 0..cols {
                for i in 0..rows {
                    let v = if joint {
                        let mut idx = vec![i];
                        idx.extend(x);
                        idx.push(j);
                        o.fits[0].tt.evaluate(&idx).unwrap()
                    } else {
                        o.fits[i + rows * j].tt.evaluate(x).unwrap()
                    };
                    e = e.max((v - buf[i + rows * j]).abs());
                }
            }
        }
        e / max_f
    };
    let bytes = |o: &Out| -> usize {
        o.fits.iter().flat_map(|f| f.tt.site_tensors()).map(|c| c.left_dim() * c.site_dim() * c.right_dim() * 8).sum()
    };
    let bond = |o: &Out| o.fits.iter().flat_map(|f| f.tt.link_dims()).max().unwrap();

    let mut times = vec![vec![]; 4];
    for _ in 0..reps {
        for (k, v) in "ABCD".chars().enumerate() {
            let t0 = Instant::now();
            drop(black_box(run(v, &t, &s)));
            times[k].push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    let case = format!("{},{}x{},{},{},{}", a[1], rows, cols, if t.corr { rank } else { 0 }, a[5], a[6]);
    for (k, v) in "ABCD".chars().enumerate() {
        let o = &outs[k];
        times[k].sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!("ROW,{case},{v},{},{},{:.2e},{},{:.1}", o.calls, bond(o), err(o, k < 2), bytes(o), times[k][reps / 2]);
    }
}
