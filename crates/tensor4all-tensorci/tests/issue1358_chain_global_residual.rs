//! Chain TCI must not report `Converged` while its global search has measured
//! a residual above tolerance (#1358, the chain path of #1352).
//!
//! The target is Simon's #1352 reproducer, frozen as f64 bits in
//! `fixtures/` (see `fixtures/README.md`): `f(a, b, c) = g(a, c)` if `b = 0`,
//! else `0`, with `a`, `b`, `c` encoded in 5 bits each and laid out on a
//! 15-site chain. Before the fix, several (layout, seed) pairs below reported
//! `Converged` with a recorded error of 0 or below tolerance, while the dense
//! relative error was 1.9e-8 to 7.1e-2. Each run is compared once against the
//! dense target.

use num_complex::Complex64;
use tensor4all_core::{MatrixLuciScalar, Scalar};
use tensor4all_simplett::{AbstractTensorTrain, TTScalar};
use tensor4all_tensorci::{crossinterpolate2, TCI2Options, TCI2Termination};

const R: usize = 5;
const N: usize = 1 << R;

const M12: &str = include_str!("fixtures/issue1352_m12_bits.txt");
const M14: &str = include_str!("fixtures/issue1352_m14_bits.txt");

/// Site order of the 15 binary sites, as `(axis, bit)` with bit 0 the most
/// significant.
#[derive(Clone, Copy, Debug)]
enum Layout {
    /// `a0 b0 c0 a1 b1 c1 ...`
    Interleaved,
    /// All bits of `a`, then `b`, then `c`.
    Abc,
    /// All bits of `a`, then `c`, then `b`.
    Acb,
}

impl Layout {
    fn sites(self) -> Vec<(usize, usize)> {
        match self {
            Layout::Interleaved => (0..R)
                .flat_map(|bit| (0..3).map(move |axis| (axis, bit)))
                .collect(),
            Layout::Abc => (0..3)
                .flat_map(|axis| (0..R).map(move |bit| (axis, bit)))
                .collect(),
            Layout::Acb => [0, 2, 1]
                .into_iter()
                .flat_map(|axis| (0..R).map(move |bit| (axis, bit)))
                .collect(),
        }
    }
}

fn decode(fixture: &str) -> Vec<f64> {
    let matrix: Vec<f64> = fixture
        .split_whitespace()
        .map(|word| f64::from_bits(u64::from_str_radix(word, 16).unwrap()))
        .collect();
    assert_eq!(matrix.len(), N * N);
    matrix
}

/// Run one case and return `(termination, recorded error, dense relative error)`.
fn run<T>(fixture: &str, layout: Layout, seed: u64, scale: T) -> (TCI2Termination, f64, f64)
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
{
    let matrix = decode(fixture);
    let sites = layout.sites();
    let n_sites = sites.len();
    let axes = |point: &[usize]| {
        let mut axes = [0usize; 3];
        for (site, &(axis, bit)) in sites.iter().enumerate() {
            axes[axis] |= point[site] << (R - 1 - bit);
        }
        axes
    };
    let value = |point: &[usize]| {
        let [a, b, c] = axes(point);
        if b == 0 {
            scale * <T as Scalar>::from_f64(matrix[a + N * c])
        } else {
            <T as Scalar>::from_f64(0.0)
        }
    };
    let (largest, _) = matrix
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.abs().total_cmp(&y.1.abs()))
        .unwrap();
    let start_axes = [largest % N, 0, largest / N];
    let start: Vec<usize> = sites
        .iter()
        .map(|&(axis, bit)| (start_axes[axis] >> (R - 1 - bit)) & 1)
        .collect();
    let options = TCI2Options {
        seed: Some(seed),
        ..Default::default()
    };
    let result = crossinterpolate2::<T, _, fn(&[Vec<usize>]) -> Vec<T>>(
        |point: &Vec<usize>| value(point),
        None,
        vec![2; n_sites],
        vec![start],
        options.clone(),
    )
    .unwrap();

    // One dense materialization; column-major, site 0 fastest.
    let tt = result.tci.to_tensor_train().unwrap();
    assert_eq!(tt.len(), n_sites);
    let (dense, _) = tt.full_tensor().unwrap();
    let (mut max_diff, mut max_value) = (0.0_f64, 0.0_f64);
    for (flat, got) in dense.iter().enumerate() {
        let point: Vec<usize> = (0..n_sites).map(|site| (flat >> site) & 1).collect();
        let expected = value(&point);
        max_value = max_value.max(Scalar::abs_val(expected));
        max_diff = max_diff.max(Scalar::abs_val(expected - *got));
    }
    let recorded = *result.errors.last().unwrap();
    (result.termination, recorded, max_diff / max_value)
}

/// Cases that reported a false `Converged` before the fix (dense relative
/// error 9.7e-5, 4.4e-4, 7.1e-2, 1.9e-8 and 5.5e-7), and two controls that were
/// already correct.
const CASES: &[(&str, &str, Layout, u64)] = &[
    ("m12", M12, Layout::Abc, 4),
    ("m14", M14, Layout::Interleaved, 3),
    ("m14", M14, Layout::Interleaved, 5),
    ("m14", M14, Layout::Abc, 1),
    ("m14", M14, Layout::Abc, 4),
    ("m12", M12, Layout::Abc, 1),
    ("m14", M14, Layout::Acb, 1),
];

fn check_all<T>(scale: T)
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
{
    let tolerance = TCI2Options::default().tolerance;
    for &(name, fixture, layout, seed) in CASES {
        let (termination, recorded, dense) = run(fixture, layout, seed, scale);
        let context = format!(
            "{name} {layout:?} seed {seed}: {termination:?}, recorded {recorded:e}, dense {dense:e}"
        );
        // Every listed case converges, and the dense comparison confirms it.
        // Requiring `Converged` also catches a regression that gives up with
        // a bad approximation instead of reporting a false convergence.
        assert_eq!(termination, TCI2Termination::Converged, "{context}");
        assert!(recorded <= tolerance, "{context}");
        assert!(dense <= tolerance, "{context}");
    }
}

#[test]
fn chain_does_not_converge_on_an_unseen_residual_real() {
    check_all(1.0_f64);
}

#[test]
fn chain_does_not_converge_on_an_unseen_residual_complex() {
    check_all(Complex64::new(1.0, 0.5));
}
