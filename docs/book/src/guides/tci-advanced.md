# TCI Advanced Topics

This guide corresponds to the Julia [Quantics TCI (advanced topics)](https://tensor4all.org/T4APlutoExamples/quantics1d_advanced.html) notebook.

## Direct `crossinterpolate2` Usage

This path bypasses the high-level quantics wrapper and works directly with quantics bits. Use `vec![2; R]` as `local_dims`, because each site is binary.

```rust,ignore
use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_tensorci::{crossinterpolate2, TCI2Options};

let r = 8;
let local_dims = vec![2; r];
let x_max = 1.0_f64;
let step = x_max / (1usize << r) as f64;

let f = move |idx: &Vec<usize>| {
    let q = idx.iter().fold(0usize, |acc, &bit| (acc << 1) | bit);
    let x = q as f64 * step;
    (-3.0 * x).exp()
};

let initial_pivots = vec![vec![0; r]];
let (tci, _ranks, errors) = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f,
    None,
    local_dims,
    initial_pivots,
    TCI2Options {
        tolerance: 1e-12,
        ..Default::default()
    },
)?;

assert!(*errors.last().unwrap() < 1e-10);

let tt = tci.to_tensor_train()?;
for bits in [
    vec![0, 0, 0, 0, 0, 0, 0, 0],
    vec![1, 0, 0, 0, 0, 0, 0, 0],
    vec![1, 1, 0, 0, 0, 0, 0, 0],
] {
    let q = bits.iter().fold(0usize, |acc, &bit| (acc << 1) | bit);
    let exact = (-3.0 * (q as f64 * step)).exp();
    let got = tt.evaluate(&bits)?;
    assert!((got - exact).abs() < 1e-8);
}
```

## Initial Pivot Selection

Choose an initial pivot where `|f|` is large. That usually helps the first local solve.

```rust,ignore
let candidates = vec![
    vec![0; r],
    vec![1; r],
    vec![0, 1, 0, 1, 0, 1, 0, 1],
    vec![1, 1, 1, 1, 1, 1, 1, 1],
];

let first_pivot = candidates
    .into_iter()
    .max_by(|a, b| {
        let va = f(a).abs();
        let vb = f(b).abs();
        va.partial_cmp(&vb).unwrap()
    })
    .unwrap();

assert_eq!(first_pivot, vec![0; r]);
```

## `CachedFunction`

Wrap expensive evaluations once, then reuse the cache across TCI sweeps.

```rust,ignore
use tensor4all_tcicore::CachedFunction;

let cf = CachedFunction::new(
    |idx: &[usize]| {
        let q = idx.iter().fold(0usize, |acc, &bit| (acc << 1) | bit);
        let x = q as f64 * step;
        (-3.0 * x).exp()
    },
    &local_dims,
)?;

let cached_f = |idx: &Vec<usize>| cf.eval(idx);
let (tci_cached, _ranks, _errors) = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    cached_f,
    None,
    local_dims.clone(),
    vec![vec![0; r]],
    TCI2Options::default().with_tolerance(1e-12),
)?;

assert!(cf.cache_size() > 0);
assert!(tci_cached.rank() >= 1);
```

## Manual Integral

For a uniform half-open grid on `[x_min, x_max)`, the quantics tensor train sum becomes a Riemann integral after multiplying by the cell width:

`integral = tt.sum() * (x_max - x_min) / 2^R`

```rust,ignore
let tt = tci.to_tensor_train()?;
let integral = tt.sum() * (x_max - 0.0) / (1usize << r) as f64;

// For f(x) = exp(-3x) on [0, 1): ∫ = (1 - e^{-3}) / 3 ≈ 0.3167
let exact_integral = (1.0 - (-3.0_f64).exp()) / 3.0;
assert!((integral - exact_integral).abs() < 1e-4);
```
