# Global pivot search: coordinate retention (issue #806)

Date: 2026-10-03. Baseline: `origin/main` `3bc704e3` (resetting axis-line scan).
Candidate: branch `fix/issue-806-global-pivot-coordinate-search` (the contributor
commit plus the review corrections: `abs_val` residual magnitude, relaxed
`EinsumScalar` bounds, corrected simplett key width, one extra regression).

Both binaries were built from the same lockfile and the same probe source in the
release profile:

```bash
cargo build -j 16 --locked --release -p tensor4all-tensorci \
  --features tensor4all-core/backend-tenferro --example global_pivot_speed
```

and run three times each, baseline first, one thread on one pinned CPU:

```bash
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 2 ./global_pivot_speed
```

Machine: AMD EPYC 7713P (64 cores), rustc 1.97.1, release profile, effective
thread count 1. Each reported time is the mean of 20 searches with
`DefaultGlobalPivotFinder::new(5, 3, 10.0)` from a fixed start, with a zero
tensor train; `best_residual` is `max |f|` over the returned pivots.

## Results

| Fixture (2 sites) | Threshold | Baseline per search | Baseline pivots | Candidate per search | Candidate pivots |
|---|---:|---:|---|---:|---|
| `f = i + j`, 16x16 | 5 | 15.0 us | `(15,0)`, residual 15 | 189 us | `(15,15)`, residual 30 |
| coupled `1000 - 4(x-3)^2 - (y-x)^2`, 8x8 | 999.5 | 8.2 us | **none** (residual 991 < 999.5) | 144 us | `(3,3)`, residual 1000 |
| same coupled function, 32x32 | 999.5 | 31.7 us | `(31,0)`, residual 3097 | 339 us | `(31,0)`, residual 3097 |

The candidate is **10-17x slower per search** on these fixtures and returns
strictly better pivots: it finds the coupled maximum the baseline misses
entirely and the separable corner (residual 30 against 15). The extra cost is
the retained-coordinate walk: up to 100 sweeps with an early stop at ten times
the acceptance threshold, each sweep evaluating one batch of candidates per
site, against the baseline's single axis-line scan. The search cost is bounded
by that sweep cap, and the reported numbers are for two-site fixtures where the
search is a large fraction of the work; the workloads in
`benchmarks/results/2026-09-30-treetci-global-search.md` evaluate far more
expensive targets per point.

## Probe source

```rust
use std::time::Instant;
use tensor4all_simplett::SimpleTensorTrain;
use tensor4all_tensorci::{DefaultGlobalPivotFinder, GlobalPivotFinder, GlobalPivotSearchInput};

struct ZeroStream;
impl rand::RngCore for ZeroStream {
    fn next_u32(&mut self) -> u32 { 0 }
    fn next_u64(&mut self) -> u64 { 0 }
    fn fill_bytes(&mut self, dest: &mut [u8]) { dest.fill(0); }
}

fn main() {
    let fixtures: Vec<(&str, Vec<usize>, f64, fn(&Vec<usize>) -> f64)> = vec![
        ("separable16", vec![16, 16], 0.5, |p| (p[0] + p[1]) as f64),
        ("coupled8", vec![8, 8], 99.95, |p| {
            let (x, y) = (p[0] as f64, p[1] as f64);
            1000.0 - 4.0 * (x - 3.0).powi(2) - (y - x).powi(2)
        }),
        ("coupled32", vec![32, 32], 99.95, |p| {
            let (x, y) = (p[0] as f64, p[1] as f64);
            1000.0 - 4.0 * (x - 3.0).powi(2) - (y - x).powi(2)
        }),
    ];
    for (name, local_dims, abs_tol, f) in fixtures {
        let input = GlobalPivotSearchInput {
            current_tt: SimpleTensorTrain::<f64>::constant(&local_dims, 0.0),
            i_set: local_dims.iter().map(|_| vec![vec![]]).collect(),
            j_set: local_dims.iter().map(|_| vec![vec![]]).collect(),
            local_dims: local_dims.clone(),
            max_sample_value: 1000.0,
        };
        let finder = DefaultGlobalPivotFinder::new(5, 3, 10.0);
        let start = Instant::now();
        let mut sample = Vec::new();
        for _ in 0..20 {
            sample = finder
                .find_global_pivots(&input, &f, abs_tol, &mut ZeroStream)
                .unwrap();
        }
        let elapsed = start.elapsed() / 20;
        let values = sample.iter().map(|p| f(p)).collect::<Vec<_>>();
        println!("{name}: per_search={elapsed:?} sample={sample:?} values={values:?}");
    }
}
```

(The baseline binary calls `find_global_pivots` without `.unwrap()`, which is the
only difference between the two probe copies.)
