# Quantics TCI target memoization (issue #747)

Date: 2026-10-04. Baseline: `origin/main` `75455d14` — the pre-change quantics
path, which records every evaluated point into a `HashMap<Vec<usize>, V>` and
never looks it up. Candidate: branch `fix/issue-747-multi-index-cache` (commits
`f7029da7`, `45130ba2` plus the review follow-ups), where the target is memoized
by `tensor4all_core::MultiIndexCache` on the quantics multi-index and the record
map and its introspection API are gone. Both binaries are release builds from
the same lockfile and the same probe source, differing only in the crate under
test.

## Machine and thread control

- AMD EPYC 7713P 64-Core, `nproc` = 64.
- Every run: `taskset -c 0` plus `RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- Effective thread count read from the running process: a sampler polls
  `/proc/<pid>/status` every 50 ms and reports the maximum `Threads:` value.
  Every run reported `max_threads=1`.
- Cargo: `-j 4`.

## Protocol

- One warm-up repetition per process invocation is discarded; the three
  following repetitions of that invocation are recorded, with `reps=4`.
- Two rounds per configuration, alternating baseline and candidate
  (`a`, `b`, `a`, `b`), so drift shows up as a difference between rounds. Six
  measured values per configuration per side; the table reports all six and
  their median.
- Primary metric: end-to-end interpolation time. Secondary metrics: points
  passed to the target callback and callback invocations (counted inside the
  probe on both sides, so they are comparable), plus the candidate's
  `num_evals` / `cache_hits` / `cache_misses` / `hit_ratio`.
- Predeclared correctness gate, checked for every run: `rep_error <= tolerance`,
  and equality of the sample digest (64 points, `f64` bits folded with FNV),
  the maximum sample error, `max_bond` and the sweep count between baseline and
  candidate. All six values below were reproduced exactly in all 24 runs.
- Grid: 2D `2^bits x 2^bits` discretized grid on `[0,1]^2`, `tolerance=1e-10`,
  `nrandominitpivot=5`, `rng_seed=0`.
- Targets: `cheap = exp(sin(x) * (1 + y))`, trivially vectorizable (worst case
  for memoization overhead); `heavy = sum_(k=1..64) sin(k x) cos(k y) / k`,
  64 transcendental terms per point (the case a memoized target exists for).

## Results (milliseconds, end-to-end)

| case | baseline | candidate | speedup (median) |
|---|---|---|---|
| cheap, 2^8 x 2^8 | 20.04, 20.28, 21.47, 21.50, 20.76, 24.09 (median 21.0) | 12.34, 12.89, 11.31, 10.99, 11.38, 11.40 (median 11.4) | 1.8x |
| cheap, 2^10 x 2^10 | 41.19, 43.31, 43.92, 43.07, 43.44, 43.24 (median 43.3) | 23.27, 23.37, 24.54, 23.96, 23.43, 23.36 (median 23.4) | 1.9x |
| heavy, 2^8 x 2^8 | 328.45, 324.13, 321.35, 319.32, 316.83, 319.21 (median 321.8) | 74.85, 76.17, 77.14, 74.71, 74.71, 75.40 (median 74.8) | 4.3x |
| heavy, 2^10 x 2^10 | 462.50, 462.98, 463.12, 468.66, 470.42, 476.53 (median 465.6) | 148.95, 149.14, 154.06, 151.41, 149.65, 154.75 (median 151.2) | 3.1x |

Observed run-to-run spread within a configuration is a few percent for the
cheap cases (largest 24.09 vs 20.04, a cold first-invocations outlier) and under
1.5% for the heavy cases.

### Target work

| case | baseline points | candidate points | reduction | callback calls (baseline / candidate) | hit ratio |
|---|---|---|---|---|---|
| cheap, 2^8 | 69,519 | 13,636 | 80.4% | 58 / 19 | 0.804 |
| cheap, 2^10 | 127,080 | 28,986 | 77.2% | 92 / 33 | 0.772 |
| heavy, 2^8 | 182,044 | 32,162 | 82.3% | 72 / 39 | 0.823 |
| heavy, 2^10 | 264,347 | 69,267 | 73.8% | 92 / 49 | 0.738 |

The candidate evaluates 74-82% fewer points and also skips the coordinate
conversion and batch assembly for cached hits, which is why the end-to-end win
is larger than the point reduction alone. The candidate's batches are in fact
smaller on average than the baseline's (e.g. cheap 2^8: 718 vs 1,198 points per
call); the saving comes from not repeating work, not from larger batches.

### Retained cache payload

For the heavy 2^10 case both sides see 69,267 distinct points (the same set of
points is evaluated; the candidate just stops repeating them). The baseline
retains one `Vec<usize>` key plus its `HashMap` entry per distinct point — one
heap allocation per point; the candidate retains one flat key and one value per
point, which `MultiIndexCache::retained_bytes` reports as
`69,267 * (8 + 8) = 1.1 MB` of logical key and value storage. (The baseline's
recording map overwrites duplicate keys, so its retained count is the distinct
count, not the 127,080/264,347 request totals above.)

## Reproduction

The probe is not committed. Place its source at
`crates/tensor4all-quanticstci/examples/issue747_probe.rs` in both worktrees,
then:

```bash
cargo build -j 4 --locked --release -p tensor4all-quanticstci \
  --features tensor4all-core/backend-tenferro --example issue747_probe
```

The baseline probe must drop the two cache-accounting groups from the final
`println!` (`num_evals`, `cache_hits`, `cache_misses`, `hit_ratio` and their
arguments), because those accessors do not exist before this change; the
callback-side counters used for the comparison are identical on both sides.

```bash
# per run: one warm-up + three measured repetitions, 1 CPU, all pools at 1
for round in 1 2; do for probe in baseline candidate; do
  env RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
      ./target/release/examples/issue747_probe <bits> 4 <cheap|heavy>
done; done
```

```rust
//! Probe for issue #747: callback work and cache accounting for a batched
//! quantics interpolation run.
//!
//! Build (release, single thread, pinned):
//! ```bash
//! cargo build -j 4 --locked --release -p tensor4all-quanticstci \
//!   --features tensor4all-core/backend-tenferro --example issue747_probe
//! ```
//! Run: `taskset -c 0 ./issue747_probe <bits> <reps> <cheap|heavy>`

use std::cell::Cell;
use std::hint::black_box;
use std::rc::Rc;
use std::time::Instant;
use tensor4all_quanticstci::{
    quanticscrossinterpolate_batch, DiscretizedGrid, QtciOptions, QuanticsBatch,
};

/// `cheap` is a trivially vectorizable target (worst case for the memo
/// overhead); `heavy` costs 64 transcendental terms per point (the case a
/// memoized target exists for).
fn target(mode: &str, x: f64, y: f64) -> f64 {
    if mode == "heavy" {
        let mut acc = 0.0f64;
        for k in 1..=64 {
            let k = k as f64;
            acc += ((k * x).sin() * (k * y).cos()) / k;
        }
        acc
    } else {
        (x.sin() * (1.0 + y)).exp()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bits: usize = args.first().map(|v| v.parse().unwrap()).unwrap_or(8);
    let reps: usize = args.get(1).map(|v| v.parse().unwrap()).unwrap_or(3);
    let mode = args.get(2).cloned().unwrap_or_else(|| "cheap".into());

    let grid = DiscretizedGrid::builder(&[bits, bits])
        .with_lower_bound(&[0.0, 0.0])
        .with_upper_bound(&[1.0, 1.0])
        .build()
        .expect("grid");
    let mode_for_target = Rc::new(mode.clone());

    for rep in 0..reps {
        let invocations = Rc::new(Cell::new(0usize));
        let points = Rc::new(Cell::new(0usize));
        let (inv, pts) = (Rc::clone(&invocations), Rc::clone(&points));
        let mode_for_target = Rc::clone(&mode_for_target);
        let f = move |batch: QuanticsBatch<'_, f64>| -> anyhow::Result<Vec<f64>> {
            inv.set(inv.get() + 1);
            pts.set(pts.get() + batch.n_points());
            Ok((0..batch.n_points())
                .map(|p| {
                    target(
                        &mode_for_target,
                        batch.get(0, p).unwrap(),
                        batch.get(1, p).unwrap(),
                    )
                })
                .collect())
        };
        let options = QtciOptions::default()
            .with_tolerance(1e-10)
            .with_nrandominitpivot(5)
            .with_rng_seed(0);

        let tolerance = options.tolerance;
        let start = Instant::now();
        let (qtci, ranks, errors) =
            quanticscrossinterpolate_batch(&grid, f, None, options).expect("interpolation");
        let elapsed = start.elapsed();

        // Correctness gate: agree with the target on a fixed sample of grid
        // points, and hash the approximations so runs are comparable.
        let mut digest = 1469598103934665603u64;
        let mut max_error = 0.0f64;
        let side = 1usize << bits;
        for i in 0..8 {
            for j in 0..8 {
                let (gi, gj) = (i * (side - 1) / 7, j * (side - 1) / 7);
                let value = qtci.evaluate(&[gi, gj]).expect("evaluate");
                let quantics = grid.grididx_to_quantics(&[gi, gj]).expect("quantics");
                let coord = grid.quantics_to_origcoord(&quantics).expect("coord");
                max_error = max_error.max((value - target(&mode, coord[0], coord[1])).abs());
                for byte in value.to_bits().to_le_bytes() {
                    digest = (digest ^ u64::from(byte)).wrapping_mul(1099511628211);
                }
                black_box(value);
            }
        }

        println!(
            "mode={mode} rep={rep} bits={bits} max_bond={} sweeps={} tolerance={tolerance:e} \
             rep_error={:e} sample_error={max_error:e} callback_invocations={} callback_points={} \
             num_evals={} cache_hits={} cache_misses={} hit_ratio={:.4} elapsed_ms={:.3} digest={digest:#018x}",
            ranks.iter().copied().max().unwrap_or(0),
            ranks.len(),
            errors.last().copied().unwrap_or(f64::NAN),
            invocations.get(),
            points.get(),
            qtci.num_evals(),
            qtci.num_cache_hits(),
            qtci.cache_stats().num_cache_misses,
            qtci.cache_hit_ratio(),
            elapsed.as_secs_f64() * 1e3,
        );
    }
}
```
