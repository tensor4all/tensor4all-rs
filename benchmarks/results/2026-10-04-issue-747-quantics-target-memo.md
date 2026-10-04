# Quantics TCI target memoization (issue #747)

Date: 2026-10-04. Baseline: `origin/main` `75455d14` (the pre-change quantics
path, which records evaluated points in a `HashMap<Vec<usize>, V>` and never
looks them up). Candidate: branch `fix/issue-747-multi-index-cache`, release
build of the same lockfile, where the target is memoized by
`tensor4all_core::MultiIndexCache` on the quantics multi-index and the record
map and its introspection API are gone.

## Protocol

- Pinned single thread: `RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0`.
  Effective thread count verified: `nproc` under `taskset -c 0` with
  `RAYON_NUM_THREADS=1` reports `1`.
- Release builds, same lockfile, same probe source; one warm-up run per
  configuration, then three measured repetitions. Times below are the
  measured repetitions (first repetition of a process is discarded as warm-up).
- Primary metric: end-to-end interpolation time. Secondary metrics: points
  passed to the target callback and callback invocations (both measured inside
  the probe, so they are identical in kind for baseline and candidate), plus
  the candidate's `num_evals` / `cache_hits` / `cache_misses` / `hit_ratio`.
- Predeclared correctness gate: identical sample digest, identical sampling
  error, identical `max_bond` and sweep count, and `rep_error <= tolerance`
  for every run. All four hold in every case below (digests and errors are
  byte-identical between baseline and candidate).
- Grids: 2D `2^bits x 2^bits` discretized grid on `[0,1]^2`, `tolerance=1e-10`,
  `nrandominitpivot=5`, `rng_seed=0`.
- Targets: `cheap = exp(sin(x) * (1 + y))`, a trivially vectorizable target
  (worst case for memoization overhead); `heavy = sum_{k=1..64}
  sin(k x) cos(k y) / k`, 64 transcendental terms per point (the case a memoized
  target exists for).

## Results

| case | baseline (ms) | candidate (ms) | target points | evaluated points (candidate) | hit ratio | speedup |
|---|---|---|---|---|---|---|
| cheap, 2^8 x 2^8 | 20.27, 20.40 | 10.72, 10.93 | 69,519 | 13,636 | 0.804 | 1.9x |
| cheap, 2^10 x 2^10 | 43.43, 46.11 | 23.42, 25.93 | 127,080 | 28,986 | 0.772 | 1.8x |
| heavy, 2^8 x 2^8 | 319.19, 319.24 | 78.11, 81.44 | 182,044 | 32,162 | 0.823 | 4.1x |
| heavy, 2^10 x 2^10 | 492.50, 517.21 | 155.32, 162.16 | 264,347 | 69,267 | 0.738 | 3.2x |

The candidate evaluates 74-82% fewer points, and the saving also covers the
coordinate conversion and batch assembly for every cached hit. The callback
invocation count drops as well (19 vs 58, 33 vs 92, 39 vs 72, 49 vs 92), so the
target is called with fewer, larger batches.

Retained cache payload after the 2^10 x 2^10 heavy run: 69,267 entries, which
the candidate reports through `MultiIndexCache::retained_bytes` as
`entries * (8 + size_of::<f64>()) = 1.1 MB` of key and value storage. The
baseline recorded `127,080`-point `Vec<usize>` keys instead (2 sites each plus
`HashMap` overhead) and kept one heap allocation per recorded point.

## Reproduction

The probe is not committed. Place its source at
`crates/tensor4all-quanticstci/examples/issue747_probe.rs` (content below) in
both the baseline and candidate worktrees, then:

```bash
cargo build -j 4 --locked --release -p tensor4all-quanticstci \
  --features tensor4all-core/backend-tenferro --example issue747_probe
# baseline binary needs the cache-accounting fields of the print removed,
# because those accessors do not exist before this change
env RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
    ./target/release/examples/issue747_probe <bits> <reps> <cheap|heavy>
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
    let mode_for_target = mode.clone();
    let mode_for_target = Rc::new(mode_for_target);

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
            qtci.cache_stats().num_cache_misses(),
            qtci.cache_hit_ratio(),
            elapsed.as_secs_f64() * 1e3,
        );
    }
}
```
