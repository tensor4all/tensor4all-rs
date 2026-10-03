# TensorCI1 pointwise evaluation reuse (issue #787)

Date: 2026-10-03. Baseline: `origin/main` `8379852e` (unmodified
`TensorCI1::evaluate`). Candidate: branch `fix/issue-787-tci1-evaluate-reuse`,
working-tree diff `9ee8a79dfc0c501d` (sha256 prefix) with
`crates/tensor4all-tensorci/src/tensorci1.rs` `0537f994d84be166` and
`crates/tensor4all-tensorci/src/tensorci1/tests/mod.rs` `c99b5dbd0c31d44c`;
the candidate is the uncommitted working tree, so re-verify the hashes before
comparing a later revision.

Both binaries were built from the same lockfile and the same probe source in
the release profile. The probe is not committed; place its source (below) at
`crates/tensor4all-tensorci/examples/issue787_probe.rs` before building:

```bash
cargo build -j 16 --locked --release -p tensor4all-tensorci \
  --features tensor4all-core/backend-tenferro --example issue787_probe
```

Each binary ran three times in two blocks (baseline, candidate, baseline,
candidate) in one session, with one thread and one pinned CPU:

```bash
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 2 \
./target/release/examples/issue787_probe
```

Machine: AMD EPYC 7713P (64 cores, `nproc` 64), rustc 1.97.1, release profile.
The probe reads the effective thread count from the backend it uses
(`default_cpu_execution_context().with_backend(|backend| backend.num_threads())`)
and prints `backend_threads=1`; it also prints all five thread environment
variables. This is a paired before/after comparison on one workload, not a
promotion gate.

## Workload

Six sites, local dimension 4, rank 2, 200 **distinct** mixed-radix
multi-indices, target function `f(idx) = 1 + sum_site (site + 1) * idx[site]`.
The probe separates

- the first ("cold") call, which pays the one-time normalized-train build, from
- the warm loop over the remaining 199 points, and

and reports, besides the timings, the pointwise maximum absolute difference
against the reference tensor train and a digest of all 200 values. Timed loops
consume their inputs and results through `std::hint::black_box`, and both
timings come from the same binary, so the baseline/candidate comparison uses
the digest rather than a cached-versus-its-own-clone difference.

## Results

All 12 runs printed the same value digest
`0xd0acc6492ed53e3a`, so the candidate produces the same pointwise values as
the baseline on this workload, and `pointwise_max|diff| = 0` in every run.

| Block | Run | cold first call | warm `tci.evaluate` | per point | reference `tt.evaluate` | per point |
|---|---|---:|---:|---:|---:|---:|
| baseline | 1 | 194.1 us | 11.746 ms | 59.03 us | 85.2 us | 0.43 us |
| baseline | 2 | 146.8 us | 11.574 ms | 58.16 us | 86.0 us | 0.43 us |
| baseline | 3 | 151.3 us | 11.184 ms | 56.20 us | 71.5 us | 0.36 us |
| candidate | 1 | 243.8 us | 197.2 us | 0.99 us | 160.7 us | 0.80 us |
| candidate | 2 | 186.7 us | 170.8 us | 0.86 us | 142.7 us | 0.71 us |
| candidate | 3 | 159.2 us | 128.8 us | 0.65 us | 126.3 us | 0.63 us |
| candidate | 4 | 4650.1 us | 144.2 us | 0.72 us | 140.7 us | 0.70 us |
| candidate | 5 | 163.5 us | 131.4 us | 0.66 us | 114.4 us | 0.57 us |
| candidate | 6 | 162.6 us | 107.1 us | 0.54 us | 104.4 us | 0.52 us |
| baseline | 4 | 5017.9 us | 12.493 ms | 62.78 us | 91.3 us | 0.46 us |
| baseline | 5 | 165.8 us | 12.016 ms | 60.38 us | 93.1 us | 0.47 us |
| baseline | 6 | 160.2 us | 11.814 ms | 59.37 us | 90.8 us | 0.45 us |

Median warm cost over the six baseline runs: **59.20 us per point**; over the
six candidate runs: **0.69 us per point** -> **86x**. End to end for 200 points,
including the cold call: 11.94 ms -> 0.31 ms (**38x**). On the same runs the
candidate's warm per-point cost was 0.54-0.99 us against a 0.36-0.80 us
prebuilt-tensor-train reference, so the remaining overhead is small but was not
attributed to a single cause by this measurement.

Two first-run effects are visible and are not used in the medians: the first
run of a block reports a multi-millisecond cold call on whichever binary ran
first after the rebuild (page-cache/first-touch effect; 5021.9 us baseline and
4650.1 us candidate), and the machine is shared, so absolute times move between
blocks. Comparing individual warm costs across blocks gives 57x to 116x.

## Probe source

```rust
use std::collections::HashSet;
use std::hint::black_box;
use std::time::Instant;
use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_tensorbackend::{default_cpu_execution_context, ExecutionContext};
use tensor4all_tensorci::{crossinterpolate1, TCI1Options};

fn fold_digest(values: &[f64]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325u64, |acc, value| {
        (acc ^ value.to_bits()).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn main() {
    let sites = 6usize;
    let dim = 4usize;
    let npoints = 200usize;
    let f = |idx: &Vec<usize>| {
        1.0 + idx
            .iter()
            .enumerate()
            .map(|(site, &v)| ((site + 1) as f64) * (v as f64))
            .sum::<f64>()
    };
    let (tci, ranks, errors) = crossinterpolate1::<f64, _>(
        f,
        vec![dim; sites],
        vec![0usize; sites],
        TCI1Options::default(),
    )
    .unwrap();

    let points: Vec<Vec<usize>> = (0..npoints)
        .map(|n| {
            (0..sites)
                .map(|site| (n / dim.pow(site as u32)) % dim)
                .collect()
        })
        .collect();
    let distinct = points.iter().collect::<HashSet<_>>().len();
    assert_eq!(distinct, npoints, "probe points must be distinct");

    let threads = default_cpu_execution_context().with_backend(|backend| backend.num_threads());
    let _ = &ExecutionContext::Cpu(default_cpu_execution_context());

    let cold_start = Instant::now();
    let cold_value = tci.evaluate(&points[0]).unwrap();
    let cold = cold_start.elapsed();

    let warm = &points[1..];
    let start = Instant::now();
    let mut acc = 0.0f64;
    for point in warm {
        acc += black_box(tci.evaluate(black_box(point)).unwrap());
    }
    let direct = start.elapsed();

    // Built after the cold call, so the cold call pays the one-time build.
    let tt = tci.to_tensor_train().unwrap();

    let mut max_err = (cold_value - tt.evaluate(&points[0]).unwrap()).abs();
    let mut values = Vec::with_capacity(points.len());
    values.push(cold_value);
    for point in &points[1..] {
        let value = tci.evaluate(point).unwrap();
        max_err = max_err.max((value - tt.evaluate(point).unwrap()).abs());
        values.push(value);
    }

    let start = Instant::now();
    let mut acc_tt = 0.0f64;
    for point in &points {
        acc_tt += black_box(tt.evaluate(black_box(point)).unwrap());
    }
    let reference = start.elapsed();

    println!("ranks={:?} final_error={:?}", ranks.last(), errors.last());
    println!(
        "distinct_points={distinct} backend_threads={threads} cold_first_call={cold:?} \
warm_tci_evaluate={direct:?} ({:.2} us/point) reference_tt_evaluate={reference:?} ({:.2} us/point) \
pointwise_max|diff|={max_err:e} digest={:#018x} acc_sink={:#x}",
        direct.as_secs_f64() * 1e6 / warm.len() as f64,
        reference.as_secs_f64() * 1e6 / points.len() as f64,
        fold_digest(&values),
        black_box(acc.to_bits() ^ acc_tt.to_bits())
    );
    println!(
        "threads: RAYON={:?} OMP={:?} BLAS={:?} OPENBLAS={:?} MKL={:?}",
        std::env::var("RAYON_NUM_THREADS"),
        std::env::var("OMP_NUM_THREADS"),
        std::env::var("BLAS_NUM_THREADS"),
        std::env::var("OPENBLAS_NUM_THREADS"),
        std::env::var("MKL_NUM_THREADS")
    );
}
```
