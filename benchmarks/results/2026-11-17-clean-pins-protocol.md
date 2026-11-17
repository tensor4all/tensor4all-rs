# Clean downstream-pinned native thread scaling

This is a fresh same-library scaling measurement. It replaces no historical
record and includes none of the withdrawn upstream repairs. The measured
production source is tensor4all-rs main `86b27b91d03c0b7027be5547c44c45829220d8de`.
Only a public-API measurement example and its dev dependencies are added.

The dependency baseline is exactly this downstream revision's Cargo.lock:
tenferro Git `32a4e98c12f9dc36ffa85b5273b9247b44936959`, Hataori Git
`9d65515706eb9c7b7414dd0d4d32644ec0c0674c`, and registry threadteam 0.5.1.
Cargo metadata must resolve all numerical/runtime packages to the original
Git/registry sources, with no path overrides or local dependency modifications.
Source checks record the fetched root base, harness commit, actual resolved
sources/features, clean Git checkout states and binary hash before measurement.
This is the exact downstream pin, not latest tenferro main.

## Frozen workload and case grid

The square-lattice Green-function evaluator integrates a physical 128×128
momentum mesh. QTT inputs: two dimensions, bounds [-6,6]×[0.8,1.6], four bits,
polynomial degree three, tolerance 1e-4, compression default 1e-12, maximum bond
dimension two. Real and complex outputs are separate scaling subjects.

The 13 configurations are: one engine B=1,2,4,8,16; P=1,2,4,8,16 lanes of B=1;
and P=2,4,8 lanes of B=2. Each real/complex configuration runs in a fresh process
in each of five blocks, for 130 independent processes. Seed 1347 randomizes all
26 configurations within every block. No historical sample is reused.

The allocation is four physical cores, logical CPUs 8,10,12,14 (their SMT peers
are unused). A one-/two-thread case receives the first one/two physical cores;
cases with four or more compute slots share all four. Eight and sixteen slots
are deliberately oversubscribed. Resources are constructed through the original
public CpuBackend::with_threads and TenferroLanes constructors. Placement is
whatever those unmodified defaults implement; no benchmark mask restoration,
explicit per-engine partition, wait-policy override or local placement patch
is added. Constructor and callback masks are recorded, including any narrowing.
All vendor/Rayon thread environment variables set by the harness are fixed at
one; the configured engine token budget is asserted at every evaluator call.

Each process primes its configured engine(s) for two seconds, then runs one
checked warmup. The primary observation is one uninstrumented public QTT API
call; process CPU clocks bracket, but remain outside, its wall timer. Callback
counters and affinity queries exist only in the warmup monomorphization.
The full 16×16 output is compared to an independent analytically reduced
1024-point oracle, including exact grid coverage and canonical prefix order.
The primary and warmup results must agree. Across configurations within each
scalar, work counts, prefixes, leaf counts and core bytes must agree; every
grid value must agree to the existing 1e-10 diagnostic tolerance. Existing
numerical test tolerances are not changed.

## Resource gate, validity and analysis

Compilation runs through ai-run build with an explicit 8 GiB cap and two build
jobs. Measurement starts only after compilation, using ai-run measure with
an 8 GiB cap, the measurement CPU domain 8–15 and a retained JSON report.
The measured command runs the precompiled example and its precompiled child
processes; it never invokes Cargo, rustc or another build. The agent's own
observer is confined to the shared CPU domain, then restored afterwards.

Only exit=0, complete 130-process numerical validation and verdict=CLEAN are
eligible for timing conclusions. CONTAMINATED attempts remain archived and a
replacement repeats the complete grid; no selective retries or sample removal.
The ai-run thresholds and its full validity observations are retained unchanged.

Report all raw samples, median wall and process CPU, relative MAD, and speedup
over the same scalar's pool-free single-thread engine. Pair the speedup samples
by block and use 10,000 bootstrap resamples of their median with seed 1347 for
a 95% interval. Relative MAD above 15% is explicitly imprecise, not a reason to
omit samples. A clear speedup requires ratio≥1.10 and paired interval lower>1;
a clear slowdown uses the reciprocal rule. Otherwise mark no clear difference.
There is no candidate repair to promote in this baseline study. No MPI result
or latest-upstream performance claim is inferred from this native grid.
