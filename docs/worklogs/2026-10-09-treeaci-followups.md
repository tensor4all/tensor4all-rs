# TreeACI umbrella follow-up repairs and investigation

[AI Supplied] Continues #854 from merged main `5a6d330a`. The user requested
continued upstream cleanup after #855 merged. No RSI algorithm or workload
was executed; its reference repository supplies only the existing DMRG data
fixture used in #784.

## Concrete changes

- #686: reuse the batch-evaluated guard starting targets and approximations.
  Coordinate walks receive their already checked initial residual, avoiding
  one operator evaluation per search start. Point diagnostics count actual
  target evaluations. Retained residual arrays are budgeted, and start input
  and expanded-coordinate buffers are released before coordinate walks.
- Core owns the shared floating-zone kernel and a checked
  `floating_zone_walk_with_initial_error` entry point. Typed errors reject
  invalid dimensions, starts, initial residuals, tolerances, and unrepresentable
  candidate storage before callbacks. The original entry point keeps its
  existing callback/error contract and shares the kernel.
- #686: propagate compact descending-node bootstrap axis suffixes in prepared
  dependency order rather than traverse and sort each complete component.
  Dimension-one sites need no digits, and each retained suffix has at most
  `ceil(log2(max_initial_rank)) <= usize::BITS` axes. Trimming after each
  incoming component also bounds high-degree merge temporaries. Enumeration
  retains the original mixed-radix digit order. Reuse one projection scratch
  buffer per initialization; overwrite dependencies before their consumers
  read them. The combined logical temporary peak is checked before allocation.
  Rank-one initialization skips suffix and projection storage.
- #626: include heap payload owned by wide `IndexKey` values in retained-cache
  ownership diagnostics. Core reports boxed storage and dynamic limb capacity
  through `IndexKey::owned_heap_bytes`; TreeTN accumulates a counter only when
  admitting a new column/key. Hits and refused or failed misses cannot increase
  retained heap accounting. Logical message admission limits are unchanged.

The batched starting approximation can choose its initial contraction center
from the whole start batch instead of the first individual point. This can
change floating-point roundoff in other fixtures; the shared search kernel
preserves mathematical scan and tie order, not a universal bitwise trajectory
guarantee. The recorded DMRG output identity is specific to that fixture.

These are logical payload/ownership estimates, not allocator-header or RSS
ceilings. No unsafe code, dependencies, numerical tolerance changes, or cache
replacement policy changes are introduced.

## Cache design reconciliation

The original #626 LRU/clear proposal is superseded by the explicit later
maintainer-approved append-only, evaluator-lifetime design:

- https://github.com/tensor4all/tensor4all-rs/pull/646#issuecomment-5316892012
- https://github.com/tensor4all/tensor4all-rs/pull/646#issuecomment-5316916280

Current main already has cross-call packed messages, bounded logical admission,
top-down hits, compact keys, varying-site hints, and diagnostics. Individual
entry eviction and public clearing would contradict the later approval.
The newly confirmed wide-key ownership omission is the remaining concrete
cache correctness item addressed here. Performance parity remains a separate
workload-dependent investigation (#671).

## #784: reproduced oscillation, no invented convergence repair

ACI-only DMRG n50, d3, input bond40, two identical inputs, f64, cap400,
relative local tolerance `1e-8`, seed1, enabled default guard, single-thread
release reproduces all 20 ranks/local errors from the original report exactly.
It still stops with `MaxSweeps`. Separate 3/20-pass histories match prefixes.
After guard/bootstrap changes the 3/20-pass output core JSON is byte-identical
to the preserved main baseline.

Independent TT-inner-product errors (about `1e-7` relative resolution):

| Pass | Relative Frobenius error | Maximum normalized local error | Max rank |
|---|---:|---:|---:|
| 3 | 3.0396351640991636e-6 | 9.96611329623192e-9 | 136 |
| 20 | 2.998203353737911e-6 | 9.999754227623127e-9 | 137 |

`||A3-A20||/||A3|| = 2.359354774057086e-6`; flat scalar accuracy does not mean
the output stopped changing. Peak-relative local residual and global
Frobenius error use different normalizers. Independent 4096 Born-weighted
samples and coordinate searches found residual/target-peak ratios below
`1e-8`, but supply lower bounds, not an error certificate or the actual
internal guard scale. No false `Converged` result or missed above-threshold
residual was proved. #784 remains open for per-edge pivot spectra/scale evidence.
The latest-only local-error gate has documented ACI provenance; a different
TreeTCI trailing-error policy is not by itself an unfixed ACI bug.

Fixture: `datasets/itensor_dmrg_mps/n50_system/psi_maxdim40_n50.h5` from
`https://github.com/zmeng137/Recursive-Sketched-Interpolation`
(revision `153b25a8aa059d0147b45955d0842b2f32fa5d1d`) as used in the
original issue. Convert the fixture using `benchmarks/treeaci_dmrg_fixture.py INPUT.h5 OUTPUT.json`
(with h5py and NumPy), then run `cargo run --release -p tensor4all-treeaci
--example followup_quality -- chain OUTPUT.json 400 1e-8 OUTPUT_PREFIX`.
The committed `followup_quality` example reruns the public API
and checks history prefixes; its chain mode never dense-materializes the
50-site target. Fixture SHA-256: `d0eac62df162b78779cbef5e2c2474c8ecf6bab5c950d8982a174eab27faeff6`.
Generated JSON, plots, binaries and timing artifacts remain
outside git.

## #794: mixed-capacity investigation remains unconfirmed

A pristine-main private diagnostic explored 120 target/seed/tolerance cases,
with 960 eligible mixed-capacity searches. Both found pivots were injectable;
there were zero non-injectable pivots or repeated no-op injections. The
137272 target evaluations stayed within the asserted per-search bound of
272 points. Local rank-limited cases were excluded rather than forcing a
search the production scheduler would skip. This is negative evidence,
not proof of absence. #794 remains open; there is no speculative public
termination change.

## Scope still open

#686 retains the low-working-budget 3+incoming scalar fallback setup and small
schedule bookkeeping investigations. #671 still needs a current real-stage
low-T/layout matrix, with actual input/output operands and bond products.
The downstream stage-replay harness also needs its obsolete diagnostic key
filter reconciled with current `input:0:<node>` namespaces before an expensive
replay; no downstream edits are included in this upstream repair.

## Validation and performance

- Debug: `cargo test --locked --offline -p tensor4all-core -p tensor4all-treetn
  -p tensor4all-treeaci -- --skip
  low_temperature_branch_convergence_does_not_hide_growth_on_smaller_cuts`:
  **2514 passed**, zero failures, 18 opt-in profiles ignored. Includes 345
  Core, 21 TreeACI, and 156 TreeTN doctests.
- The R=9 low-T regression is impractically slow in debug; its focused release
  run **passed** in 46.14 seconds with its existing convergence/witness bounds.
- Diagnostics feature: **2 integration tests passed**, including separate
  input operands/output/frame labels and warm cache timing accounting.
- Scoped all-target Clippy with warnings, missing error docs, and missing panic
  docs denied; `cargo fmt --all --check`; public-error-doc audit; complete API
  inventory; repository-rules review: all **passed**.
- The committed DMRG converter produces byte-identical JSON to the converter
  used in the original reproduction. ACI core outputs at 3/20 passes also match
  the separately preserved pre-edit quality baseline byte-for-byte.

The initial shared-target timing baseline incorrectly reused candidate
artifacts across worktrees. Candidate-only API/resource strings and warnings
confirmed the provenance failure; that preliminary timing report is excluded,
with raw data preserved. The usable performance comparison rebuilds the
baseline in an isolated target and records executable hashes.

Valid paired release experiment: baseline main `5a6d330a` built in a separate
`CARGO_TARGET_DIR`; candidate code commit `797f644edd30feb3472c981ba942e5919c8577c5`.
Candidate-only API/resource-context strings are absent from the baseline and
present in the candidate. The baseline code tree is identical to merged main.

Protocol: committed `scripts/run-treeaci-branch-cost.py`, 120 bounded cases
(f64/c64, four bond profiles, degree2/3/4, end-to-end ACI and cold/warm query
batches8/32), three alternating baseline/candidate pairs, three measured
repetitions after each warm-up, CPU2 affinity, all provider/BLAS/Rayon threads1,
production features with diagnostics disabled. Decision thresholds are the
script defaults, written before sampling. Verdict: **DESCRIPTIVE**, **zero
validity failures**; no regression threshold was pre-registered. Host frequency
sampling was unavailable. All independent dense-oracle assertions passed;
maximum relative oracle residual across both binaries was `2.655255541087604e-15`.

| Degree | ACI cases | Median candidate/baseline wall ratio | Case-ratio range |
|---|---:|---:|---:|
| 2 | 8 | 0.921127 | 0.865059–0.944511 |
| 3 | 8 | 0.938990 | 0.922700–0.952464 |
| 4 | 8 | 0.944612 | 0.918957–0.960005 |

The ACI fixture represents the same global function under these topologies.
Most cases save 60 target evaluations per run (two passes ×30 guard starts);
some degree-two cases save132 because changed initial contraction centers also
alter roundoff-level walk continuations. These are fixture-specific descriptive
timings, not a general branching/TT parity verdict. No blanket ACI-versus-TCI
or real-stage speedup is claimed.

Reproduce with separately built executables:
`scripts/run-treeaci-branch-cost.py --baseline BASE_BINARY --candidate
HEAD_BINARY --baseline-commit BASE_SHA --candidate-commit HEAD_SHA --pairs 3
--repeats 3 --cpu AVAILABLE_CPU --output NEW_REPORT.json`. Compile the example
with `T4A_BENCH_GIT_COMMIT` set to the matching recorded revision. Raw generated
reports remain in the ignored downstream experiment directory.

Baseline binary SHA-256:
`1b1accb92e5bdd1a9f3f43335b3be915180ca7502f1c79bdcbc8b130deff8a9a`.
Candidate binary SHA-256:
`68e4438284782cad8bd1854a29e416a040ff1122868affdc16859891f5d5ed96`.


Coverage impact: the original floating-zone tests move to their owning
module's test subdirectory with all assertions retained. New trajectory-parity,
callback/error-boundary, mixed-dimension ordering, long-chain suffix bounds,
scratch reuse, combined budget, exact guard point-count, and wide-key ownership
regressions exercise the replacement paths. No coverage thresholds or existing
numerical tolerances were lowered.
