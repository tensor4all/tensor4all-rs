# CI fixture measurements for issue #769

## Scope and retained contracts

The fixture baseline is `2ce532a8336f299897ce5cef44113e768acf9757`.
These are local release execution measurements, not estimates of hosted CI
wall time, compilation savings, or aggregate runner usage.

| Candidate | Decision | Retained coverage |
|---|---|---|
| `single_scale_interpolation_n3` | Reduce bit depth from 4 to 3; retain degree 15. | The same non-polynomial function, asymmetric three-dimensional box, fused site dimensions, default SVD compression, rank bound, train length, and full-grid `1e-7` accuracy assertion remain. Three sites exercise left, interior, and right cores; all 512 grid points are checked. The adjacent `single_scale_interpolation_n2` keeps four sites and repeated interior-core propagation. This is a general interpolation test, not a reduced historical bug reproducer. |
| Tutorial `qtt_r_sweep` | CI sets maximum bit depth to 10; default remains 15. | Every depth from 2 through 10, complete sampling at each retained depth, unchanged `1e-12` interpolation tolerance, fixed initial pivots, real-valued target, error/statistics CSV schemas, and plotting inputs remain. A new analytic error assertion uses the live tutorial's `1e-8` validation bound. Default full-range output is still supported and validated. |
| Other 12 tutorial binaries | Retain every flow and their existing fixtures. | Includes real/complex transforms, multivariate interpolation, affine boundary modes, partial Fourier transforms, and their numerical checks. CSV existence checks supplement these tests; they do not replace them. |
| TreeACI low-temperature convergence, #741 | Retain `r=9` and all five witnesses. | The original failure is a localized, topology-dependent false convergence that sparse checks can miss. No smaller fixture has been established to fail the pre-fix algorithm, so this audit makes no equivalence claim for a smaller grid. |
| TreeTN global-subspace-expansion AD regression | Retain. | Already two dimension-2 sites; the backward/local-density assertions exercise a distinct AD path. |
| Two-coordinate-axis QFT scheduling | Retain. | Already four binary sites (16 values); both partial and full refinement and both coordinate axes remain. |

No numerical tolerance, coverage threshold, skip list, or production numerical
algorithm is changed by the fixture reduction.

## Redundant execution audit

- The Cargo book harness includes the root README plus 10 guide pages. mdBook
  additionally traverses the published chapter/tutorial set through `SUMMARY.md`
  and exercises its own standalone snippet compilation. Root README validation
  is unique to the Cargo harness; extra tutorial snippets and mdBook processing
  are unique to the mdBook path. Both stay. Their expensive preparation rebuild
  was addressed by the preceding CI work; overlap alone does not justify
  deleting either contract.
- `qtt_integral` and `qtt_integral_sweep` share numerical setup but have different
  terminal-versus-CSV and multi-resolution contracts. Both stay. Similar QTT
  tutorial names likewise do not establish redundant regression protection.
- The coordinated core integration-suite change removes only an exact index
  alias duplicate, with its own retained-test mapping. This fixture audit does
  not claim broad numerical test duplication or delete any shared helper.

## Measurement protocol

Host: Apple M4, 10 logical CPUs, 24 GiB RAM, macOS 27.0,
`rustc 1.98.1 (48a229cea 2026-09-01)`. Builds use the locked dependency graph,
release profile, four Cargo jobs, and the existing warm local artifact cache.
Build time is excluded from execution measurements. Doctests are excluded from
these unit/integration/binary suites and remain a separate hosted check.

`RAYON_NUM_THREADS=1` configures the pinned tenferro `CpuContext::from_env` used
by the default backend. A separately compiled probe calls
`tensor4all_tensorbackend::with_default_backend` and asserts
`backend.num_threads() == 1`; the effective count was one. `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` also equal
one. Libtest uses one test thread. The interpolative fixture is deterministic.
Tutorial QTCI call sites set `n_random_init_pivot=0` and use explicit pivots;
`to_treetci_options` disables global pivot search and `run_treetci_batch` uses
`DefaultProposer`. Thus those interpolation paths make no random pivot choices
and need no RNG seed. The sweep retains the same three explicit starting pivots. No CPU affinity is imposed. Other task build
and timing processes were held during the retained measurements.

Baseline test executables are retained before the edits. Baseline and candidate
executables run alternately in the same session, with one discarded warmup
followed by three retained samples. The isolated n3 test and complete affected
unit/integration/binary suites are measured separately. Tutorial suite samples
were refreshed after adding the depth-override boundary test; its candidate
suite includes the new test while the retained baseline executable does not. The tutorial suite includes its library,
all binary test targets, the 13-binary integration flow, and verification tests;
it is not a whole-workspace measurement. The old tutorial integration executable
runs the updated sweep executable with no override, preserving default depth 15
and also validating the new numerical assertion on the full default range.

The two packages are selected together when building. Selecting only
`tensor4all-interpolativeqtt` exposes a pre-existing missing `global-defaults`
feature combination in its core dependency; the tutorial package enables the
normal core defaults used by the workspace. This experiment does not change
that unrelated feature contract.

## Results

Raw samples, including discarded warmups, are in
[`ci769-fixtures.json`](ci769-fixtures.json). Reported medians use trials 1–3.

| Execution scope | Baseline median | Candidate median | Change |
|---|---:|---:|---:|
| `single_scale_interpolation_n3` | 27.428019 s | 12.957020 s | 52.76% lower |
| `interpolative_full_unit_suite` | 28.867090 s | 14.845848 s | 48.57% lower |
| `qtt_r_sweep` | 0.062391 s | 0.008403 s | 86.53% lower |
| `tutorial_full_package_suite` | 4.501759 s | 4.492161 s | 0.21% lower |

The 27-test interpolative suite improves by about 49%. The tutorial sweep saves
about 54 ms in isolation while retaining all 13 tutorial flows. The complete
tutorial suite distributions overlap; its approximately 0.2% median difference
is inconclusive and is not a claim of a meaningful suite speedup. The candidate
also adds a depth-override boundary test: 23 baseline tutorial tests become 24.
It covers malformed input, both rejected range endpoints, non-UTF-8 input on
Unix, and the accepted minimum with its four-point CSV. The maximum/default
range 15 is exercised by each retained baseline integration run against the
updated executable and its analytic assertion. Interpolative test counts remain
27 before and after. The coordinated core suite has separate measurements.

A diagnostic pass through each tutorial process identified the remaining
execution cost. These are single observations, not controlled speedup estimates:

| Tutorial binary | Candidate diagnostic time |
|---|---:|
| `tensor4all_tutorial_code` | 0.0032 s |
| `qtt_function` | 0.0039 s |
| `qtt_interval` | 0.0040 s |
| `qtt_integral` | 0.0034 s |
| `qtt_integral_sweep` | 0.0074 s |
| `qtt_r_sweep` | 0.0083 s |
| `qtt_multivariate` | 0.0072 s |
| `interpolative_qtt` | 0.0047 s |
| `qtt_elementwise_product` | 0.0350 s |
| `qtt_affine` | 2.3686 s |
| `qtt_difference_kernel` | 0.0143 s |
| `qtt_fourier` | 0.3998 s |
| `qtt_partial_fourier2d` | 1.6247 s |

Affine and partial Fourier demonstrations dominate the reduced tutorial suite.
Their boundary/layout and sampled transform contracts are retained here; this
change does not establish smaller equally representative fixtures for them.
The raw diagnostic baseline sweep observation includes first execution of the
saved executable and is not used for the sweep speedup claim. The repeated,
warmed sweep samples above are the basis for that comparison.

The default 2-through-15 sweep samples 65,532 points; CI's 2-through-10 sweep
samples 2,044 points. Every point in the retained grids is still checked rather
than replacing numerical validation with a sparse or CSV-existence check.

## Validation evidence

All 27 interpolative tests and all 24 final tutorial tests passed in the direct
release-suite measurements. Nextest passed the original 50 combined tests; a
focused nextest run passed the subsequently added boundary test (23 unrelated
tutorial tests filtered out). Clippy with all targets and denied warnings passed
for core, interpolativeqtt, and tutorial-code together. Formatting and whitespace
checks passed. The live tutorial edit is prose only; its existing runnable code
block and linked source path are unchanged.

The release build selects both packages, followed by nextest using the same
feature union:

```bash
cargo test --locked --release -p tensor4all-interpolativeqtt -p tensor4all-tutorial-code --no-run -j 4
RAYON_NUM_THREADS=1 cargo nextest run --locked --release -p tensor4all-interpolativeqtt -p tensor4all-tutorial-code --test-threads 2
```

For the isolated execution samples, the retained release executables are called
with `--test-threads=1`; the n3-only sample additionally selects
`tests::single_scale_interpolation_n3 --exact`. The full tutorial sample runs all
its libtest executables sequentially, including executables with zero tests, to
retain the same process set before and after.
