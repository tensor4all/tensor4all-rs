# Issue #1130: measured acceptance results

Decision: both cases satisfy the pre-registered **10.0x hard runtime limit**,
the applicable RSS limits, and every numerical check. Both exceed the **1.5x
soft review trigger**; runtime overhead remains a review item.

Measured revision: `d60f07514d574a059070e258bb641575947a7ba5`. Both arms are policies of the **same library**
and use the same finished release binary. This is an internal policy comparison.
Five fresh-process paired repetitions per case, with alternating order, on
logical CPU 8 of AMD Ryzen 9 6900HX (Linux/WSL, platform faer provider, one thread).
Every process primes the CPU for one second outside the scheduler interval.

Case names identify fixed synthetic scheduler fixtures, not production Green
functions: `original` is a six-bit single-axis unitary DFT; `interleaved` is
a two-axis product DFT with three interleaved bits per axis. Both use the same
seed `0x5eed_0006` SplitMix64 complex nine-site chain (six transformed sites,
three spectators), input bonds at most four, and all 64 full-depth input
patches, with `rtol=1e-8`, `atol=0`. The recorded harness constructs them explicitly.

The [pre-registration comment](https://github.com/tensor4all/tensor4all-rs/pull/1407#issuecomment-7025598593) and
[maintainer decision](https://github.com/tensor4all/tensor4all-rs/pull/1407#issuecomment-7025360934)
precede every timed run. Limits were not changed after seeing measurements.

| Case | RankGoal(8) median scheduler time | Tolerance median scheduler time | Median paired ratio | Descriptive bootstrap 95% interval |
|---|---:|---:|---:|---:|
| original | 0.390846 s | 0.639912 s | 1.6254x | [1.6144, 1.6837] |
| interleaved | 0.431963 s | 1.030759 s | 2.3832x | [2.3241, 2.4057] |

The primary statistic is the median of the five paired ratios, not the ratio
of the two marginal medians. The percentile bootstrap uses 10,000 resamples,
seed 1130; with five pairs its interval is descriptive and does not replace
the pre-registered median gate.

| Case | RankGoal(8) maximum process RSS | Tolerance maximum process RSS | Worst paired RSS ratio |
|---|---:|---:|---:|
| original | 34.500 MiB | 39.512 MiB | 1.2357x |
| interleaved | 34.848 MiB | 35.434 MiB | 1.1043x |

The original case stays below the 3.0x paired RSS ceiling and 5 GiB absolute
limit. Every process also runs under a 5 GiB virtual-address limit and 6 GiB
gate memory cap. Process RSS is VmHWM sampled immediately after the scheduler,
including input/operator setup and CPU priming, before the dense oracle.

| Case / policy | Logical parameters | Terms | Compression attempts | Accepted compressions | Maximum dense-oracle L2 error | Maximum reported bound |
|---|---:|---:|---:|---:|---:|---:|
| original / goal8 | 712 | 1 | 47 | 47 | 1.957535958e-08 | 2.972222848e-08 |
| original / tolerance | 712 | 1 | 127 | 127 | 2.989798202e-08 | 5.253295304e-08 |
| interleaved / goal8 | 4296 | 9 | 63 | 49 | 4.013304833e-12 | 1.894636328e-11 |
| interleaved / tolerance | 712 | 1 | 255 | 255 | 2.025877801e-11 | 5.225007392e-11 |

Every cell checks its own output against an independent dense product-DFT
oracle, including all retained terms, and checks both L2 and maximum absolute
error against the reported bound plus the registered 1e-9 roundoff allowance.
Every reported bound is inside the global allowance. Tolerance returns one
strict region/term in both cases; the interleaved goal baseline legitimately
retains nine terms. No assertion tolerance was changed.

Both cases cross the soft review trigger. The executed compression-attempt
counts increase from 47 to 127 (original) and 63 to 255 (interleaved). These
counters describe extra work consistent with always probing eligible axes;
they are not a measured phase-time attribution. The interleaved tolerance
output stores 712 parameters instead of 4,296, at the measured runtime cost.
On the original fixture both policies already retain 712 parameters in one
term: there is no measured final-storage improvement for its 1.6254x paired
runtime ratio. Tolerance supplies rank-independent accuracy-driven selection;
that capability, not an original-case speed or storage gain, is the trade-off
submitted for review. These numbers do not justify choosing it for every input.
No phase microbenchmark or post-hoc optimization was substituted for the
registered complete workload.

Validity and evidence:

- One complete attempt: 20/20 CLEAN reports and numerical checks; no retries
  or omitted samples. Every row and every failed/contaminated attempt would
  have been retained; none occurred.
- [metadata.json](metadata.json) records source pins, binary/controller/harness
  SHA-256, compiler/build log, provider, environment, host and topology.
- [attempt-1/samples.json](attempt-1/samples.json) retains arm order, raw
  numbers, commands and per-process gate report names. The adjacent stdout,
  stderr and gate JSON files retain the complete cell evidence.
- [outcome.json](outcome.json) contains all paired ratios, intervals and gates.
  Agent affinities were confined to observer cores and restored after the run.
- This evidence covers these two fixed seeded inputs on this Linux guest.
  The five-pair intervals are descriptive; hypervisor scheduling outside the
  Linux noise observables is not independently measured. The cap-32 Green
  function diagnostic was not run and is not claimed as covered.

Reproduce using the committed [controller](../run_acceptance.py) and
[plan](../benchmark-plan.md) at the measured revision. Write to a fresh scratch
directory outside the checkout. The controller builds through `ai-run build`
and runs the finished binary only through `ai-run measure`.
