# Issue #1130 performance acceptance

The complete paired experiment passes the maintainer's pre-registered 10.0x
hard runtime gate for both cases, the RSS limits, and every error-bound check.
Both cases exceed the 1.5x soft review trigger, so the runtime overhead is
explicitly handed over for review rather than described as a speed improvement.

The original six-qubit case has median paired ratio 1.6254x, and the interleaved
two-axis case 2.3832x. Original-case tolerance maximum process RSS is 39.512 MiB;
the worst paired RSS ratio is 1.2357x (limit 3.0x), below the 5 GiB process cap.
All twenty cells are CLEAN and numerically valid. No sample was omitted or
retried. The registered full workload, rather than a replacement microbenchmark,
was measured.

The [generated report and complete raw evidence](../experiments/issue-1130-implementation/acceptance-2026-11-20/)
record all five paired ratios per case, descriptive bootstrap intervals, gate
observables, errors, counters, source/binary hashes and reproduction settings.
The [maintainer's decision](https://github.com/tensor4all/tensor4all-rs/pull/1407#issuecomment-7025360934)
and [published pre-registration](https://github.com/tensor4all/tensor4all-rs/pull/1407#issuecomment-7025598593)
precede the run. Neither hard threshold was changed after timing results.

## Revisions and method

- Synchronized the implementation branch with current main before registration.
  The merged main change concerns independent one-site truncate validation.
- Both policies use the same immutable finished binary from the registered
  revision, resolved through Git and recorded in the measurement metadata.
- Release build through `ai-run build --mem 6G`, faer platform CPU provider,
  logical CPU 8, five alternating fresh-process pairs per case. Each process
  primes for one second outside scheduler timing. Observers use CPUs 0-7;
  their original affinities were saved and restored afterwards.
- Each timed process uses `ai-run measure --mem 6G --cpus 8`, the 5 GiB
  virtual-address cap, and one-thread Rayon/BLAS environment. The completed
  binary's hash and clean source revision were checked across the suite.
- Every timed cell checks a dense product-DFT oracle after the timed interval,
  its numerical bound and global allowance, and disjoint regions. Tolerance
  additionally checks strict partition conversion. No numerical tolerance or
  test threshold was modified.
- The controller builds and checks formatting before measurement. Its output
  status is `PASS`. Raw evidence was copied into the repository only after the
  full run ended; no sources or commits changed during measurements.

## Review implications and limits

The interleaved output reduces logical storage from 4,296 parameters in nine
terms to 712 parameters in one term. The original output stores 712 parameters
in both policies. Compression attempts grow from 47 to 127 and from 63 to 255,
respectively. Those are executed-work counters, not a phase-time attribution.
Both soft runtime triggers should be reviewed with the per-case raw timings.

This evidence covers the two registered seeded workloads on this Linux/WSL
host. Five-pair bootstrap intervals are descriptive, and host hypervisor noise
is not independently observable beyond the recorded Linux gates. The cap-32
Green-function diagnostic remains outside the measured acceptance suite.

The public API and Rust scheduler are unchanged by this acceptance step.
Cross-provider code review and hosted CI for the final evidence commit remain
part of the handover; this work does not authorize a merge.
