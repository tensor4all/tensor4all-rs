# Opt-in convergence confirmation comparison

## Protocol (declared before the cost campaign)

The regression baseline is origin/main at `35174fdd`; the historical cost arm is
`501ebb4a`. Exact commits are resolved from each checkout by the runner. Candidate
is the committed opt-in implementation containing this protocol; its exact HEAD
is frozen in the generated manifest before execution. Previous fixture probes
are correctness diagnostics, not cost observations.

Four arms use one driver, `crates/tensor4all-treetci/examples/confirmation_comparison.rs`:
historical/off, main/off, candidate/off, candidate/on. Reference copies remove
only the two unavailable option fields and replace confirmation counters with
zero; retain those adapted sources with their hashes. All use original local LU,
no oracle memo, default nsearch=5, 30 sweeps, initial all-zero pivot, root 0,
Linux faer, release profile and an explicit one-thread CpuBackend. The global
validation backend is configured to one thread too. Provider budgets are set by
API, not inferred from environment variables.

Checked grid per arm: 16-site Gaussian, tolerances 1e-4/1e-6/1e-8, seeds 1–6;
60-site smooth comb, tolerance 1e-8, seeds 1–3. Targets follow Simon's #1371 / #1264
reproducers. The comb is the smooth control, not the distinct rank-216 slow-decay
case. Gaussian validation materializes 2^16 points once per result. Comb validation
uses the fixed 20,000-point xorshift stream in the driver. Record independent
relative max/L2, reported error, full rank/error-bit histories, verdict, oracle
and requested counts, and confirmation counts from the same run. Retain negative
accuracy outcomes; sampled confirmation is not a full-domain bound.

Primary correctness gates: candidate/off equals main on ranks, error bits,
termination, oracle/request counts for every checked cell; the ordinary seed-4
real/complex #1352 regressions converge within their unchanged 1e-10 dense error.
Non-finite diagnostics or different deterministic outputs across repetitions
invalidate the experiment. Historical and option-on accuracy differences remain
visible; they do not redefine the default or resolve #1371. There is no promised
speedup or optional-confirmation overhead acceptance threshold: timing is cost
evidence for review, not a performance optimization claim.

Timed grid: all three comb seeds in each arm, two complete paired rounds. Round 1
arm order historical/main/off/on; round 2 on/off/main/historical. Every cell has
identical two-second time-based spin priming. The internal timer covers the full
interpolation call, including confirmation history replay and final requested
materialization; context construction, priming and independent validation are
outside it. Report the median per seed and arm, plus paired ratios, retaining
both rounds. Counts from each timed cell must match its checked cell exactly.

Build all finished binaries before execution. Use `ai-run build --mem 6G --jobs 2`
for builds/checks; each timed arm runs only its finished binary through
`ai-run measure --mem 6G --cpus 8-15 --report ...`. The host is a Ryzen 9 6900HX
(8 cores/16 logical CPUs, 20 GiB RAM), observer set 0–7. Record actual affinity,
CPU topology, compiler, lock/driver/binary hashes, environment thread settings,
source clean states and guard JSON. Source/HEAD/binary changes, nonzero exits,
resource kills, missing cells or non-CLEAN reports make the whole paired campaign
INCONCLUSIVE. Retain every attempt. A CONTAMINATED campaign is rerun in full under
the same protocol, per the machine policy; never selectively retry a case. No
build, edit or commit while timing; observer commands stay on CPUs 0–7.

[Complete results](RESULTS.md), [all checked cells](checked.csv),
[raw timing samples](timed.csv) and [guard/source metadata](data/) record the
completed campaign. All eight guards are CLEAN; no campaign attempt was discarded.
