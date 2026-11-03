# Fit and Zipup diagnostic protocol (revision 2)

This supplements the decomposition comparison with actual operation-level
cost and error checks. It is a separate shared-host diagnostic, with no
performance promotion or parallel scaling claim.

- Same release/default-faer build and thread/affinity budgets as the
  decomposition protocol: 1 thread on CPU 8 and 8 threads on CPUs 8-15.
- The full `(chi,d,length)` grid is `(16,2,6), (32,2,6), (16,4,4), (32,4,4)`.
  Output rank cap is `chi`. Inputs are normalized Gaussian MPOs, seed 1108.
- Both Fit (two sweeps) and Zipup run deterministic SVD and RSVD, with RSVD
  seed 1108, oversampling 8, two power iterations. No threshold policy is used.
- One second of time-based priming per method/case, one discarded warmup per
  algorithm, three timed samples, alternating SVD/RSVD order across repeats.
- A small exact dense oracle is constructed before timing and bounded to at
  most 65,536 entries. Every result is materialized once after its timer stops
  and compared with that oracle by a full Frobenius residual.
- Every RSVD residual must be at most `1.05 * deterministic_error + 1e-12`.
  Do not relax that gate or selectively retry cells. Random-input errors can
  be large; matched error here does not establish scientific usefulness.
- Retain all raw samples and resolved source/binary hashes. Report medians
  and errors for the entire grid. No concurrent local build/test during timing.

Both executables check their actual backend/Par worker budget before timing;
the core harness also prints each worker CPU mask. A clamped budget is an error.


The [one-power-iteration attempt](contraction-power1/manifest.json) failed the
5% error gate for the six-site `(chi,d)=(32,2)` Zipup warmup: relative error
0.6815967794202312 versus deterministic 0.6375676163114457. That attempt is
invalid as a matched-error comparison and is retained in full. Revision 2
predeclares two power iterations for every RSVD cell, then repeats the entire
grid and both budgets. The error gate and library defaults are unchanged.
