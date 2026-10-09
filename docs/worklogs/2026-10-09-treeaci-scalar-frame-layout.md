# Reuse scalar frame-builder batch layouts

Tracking: [#686](https://github.com/tensor4all/tensor4all-rs/issues/686), under
[#854](https://github.com/tensor4all/tensor4all-rs/issues/854).

## Decisions

`FrameBuilder::compute_batch` previously delegated every leaf or three-or-more
incoming sample to scalar `compute`, repeating bond/axis discovery for the same
immutable core. A batch now prepares one layout lazily at its first genuine
contraction and reuses it for subsequent samples. Memo hits and existing-prefix
pulls return before preparation. Recursive dependencies use their own layout.
The layout lives only within the current batch and holds O(physical axes +
incoming cuts) metadata, with no persistent per-cut cache or O(E^2) retained
storage. Prepared input cores and their sharing through extensions are unchanged.

The common scalar contraction accepts a borrowed iterator rather than allocating
an owned-vector-to-slice adapter per sample. It preserves incoming order and the
same recursive multiply/add reduction. Ordered cut identities and frame lengths
are checked before contraction. The obsolete owned-vector wrapper is removed;
its existing numerical test now uses the retained slice seam. Candidate-frame
cache admission, ownership, sample order, transaction behavior and resource
ceilings remain unchanged; no numerical or dependency policy is introduced.

## Verification conclusions and constraints

The focused regression fails on pristine merged `66ecc870`: a two-sample leaf
batch performs four axis lookups instead of two. The candidate checks exact
frame values, identical core reads and constant axis discovery for all four
scalar types, leaf/three/four incoming cuts, permuted unequal axes and multiple
physical axes. Repeat and empty ranges, existing-prefix-only batches, mixed
old/new ranges and actual append-only extensions preserve their contracts.
Malformed cut order/count, unknown directed edges and incorrect frame lengths
are rejected. Existing kernel tests retain the scalar reference through degree
zero to four and every axis ordering.

257 checks pass: 254 debug tests/doctests, two diagnostics integration tests,
and the focused low-temperature R=9 release regression. The R=9 numerical
workload uses optimization because its debug run is impractically slow; its
tolerance is unchanged. Deny-warning all-targets diagnostics Clippy passes.
Removed paths were reviewed for coverage: the obsolete adapter's sole test
still exercises the common scalar contraction, and existing axis/length error
checks plus the new layout checks exercise its validation paths. No tolerance
or coverage threshold is relaxed. Root README remains accurate.

The [complete paired release report](../../benchmarks/results/2026-10-09-scalar-frame-batch-layout.md)
retains all 120 case summaries and intervals. There are zero oracle/validity
failures, and all evaluated-point counts and maximum relative errors are exactly
unchanged. ACI median wall ratios for degrees 2/3/4 are 1.004045, 0.997207 and
1.007103. The comparison is descriptive: no preregistered regression threshold
or general speedup claim. The exact regression establishes reduced metadata
work; the default matrix does not isolate that cost from the full algorithm.

#686 remains partially open: individual scalar candidate layout preparation and
the low-priority rank/capacity scan investigation remain. #784, #794 and #671
are outside this repair. RSI algorithms are excluded.
