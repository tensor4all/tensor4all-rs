# TreeTCI retained-coordinate global pivot search

## Decisions

- Reuse the core `floating_zone_walk` used by chain TCI. Retain each coordinate
  move and repeat sweeps, with the same 100-sweep bound and early stop at ten
  times the acceptance threshold. Keep strict threshold acceptance, stable
  ordering across starts and deduplication.
- Own one cached tree evaluator for the whole search. Name the varied site
  on each scan, including binary-site single-candidate scans; reuse flat
  coordinate scratch bounded by one scan. Keep pointwise readout injection
  for numerical/reference checks.
- Preserve the lower-value tie rule within a coordinate scan. The search
  intentionally changes fixed-seed trajectories; pivot recordings were
  regenerated from the pointwise reference. Existing numerical test
  tolerances remain unchanged.

## Verification conclusions and constraints

- Original-start axis scans miss above-threshold pivots on the coupled
  residual regressions; retaining moves and repeating sweeps finds them.
  Coverage includes real/complex single/double precision and a branched tree
  with a site-free junction, plus ordering, duplicates, failures and bounds.
- The issue's `(0,0)` start on an `i*j` residual has flat zero fibers. A greedy
  floating-zone walk can stall there too; this repair does not promise global
  maximization or certify full-network accuracy.
- Algorithmic cost is compared in the linked [complete paired experiment](../../benchmarks/results/2026-10-07-treetci-global-walk.md).
  The walks visit different candidates, so timings do not isolate readout
  speed and need not retain identical rank/error histories.
