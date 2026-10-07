# Continued TreeTCI optimization (#833)

## Decisions

- Support raising or removing the cap on a state for the same function,
  topology and dimensions. Current pivots and sampled normalization scale
  carry over; each call owns new diagnostics, convergence window and budget.
  Seeded calls restart streams; caller-stream calls advance the supplied RNG.
- Remove full pivot-map history rather than retain one copy. Each pass visits
  every edge once and its two canonical subtree keys are unique to that cut.
  Other edge updates leave those pivots unchanged, so built-in proposers can
  retain current edge pivots with the same ordering and deduplication.
  Custom proposers read `ijset` instead of the removed `ijset_history` field.
- The global-search benchmark now fingerprints only current pivot sets;
  fingerprints in historical results that included all snapshots remain
  historical and cannot be compared to the new fingerprint convention.

## Verification conclusions and constraints

- A real/complex three-arm comb with a site-free junction is optimized with
  χ=4, then uncapped, with global search enabled and disabled. The capped
  result is demonstrably inaccurate; continued and fresh uncapped results
  recover rank 8 and match all 512 target entries and each other within
  1e-12 times the target maximum magnitude. Initial pivots provide independent
  arm variations so the local-only configuration can discover junction rank.
- All 169 treetci/quanticstci unit/integration tests and 70 doctests pass.
  Strict Clippy, strict documentation, API inventory, public-error documentation
  and the full library panic audit pass.
  This includes existing branching/truncated-proposer regressions and the
  runnable continued-state example. No tolerance or coverage threshold changes.
- Continuation is not equivalent to a fresh run or one longer call: sampled
  convergence and each call's final-search skip still apply. Evaluation failure
  can leave partial updates. No universal convergence guarantee or performance
  speedup is claimed; only the copied/retained pivot-map history is eliminated.
