# SimpleTT shared cache keys (#627)

## Decisions

- Reuse core's bit-packed `FlatIndexer` / `IndexKey` for batch deduplication
  and persistent left/right environments; remove the private mixed-radix copy
  and its 1024-bit limit. Keep public evaluation signatures and contraction
  order unchanged.
- Use the existing workspace `rustc-hash` dependency and the entry API for
  integer-key batch maps. No result depends on map iteration order.
- Count the three temporary split candidates using borrowed slices and reuse
  the two scratch sets. These sets contain no cached evaluations or owned
  index vectors; checked integer encoding is retained for the selected split
  and persistent environments. Encoding all discarded candidates showed an
  automatic-split regression in the first diagnostic experiment.
- Construct persistent indexers lazily per visited prefix/suffix length. This
  preserves the infallible constructor and valid partial reads around a
  zero-dimensional site. Metadata is reused across entries and cache clears;
  touching every length retains O(n²) dimension/offset metadata. Memory was
  not measured, so fewer retained bytes are not claimed for sparse workloads.
- Validate partial reads once before recursion; core's checked encoding still
  validates each key. Leave grouped BLAS3 misses and cache admission/statistics
  for the remaining #627 work.
- Fix zero-width singleton fields in core by assigning them offset zero.
  Their validated value is always zero, so this preserves keys while avoiding
  shifts by the storage width and writes past a final limb. Both fixed-width
  and dynamic placement paths were checked; a shared regression covers them.

## Verification conclusions and constraints

- The >1024-bit consumer regression fails on the original private indexer and
  passes with shared keys. Mixed dimensions, zero-width fields, partial
  lengths, boundary splits, warm reuse, and clearing preserve known values
  and errors. The core singleton regression independently reproduced a shift
  overflow before the offset fix.
- Local suites run: `cargo nextest run -p tensor4all-core -p
  tensor4all-simplett`, `cargo nextest run -p tensor4all-treetn -E
  'test(cached_evaluator)'`, and `cargo test --doc -p tensor4all-core -p
  tensor4all-simplett`. Hosted CI on the pushed head is the record of results.
- Removed code paths were reviewed for coverage impact. Private-indexer unit
  tests were replaced by cache-level wide-key/value regressions; core retains
  the encoder's width, injectivity, and boundary tests at its owner.
- [Complete performance evidence](../../benchmarks/results/2026-11-05-simplett-index-keys/README.md)
  retains two complete paired experiments, all per-cell confidence intervals,
  raw samples and failed noise observations. Both experiments are
  **inconclusive**; a fixed 120-second idle check also failed, so a third timed
  suite was not started. The final observed batch ratios meet the timing
  thresholds, but no performance win is certified. A quiet-host rerun remains
  required for performance signoff; the implementation stays local for review.
- No memory, large-rank or multithread scaling claim is made. Prefix metadata
  can grow quadratically, and the persistent caches remain unbounded under
  the existing policy pending the separate admission/statistics work.
