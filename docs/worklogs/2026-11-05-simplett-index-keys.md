# SimpleTT shared cache keys (#627)

## Decisions

- Reuse core's bit-packed `FlatIndexer` / `IndexKey` for batch deduplication
  and persistent left/right environments; remove the private mixed-radix copy
  and its 1024-bit limit. Keep public evaluation signatures and contraction
  order unchanged.
- Use the existing workspace `rustc-hash` dependency and the entry API for
  integer-key batch maps. No result depends on map iteration order.
- Construct persistent indexers lazily per visited prefix/suffix length. This
  preserves the infallible constructor and valid partial reads around a
  zero-dimensional site. Metadata is reused across entries and cache clears;
  touching every length retains O(n²) dimension/offset metadata, a tradeoff
  to assess alongside the timing results for this issue's proposed design.
- Validate partial reads once before recursion; core's checked encoding still
  validates each key. Leave grouped BLAS3 misses and cache admission/statistics
  for the remaining #627 work.

## Verification conclusions and constraints

- The >1024-bit regression fails on the original private indexer and passes
  with shared keys. Mixed dimensions, zero-width fields, partial lengths,
  boundary splits, warm reuse, and clearing preserve known values/errors.
- The changed crate passes 252 Nextest tests and 50 doctests. Crate-only strict
  Clippy passes; dependency linting encounters existing errors in core.
- Performance evidence is pending the paired comparison of retained original,
  shared-key-only, and persistent-key binaries on this Mac. CPU affinity is
  unavailable, so local results are diagnostic rather than isolated-host
  certification.
