- `tensor4all-quanticstci`: the components of one `quanticscrossinterpolate_multicomponent`
  call now share one memo of the callback's answers (#1514), so a point already evaluated
  for one component is not requested again for another. The callback evaluates
  `n_components * |union of the sampled points|` values instead of
  `n_components * sum of the per-component samples` (23.5x fewer at 8x8, 2.4x at 2x3);
  fits, ranks, errors and the random stream are unchanged. The callback must be a pure
  function of the point, independent of batching and call order: later components now
  send fewer or no batches. An error from one component now names it
  (`component k of n: ...`) with the same error variant.
