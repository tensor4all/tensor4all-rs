- `tensor4all-partitionedtreetn` finds overlapping patch projectors through a
  trie instead of pairwise loops (#1293). This covers partition validation,
  `append`, `add`, `Projector::are_disjoint`, and the overlap check and pair
  enumeration of `contract`. Results and pair order are unchanged. Two
  operands whose patches split on different indices, contracted into 16,384
  output patches, take 36 s instead of 106 s in `contract` (release, one
  thread); at about 100 patches nothing changes measurably.
