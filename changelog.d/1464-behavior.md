- `tensor4all-treeaci`: the unit-vertex expansion of #1435 no longer makes a commit
  fail on a shape limit it fits without the expansion (#1464). An expanded bond is
  charged against the limit of the side it enlarges (`max_candidate_rows` or
  `max_candidate_cols`, counted over the whole enlarged side) and against the
  local-matrix element ceiling, and is skipped for that commit when it would
  exceed one; before, the commit returned `TreeAciError::ResourceLimit`. With the
  default limits no measured run reaches them: the #1435 table, the same table
  without the unit vertex and the G0 and star workload sweep (213 runs) are
  bit-identical. The working-byte, frame-byte and sample-arena budgets are still
  hard errors for an expanded commit. A failed proposal that undoes an expansion
  now also drops the memoized candidate frames over the removed samples, whose ids
  the arena hands out again.
