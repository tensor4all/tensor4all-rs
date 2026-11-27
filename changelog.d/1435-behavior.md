- Tree ACI also expands candidates across a non-unit branching vertex when
  at least two committed pivots agree at its local coordinate (#1435).
  The degree must be at least 3; the existing incoming-rank product slack
  and largest-rank neighbour choice are unchanged. Every local value remains
  in the candidate grid, and non-unit chain sites keep their previous rule.
  This unlocks the binary delta junction in #1435 case B without evaluating
  retained pivots again or replaying earlier matrices.
