# Spectator-aware merge-refine scheduling (#1299)

## Decisions

- Keep one global schedule and error allowance. Identify output regions by both
  their selected-coordinate prefix and a disjoint spectator cell. Different
  spectator groups never merge with each other.
- Reuse mutually disjoint spectator constraints directly. When constraints
  overlap, refine only the varying spectator indices to a common coordinate
  layout. Check its size and the resulting leaf count against the caller's work
  and term limits before transforming or projecting any tensor. This bounded
  refinement is deliberately simpler than reconciliation of overlapping outputs.
- Apply the operator once per original patch, then restrict its image to the
  spectator cells. Charge each original application error once globally;
  restriction is nonexpansive. Later residuals at a level have disjoint selected
  and spectator supports, so their existing Euclidean accounting still applies.
- Validate complete selected-coordinate coverage separately in each represented
  spectator cell. Missing spectator space remains outside the input support;
  missing selected leaves require the explicit zero-for-missing contract.

## Verification conclusions and constraints

- Small dense DFT comparisons pass on chains and degree-three trees for real
  and complex inputs, including same-ID indices at different prime levels. An
  actual patched-elementwise producer with every site in its patch order feeds
  Fourier directly. Exact and adaptive results preserve topology and disjoint
  output supports; selected-output and explicit sparse coverage compose.
- A measured-application regression confirms that the same original error is
  charged once across spectator and output refinements. The reported bound
  covers the dense whole-result residual and stays inside the original allowance.
- The real-input regression exposed an existing exact zip-up identity-skip
  defect: untouched real spectator nodes bypassed complex scalar promotion.
  The fix belongs in the TreeTN contraction owner and now has a direct
  identity-skip regression. Related operator tests retain ordinary real behavior.
- The shared bounded grid helper also refuses overflowing element counts and
  vector layouts even with an unlimited caller budget; those paths have direct
  boundary coverage without allocating large tensors.
- Changed-crate suites and affected doctests, operator regressions, Clippy,
  public-error documentation, crate boundaries, panic audit and mdBook checks
  pass. Initial integration validation caught an incorrectly named test operator;
  the repaired test and both QFT targets pass. No test tolerance was relaxed.
- Integration with #1329 retains projector-trie overlap validation within each
  spectator cell. Original spectator constraints in that cell are mutually
  compatible, so projector overlap is equivalent to selected-prefix overlap.
- Root README claims remain accurate. Removed error-variant paths were reviewed
  for coverage impact: strict-addition error coverage remains, and the obsolete
  rejection has been replaced by acceptance, per-cell coverage and resource-limit
  regressions. No speedup, new branching-selection support or maintainer
  acceptance is claimed.
- Overlapping constraints can multiply the work items. The bounded common grid
  can refuse a layout that a more elaborate adaptive refinement could handle;
  callers can raise the limits or use a partition restricted to selected indices.
