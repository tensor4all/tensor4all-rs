# Changelog

## Unreleased — the default edge visitor alternates direction (behaviour change)

`AllEdges`, the default `EdgeVisitor`, now sweeps the sorted edges forward on
the first inner pass of an iteration and backward on the second (#899).
Before, both passes used the sorted order. `INNER_EDGE_PASSES` stays `2`, so an
iteration still updates every edge twice.

- **Added**: `EdgeVisitor::visit_order_in_pass(state, pass)`, which the
  optimizer now calls for each inner pass. Its default returns `visit_order`,
  so custom visitors that implement only `visit_order` behave as before.
- **Changed**: default runs follow a different trajectory, so the selected
  pivots, the iteration count, and the per-iteration ranks and errors can
  differ from earlier releases at the same options and seed. Tests that pin a
  trajectory must be re-recorded; the three in this workspace were, with the
  previous values kept in comments.
- **Effect**: on a comb (three binary chains of 10 sites joined by one site of
  local dimension 1; `tolerance = 1e-6`, global pivots on, seed 11), the
  number of distinct function evaluations to convergence falls from 61,458 to
  43,167, from 705,522 to 384,659, and from 2,880,748 to 2,329,457 for three
  targets of increasing rank (ratios 0.70, 0.55, 0.81). Final ranks (20, 48,
  80) and the sampled maximum relative error (1.4e-6 to 1.8e-6) are unchanged,
  and both orders end in `Converged`.
