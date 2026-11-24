# Changelog

## Unreleased — adaptive local threshold under the sampled stop rule (behaviour change)

Under `StopRule::Sampled`, the local rrLU truncation threshold of each edge
update now starts at the absolute tolerance and is halved after each iteration
whose recorded error is not below `tolerance`, down to `1 / E` of it for E tree
edges (#1371). The search acceptance and the stop tolerance are unchanged.

- **Added**: `TreeTciTermination::LocalThresholdFloor`, reported under
  `Sampled` when three trailing sweeps at the `1 / E` floor all fail the
  tolerance. It is armed only for trees with at least two edges: with one
  edge or none the floor would be `1.0`, the threshold cannot descend, and
  the run ends `MaxIterations` or `Converged` as before.
- **Breaking**: `TreeTciOptimizationResult` and `TreeTciRunResult` have a new
  public field `local_threshold_factor`, the factor the last sweep applied
  (`1.0` under `PerBond` and for trees with fewer than two edges). Struct
  literals and destructuring patterns without `..` must add it.
- **Changed**: `Sampled` runs whose recorded error fails the tolerance in some
  iteration follow a different trajectory, so `Sampled` is no longer exactly
  the behaviour before #1405. On the full-run chain golden it now takes 13754
  evaluations, 1.08x the per-bond stop (1.74x in the entry below). This
  includes `TreeTciInterpolator`, which always runs `Sampled`.
- **Not changed**: under the default `StopRule::PerBond` the factor stays
  exactly `1.0`, `LocalThresholdFloor` is never reported, and runs are the
  previous ones. The pinned `PerBond` runs match in termination, ranks and
  points exactly and in the recorded errors within 1e-9 relative (bit-identical
  in local measurements on Linux x86_64).

## Unreleased — per-bond stop rule as default (behaviour change)

`TreeTciOptions::stop_rule` selects how the optimizer records its error and
decides `Converged` (#1405). The default is `StopRule::PerBond` again, the rule
of TensorCrossInterpolation.jl, TreeTCI.jl and xfac: the recorded error is the
largest bond pivot error, and the global search accepts a pivot only above
`tolerance * tol_margin_global_search`, also in its fallback stages.
`StopRule::Sampled` is the previous behaviour (#1357): the recorded error also
includes the residuals the global search measured.

- **Changed**: with the global search enabled, default runs can stop earlier
  and record different `errors` than the previous release at the same options
  and seed. To keep the previous behaviour, set `StopRule::Sampled` (Python:
  `stop_rule="sampled"`).
- **Not changed**: `TreeTciInterpolator` always runs `StopRule::Sampled`,
  because the adaptive drivers accept a patch on its error estimate; the
  public `find_global_pivots*` functions keep the sampled acceptance.
- **Effect**: on the 16-site quantics chain of the full-run golden the sampled
  stop takes 1.74x the evaluations of the per-bond stop (22073 against 12720).

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
