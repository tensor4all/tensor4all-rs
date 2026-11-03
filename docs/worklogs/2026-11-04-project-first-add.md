# Project-first `add_with_patching` (#1131)

## Decisions

- `add_with_patching` now follows the same project-first rule as
  `contract_adaptive`: the budget-truncated form of a patch is only a probe for
  the split decision, and the children of a split are projections of the exact
  addend patch. The cutoff budget and `max_bond_dim` are spent once, by the
  closing `truncate_adaptive` pass, instead of once per split level.
- The pre-split truncation is kept as the probe rather than dropped: it decides
  whether a patch is over the cap and (with `ExactParameterGain`) which site
  splits cheapest. Probing without the cap is deliberate; a capped probe cannot
  distinguish an exact rank from a truncated one.
- A patch at or below its assigned budget is not dropped at the decision step
  any more: it stays in the result set so the closing pass still derives every
  other leaf's budget from the whole input weight, and drops the patch itself.
  Dropping it earlier made the survivors' thresholds grow (for two equal-volume
  patches with squared spectra `[93, 7]` and `[1, 0]` at `cutoff = 0.1`, the
  heavy patch's budget became 10 instead of 5.05 and its weight-7 mode was
  discarded).
- Split children are built with the same `project_if_present` helper the
  contraction path uses, so projection-created null bond space is compressed at
  the `64 * epsilon` numerical-rank threshold instead of being charged to the
  caller's budget.
- The test-only `add_project_first` prototype is gone; its own test now drives
  the public routine, so there is one implementation of the rule.
- The budget keeps its meaning: `cutoff` is still a fraction of the input
  `||F||^2` under `Volume` and a per-factorization relative cutoff under `Norm`,
  and the closing pass re-derives the same local thresholds from the
  input-referenced totals (splitting preserves both the total volume and the
  total squared norm of the exact leaves).

## Verification conclusions and constraints

- Issue reproducer (6-site chains, site dimension 2, two addends, center 2),
  relative squared error `||F - sum_p R_p||^2 / ||F||^2`, release build, one
  core, worst cases of the 144-run grid:

  | allocation | cutoff | cap | r | decay | before (`56c25d4a`) | after | ratio |
  |---|---|---|---|---|---|---|---|
  | `Volume` | 1e-3 | 1 | 3 | 0.15 | 1.3056e-3 | 3.2951e-5 | 39.6x |
  | `Norm` | 1e-2 | 1 | 3 | 0.3 | 1.1479e-2 | 9.1650e-5 | 125x |

  Both cases keep their patch count (30 leaves).
- Cost of the exact carried patches: the 144-run grid takes 4.6-4.7 s before
  and 5.7 s after (1.22x), the `Volume` worst case 0.042 s before and 0.061 s
  after (1.44x), and the grid's total leaf count rises from 1499 to 1541
  (+2.8%). No patch of this grid is dropped before the closing pass, so these
  numbers come from the split rule alone. The added work is the projection of exact patches at every
  split level plus their round-off compression; no split is pruned by an
  already-truncated parent any more.
- In every run where the exact leaf projections fit the cap (25 of 144 after,
  32 before), the result is exact to round-off (`< 1e-12`); before, all 32 such
  runs were lossy (up to 1.17e-2). This is the property the regression tests
  pin.
- Mutation checks: (a) splitting the probe instead of the exact patch restores
  exactly the pre-fix numbers (1.3056e-3, 1.1479e-2, 7.7217e-5) and turns all
  three issue-reproducer tests red; (b) dropping a below-budget patch at the
  decision step again turns the dropped-patch test red (surviving weight 93
  instead of 100).
- Not measured: trees with more than one branch, dtypes other than `f64`, and
  the `Nary`/`NaryWithin` sum engines; the change is independent of the engine
  that builds the exact leaves.
