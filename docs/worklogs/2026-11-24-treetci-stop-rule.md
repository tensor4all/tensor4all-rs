# TreeTCI stop rule: per-bond default, sampled opt-in (#1405)

## Decisions

- `StopRule { PerBond, Sampled }` lives in `TreeTciOptions` next to `tolerance`; `PerBond` is the default. The contract is in [treetci-termination.md](../design/treetci-termination.md#stop-rules).
- The rule enters the optimizer at one `match`, where the search residual is or is not folded into the recorded error; nothing else feeds the search residual into the stop.
- The search computes, in one binding, the acceptance floor of the ordinary walks and `fallback_threshold`, the one acceptance threshold of every fallback stage (probes, restarts and the stage-3 enumeration of #1433, rebased onto `8c277c48`). The walks, their early stop on `abs_tol`, the enumeration, and the preference for discoveries above the margin are unchanged on both rules.
- The public `find_global_pivots*` signatures are unchanged and keep the `Sampled` acceptance; a rule argument would collide with #1433's edits in the same file.
- Existing goldens and regressions pin `Sampled` explicitly; `PerBond` has its own goldens. The #1352 regressions and the #1433 box tests run under both rules.
- `TreeTciInterpolator` always runs `Sampled` (review of #1446): the adaptive drivers accept a patch on its `error_estimate` without verification, and only `Sampled` folds the measured search residuals into it. So `partitionedtreetn` behaves as on `main`, and its M2 golden stays the check of that; the separate `PerBond` M2 golden was removed.

## Verification conclusions and constraints

New `PerBond` goldens compared with `501ebb4a` (same fixtures, default options otherwise):

| Golden | PerBond | 501ebb4a |
| --- | --- | --- |
| Chain full run (16 sites, seed 1) | ranks [4, 8, 8, 8], `Converged`, 12720 points | identical, errors to the bit |
| Branched full run (13 sites) | ranks [8, 16, 16, 16], errors 0, 161713 points | identical |
| Capped sparse chain (cap 8) | [8, 8, 8], `MaxBondDimension`, errors 0.998, 10299 points | identical |
| Capped star tree (cap 2) | [2, 2, 2], `MaxBondDimension`, errors 0.259, 1374 points | identical |

- Under `Sampled` the chain takes six iterations and 22073 points (1.74x).
- Before the interpolator was pinned to `Sampled`, the two M2 TreeTCI scenarios under `PerBond` matched `501ebb4a` in splits, evaluations, terminations, bond dimensions and node values. Observed, not isolated: their cache hits (14717 / 14467 against 10196 / 10081; 13917 / 13667 before the rebase onto #1433), and engine error estimates that differ from `501ebb4a` by at most 2e-16 against an absolute tolerance of 1.3e-8 (all calls stay outside the 10x screen). Hypothesis, unverified: the unit-endpoint probes and restarts, kept on `PerBond`, re-read memoized points; the same effect was reported when the M2 goldens were re-recorded on top of #1356 (`2bf0bdc8`). The rebase onto #1433 adds 800 hits to each scenario, the same increase main's re-pin shows under `Sampled` (`96149dbe`). Function evaluations are identical, so the oracle cost is not affected.
- `Sampled` reproduces `main` (`050eb4ba`, then `8c277c48`, `3f592dd6` and `31408d96` after the rebases onto #1433, #1444 and #1445): the full-run goldens (ranks, errors, evaluated points), the optimizer and search regressions, and `adaptive_m2.json`. Re-recording that file on the same machine at `main` and on this branch under `Sampled` gives byte-identical output, at each of the four. Both differ from the committed file in the last digit of 158 floats, all in the dense-engine scenarios, which do not use TreeTCI; the TreeTCI scenarios are byte-identical to the committed file.
- Large-target acceptance (Gaussian chain of #1371, 60-site comb of #1372, #1352 table, the R = 7 box of #1433) is reported in the PR, not here.
