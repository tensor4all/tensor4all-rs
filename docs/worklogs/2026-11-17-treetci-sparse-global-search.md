# TreeTCI global confirmation on sparse branches

## Decisions

- Keep this separate from #1356's exhausted-candidate diagnostic. Simon's
  [three-branch reproducer](https://github.com/tensor4all/tensor4all-rs/issues/1352#issuecomment-6939597508)
  has non-exhausted, rank-deficient candidates and a measured zero residual.
- When exploratory global searches find no pivot, verify retained tail pivot
  pairs at dimension-one endpoints, then restart on their observed subtree
  support while randomizing the complementary subtree. This exposes
  inconsistencies between neighboring updates and avoids starts confined to
  a zero-valued branch. The requested `nsearch` bounds each extra family of
  walks; tree size does not multiply that budget.
- When no exploratory candidate crosses the discovery margin, retain already
  observed tolerance violations rather than repeatedly reporting them without
  a new pivot. Confirmation uses the requested absolute tolerance, not the discovery
  margin. Otherwise the 14-term variants can retain an error between one and
  ten times tolerance without triggering a new pivot.
- Preserve the largest residual actually observed during global search and
  include it in recorded errors, observers, and convergence decisions even
  when no pivot crosses the discovery margin. These remain sampled errors,
  not a global error bound. Rank-cap precedence and final-iteration search
  skipping remain unchanged.
- Retain only per-search metadata and scratch; no pivot history or full-domain
  production materialization is added. Start construction is bounded by
  `nsearch * n_sites`, and existing cached, batched, session-aware readout is
  reused. Caller-owned RNGs drive complementary-subtree restarts directly.

## Verification conclusions and constraints

- The original seed-1 program on unmodified `33d75503` reproduces the reported
  ranks 11/1/10 and a dense relative error of 1.4902070749e-5 with a recorded
  zero. The frozen matrix fixtures preserve the original f64 bits without a
  test dependency on `StdRng`.
- On that run, the final 15 exploratory walks all stayed on `b != 0` zero
  fibers. A retained pivot pair on `b = 0` had residual 6.4361066202e-8, and
  the cached support-slice maximum matched the dense residual. Re-evaluating
  the final candidate matrices confirmed their lower sampled rank and zero
  rrLU residual: the evidence points to missed global information, not an
  incorrect matrix residual or cached readout.
- Regressions cover both 12-term and 14-term targets, failing and exact seeds,
  and real/complex scalars, with one dense whole-result comparison per run.
  The original pointwise readout is unsuitable for ordinary test loops; the
  dense reference preserves the same target and makes that check practical.
- Constant-rank and exact cases must still converge. The proposed search
  does not certify every unseen direction of an arbitrary black-box target.
