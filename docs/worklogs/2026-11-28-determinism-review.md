# Default-planner determinism revalidation

## Decisions

- Separate the historical omeco tie defect from current evidence. The pinned
  default planner contains tenferro-rs#2014; no current failure is attributed
  to the closed cause.
- Restore the measurement, DenseEngine and TreeTCI generic-path fresh-thread
  regressions instead of retaining skips with obsolete reasons.
- Keep the accepted generic-path M7 steps (c)/(a): raw messages remove
  generic overhead, and positional contractions remove planner dependence.
  Passing fixture tests does not change that maintainer decision.
- Limit the Fourier call-chain argument to level-zero stored-patch zip-up.
  General product terms use partial contraction and need their own argument.

## Verification conclusions and constraints

- The restored regressions pass at the pinned default planner on their `f64`
  fixtures, with fresh threads and rebuilt index identities. This establishes
  those cases, not all generic layouts, dtypes or pooled evaluator histories.
- Cross-process values have no two-process regression here. #1497's Fourier
  cross-lane result and N-ary operand-sequence test also remains outstanding;
  #1528's interpolation tests do not establish it.
- No numerical tolerance or production algorithm is changed.
