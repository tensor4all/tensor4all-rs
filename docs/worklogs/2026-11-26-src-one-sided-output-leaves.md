# SRC one-sided output leaves (#1472)

## Decisions

- Keep intentional disconnected products in the SRC factor-join abstraction.
  Core's connected-network requirement is useful validation and stays intact.
- Plan connected components from full index values and contraction
  compatibility. Retained batch axes connect their factors before contraction,
  so no outer product can mix probe columns. Contract independent components
  before joining their results, avoiding premature local intermediates.
- Use the same join for batched probes and ordinary projections. Fixing only
  probe assembly exposes the corresponding disconnected leaf projection in
  both rooted-tree and chain schedules. Arithmetic stays in tensor operations
  without whole-network materialization, host conversion, or backend creation.

## Verification conclusions and constraints

- Regression tests reproduce the disconnected-network failure before the fix.
  They cover either operand carrying outputs, real and complex data, nonunit
  bonds, fixed and adaptive SRC, stars, endpoint-rooted chains, and
  interior-rooted chains, comparing each complete result to a dense reference.
- Local probe tests compare a width-three batch starting at column one with
  independent columns. Additional checks cover multiple disconnected
  components, scalar factors, same-ID primed legs, invalid inputs, and reverse
  gradients against an analytic oracle.
- All 1,209 ordinary TreeTN tests pass (five existing tests remain skipped),
  including connected SRC and both-site-free regressions. TreeTN Clippy,
  formatting, the deterministic repository-rules review, and the implicit-engine
  ratchet also pass. The root README's public-surface claims remain accurate.
- Verification is on CPU; no CUDA or performance claim is made.
