# Patched QTT follow-up cleanup

## Decisions

- Make `NInterval::split` own midpoint representability checks, so all refining
  constructors share the rule and retain axis-zero-fastest child ordering.
- Keep one compression sweep implementation and supply its factorization at the
  caller boundary. The explicit SVD route needs storage and backend linalg
  capabilities, without LU/CI pivoting. Unsupported session methods return a
  typed error before mutation or one-site shortcuts.
- Generate node and validation coordinates directly into one column-major
  buffer, through one shared node enumeration path.
- Precompute normalized 1D Lagrange values once and share them across axes.
  Use a call-local TTCache with a full left split to reuse prefixes without
  reassociating floating-point contractions.

- The initial matched experiment found a 1.326 time ratio, so the first cache
  integration was rejected on performance. The full-left batch boundary now
  uses a lexicographic traversal in simplett instead of a two-sided mapper and
  persistent per-prefix entries. Its temporary stack holds one vector per level;
  other TTCache splits retain their persistent-cache behavior.

## Verification conclusions and constraints

- Real/complex residual comparisons, shared multiscale values and lane
  determinism pass. The final matched release experiment has a geometric mean
  time ratio of 0.691, with all six cases faster and every residual bitwise equal.
  See the [complete results](../experiments/2026-11-14-compression-check/results.md)
  and [predeclared protocol](../experiments/2026-11-14-compression-check/protocol.md).
- Full crate tests, workspace doctests, affected doctests after the cache
  refinement, strict workspace Clippy, mdBook and the four required audits pass.
  Local execution covers Linux/native faer; Accelerate remains CI-owned.
- Cache memory is bounded by the validation batch and remaining sites and is
  released after each compression check. It trades temporary prefix storage
  for repeated contractions.
