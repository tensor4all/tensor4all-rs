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

## Verification conclusions and constraints

- Test-only references retain the previous residual calculation for bitwise
  comparison and a paired release experiment. See the
  [predeclared experiment](../experiments/2026-11-14-compression-check/protocol.md).
- Cache memory is bounded by the validation batch and remaining sites and is
  released after each compression check. It trades temporary prefix storage
  for repeated contractions.
