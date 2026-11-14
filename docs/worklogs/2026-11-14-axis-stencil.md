# Axis-wise patched QTT validation

## Decisions

- Store the tensor-product validation stencil as one 1D matrix and contract
  successive axes on the supplied engine. Column-major rotations restore the
  existing axis-zero-fastest test order after all axes are processed.
- Remove the dense-stencil resource option outright. The remaining checked
  shape and byte products cover node/test batches, intermediate validation
  values and interpolation cores.
- Keep the original dense implementation only in small real/complex reference
  tests. It checks both individual predictions and maximum residuals against
  floating-point accumulation error.

## Verification conclusions and constraints

- Validation results and lane determinism are recorded in the PR verification.
- The algorithm avoids the dense Kronecker stencil; interpolation cores still
  scale with polynomial order and dimension. This does not remove their cost.
