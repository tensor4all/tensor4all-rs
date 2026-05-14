# Batch Issues 464 and 475 Design

## Goal

Address the approved batch prefix for issues #464 and #475 on one branch:
verify the TreeTN cached evaluator surface from #464, and add a safe constraint
row normalization substrate for #475.

## Context

Issue #464 asks for an allocation-light batched TreeTN evaluator. Current
`origin/main` already exposes `TreeTNCachedEvaluator`, `CachedEvaluatorOptions`,
C API evaluator handles, tests, and benchmarks. The batch should verify that
surface instead of duplicating it.

Issue #475 asks for normalization of affine/halfspace constraint rows such as
`16*x <= 64` to the primitive row `x <= 4` before those rows drive quantics
operator construction. This is not the same operation as normalizing an affine
map `y = A*x + b`: changing `y = 16*x + 64` into `y = x + 4` would be a
behavioral regression. The safe Rust-side addition is a dedicated constraint
row normalization API that callers and future constraint-operator builders can
use before constructing affine/halfspace transform operators.

## Approach

1. Keep `affine_operator` semantics unchanged. It represents an affine map, not
   a scale-invariant constraint.
2. Add a small public constraint-row type in `tensor4all-quanticstransform`.
   It clears rational denominators, divides coefficients and RHS by a positive
   row gcd, and preserves the represented equality or inequality set under
   positive scaling.
3. Export the type from the crate root and document when to use it. The docs
   must explicitly say it is for constraints, not affine maps.
4. Verify #464 with focused TreeTN/C API evaluator tests. If those tests pass
   without new code, record #464 as already implemented on the base branch.

## Testing

The #475 implementation uses TDD:

- Add failing tests for integer normalization, rational denominator clearing
  plus gcd reduction, negative rows with positive gcd, zero rows, and the
  guarantee that `AffineParams::to_integer_scaled` is not constraint-normalized.
- Implement the minimal normalization helper.
- Run focused `tensor4all-quanticstransform` tests.

The #464 close-out uses focused evaluator tests:

- `cargo nextest run --release -p tensor4all-treetn cached_evaluator`
- `cargo nextest run --release -p tensor4all-capi treetn_evaluator`

No test tolerances or coverage thresholds are changed.
