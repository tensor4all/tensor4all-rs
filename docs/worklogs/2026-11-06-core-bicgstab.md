# Core BiCGSTAB

## Decisions

- Follow [the BiCGSTAB design](../design/core-krylov-bicgstab.md), with CG's
  options/result conventions and a pinned KrylovKit implementation as reference.
- Track power-of-two scales on the short recurrence rather than relying on
  physical inner products. This preserves tiny/large systems across all four
  scalar types while keeping complex coefficients in backend scalar arithmetic.
- Reuse private vector operations with CG. Its recurrence and public contract
  are unchanged. The current branch temporarily includes the pending CG PR;
  synchronize against `origin/main` after that dependency lands.
- Combine scalar multiplication and division with exponent normalization when
  an intermediate quotient would leave the safe range. This fixes small shadow
  projections without changing ordinary-range evaluation or dropping scalar
  gradient metadata.
- Fix finite complex division at the shared native scalar seam, carrying each
  component product's exponent separately. This preserves both extreme
  quotients and small components beside large ones. The traced division route
  already handles the reviewed scales and is covered without changing it.
- Wrap callback failures before numerical recovery. Nested solver breakdowns
  retain their typed source and propagate as operation failures, rather than
  being mistaken for recoverable BiCGSTAB arithmetic.

## Verification conclusions and constraints

- Focused tests cover real/complex nonsymmetric systems, intermediate and full
  convergence, all scalar types, extreme scales, block tensors, cycle budgets,
  source errors, and recurrence breakdowns. Natural roundoff fixtures exercise
  residual replacement at both convergence points.
- The pinned upstream source passes the same four scalar-type known-solution
  fixtures through a minimal dense-vector adapter; this is not a full upstream
  package integration test.
- The complete core/backend suites pass (1,495 tests, six existing skips), as
  do 109 affected TreeTN tests, 108 optimized Krylov tests, and two optimized
  complex-division tests. Removing either convergence verification path makes
  the roundoff regression fail.
- Review regressions cover real/complex small shadow products at `1e-200` and
  `1e-320`, subnormal and maximal complex quotients, mixed component scales,
  exceptional input classification, scalar gradients, and nested typed callback
  errors at initialization, operator application, and residual verification.
- Coverage impact of the moved helpers was reviewed: unchanged CG regressions
  and the new BiCGSTAB cases exercise their original and shared paths.
- Formatting, API inventory, public-error docs and deterministic rules pass.
  Strict Clippy retains 43 existing findings in unchanged files; the check
  excluding that lint passes. All 621 core/backend doctests pass. Rustdoc
  succeeds with three existing core link warnings and one existing backend
  link warning.
  Local mdBook testing is unavailable, and hosted CI for this uncommitted
  feature has not run.
- Direct integration review found no remaining implementation findings.
  No performance or differentiability claim is made. The maintainer makes all
  commits and approves later publication of this branch.
