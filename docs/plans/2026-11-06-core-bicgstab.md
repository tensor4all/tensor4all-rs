# Core BiCGSTAB implementation plan

> Execute inline with `superpowers:executing-plans` and test-driven development.
> The maintainer makes every commit. Leave these changes uncommitted.

**Goal:** Add matrix-free BiCGSTAB for real nonsymmetric and complex
non-Hermitian systems in `tensor4all-core`.

**Architecture:** Use `TensorVectorSpace` and `AnyScalar` for a scaled short
recurrence, with true-residual checks at both convergence points and at budget
exhaustion. Reuse CG's scaling primitives privately; preserve its behavior.

**Tech stack:** Existing Rust core abstractions; pinned KrylovKit reference.

**Spec:** `docs/design/core-krylov-bicgstab.md` (approved in this chat).

## Global constraints

- `max_iter = 100`, `rtol = 1e-10`, `atol = 0` by default.
- Result residual is absolute; convergence uses `max(atol, rtol * ||b||)`.
- Preserve complex conjugation, typed failures and operation source chains.
- Every cycle and recurrence restart consumes the original cycle budget.
- No dependency additions or existing test-tolerance changes.
- Temporary CG dependency is a fast-forward of the branch created from
  fetched `origin/main`; synchronize after CG lands before an independent PR.

## Review focus

- Intermediate convergence must precede stabilization denominators.
- Residual replacement must restart the shadow residual and direction together.
- Complex recurrence coefficients must retain their imaginary parts.
- Rescaling must preserve tiny/large physical residuals and update units.
- Error and exhausted-budget results must never claim estimated convergence.

## Task 1: Public solver and ordinary systems

**Files:** `src/krylov/bicgstab.rs`, `src/krylov.rs`,
`src/krylov/tests/bicgstab.rs`, `src/krylov/tests/mod.rs` under the core crate.

**Interfaces:** Produce `bicgstab`, `BiCgStabOptions`, `BiCgStabResult<T>`;
consume `TensorVectorSpace` and `KrylovError`.

- [x] Write known-solution real and complex nonsymmetric tests and identity
  intermediate-step test; run and confirm failure from missing public API.
- [x] Implement the bounded recurrence with intermediate/full true-residual
  checks and typed breakdowns; run the tests to green.
- [x] Add initial-guess, tolerance, budget, callback, index and breakdown cases.

## Task 2: Scaled recurrence and residual replacement

**Files:** `src/krylov/scaling.rs`, `src/krylov/cg.rs`, BiCGSTAB and its tests.

**Interfaces:** Consume private power-of-two vector/scalar helpers extracted
from CG; no new public helpers.

- [x] Write scale regressions for all four scalar types, including smallest
  subnormals, huge/tiny operators and nondiagonal systems; observe failures.
- [x] Track residual and operator-image scales through the BiCGSTAB recurrence.
  Keep coefficients in `AnyScalar`, with scalar extraction only for decisions.
- [x] Add false-convergence/restart and block-vector regressions. Run all new
  tests and unchanged CG tests, including optimized numerical checks.

## Task 3: Documentation and final validation

**Files:** core/module/root README, provenance, design index, bundled skill
capability reference; design and this plan.

**Interfaces:** Consume the completed public API and its exact diagnostics.

- [x] Add runnable asserted rustdoc, update capabilities and source/license
  attribution; cross-check representative fixtures against pinned KrylovKit.
- [x] Run core Nextest, core doctests, relevant optimized regressions, Clippy,
  rustdoc, fmt, diff checks, API inventory, public-error docs and deterministic
  repository-rules checks. Record unavailable checks and baseline findings.
- [x] Review the whole change against the spec and other solvers. Leave all
  edits uncommitted and report the branch, worktree and material limitations.

## Initial validation outcome

- Core Nextest: 1,170 passed; three existing skips.
- Core doctests: 401 passed. Optimized Krylov regressions: 103 passed.
- All 21 BiCGSTAB regressions pass, including all four scalar types,
  subnormal/large norms, scaled nonsymmetric operators, four-dimensional
  recurrences, block vectors, breakdowns, and true-residual replacement.
- The roundoff regression fails when either intermediate true-residual checking
  or full-step replacement is removed. The restored source passes debug and
  optimized checks. The upstream reference passes all four scalar fixtures.
- Formatting, diff checks, public-error docs, API inventory completeness,
  local documentation links, and deterministic repository rules pass.
- Strict Clippy has 43 existing `nonminimal_bool` findings, all in files
  unchanged by this branch; excluding that one lint passes. Rustdoc succeeds
  with the same three existing link warnings. `mdbook` is unavailable locally.
- Final integration review was performed directly, following the shared rule
  against unrequested independent AI reviews. The short recurrence, complex
  coefficients, scale units, error paths, moved CG helpers, and public docs
  were checked against the approved design. No implementation findings remain.
- The original CG checkout is clean. All new changes remain uncommitted in
  `feat-core-bicgstab`; no push or PR was made for this feature. Its CG
  dependency still needs to land before an independent PR against main.

## Review corrections

- [x] Add failing regressions for direction-coefficient range loss, shared
  complex division, and callback errors entering solver recovery.
- [x] Implement combined scaled coefficient arithmetic with gradient coverage.
- [x] Fix shared native complex division; cover subnormal, huge, mixed-scale,
  pure-real/imaginary, and exceptional operands and verify the traced route.
- [x] Preserve callback error origin and source chains before recovery.
- [x] Run full core/backend suites (1,495 passed, six existing skips), affected
  TreeTN tests (109 passed), optimized Krylov tests (108 passed), and optimized
  scalar regressions (two passed).
- [x] Complete doctests, Clippy, rustdoc, API inventory, documentation consistency,
  and final integration review. Leave all changes uncommitted.

Review corrections also pass 621 core/backend doctests, API inventory,
formatting, public-error documentation and deterministic rules checks. Strict
Clippy reports the same 43 existing `nonminimal_bool` findings in unchanged
files; excluding only that lint passes. Rustdoc succeeds with four existing
link warnings across the two crates. Local mdBook testing remains unavailable.
No test tolerances or lint configuration were changed. The three review
reproductions now succeed or preserve the requested callback error, and direct
integration review found no remaining findings in the fixes.
