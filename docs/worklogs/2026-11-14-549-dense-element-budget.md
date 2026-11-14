# Shared dense element budget for GSE and the quantics 4^n embedding

## Decisions

- One shared `tensor4all_core::DenseElementBudget` in element counts, with
  `DEFAULT_MAX_DENSE_ELEMENTS = 1 << 24`, instead of a per-call limit. Both guarded
  sites check it **before** the first dense allocation: TreeTN GSE against
  `q_dim × q_dim` (after the existing checked `q_dim` product, before
  `build_reference_density`), and the multivariable quantics embedding against the
  summed per-site `left × 4^nvariables × right` shapes (after
  `checked_multivar_dims`, before the site tensors exist).
- A defaulted budget rather than a mandatory one: a mandatory `Option` would have
  broken every current GSE and `*_operator_multivar` caller and doctest, and an
  unbounded default would have added no guard. The owner chose the defaulted
  variant on issue #549, which also fixes the value.
- The three `*_operator_multivar` constructors take the budget as a trailing
  parameter instead of gaining `_with_budget` twins: early development allows the
  breaking change, and one constructor per operation stays the only path.
- Failure is a typed variant on the existing crate error enums
  (`GseError::DenseBudget`, `QuanticsTransformError::DenseBudget`) wrapping
  `DenseBudgetError`, so the element count and the limit reach the caller. No new
  `anyhow::Result` on a public function.
- Ordering is deliberate: `checked_multivar_dims` and `product_dim` still run
  first, so overflow inputs keep reporting their existing errors and only the
  over-budget case is new.
- Out of scope, per the same owner decision: the TCI2 `Full` pivot candidate budget,
  migrating `ContractionOptions::dense_reference_limit` to the shared type, a
  structured identity/copy representation for the 4^n embedding, and the extra
  retained quantics copies measured in the issue.

## Verification conclusions and constraints

- The guard rejects before allocating: the regression fixtures fail with the budget
  error without the `q_dim²` / `4^n` object being built, and the accepted cases
  (exactly at the limit) keep the previous numerical results.
- A shape that cannot be allocated at all still reports the structural byte-length
  error; the budget is a policy limit layered on top of the existing checks, not a
  replacement for them.
- Element-count semantics are fixed by unit tests: equality passes, a zero extent is
  empty even when the other extents overflow, a scalar is one element, and an
  overflowing product or sum is an error rather than a saturated count.
- `DEFAULT_MAX_DENSE_ELEMENTS` admits the measured `q_dim = 4096` GSE case exactly
  (`q_dim² = 1 << 24`) and rejects `q_dim = 4097`; it admits `r · 4^n` up to the
  measured ~1.5 GiB quantics configuration. The budget bounds one dense object or
  one object group, not the process peak: the issue's measurements show a peak of
  roughly 3–10 × one `q_dim²` object for GSE, so a caller with a hard memory bound
  must set the budget below that factor.
- Local verification: focused crate tests, changed-crate doctests, rustdoc, Clippy
  with the repository's lint set, and the deterministic repository-rules review
  pass. Not run here: the workspace-wide suite and the compiler-backed panic audit
  (that audit builds every workspace target and this host has no HDF5 development
  library). Hosted CI remains authoritative.
- Independent plan review: `ollama-cloud/glm-5.3-flash:high` returned
  `VERDICT: APPROVE` (no blockers, no important findings; four minor findings were
  addressed in `IMPLEMENTATION-PLAN.md`).
- Independent post-implementation review (`independent-review` skill) by
  `openai/gpt-6.1-sol` on the finished diff returned `VERDICT: CHANGES` with two
  important findings, both fixed: an unallocatable site shape keeps its structural
  byte-length `Operation` error instead of the budget error, and the exact-limit case is
  covered by a numerical test (the derived element count reproduces the same operator,
  and one element less is rejected). The two minor findings (a phase-rotation `# Errors`
  sentence and the guide's site-shape wording) were corrected in the same change.
