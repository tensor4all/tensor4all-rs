# Issue #752: measured approximate operator application

Session: 2026-09-16 (fourth batch). Base: `a79900e7` (squash of #757). Branch:
`feat/752-approximate-application`. Closes the item "Approximate application with
its own retained error allowance" that #753 deferred when it removed the
caller-supplied apply options.

## What changed

- `MergeRefineOptions::apply_options: Option<ApplyOptions>` (default `None`). The
  schedule now always computes the exact local application of every input leaf and,
  when the caller supplies truncating options, also applies with those options and
  measures the per-leaf deviation `||exact - approximate||`.
- The measured application error enters each level-zero item's bound, so it flows
  through the existing ledger: restriction carries it unchanged, merges add it, and
  the report's `error_bound` covers it together with the compression residuals.
- The l1 allocation now charges the application error first: the compression share
  is `(allowance - application error) / (2^d * output_depth)`. A measured
  application error above the global allowance is rejected with repair guidance
  instead of returning a result that violates the pinned contract.
- `ReconstructionTarget::from_subset_operator` still applies exactly and now says
  so explicitly, because an immutable target has no place to carry an application
  error; that asymmetry is documented in the rustdoc, the design record, the
  README, the guide, and the skill reference.

## Verification

- 117 crate tests plus 51 doctests. The new regressions build a two-node
  `I ⊗ I + 0.1 · X ⊗ X` MPO whose dyeary leaf image is a rank-two superposition, so
  truncating the application to bond dimension one has a real, measurable effect:
  one test asserts the bound is non-trivial and covers the deviation from the exact
  trajectory, the other asserts a near-exact tolerance rejects the same truncation
  with an application-error message.
- Release Clippy `--all-targets --no-deps -- -D warnings`, `cargo fmt --check`,
  the runnable example, the mdBook snippet suite and book build, and the
  repository-rules dry-run review.

## Notes and limits

The Fourier operator turned out to be a poor test vehicle for this feature: its
image of a dyadic input leaf is already (numerically) rank one, so truncating the
application has no effect there. The regression therefore uses an operator whose
leaf images are genuinely rank two. This is also the honest scope statement: the
option matters for general operators, and for a unitary Fourier operator the exact
application is already cheap.

Requesting the option costs a second application per input leaf, because the exact
application is the measured reference. No performance claim is made, operator
construction error (for example `FourierOptions::tolerance`) is still outside this
bound, and nonuniform input trees, item dropping, and automatic padding remain
open.
