# Tutorial Review — April 30, 2026

Review scope: the live mdBook tutorial pages under `docs/book/src/tutorials/`,
the runnable sources under `docs/tutorial-code/`, the tutorial planning notes,
and the repository rules that keep online tutorials and code from drifting.

## Executive Summary

The tutorial section is wired into the online mdBook and the basic progression is
sound: scalar QTT, physical intervals, integrals, bit-depth sweeps,
multivariate functions, then QTT operations. The local validation checks pass.

The remaining issues are mostly documentation-quality and maintenance-process
issues rather than compile failures. The largest content risk is the 2D partial
Fourier tutorial: the live code block is runnable, but it does not show the
operator node remapping / interleaved-state expansion that the prose and the
runnable tutorial source describe.

## Verification Performed

- `./scripts/test-mdbook.sh` passed.
- `cargo test --release -p tensor4all-tutorial-code` passed: 14 library unit
  tests, 1 binary unit test, and 26 integration tests.
- No `ignore` or `no_run` Rust tutorial blocks were found in
  `docs/book/src/tutorials`.
- `CI_rs.yml` runs `./scripts/test-mdbook.sh` on pull requests, and
  `deploy-docs.yml` runs the same check before publishing GitHub Pages from
  `main`.

## Priority Findings

### P0 — None Found

No live tutorial currently fails the mdBook doctest workflow, and the
`tensor4all-tutorial-code` crate passes its release-mode tests.

### P1 — 2D Partial Fourier Code Block Does Not Match The Described Tutorial

File: `docs/book/src/tutorials/computations-with-qtt/partial-fourier2d.md`

The prose says the tutorial builds an interleaved 2D QTT and maps a 1D Fourier
operator onto x-sites at positions `0, 2, 4, ...`. The runnable tutorial source
does that via `build_input_grid`, `build_partial_fourier_operator`,
`rename_operator_nodes`, `expand_operator_to_interleaved_state`, and
`transform_x_dimension`.

The live mdBook block instead:

- builds `sizes = vec![2usize; bits * 2]`, which reads as a flat 8-variable
  binary tensor rather than a two-variable `DiscretizedGrid`;
- uses a constant callback `_idx -> 1.0`, so it does not show `f(x, t)`;
- builds `quantics_fourier_operator(bits, ...)` but does not show node mapping
  onto the even interleaved x-sites;
- calls `operator.align_to_state(&state_tn)` directly, which hides the actual
  expansion required to leave t-sites untouched.

Recommended fix: rewrite the mdBook block so it either shows the real public
workflow used by `docs/tutorial-code/src/qtt_partial_fourier2d_common.rs`, or
explicitly presents a smaller helper-backed version where the helper is named
and linked. As written, the snippet passes tests but teaches the wrong shape of
the partial-transform algorithm.

### P1 — Several Tutorial Assertions Are Too Weak For Repository Rules

Repository rule: mdBook examples must include assertions verifying correctness,
not merely execution.

Weak assertions remain in:

- `qtt-definite-integrals.md`: `assert!(integral > 0.0)` should compare against
  the analytic value for `x^2` on `[-1, 2]` within tolerance.
- `elementwise-product.md`: `assert_eq!(product.node_count(), 3)` checks only
  structure, not that the product evaluates as `f_a(i) * f_b(i)`.
- `affine-transformation.md`: `assert_eq!(external.len(), 3)` checks only a
  shape fact. It should evaluate at a small point and compare with the affine
  pullback reference.
- `fourier-transform.md`: `assert!(result.node_count() > 0)` checks only that
  an output exists. It should compare at least one transformed coefficient, or
  check a norm/reference value with the documented scaling.
- `partial-fourier2d.md`: `assert!(result.node_count() > 0)` is not a
  correctness assertion for the partial transform.

Recommended fix: keep the blocks short, but make every assertion numerical or
semantic enough to catch a stale API or wrong operation.

### P2 — Tutorial Maintenance Policy Is Split And Partly Stale

Files:

- `docs/tutorial-code/README.md`
- `AGENTS.md`
- `REPOSITORY_RULES.md`

The rules do say that mdBook snippets and examples must not drift from the
public API, and `docs/tutorial-code/README.md` says tutorial code and mdBook
pages should be updated together. However, the maintenance policy is still
split across files and not explicit enough for the new online tutorial section.

Specific gaps:

- `docs/tutorial-code/README.md` lists only seven tutorial markdown files in its
  suggested order and omits affine transformation and 2D partial Fourier.
- The README points to the legacy tutorial-code markdown files under
  `docs/tutorial-code/docs/tutorials/`, while the live online tutorials are in
  `docs/book/src/tutorials/`.
- `AGENTS.md` and `REPOSITORY_RULES.md` do not explicitly say that agents who
  change tutorial code, generated CSV/PNG artifacts, or public APIs must check
  and update the live online mdBook tutorials in the same branch.

Recommended fix: add a short "Tutorial Maintenance" subsection to
`AGENTS.md` and/or `REPOSITORY_RULES.md` that names the live source of truth:
`docs/book/src/tutorials/` for online pages and `docs/tutorial-code/src/bin/`
plus shared helpers for runnable demos. Also update `docs/tutorial-code/README.md`
to list all nine live tutorials and clarify the role, if any, of the legacy
markdown files in `docs/tutorial-code/docs/tutorials/`.

### P2 — No Single Refresh Command For Tutorial Artifacts

The README documents how to refresh one tutorial manually, and
`docs/tutorial-code/scripts/check.sh` provides a smoke check that avoids
changing tracked artifacts. There is still no single tracked command that
regenerates all tutorial CSVs and PNGs, copies the PNGs into the live mdBook
locations, and builds/tests the book.

Recommended fix: add a deliberate refresh script, for example
`scripts/refresh-tutorial-artifacts.sh`, that:

1. runs all tutorial binaries that own tracked data;
2. runs all Julia plotting scripts;
3. copies or verifies PNGs in `docs/book/src/tutorials/**`;
4. runs `./scripts/test-mdbook.sh`;
5. prints the changed artifact list for review.

### P3 — Some Advanced Pages Still Jump Too Quickly For Beginners

The beginner-oriented goal is mostly met in the first five tutorials. The
operation tutorials remain terse for a new master's or early-PhD reader.

Pages to improve:

- `elementwise-product.md`: briefly define `TreeTN`, `diagonal_pairs`,
  `PartialContractionSpec`, and why pairing site indices diagonally gives a
  pointwise product.
- `affine-transformation.md`: explain `Fused`, passive pullback via
  `transpose()`, and why boundary conditions matter.
- `fourier-transform.md`: mention the bit-reversed Fourier output convention
  or link directly to the guide section that explains it.
- `partial-fourier2d.md`: add one compact diagram or table showing interleaved
  node positions: `x0, t0, x1, t1, ...`.

Recommended fix: add a few explanatory sentences and two or three inline
comments per advanced code block. Keep the pages short, but make each new
concept visible before it appears in code.

## What Is Working Well

- All nine online tutorials are present in `docs/book/src/SUMMARY.md`.
- The live pages link to runnable source files under `docs/tutorial-code/src/bin/`.
- The first five tutorials form a reasonable onboarding path for QTT basics.
- mdBook examples are runnable and are checked in CI.
- The tutorial-code crate has focused tests for generated CSV shape, numerical
  behavior, plotting inputs, and the more complex affine/Fourier helpers.

## Suggested Fix Order

1. Fix `partial-fourier2d.md` so the online code block matches the actual
   partial-transform workflow.
2. Strengthen weak assertions in the five affected mdBook snippets.
3. Update the tutorial maintenance policy in `AGENTS.md`,
   `REPOSITORY_RULES.md`, and `docs/tutorial-code/README.md`.
4. Add a full artifact refresh script.
5. Add short beginner-facing explanations to the advanced operation tutorials.
