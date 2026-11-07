# One-site ALS linear solves

## Decisions and scope

- Started `feat-treetn-one-site-als` in an isolated worktree from freshly fetched
  `origin/main` (`097e2927d640ea68ff365ac56886be7676e79c8d`). The CG and BiCGSTAB
  draft PRs remain separate. The user approved the one-site ALS design.
- TreeTN and the MPS facade select ALS with `with_nsite(1)`; the default stays
  two-site MALS. Existing projected GMRES handles local equations. Exact QR
  transports the center; the existing two-node reference/cache hooks see both
  modified tensors. Single-node systems get one update per sweep.
- One-site mode rejects bond caps and SVD policies. Identity-only solves also
  use local projections to respect the initial rank limits. Redundant dimensions
  may shrink; insufficient ranks return a nonconverged result without growth.
- Regression tests exposed the shared GMRES homogeneous-system shortcut returning
  a nonzero initial guess. All four paths now return zero for a zero-norm RHS;
  tiny nonzero RHS values follow the normal solve. Invalid affine coefficients
  are checked before the shortcut. No existing assertion tolerances were changed.
- Real and complex affine references cover chains, branching trees, repeated
  sweeps, zero/exact initial guesses, zero RHS, zero sweeps, residual controls,
  invalid modes, rank limits, and facade forwarding. A six-site rank-two solve
  requires multiple sweeps, with local tolerance tighter than the sweep target.

## Validation completed

- Baseline linear-solver selection: 70 tests passed.
- Changed-crate Nextest: 2,290 tests passed, 10 existing skips.
- Optimized ALS integration selection: 12 tests passed.
- Optimized GMRES homogeneous-system selection: 6 tests passed.
- Deterministic repository-rules preview including file growth: pass, no findings.
- Repository-rules helper self-tests: 114 passed.
- Public-error documentation audit, local Markdown links, formatting and diff
  whitespace checks passed.
- All 632 doctests passed (395 core, 35 itensorlike, 202 treetn).
- Complete public API inventory generated and verified; the new TreeTN and
  facade option methods appear in it.
- Clippy across all three crates passes with only `nonminimal_bool` suppressed.
  Strict Clippy encounters 43 existing findings in unchanged core files
  (`defaults/direct_sum.rs`, `defaults/idx_tensor.rs`, `floating_zone.rs`), plus
  20 in unchanged TreeTN files (`operator/linear_operator.rs`, `simplett_bridge.rs`,
  `treetn/evaluator.rs`). No new lint configuration was added. The separate
  strict TreeTN/facade invocation confirms no finding in the ALS source files.
- Rustdoc completed with five existing warnings: three in core (an unresolved
  `TensorElement` link and two private-storage links) and two unresolved
  `IdxTensor` links in TreeTN `treetn/decompose.rs`.
- Final runtime source review confirmed the one-site branch does not contract
  the transport pair or materialize the full state, and retains the shared
  exact QR and reference/cache synchronization paths. The removed validation
  restriction retains coverage through the two-site single-node rejection test
  and the new one-site single-node solves.

## Boundaries

CG/BiCGSTAB selection, preconditioning, enrichment, compression-aware Krylov
methods and performance benchmarks remain follow-ups. The separate CG PR
improves shared norm range; the ALS branch does not duplicate it. No full-workspace
or alternate-backend validation was requested. Hosted checks have not run for
this uncommitted branch. No agent commits, pushes or PRs were made.

## PR preparation

- Fetched origin and confirmed the branch base still equals current `origin/main`.
- Prepared a draft PR title and description under ignored `target/als-pr/`.
  Changes are staged; publication awaits the user's commit, as instructed.
- A fresh Nextest rerun stopped before execution because the linker ran out of
  disk space (about 648 MiB free). Cargo now resolves to Homebrew 1.98.1 rather
  than the Rustup stable toolchain used for the successful checks above. Shared
  artifacts were preserved; no cleanup was attempted.
- Reran the existing optimized binaries directly without rebuilding: all 12 ALS
  and 6 GMRES regressions passed. Rustup-stable formatting and the staged diff
  whitespace checks also passed.
