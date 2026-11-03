# Randomized factorization (#1108)

## Decisions

- Implement index-aware Gaussian range finding and dense RSVD in core using
  existing contraction, QR and SVD primitives. Keep matrix-free drivers private.
- The first version has an explicit fixed rank; reject threshold policies and
  bypass global QR/SVD truncation defaults. The projected spectrum does not
  certify the full approximation error.
- Keep caller-owned RNG streams continuous across decompositions and sweeps.
  Seeded convenience entries use ChaCha8Rng. Preserve the global, context and
  explicit-session execution boundaries.
- The latest issue decision requires `UnsupportedTruncationPolicy` with a
  fixed-rank/deterministic-SVD remedy; no implicit fallback. The standalone
  Lanczos reference links the exact transitive faer artifacts outside the
  workspace instead of adding a library dependency.
- Retire simplett's unimplemented RSVD option instead of adding another
  implementation to a retiring crate.

## Verification conclusions and constraints

- Baseline attribution precedes RSVD wiring. The diagnostic grid is six-site
  Gaussian MPO pairs with `(chi, d) = (8, 2), (16, 2), (32, 2), (16, 4)`,
  output rank cap `chi`, two Fit sweeps, one warmup and three samples per cell.
  Release/default faer, one thread, CPU affinity 8. This is an attribution
  probe, not a guarded performance comparison or a speedup claim.
- Real/complex reconstruction, both canonical directions, seeded streams,
  projected two-site contractions, context/session routing, and a partial-rank
  finite-difference AD oracle pass. After synchronizing with main at `ca4819ca`,
  the four affected crates pass 2,467 unit/integration tests (four existing
  ignored tests), Clippy, and all 13 directly affected core doctests. Full
  core/treetn doctests and focused release regressions also pass.
- The retired simplett stub and unused C enum reverse conversion were reviewed
  for coverage impact. No shared helper loses its sole exerciser; C-to-Rust
  enum coverage remains. The C ABI is unchanged.
- Header regeneration matches the checked-in header, which compiles as C11
  and C++17. The mdBook tests and API inventory also pass after synchronization.
  No GPU or hosted CI validation is claimed.
  [Validation details](../../benchmarks/results/2026-11-03-randomized-factorization/validation.md)
  record the checks and local tooling limitations.
- [Diagnostic results and declared protocols](../../benchmarks/results/2026-11-03-randomized-factorization/README.md)
  retain the baseline, Lanczos comparison, and complete experiment evidence.
  The final grids contain 72 decomposition and 96 contraction samples, all
  within their predeclared accuracy gates. GEMM/QR accounts for 50.9–78.0% of
  RSVD time on this grid. With two power iterations, RSVD is slower on the
  tested d=2 contractions and faster on d=4. These shared-host diagnostics do
  not establish a general speedup, scaling benefit, or upstream placement win.
- The first Lanczos comparison is invalid: the pinned public driver clamps
  its default maximum subspace to 64 for a 64x64 matrix, while its private
  driver asserts that maximum is strictly smaller than the matrix dimension.
  Both public `partial_svd` and its scratch planner use this clamping rule.
  The failed attempt is retained in the experiment evidence. Revision 2
  predeclares a complete doubled-chi grid with unchanged accuracy gates.
  No upstream report or patch was prepared; that needs separate authorization.
- One power iteration did not meet the declared 5% end-to-end error gate on
  a flat-spectrum random Zipup case (0.68160 vs deterministic 0.63757).
  The complete failed attempt is retained. A separate complete comparison
  predeclares two power iterations in every RSVD cell, without changing the
  gate or the library default. A local SVD speedup alone is not an application
  accuracy result.
- Repository rules review passes with three advisory file-growth findings.
  The algorithms live in dedicated randomized modules. The additions to the
  existing tensor trait, contraction dispatch, and Fit updater keep their
  owning API/configuration and execution responsibilities together.
- A trial merge with #1115 at `cb6f711e` has no text conflicts and passes
  workspace compilation, partitioned TreeTN tests/doctests/Clippy, and all
  RSVD regressions. After #1115 merged into main at `67583479`, the actual
  synchronization also has no conflicts and produces identical production
  code to the tested trial tree. The pinned compatibility check is recorded
  in the validation details.
- R2 environment projection remains deferred until inverse-canonical parallel
  Fit. Adaptive error-certified rank selection is also out of scope.
