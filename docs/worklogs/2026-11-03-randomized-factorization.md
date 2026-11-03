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
  finite-difference AD oracle pass. The existing core/treetn/simplett suite and
  affected doctests passed before the final main synchronization; final checks
  are recorded below when complete.
- The retired simplett stub and unused C enum reverse conversion were reviewed
  for coverage impact. No shared helper loses its sole exerciser; C-to-Rust
  enum coverage remains. The C ABI is unchanged.
- C header regeneration is unavailable locally because cbindgen is absent.
- [Baseline samples and the declared comparison protocol](../../benchmarks/results/2026-11-03-randomized-factorization/protocol.md)
  retain the attribution evidence and measurement constraints.
- R2 environment projection remains deferred until inverse-canonical parallel
  Fit. Adaptive error-certified rank selection is also out of scope.
