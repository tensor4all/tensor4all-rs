# Post-hoc TreeTN contraction outcomes

## Decisions

The first [outcome slice](../design/treetn-contraction-outcome.md) adds final
stored-rank reports over the existing contraction entries. Ordinary calls are
unchanged and allocate no report. Outcome calls finish all numerical work,
then inspect only topology and bond metadata; no speedup or accuracy guarantee
is claimed. Equality with the independent positive threshold saturates, while
edgeless results complete.

Compatibility, caller-context and caller-RNG routes delegate to their current
owners. Caller-RNG dispatch supports RSVD Zipup/Fit and SRC directly; seeded
dispatch retains the existing fields and named RNGs. This avoids duplicating
the numerical schedules, context validation or AD decisions. Saturation frees
the completed network and returns the first qualifying report edge.

The completed variant keeps `TreeTN` by value, as the existing entry does.
The enum has a documented local Clippy exception for its variant-size
difference to avoid adding a separate heap allocation for the network.

The early-abort hook, adaptive lazy grouping and sibling merging remain
separate stages. The existing adaptive analytical baselines remain unchanged.

## Verification conclusions and constraints

Known-GHZ references and metadata fixtures cover chains and degree-three
trees, real/complex values, same-ID primed output indices, unequal stored
ranks, root fallback after center pruning, scalar outcomes, threshold/cap
independence and every method. Seeded RSVD and mid-stream caller RNGs are
compared through both context routes, including RNG consumption on saturation.
Source-chain, invalid-option and foreign-context behavior is inherited, and a
scalar derivative oracle checks that post-hoc reporting preserves AD.

Single-precision tests retain existing support/refusal and promotion behavior;
they do not claim new f32/c32 factorization support. Existing implicit
`to_dense` may mix the global eager owner with a caller-context result, and
`adopt_into_in` deliberately refuses rebinding that eager owner. Owner-level
tests therefore materialize a scoped result once through the existing explicit
leaf seam and read back only the final tensor for the host oracle. A separate
audit of the public dense-readout/context contract is recommended; this slice
does not add a parallel conversion API or alter those semantics.

No existing tolerance, coverage threshold or default numerical dispatch is
changed. Non-default providers, CUDA and performance scaling require their
own verification; no early-abort saving is claimed here.

## Local validation

All Cargo checks used `--locked`. Numerical tests used the ordinary test
profile, `RAYON_NUM_THREADS=1` and two test threads; no optimized-behavior or
performance claim needs release validation in this metadata-only slice.
`cargo-nextest` was unavailable, so the changed crate used `cargo test`.

- `cargo test -p tensor4all-treetn --lib --tests`: 990 passed, two existing
  ignored tests unchanged. This includes 14 outcome regressions.
- `cargo test -p tensor4all-treetn --doc`: 207 passed, including seven new
  outcome examples.
- `cargo test -p tensor4all-partitionedtreetn --test integration
  adaptive_contraction_baseline`: three passed.
- `cargo clippy -p tensor4all-treetn --all-targets -- -D warnings
  -D clippy::missing_errors_doc -D clippy::missing_panics_doc`: passed.
- `RUSTDOCFLAGS='-D warnings' cargo doc -p tensor4all-treetn --no-deps`:
  passed after qualifying two existing `IdxTensor` links in `decompose.rs`;
  its numerical implementation and examples are unchanged.
- `TENSOR4ALL_CARGO_PROFILE=dev ./scripts/test-mdbook.sh` with the shared
  target's dependency search path: all 32 chapters passed.
- API inventory: all 17 public crates accounted for; root README crate map
  and existing examples remain accurate.
- Formatting, public error docs, crate boundaries, test-target registration,
  local Markdown paths and deterministic repository-rules review passed.

Hosted CI, coverage and the hosted rules review remain outstanding until an
approved push/PR. No remote publication was performed.
