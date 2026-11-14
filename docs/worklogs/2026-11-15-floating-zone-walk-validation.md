# `floating_zone_walk` input validation

## Decisions

- Give `tensor4all_core::floating_zone_walk` the same contract as its sibling:
  it returns `FloatingZoneError<Err>` and validates its arguments before the
  seed evaluation. One private `validate_search` owns the metadata checks
  (starting length, zero local dimension, coordinate bounds, candidate-storage
  overflow, stopping tolerance); `floating_zone_walk_with_initial_error` keeps
  only its `initial_error` check, so the two entry points share one validation
  path instead of two. The private walk body is unchanged.
- No deprecated sibling and no compatibility wrapper: the repository is in
  early development and removes superseded API immediately.
- Callers keep their own pre-validation. Removing it would change each caller's
  public error type and messages, which is a separate change.
- Candidate-storage overflow stays gated on `max_sweeps > 0`, matching the
  sibling: with zero sweeps neither entry point allocates a scan batch.
- Accepted order change: an input with both an invalid `initial_error` and an
  invalid `early_stop_tol` now reports `InvalidTolerance`, because the tolerance
  check moved into the shared helper. No test, doc or caller distinguishes the
  two, and `FloatingZoneError` is `#[non_exhaustive]`.

## Verification

- Reproduced the issue's input on the pre-change tree: `&[2]` with an empty
  starting point panicked at `floating_zone.rs:298` with `index out of bounds:
  the len is 0 but the index is 0`. After the change the same input returns
  `StartingPointLength { expected: 1, actual: 0 }` without calling the
  evaluator.
- New regressions in `floating_zone/tests`: wrong starting length (both
  directions, zero and one sweep), zero local dimension, out-of-bounds
  coordinate, allocation overflow at one sweep with the zero-sweep success
  asserted beside it, invalid tolerance, and seed/scan callback failures wrapped
  in `Evaluation`.
- `cargo fmt --all -- --check`, strict Clippy (`-D warnings`, `missing_errors_doc`,
  `missing_panics_doc`) on `tensor4all-core`, `tensor4all-tensorci`,
  `tensor4all-aci`, `tensor4all-treetci`, `tensor4all-treetn`;
  `cargo test --doc -p tensor4all-core` (416 passed); `cargo test -p
  tensor4all-tensorci -p tensor4all-aci -p tensor4all-treetci` all green;
  `cargo doc` for the four changed crates clean.
- `RUSTDOCFLAGS="-D warnings" cargo doc -p tensor4all-core` fails on five
  pre-existing broken links in files this change does not touch
  (`IdxTensorError`/`idx_tensor` docs); they were left alone.
- The change is crate-local in compile impact, so the gate is the five crates
  above rather than a workspace build: `grep -rn floating_zone_walk` over every
  `*.rs` file finds the four production call sites, seven test call sites and
  the `lib.rs` re-export, all covered.

## Review records

- Plan review, before implementation: `ollama-cloud/glm-5.3-flash`, verdict
  APPROVE. Its one IMPORTANT finding (the overflow regression must be gated on
  `max_sweeps > 0`) and two MINOR findings (test-site count, line references)
  were applied to the plan before coding.
- Finished diff review: `openai/gpt-6.1-sol`, verdict PASS with one MINOR
  finding. The finding was real — the new `tensorci::error` helper's rustdoc
  link `[FloatingZoneError::Evaluation]` was unresolved because the type was
  only named by its full path in the signature — and was fixed by importing
  `FloatingZoneError` in that module.
