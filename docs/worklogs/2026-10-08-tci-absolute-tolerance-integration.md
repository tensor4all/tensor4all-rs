# Absolute two-site tolerance integration (#849)

#810 was closed while its fix in fork PR #811 remained unmerged. This branch
integrates Sakurai Rihito's existing implementation and regression tests with
`git cherry-pick -x e1bc26ee176c6dde77967ecc90aad5e12c98d764`, preserving authorship
and recording provenance. It does not require executing or waiving the fork's
repository-rules workflow.

Both public two-site sweeps and the optimizer freeze the absolute truncation
threshold at the start of each half-sweep. With normalization disabled the
threshold is `tolerance`; otherwise it is `tolerance * max_sample_value`.
Full and Rook pivot paths receive the same threshold and retain LU's distinct
numerical relative floor. Invalid scaled tolerances fail before public sweep
state mutation. Final one-site cleanup uses the same conversion helper.

All five new regressions fail on the unchanged main `tensorci2.rs`; all pass
with the integrated fix. They exercise Full/Rook, scalar/batched callbacks,
both sweep directions, absolute/relative rank selection, global versus local
normalization scale, overflow and complex values. Related TreeTCI pivot
options already use a separate numerical relative floor and an absolute
sample-scaled cutoff; no duplicate repair is needed there.

The complete tensorci non-release suite passed 107 tests, and its 33 doctests
passed. Strict changed-crate Clippy and deterministic repository-rules preview
passed. Nextest is unavailable, so the suite used `cargo test --lib --tests`.
The five focused release regressions also passed, as did all mdBook chapters
and the library panic audit (zero new/stale findings). The release regression
selected both tensorci and book-tests to reuse the validated guide dependency
configuration; only the named tensorci regression target ran.
Validation uses `--features tensor4all-core/default`:
isolated tensorci's dependency configuration otherwise omits the legacy
convenience backend required by its existing tests. No manifest change is needed.

The guide describes absolute versus relative tolerance and the local nature
of the error criteria. Root README and current examples remain accurate.
No numerical test tolerance, coverage threshold, public API or dependency is
changed by this integration. Hosted CI remains authoritative.
