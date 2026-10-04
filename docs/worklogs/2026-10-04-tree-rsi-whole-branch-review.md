# Tree RSI whole-branch review, 2026-10-04

## Verdict and scope

Review complete: **0 blocker, 1 major, 3 important, 1 minor finding group**.
These findings remain open. This was a new review, not a repair round. No
implementation, existing tests, tolerance, index, branch, or remote was changed.
This record is the only added repository file.

The effective implementation is `rewrite/tree-rsi` in the main checkout at
`9ad67f2c494d82ec0addfab201f9133533143f98`. Its implementation is not committed;
the review covers the combined staged/unstaged tracked changes and all 59
untracked source/documentation files, **86 files in total**. A HEAD-only diff
would omit the new crate. The source manifest is under ignored
`target/rsi-review-20261004/reviewed-source-manifest.json`, with canonical JSON
SHA-256 `3cad14c3b87278aceeb86822b9dc2dffe1851e3ce5d2f6bfca55c72ceccd4cf4`.
All recorded source hashes remained unchanged during the review.

`plan/tree-rsi-implementation` at `29392e3f` is the withdrawn predecessor,
including its historical outputs. It is not the candidate being reviewed.
No old branch commits were imported. The separate `review/non-rsi-staged-work`
checkout was inspected only to identify scope. The effective RSI changes add
context-aware matrix operations, row-only LU interpolation and diagnostics in
core/backend; they do not modify ACI or TCI algorithms.

After fetching origin, this branch is 12 commits behind `origin/main`. Tests
below validate this working source against its own base, not a synchronized
merge candidate or hosted CI. No commit, push, PR or issue was created.

## Open findings

### R1 — Major: paper workers are not bound to the recorded source revision

Locations: `benchmarks/tree-rsi/paper-coverage/build.py:10–38`,
`run.py:140–163`; the repeat, supplement and matched runners use the same
worker selection pattern.

Build copies a worker executable to a reusable path without writing a receipt
binding its hash to the library/dependency source, baseline snapshot, lockfile
and build configuration. Run hashes the current working source independently,
then executes whichever existing worker is at that path. Comparing the source
before and after a run only establishes that it did not change during the run;
it does not establish that the worker was compiled from it.

**Ran:** a temporary mocked run with unchanged new-source identity and workers
containing unrelated old-build bytes was accepted through all six scheduled
attempts and completion, recording `candidate_source_unchanged=true`. The
attempts were test doubles, not numerical or performance measurements.
`target/rsi-review-20261004/stale_worker_probe.py` preserves the harness.

The actual retained September runs also have a different candidate fingerprint
from the current implementation (`b72d8df6…` versus `b412cf29…`). That difference
is historical provenance, not evidence that those runs were fabricated. Their
results cannot be relabeled as measurements of this reviewed implementation.

Required repair: capture an immutable source/build receipt at compilation,
validate it before every worker run, and reject stale workers or rebuild them.
Include the baseline snapshot, worker source, dependency source/configuration,
lockfile and binary. Retain the original measurements under their own identity.
Do not update performance winners until fresh, correctly bound measurements
exist. This report makes no new speedup claim.

### R2 — Important: zero-tolerance exact interpolation silently loses a finite pivot

Locations: `crates/tensor4all-core/src/matrixlu.rs:768,819,840`, its pivot
selection helper around line 483, and the RSI call at `engine.rs:697`.

**Ran:** a two-site single operand representing `diag(1, 1e-200)`, with
`max_bond_dim=2`, `sketch_dim=2`, and `rel_tol=0`, returns dense column-major
`[1,0,0,0]`, rank one, `exact_columns=true` and `relative_pivot=0`. The expected
last entry is the representable value `1e-200`. Squaring the tiny pivot in
`abs_sq()` rounds to zero before taking its square root. Binary block scaling
and the GEMM range checks do not detect this loss inside LU.

This is a **pre-existing shared LU numerical issue exposed by the new RSI
path**, not an incorrect tree recursion or a large global relative error:
the lost entry is extremely small compared with the unit entry. Zero tolerance
and a reported zero residual nevertheless cannot establish its preservation.
Related checks found the same magnitude calculation in full LU diagnostics and
row-only pivot extraction. The public row-only facade also treats a nonzero
`[1e-20]` matrix as rank zero with default options because it inherits LU's
absolute epsilon floor.

Required repair: investigate stable magnitude/pivot selection in core and its
related full-factor APIs as a shared numerical follow-up; do not introduce an
RSI-only decomposition or an unrelated ACI/TCI patch. Add zero-tolerance,
real/complex and scalar-width regressions and make scale/floor semantics clear.

### R3 — Important: new row-only public API returns success on invalid inputs

Location: `crates/tensor4all-core/src/matrix_luci.rs:152–168`.

**Ran:** `matrix_luci_row_interpolation_owned_in` returns `Ok` for 1×2 matrices
`[1, NaN]` and `[1, infinity]`; the former even reports `[1,0]` as its pivot
magnitudes. It also returns `Ok` for negative or NaN relative tolerances.
Its `# Errors` section promises rejection of invalid controls and nonfinite
input, but it delegates directly to LU without validating them. Removing the
unused upper factor also removes a place where NaNs in that factor could be
noticed. RSI itself validates its controls and normalizes finite blocks before
calling this facade, so this is not evidence that ordinary RSI inputs bypass
its validation.

Required repair: validate every input component and the supported control
constraints at the new public boundary; exercise invalid values even when they
lie outside the left factor. Review the full-factor facade for the same
contract issue rather than duplicating inconsistent validation.

The executable Rust reproducer and output for R2/R3 are under ignored
`target/rsi-review-20261004/probe.rs` and `rust-probe-output.txt`. It was compiled
against the same dependency graph as the tested RSI library. An optional
standalone Cargo probe build was cancelled when it resolved a different graph;
it is not counted as validation.

### R4 — Important: artifact verification does not validate the experiment schedule

Locations: `benchmarks/tree-rsi/paper-coverage/verify_artifacts.py:6–41` and
`summarize.py:6–15`.

**Ran:** an expected 24-observation protocol (two cases, two seeds, two
algorithms, three blocks) passes `records` and `verify_run_evidence` when all
24 unique tags instead contain an unrequested fixture, seed 999, algorithm RSI
and block zero. Count and tag uniqueness do not verify the Cartesian schedule.
Removing `source_files` also passes because its default is an empty dictionary.
The current evidence test even supplies rows containing no experiment fields.

Required repair: compare actual `(case including options, algorithm, seed,
block/phase)` identities with the protocol schedule, require the source/build
manifest appropriate to each run type, and reject missing/unexpected/duplicate
identities. Validate trace schedules against their original measurements too.
The stronger schedule validation already present in the small benchmark and
main-comparison summarizer provides a related pattern. Reproducer:
`target/rsi-review-20261004/verify_probe.py`. These probes demonstrate an
integrity-check gap; they do not assert that existing observations are wrong.

### R5 — Minor: two required local checks fail

**Ran:** `scripts/check-public-error-docs.py` exits 1 for the new
`mat_mul_owned_in` and `hadamard_many_with_rng_in` error sections. The latter
only refers to another API; the checker requires concrete conditions/variants
at the entry point. `scripts/audit-library-panics.py` exits 1 with nine
unbaselined findings and ten stale baseline entries after source movement.
Inspection of the backend locations shows existing assertions shifted by the
new imports; this audit result does not establish nine new runtime panics.
Update documentation and review/update the moved baseline entries before
claiming the local validation gate passes. Do not lower thresholds or suppress
new findings indiscriminately.

## Mathematics, architecture, performance and artifacts

The local product-of-sketches, selected exact operand frames and root product
match the documented algebra. The primary comparison was the paper's TT
construction in III.A.1–4 and the pinned author code, not earlier AI prose:

- https://arxiv.org/html/2602.17974v1
- https://github.com/zmeng137/Recursive-Sketched-Interpolation/tree/153b25a8aa059d0147b45955d0842b2f32fa5d1d

The tree traversal, directed-message requirement closure, exact-component
caches and binary scaling are extensions. A local fit of the lifted frame's
sketch does not alone prove a fit of its physical complement. The deterministic
rank-one counterexample and the public accuracy caveats remain appropriate.
No general tree error theorem or arbitrary nonlinear/GW capability is implied.

The library dependency tree has no ACI, TreeACI, TreeTCI, SimpleTT or TensorCI.
Comparison/preparation executables own their additional algorithm dependencies.
Dense matrix operations cross crate boundaries through the existing Matrix and
context-aware backend seams. The static performance scan found no additional
confirmed violation: contractions use GEMM/batched GEMM, bounded exact messages
are cached, and rank-one high-degree prefix/suffix paths avoid sibling
recontraction. General high-degree cores retain the documented degree-dependent
work; no general cubic scaling conclusion or measured regression is claimed.

The candidate's new RSI/benchmark Rust and Python source is 6,942 lines, not the
old branch's hundreds of thousands of generated output lines. Existing roughly
40 GiB paper artifacts are under ignored `target/tree-rsi/paper-coverage`;
`git check-ignore` confirms this and the generated results synopsis. Generated
data and old JSON protocols are not in the 86-file candidate. No artifacts
were deleted. README/crate map, guide registration, architecture selection,
llms and skill references were checked; the relevant local Markdown link
targets exist and the asserted product example is nonconstant and sketched.

A read-only downstream check still finds `tree_rsi_elementwise` and `abs_tol`
in `gw-rs/g0_rsi/src/main.rs` and `gw-rs/sgw_rsi/src/ops/rsi.rs`. They do not
match this product-only API. Downstream migration and G0/Pi/Sigma/iteration
acceptance remain separate unfinished work; no GW build or acceptance run was
performed.

## Validation

- `cargo run -p xtask --release -- api-dump`: exit 0; complete inventory verified.
- `cargo test -p tensor4all-treersi -p tensor4all-core -p tensor4all-tensorbackend`:
  exit 0, 1,627 passed, 5 ignored, including 43 RSI tests/doctests.
- `cargo test -p tensor4all-treetn -p tensor4all-treetci`: exit 0,
  1,063 passed, 3 ignored, including doctests.
- `cargo clippy -p tensor4all-treersi -p tensor4all-core -p tensor4all-tensorbackend
  --all-targets -- -D warnings`: exit 0.
- `cargo check -p tensor4all-treersi --no-default-features --features
  tenferro-cpu-faer`: exit 0. This does not imply every dependency's global
  feature was disabled; Cargo features are additive.
- Three Python unittest discovery runs: 7 + 5 + 13 tests passed. Paper tests
  used the existing ignored virtual environment; system Python lacked h5py.
- `./scripts/test-mdbook.sh`: exit 0, complete chapter run.
- `cargo fmt --all -- --check`, tracked/index whitespace checks, and
  `scripts/check-crate-boundaries.py`: exit 0.
- Public-error documentation and panic audit: **exit 1**, as R5 above.
- Repository-rules dry run: exit 0, deterministic pass only. This is not the
  hosted/LLM rules review and its tracked Git diff does not cover new untracked
  implementation files; those were reviewed directly.
- Rust and Python probes reproduced R1–R4. No existing tolerance changed.

No full workspace test, hosted CI, full large benchmark rerun, new performance
measurement, or downstream GW validation was performed. Reusable build output
and ignored review reproducers were retained for a future authorized fix round.

## RSI-scoped disposition

The [follow-up fix record](2026-10-04-tree-rsi-review-fixes.md) supersedes this
review's branch-base and validation status. R1, R3, R4 and R5 are addressed in
the synchronized RSI draft. R2 belongs to the existing independent
[core issue #779](https://github.com/tensor4all/tensor4all-rs/issues/779) and is
explicitly excluded from this branch. Unrelated full-factor/dense/rook LUCI
residual corrections were also removed from the RSI draft.
