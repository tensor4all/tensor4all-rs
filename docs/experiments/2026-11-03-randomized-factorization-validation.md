# Local validation

The final branch incorporates `origin/main` at `635f9a8e`, including #1115
and the f32/Complex32 direct-sum support in #1117. RSVD's separate scalar
restriction is unchanged and now explicitly documented and tested for both
unsupported 32-bit types.
Benchmark manifests pin the measured commits separately; both remain ancestors
of the branch. Synchronization does not change the RSVD algorithm or benchmark
protocols. This document records local checks; hosted checks are reported on
[PR #1116](https://github.com/tensor4all/tensor4all-rs/pull/1116). No GPU execution
is included in the local evidence.

## Review revision checks

The review adds a rotated-spectrum rank/error regression with both decay
ratios `0.6` and `0.95`, ranks `2, 5, 9`, both canonical directions and seeds
`0..8`. All 96 randomized decompositions are checked against dense SVD
references, the known spectral tails, rank monotonicity, and the derived
expected squared-error reference. The original `1.02` regression threshold
is unchanged. The derivation is in the
[design note](../design/randomized-factorization.md#expected-error-used-by-the-numerical-regression).

- `cargo test --locked -p tensor4all-core randomized_factorization`:
  18 tests passed, including explicit f32/Complex32 rejection.
- `cargo test --locked -p tensor4all-treetn randomized_factorization`:
  seven tests passed.
- Core and TreeTN `cargo clippy --locked --all-targets` with the repository's
  warnings/error-doc/panic-doc gates passed.
- The `randomized` and `rsvd` core doctest filters passed ten examples.
- The new rank/error curve test also passed with `cargo test --locked
  --release -p tensor4all-core slow_and_fast_spectra_follow_dense_svd_rank_error_curves`.

The benchmark raw data is retained in separate
[PR #1118](https://github.com/tensor4all/tensor4all-rs/pull/1118), with
measurement and protocol files verified byte-for-byte against the original
snapshot. This PR retains the harness, summary and validation conclusions.
The previous implementation head `b3bc9a24` passed all hosted checks, including
macOS Accelerate; CI is rerun for the review revisions before merge.

## Earlier implementation checks

Commands below use the existing shared Cargo target via `CARGO_TARGET_DIR`.
Ordinary checks use the default development/test profiles. The release check
exercises the new numerical regressions under optimization.

| Check | Result |
|---|---|
| `RAYON_NUM_THREADS=1 cargo test --locked -p tensor4all-core -p tensor4all-treetn -p tensor4all-simplett -p tensor4all-capi --lib --tests -- --test-threads=2` | At main `ca4819ca`: 2,467 passed, zero failed, four existing ignored tests. |
| `cargo clippy --locked -p tensor4all-core -p tensor4all-treetn -p tensor4all-simplett -p tensor4all-capi --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc` | At main `ca4819ca`: passed. |
| `cargo clippy --locked -p tensor4all-benchmarks --example benchmark_randomized_factorization -- -D warnings` | Passed for the final benchmark harness. |
| `cargo test --locked --release -p tensor4all-core randomized_factorization` and the corresponding `tensor4all-treetn` command | 17 core and seven TreeTN regressions passed. |
| `cargo test --doc -p tensor4all-core -p tensor4all-treetn` | 385 core and 194 TreeTN examples passed before the final main synchronization. |
| `cargo test --locked --doc -p tensor4all-core randomized` and the `factorize_with_rng` filter | At main `ca4819ca`: nine randomized and four caller-RNG examples passed. |
| `TENSOR4ALL_CARGO_PROFILE=dev ./scripts/test-mdbook.sh` | At main `ca4819ca`: guide chapter tests and the shared randomized example passed. |
| `cargo doc --no-deps -p tensor4all-core -p tensor4all-treetn` | Passed with four existing rustdoc link warnings; see limitations below. |
| `cargo run -p xtask --release -- api-dump` | After synchronization: public crate inventory generated and verified. |
| `python3 scripts/check-public-error-docs.py`, `check-crate-boundaries.py`, and `check-test-targets.py` | Passed. |
| `python3 scripts/audit-library-panics.py` | Passed; no unbaselined findings or stale entries. |
| `python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run --file-growth` | Passed; advisory growth findings in the tensor trait, contraction dispatch, and Fit updater. |
| `python3 scripts/test-repository-rules-review.py` | 114 passed; retry/error messages are mocked test cases. |
| `./scripts/check-capi-header.sh` with cbindgen 0.29.2 | Three helper tests passed; regenerated header matches; C11 and C++17 compile checks passed. |
| `cargo fmt --all -- --check` and `git diff --check` | Passed. |

The panic baseline tracks an existing `fuse_indices` debug assertion at line
8285 instead of 8255 after the additions to `idx_tensor.rs`. The assertion and
surrounding code are unchanged; only its reviewed location was updated. No
additional panic path was accepted into the baseline.

## Tooling limitations

`cargo-nextest` is absent, so the affected crate suite uses `cargo test`.
cbindgen 0.29.2 was installed in an isolated temporary prefix for the header
check. It emits an existing `DefaultTreeTN` generic-path warning; regeneration
and the exact header comparison succeed. The compile checks use
`-Wall -Wextra -Werror -fsyntax-only`.

Rustdoc's warnings concern two existing links to the private
`IdxTensorStorage::Materialized` and two unresolved `IdxTensor` links in the
TreeTN decomposition module. A warnings-as-errors documentation build is
therefore unavailable without unrelated fixes. The new API examples are
compiled and asserted.

The local repository rules check is the deterministic dry run. Hosted rules
review and CI remain required before merging; the hosted workflow permanently
disables its external LLM pass.

## Compatibility with PR #1115

An uncommitted trial merge combined this branch at `a668d4a9` with
[#1115](https://github.com/tensor4all/tensor4all-rs/pull/1115) at
`cb6f711e5f62a0a7d48de4528a33c16aefd3b816`, using main at `ca4819ca`.
There were no merge conflicts. The only overlapping file was
`skills/use-tensor4all-rs/references/crates.md`; its independent additions
merged automatically. The combined tree was
`ac2bcc9cad9623687ecc2958d822808751b02711`.

Checks on that combined tree:

- `cargo check --locked --workspace --all-targets --exclude tensor4all-hdf5`
  passed. HDF5 can still compile transitively; its test targets were excluded.
- `RAYON_NUM_THREADS=1 cargo test --locked -p tensor4all-partitionedtreetn --lib --tests -- --test-threads=2`
  passed 369 tests with 13 existing ignored tests.
- `cargo test --locked -p tensor4all-core -p tensor4all-treetn randomized_factorization`
  passed all 24 RSVD regressions.
- `cargo clippy --locked -p tensor4all-partitionedtreetn --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc`
  passed.
- `cargo test --locked --doc -p tensor4all-partitionedtreetn` passed 121 examples.

The trial merge was discarded after validation. Subsequently #1115 was merged
into main at `67583479a6c67441fd56beeaecb530bb3a0699a8`, and integration head
`b3bc9a24` merged that main revision without conflicts. At that integration
head, the combined production tree was byte-identical to the trial tree: only
the validation document, worklog and relocated panic baseline differed.
Those compatibility results cover the #1115 integration; the subsequent #1117
synchronization and review checks are described above. Hosted CI is rerun on
the synchronized review head.
