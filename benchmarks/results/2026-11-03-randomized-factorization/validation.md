# Local validation

The final branch incorporates `origin/main` at `ca4819ca`. Benchmark manifests
pin the measured commits separately; both remain ancestors of the branch.
The synchronization adds the opt-in N-ary sum and does not change the RSVD
algorithm or benchmark protocols. No push, PR, hosted checks, or GPU execution
is included in this evidence.

Commands below use the existing shared Cargo target via `CARGO_TARGET_DIR`.
Ordinary checks use the default development/test profiles. The release check
exercises the new numerical regressions under optimization.

| Check | Result |
|---|---|
| `RAYON_NUM_THREADS=1 cargo test --locked -p tensor4all-core -p tensor4all-treetn -p tensor4all-simplett -p tensor4all-capi --lib --tests -- --test-threads=2` | After synchronization: 2,467 passed, zero failed, four existing ignored tests. |
| `cargo clippy --locked -p tensor4all-core -p tensor4all-treetn -p tensor4all-simplett -p tensor4all-capi --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc` | After synchronization: passed. |
| `cargo clippy --locked -p tensor4all-benchmarks --example benchmark_randomized_factorization -- -D warnings` | Passed for the final benchmark harness. |
| `cargo test --locked --release -p tensor4all-core randomized_factorization` and the corresponding `tensor4all-treetn` command | 17 core and seven TreeTN regressions passed. |
| `cargo test --doc -p tensor4all-core -p tensor4all-treetn` | 385 core and 194 TreeTN examples passed before the final main synchronization. |
| `cargo test --locked --doc -p tensor4all-core randomized` and the `factorize_with_rng` filter | After synchronization: nine randomized and four caller-RNG examples passed. |
| `TENSOR4ALL_CARGO_PROFILE=dev ./scripts/test-mdbook.sh` | After synchronization: guide chapter tests and the shared randomized example passed. |
| `cargo doc --no-deps -p tensor4all-core -p tensor4all-treetn` | Passed with four existing rustdoc link warnings; see limitations below. |
| `cargo run -p xtask --release -- api-dump` | After synchronization: public crate inventory generated and verified. |
| `python3 scripts/check-public-error-docs.py`, `check-crate-boundaries.py`, and `check-test-targets.py` | Passed. |
| `python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run --file-growth` | Passed; advisory growth findings in the tensor trait, contraction dispatch, and Fit updater. |
| `python3 scripts/test-repository-rules-review.py` | 114 passed; retry/error messages are mocked test cases. |
| `./scripts/check-capi-header.sh` with cbindgen 0.29.2 | Three helper tests passed; regenerated header matches; C11 and C++17 compile checks passed. |
| `cargo fmt --all -- --check` and `git diff --check` | Passed. |

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

The repository rules check here is the deterministic dry run, not the hosted
LLM review. Hosted review and CI remain required before merging.
