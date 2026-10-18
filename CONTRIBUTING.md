# Contributing to tensor4all-rs

Thank you for contributing. tensor4all-rs is in early development: public APIs may change, and fixes should remove obsolete paths rather than add compatibility shims.

## Start with an issue

Use the repository issue forms so the scope and acceptance criteria are visible before substantial work begins:

- [Bug report](https://github.com/tensor4all/tensor4all-rs/issues/new?template=bug_report.yml): include a minimal reproducer, expected behavior, Rust version, and platform when relevant.
- [Feature request](https://github.com/tensor4all/tensor4all-rs/issues/new?template=feature_request.yml): explain the use case, proposed behavior, and acceptance criteria. Link the related Tensor4all.jl issue for cross-repository work.

Small documentation and typo fixes may go directly to a pull request. For nontrivial, public-API, numerical, performance, or cross-crate changes, comment on the issue before implementation so maintainers can confirm the scope and dependency boundary.

## Prepare the change

1. Branch from the current `origin/main`.
2. Read [`AGENTS.md`](AGENTS.md), [`REPOSITORY_RULES.md`](REPOSITORY_RULES.md), and the relevant [architecture guide](docs/book/src/architecture.md).
3. Keep source code and documentation in English.
4. Add the smallest test that would fail without the change. Numerical tests must check a meaningful value, identity, residual, or reconstruction error rather than only shapes or finiteness.
5. Update rustdoc, guides, examples, and generated/public API claims affected by the change.
6. Follow the shared Work Logs And Design Records policy and the [local format](docs/worklogs/README.md) when a decision record is needed; small fixes and AI assistance alone do not require one.

### CI time budget and heavy tests

The standard pipeline (`CI_rs`) is declared to stay fast, and it carries one total
budget instead of per-test or per-job caps:

- **40 minutes for the whole workflow.** The required `rollup-rs` gate measures the
  span from the run's first job start to its last job completion and fails above it,
  printing the per-job elapsed times. This gates completion; it does not cancel the
  run, and the gate's own scheduling wait is not charged to the pipeline.
- **5 minutes of suite execution** (`--budget-seconds 300`) in the test jobs, so a
  single test that makes the suite slower than that fails the job that ran it.

Do not raise either number to accommodate a slow test. A test that needs more is a
heavy test:

- keep it in `crates/*/tests/heavy_*.rs`, declared as a module in that package's
  `tests/main.rs` integration target, and mark it
  `#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]`;
- [`.github/workflows/heavy-tests.yml`](.github/workflows/heavy-tests.yml) runs those
  files nightly (`-E 'test(/^heavy_/)' --run-ignored ignored-only`), builds every
  target for the compile check, and opens or updates a `ci-nightly` issue when it fails.

Non-default feature configurations (system BLAS, provider injection, MPI,
diagnostics, HDF5 runtime loading, CUDA) cannot share the default build, so they
run in [`.github/workflows/feature-matrix.yml`](.github/workflows/feature-matrix.yml)
nightly, also outside this budget and with its own bound per configuration. See
[Supported feature configurations](#supported-feature-configurations) for the
declared matrix.

Run the heavy suite locally with:

```bash
cargo build --locked --profile ci --workspace --all-targets
cargo nextest run --locked --cargo-profile ci -p tensor4all-treeaci \
    --run-ignored ignored-only -E 'test(/^heavy_/)'
```

Example and bench targets stay out of the default suite for the same reason: they add
no test execution but roughly a quarter of its compile CPU. `cargo clippy --workspace
--all-targets` keeps them compiling on every pull request, and the nightly workflow
builds them. The cross-repository policy is the shared `CI Total Time Budget` rule
(`rules/common/docs-and-tests.md`); local validation is unchanged, see the
[local validation table](#validate-locally).

### API naming

Follow the repository vocabulary documented in the [architecture guide](docs/book/src/architecture.md#vocabulary-conventions):

- unsuffixed method for the ordinary operation;
- `_mut` for in-place mutation;
- `_into` when consuming `self`;
- `_batched` for batched input;
- `_owned` only when input ownership enables a distinct optimization.

Use `max_bond_dim: Option<usize>` for bond caps and `SvdTruncationPolicy` for SVD truncation. Dense flat buffers are column-major. Do not add scalar-specific Rust entry points when a generic API works; scalar-specific names belong at the C FFI boundary.

## Validate locally

Choose checks by the changed surface and risk, rather than duplicating hosted
CI before every push. Ordinary local checks use non-release profiles; the
existing `debug = 0` settings remain unchanged.

| Change | Local validation before pushing |
|--------|---------------------------------|
| Crate-local code | Formatting, focused changed-crate regression tests (or `cargo check` for compile-only changes), changed-crate Clippy, and relevant deterministic scripts. |
| Documentation/API | For prose only, check links and consistency. For changed examples/APIs, run affected crate doctests and rustdoc; run `./scripts/test-mdbook.sh` when guide snippets or their APIs change. |
| Cross-crate, manifests, dependencies, features | Expand checks to affected dependents and feature configurations; use the full workspace when the impact is workspace-wide or uncertain. Run boundary checks for dependency changes. |
| Numerical, FFI, unsafe, optimization-sensitive code | Run focused numerical/boundary regressions and relevant audits; include release tests where optimized behavior matters (including unsafe changes), and benchmarks for performance claims. C API changes require the header check and affected binding validation. |
| CI/helpers | Run the changed helper's tests and relevant deterministic checks; reproduce affected jobs as needed. Unknown changes use the conservative code-change tier. |
| Release preparation or hosted-CI failure reproduction | Run the applicable release gates or failing job commands; full local CI-equivalent validation is appropriate for broad failures or explicit maintainer requests. Publication remains human-only. |

Typical crate-local commands (replace the placeholders):

```bash
cargo fmt --all -- --check
cargo test -p <changed-crate> <test_filter>
cargo clippy -p <changed-crate> --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
```

Use `cargo test`, not Nextest, for HDF5 tests. A numerical regression that is
impractically slow without optimization may use release mode with the reason
recorded. Do not routinely run the same tests in both `release` and `ci`.
Numerical tolerances, coverage requirements, unsafe/FFI safety, and release or
publication approvals are not weakened by choosing focused local checks.

### Full validation reference (not an unconditional local gate)

Hosted CI remains required and authoritative for comprehensive tests, coverage,
documentation and feature/backend checks. Coverage and its per-file thresholds
run nightly in `.github/workflows/coverage.yml`, not on pull requests; a failing
night files a `ci-nightly` issue. That workflow pins its Rust toolchain because the
thresholds are measured percentages; bump it in a dedicated PR and re-measure there. Consult `.github/workflows/CI_rs.yml`
for the current commands and platform setup. The following commands are useful
when broader local validation is needed, not as a checklist for every push.
They require `cargo-nextest`, mdBook 0.5.2, the HDF5 development library, and
(for C-header checks) cbindgen 0.29.2.

```bash
cargo fmt --all -- --check
cargo check -p tensor4all-tensorbackend --no-default-features --features explicit-context,tenferro-cpu-faer
cargo doc -p tensor4all-tensorbackend --no-deps --no-default-features --features explicit-context,tenferro-cpu-faer
cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo nextest run --cargo-profile ci --workspace --exclude tensor4all-hdf5
cargo test --profile ci -p tensor4all-hdf5
cargo test --doc --profile ci --workspace -j 8
TENSOR4ALL_CARGO_PROFILE=ci ./scripts/test-mdbook.sh
cargo doc --workspace --no-deps
./scripts/check-capi-header.sh
cargo run -p xtask --release -- api-dump
```

Also run maintenance checks relevant to changed public APIs or crate boundaries:

```bash
python3 scripts/audit-library-panics.py
python3 scripts/check-public-error-docs.py
python3 scripts/check-crate-boundaries.py
```

### Supported feature configurations

Cargo features here are not additive: `tenferro-cpu-faer` and
`tenferro-system-blas` are mutually exclusive CPU backends (tenferro-rs #2004),
so `--all-features` is not a valid build target. The repository declares the
configurations below and builds or audits exactly these. A configuration selects
its packages explicitly; `--workspace --no-default-features` is never used,
because the guide, tutorial, and benchmark packages are documented and built
with their default features.

| Configuration | Cargo selection | Verified by |
|---|---|---|
| default | Each crate's default features | PR CI Test, Doctests, and Lint (`cargo clippy --workspace --all-targets`); PR CI panic audit; nightly heavy-tests |
| explicit-context | `-p tensor4all-tensorbackend --no-default-features --features explicit-context,tenferro-cpu-faer` | PR CI Lint (`Check explicit-context-only build`) |
| system-blas | Every compute crate with `--no-default-features --features tenferro-system-blas`; HDF5 with `link,tenferro-system-blas` | Nightly `.github/workflows/feature-matrix.yml`, including a panic audit |
| provider-inject | Every compute crate with `--no-default-features --features tenferro-provider-inject`; HDF5 with `link,tenferro-provider-inject` | Nightly feature matrix, including a panic audit |
| mpi | `-p tensor4all-partitionedtt --features adaptive-hataori-mpi` (it implies the rayon path) | Nightly feature matrix, including a panic audit |
| diagnostics | `-p tensor4all-treetn -p tensor4all-treeaci --features diagnostics` | Nightly feature matrix, including a panic audit |
| hdf5-runtime-loading | `-p tensor4all-hdf5 --no-default-features --features runtime-loading,tenferro-cpu-faer` | Nightly feature matrix, including a panic audit |
| cuda | `-p tensor4all-core -p tensor4all-treetn --features tenferro-cuda` | Nightly feature matrix, compile only; CUDA execution is verified locally on a GPU only |

The PR CI panic audit compares the default configuration against
`scripts/library-panics-baseline.json`. A non-default configuration reuses that
baseline as an allowlist and does not report unmatched entries as stale, because
it selects a subset of the workspace; pass `--config-name NAME --package ...`
with the Cargo selection flags to audit one configuration:

```bash
python3 scripts/audit-library-panics.py --config-name diagnostics \
  -p tensor4all-treetn -p tensor4all-treeaci --features diagnostics
```

Adding a supported feature configuration means adding it to the matrix in
`.github/workflows/feature-matrix.yml` and to this table.

### Build artifacts

At session start, agents report the total allocated size of registered
worktrees' targets, including configured external targets, once per session.
See [inventory and cleanup](docs/design/build-profiles.md#artifact-cleanup) for
scope, deduplication and unavailable-path reporting. Keep reusable active
output, remove owned disposable output at task/worktree retirement, and review
long-lived targets periodically based on free space. Never automatically delete
shared or another task's output. No fixed machine-specific size limit applies.

Do not commit `target/`, coverage output, generated API dumps, or other build artifacts.

## Open the pull request

- Link the issue with `Closes #…` when the PR fully resolves it.
- Explain the behavior change and why it belongs in the selected crate/layer.
- Summarize validation conclusions and unverified areas; link supporting evidence or special reproduction conditions when needed, not a command history.
- Call out public API breaks, numerical tolerance changes, dense materialization, new dependencies, or cross-repository sequencing explicitly.
- Keep unrelated changes in separate PRs.

Required CI and the repository-rules review must pass. Maintainers normally squash-merge accepted PRs and delete the completed branch.
