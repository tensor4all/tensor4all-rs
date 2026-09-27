# Build Profiles

## Goal

Keep ordinary tensor4all-rs development and verification builds small while retaining an explicit full-debug release profile for debugger sessions and detailed Rust diagnostics through Tensor4all.jl or the C API.

## Workspace dependency lock

Commit the root `Cargo.lock` to keep workspace CI and contributor builds on the
same resolved dependency graph, and pass `--locked` to CI Cargo build and test
commands. Refresh dependency versions deliberately with `cargo update` in a
reviewed change. The lockfile governs this workspace; downstream projects that
depend on an individual tensor4all crate continue to resolve their own graph.

## Profile contract

Ordinary profiles generate no Rust debug information:

```toml
[profile.dev]
debug = 0

[profile.test]
debug = 0

[profile.release]
debug = 0
```

Debug assertions and overflow checks remain enabled for dev and test builds. Release optimization and runtime semantics are unchanged.

Full source-level release diagnostics use a separate profile:

```toml
[profile.release-debug]
inherits = "release"
debug = true
```

Build the C API for a Rust debugger or file-and-line-rich backtrace with:

```bash
cargo build --profile release-debug -p tensor4all-capi
```

Set `RUST_BACKTRACE=1` or `full` when running the Tensor4all.jl/C API caller. Keeping this output under `target/release-debug` avoids rebuilding or retaining full debuginfo in the ordinary release profile.

One-command dev/test debugger overrides remain available through `CARGO_PROFILE_DEV_DEBUG=2` and `CARGO_PROFILE_TEST_DEBUG=2`.

Comprehensive CI uses release optimization without retaining incremental state
or linker symbols:

```toml
[profile.ci]
inherits = "release"
incremental = false
strip = "symbols"
```

Run release-equivalent CI tests locally with `cargo test --profile ci` or
`cargo nextest run --cargo-profile ci`. Coverage remains on its
instrumentation-owned profile and does not use `ci`.

Coverage includes ordinary library artifacts, including the C API shared
library. Check the full C API package, including unit and integration tests,
when validating coverage locally:

```bash
cargo llvm-cov nextest --locked --release -p tensor4all-capi --json --output-path coverage.json
```

C API entry points use `#[cfg_attr(not(test), unsafe(no_mangle))]`. This retains
C ABI symbols in the ordinary library while giving unit-test functions distinct
symbols, so LLVM can combine both builds without collisions between coverage
records. Integration tests exercise the ordinary library through its public
boundary; unit tests also cover private validation and error paths. Hosted
workspace coverage and the unchanged per-file thresholds remain authoritative.

## CI artifact reuse and guide verification

The workspace lockfile and `--locked` keep dependency resolution stable across
CI jobs. Rust artifacts are restored with workspace crates included, but a
cache hit alone does not establish reuse: Cargo still checks profile, features,
compiler flags, dependency fingerprints, and source timestamps. A fresh checkout
can make unchanged workspace source newer than restored dep-info and trigger
recompilation while registry and Git dependencies remain fresh. Do not rewrite
source timestamps or change cache keys merely to make the hit indicator green.

Test, Doctests, and Coverage retain `build-diagnostics-*` artifacts for seven
days. Their JSON summaries and timestamped Cargo logs distinguish `Fresh`
artifacts from compilation and record the build-completion boundary. Cargo
`--timings` reports complement these logs; command time after that boundary
includes execution and reporting, not just tests. See the
[CI reuse measurements](../experiments/ci769-build.md) for an observed case and
runner-time accounting conventions.

Both the Cargo book harness and standalone mdBook configuration select Rust
2024, allowing eligible examples to share compiled doctest executables. Keep
these two edition settings aligned when changing the book's test strategy;
changing only a code fence does not migrate the Cargo harness.

CI and Pages verification pass `TENSOR4ALL_RUSTDOC_LOG` from the preceding
verbose workspace rustdoc run to `scripts/test-mdbook.sh`. This preserves exact
`--extern` paths without a second preparation build. Standalone calls prepare a
`book-tests` doctest log automatically. An explicitly supplied nonexistent log
is an error. Native HDF5 link paths are forwarded in both modes.

## Rationale

Ordinary local checks use non-release profiles; comprehensive CI and benchmarks
use optimized builds. Choose local checks using the
[change/risk table](../../CONTRIBUTING.md#validate-locally), not full CI parity.
Full debuginfo or line tables multiply across independently linked test and
example executables. A controlled full-workspace measurement found that changing ordinary release from line tables to `debug = 0` reduced allocated target output from 14,759,100,416 to 3,354,374,144 bytes (77.27%); see `docs/worklogs/2026-08-09-release-debug-info-reduction.md`.

Most development does not debug Rust through Tensor4all.jl or the C API and does not benefit from this metadata. The named profile preserves the diagnostic capability without imposing its storage cost on every release build.

## Artifact cleanup

Profile settings reduce individual builds but do not delete older dependency,
feature or profile variants, nor prevent duplication across worktrees.

### Session-start inventory

Once at the start of each agent session (not on every `AGENTS.md` read):

1. Enumerate this repository's registered worktrees with
   `git worktree list --porcelain`. Include each existing `<worktree>/target`,
   even if its current configuration points elsewhere.
2. Resolve each worktree's effective target using
   `cargo metadata --offline --no-deps --format-version 1` from that worktree
   and read `target_directory`. This accounts for the current environment and
   Cargo configuration without compiling or downloading dependencies. Include
   known task-specific `--target-dir` or `CARGO_TARGET_DIR` paths as well.
3. Canonicalize paths, count shared targets once, and omit nested targets
   already covered by a parent. Measure allocated disk usage with `du -sk`
   (or the platform equivalent), sum it, and report the total and largest paths.
4. Report inaccessible paths, missing worktrees, metadata failures or timed-out
   measurements explicitly; label the total partial if anything is unmeasured.
   Missing target directories contribute zero. Do not build to obtain a number.

This is a scoped inventory, not a whole-disk search: unregistered checkouts and
unknown historical external targets are not included. A shared target may also
contain other repositories' artifacts; report that limitation rather than
claiming exact attribution. Do not delete anything as part of this report.

### Ownership and cleanup

- Before a large build, identify the effective target and its owner (the task
  or developer), whether it is shared, and whether its output will be reused.
  Do not build every worktree merely because it exists.
- Keep reusable output for active worktrees; cleaning on every edit or push
  wastes subsequent rebuilds.
- Give one-off large builds a dedicated target when practical and remove their
  owned disposable output at completion. Before retiring a disposable worktree,
  remove its dedicated build output, including external targets; removing the
  worktree alone does not remove external output.
- Review long-lived and self-hosted targets periodically and before large builds
  when free space is low relative to expected build growth, or when stale
  variants dominate. Propose cleanup rather than imposing a fixed GiB limit.
- Clean only output owned by the completed task, after confirming no build uses
  it. Shared or other tasks' targets require owner approval; if ownership is
  unclear, ask rather than delete. Report deferred cleanup at handoff.

For an identified target, `cargo clean --target-dir <path> --profile release`
removes that profile's output; `cargo clean --target-dir <path>` discards all
Cargo build output there when a cold rebuild is acceptable. Check the path and
ownership first. No automatic deletion or background cleanup service is needed.

## Compatibility and non-goals

Optimization, numerical behavior, public APIs, ABI, feature selection, and release assertions are unchanged. Ordinary release backtraces may lack Rust source file/line detail. This policy does not introduce shared worktree targets, automatic artifact garbage collection, sccache, or dependencies. Existing target directories retain historical variants until explicitly cleaned.
