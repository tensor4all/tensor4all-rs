# tenferro-rs #2004 consumer migration (work in progress)

Consumer-side migration of `tensor4all-rs` onto the tenferro-rs revision that
contains #2004 (upstream PR #2026, merge commit `b3f472962`). `main` pins
`0457a2ed`, which predates #2004 and several later API changes.

Worktree: `tensor4all-rs-2004-consumer`, branch
`chore/2004-tenferro-cpu-boundary` (off `origin/main` `f11e30d8`). The whole
workspace now compiles against the new revision; the remaining work before the
PR is the hosted CI run and the merge.

## Done

- Pin `0457a2ed…` → `b3f472962…` in the root manifest; `Cargo.lock` regenerated
  as part of the same commit.
- Upstream feature forwarding repointed (`/cpu-faer` → `/native`,
  `/cpu-blas` → `/blas`) in `tensor4all-core`, `tensor4all-simplett`,
  `tensor4all-tensorbackend`. This crate's own feature names
  (`tenferro-cpu-faer`, `tenferro-system-blas`, `tenferro-provider-inject`)
  are kept; only the upstream edges change, so no downstream feature
  selection breaks.
- Session API: `CpuExecutionContext::with_session` and
  `run_canonical_session` return `Result<_, CpuExecutionContextError>`, with a
  new `SessionEntry` variant carrying tenferro's diagnostic. The crate-internal
  `with_default_session` keeps its previous call shape (one tenferro error type)
  so the ~20 existing callers need no change; the two callers whose operation
  has its own error type map the entry rejection explicitly
  (`GroupedGemmError::Backend`).
- `LogicalTensor::from_native` rejects the new `DType::External(_)` variant with
  a typed unsupported tenferro diagnostic instead of an incomplete match.
- `matrix.rs` Hermitian eigenpath uses the runtime-bound borrowed eager session
  (`EagerSessionLinalgExt::eigh`) instead of the removed `EagerTensor::eigh`.
- `context.rs` tests: cross-context eager addition goes through the session
  (`EagerSessionLinalgExt::add`); the caller-managed external CPU domain test was
  removed because tenferro-rs no longer exposes `ExternalCpuDomain` /
  `CpuBackend::from_external_managed_domains` (`resource_domain` is private in
  the new revision; the retained owner-scoped seam is the allocation domain).
  That deletion removes the only exerciser of that removed API, not of retained
  behaviour.

`cargo check -p tensor4all-tensorbackend --all-targets` is clean in both the
default and the `explicit-context,tenferro-cpu-faer` configurations.

Further `tensor4all-tensorbackend` adaptations in the same pass:

- `DType::External(_)` is rejected with a typed diagnostic at every site that
  materializes, promotes, conjugates, scales, does axpby, or names a dtype
  (`tenferro_bridge.rs`, `any_scalar.rs`, `tensor_element.rs`); `dtype_size_bytes`
  documents its zero arm as unreachable because every entry point rejects the
  kind first.
- The explicit-provider grouped GEMM flattens the new
  `Result<Result<…>, SessionEntryError>` into `GroupedGemmError::Backend`.
- `with_default_session` keeps one tenferro error type; the two callers whose
  operation has its own error type (`GroupedGemmError`, `tenferro_einsum::Error`)
  map the entry rejection themselves through the new
  `pub(crate) use defaults::default_context`.
- `tensor.reduce_sum` now takes `Option<&[usize]>`.

## Review round (independent post-review of the finished diff)

The committed diff was reviewed independently before the PR. Findings fixed:

- `CudaExecutionContext`'s `BackendSessionHost` impl still assumed an infallible
  entry; the CUDA configuration did not compile. The wrapper and its
  upload/download callers now report the fallible entry as the new
  `CudaExecutionContextError::SessionEntry` (verified with
  `cargo check -p tensor4all-tensorbackend --features tenferro-cuda` on a host
  with the CUDA toolkit and an A100 present; no device test was run locally).
- Session-entry and external-dtype rejections that were formatted into message
  strings now preserve their typed tenferro cause
  (`anyhow::Error::new(…).context(…)`, `Error::unsupported(…)`), including the
  native einsum entry path and tensor conversion.
- Added focused tests: an external target dtype is rejected as
  `ErrorKind::Unsupported`; the external kind is named `external`; the Rayon
  worker rejection preserves tenferro's typed `Contended` cause; and the
  caller-supplied context stays caller-owned after the context drops (through the
  remaining `CpuBackend::from_context` seam, replacing the coverage the removed
  external-domain test carried).
- The rustdoc of `with_default_session` names the error it actually produces
  (runtime-state classification with the admission diagnostic as source).

Not reproducible deterministically, so not added: a grouped-GEMM admission
rejection test (it needs a second contending thread); the validation-error paths
that leave the output untouched are already covered by the existing grouped-GEMM
tests, and the admission classification is covered by the worker test.

## Known gap found by the hosted CI gate

`python3 scripts/audit-library-panics.py` ran `cargo clippy --workspace
--all-features`, which cannot build any more: tenferro-rs #2004 made the CPU
backend features mutually exclusive, so `--all-features` enables `native` and
`blas` together and upstream rejects it. The audit tool now accepts complete
feature selections through `T4A_PANIC_AUDIT_FEATURE_SETS` (semicolon-separated,
each passed as `--no-default-features --features <selection>`) and the CI job
lists the native selection.

The audited pass therefore leaves out every feature that would select the blas
backend (`tenferro-system-blas`, `tenferro-provider-inject`). The blas
configuration is **not** audited yet: `cargo clippy --workspace
--no-default-features --features tenferro-system-blas` still ends up with both
upstream CPU backends enabled through feature unification between workspace
members, so that configuration does not build workspace-wide. The same applies
to the documented `--no-default-features --features tenferro-system-blas`
benchmark commands in `benchmarks/README.md`. Fixing that (finding the member
that re-enables upstream defaults and making the workspace blas-consistent) is
the follow-up that lets the audit cover the blas configuration again.

## Open: worker session entry is rejected by the new arbiter

The workspace test command CI runs
(`cargo nextest run --cargo-profile ci --workspace --exclude tensor4all-hdf5`)
leaves exactly one deterministic failure after this migration:

    tensor4all-partitionedtt adaptive_interpolation::tests::
      hataori_workers_finish_a_split_interpolation_issue830

Instrumenting the failing operation (temporarily, reverted) shows the error chain
is a session-entry rejection, not a numerical problem:

    matmul error chain: RuntimeStateSource { op: "canonical session entry",
      source: SessionEntry { source: Contended { backend: "CpuBackend",
      message: "a Rayon worker cannot wait for conflicting CPU resources" } } }

Root cause: in the new revision a CPU backend grants its allowed CPU set to one
owner at a time (`tenferro-cpu/src/backend.rs::acquire_execution_permit` →
`arbiter.rs::acquire_request_waiting`), and a Rayon worker may never park behind
another owner, so it is rejected as `Contended`. The interpolation's worker
callbacks reach tenferro through the crate's *process-global* convenience path
(`IdxTensor` ops such as `contract`, which have no context argument), so every
worker session lands on the same default backend and arbiter. Upstream's own
message names the intended shape: "pass the entered session to nested operations
instead of entering the backend again".

A per-entry inline backend in `CpuExecutionContext::inline_backend` does not fix
this (the inner operations still use the global default); that experiment was
reverted.

### Session conclusion: the fan-out is scoped, so borrowing works

Checked against the hataori source (checkout
`~/.cargo/git/checkouts/hataori-rs-344937bffdccfbcf/59d0ccd`):

- `src/local.rs:95-103`: "The callback and input, output, and error types must
  satisfy `Send`/`Sync` as required by Rayon, but **none needs to be `'static`**",
  and `LocalMode::Outer` "evaluates every input exactly once in parallel,
  preserves result order, and reports the lowest callback-error index **only
  after all callbacks have finished**". `map_in` returns the collected results,
  which only exists if every callback completed.
- The consumer uses exactly that: `adaptive_interpolation.rs:688`
  `hataori::map_in(domain, LocalMode::Outer, wave, |patch| …)`.
- `src/pmap.rs:1474-1486`: the MPI entry takes `F: FnMut(T) -> Result<U, E>`
  with no `Send`/`'static` bound and is a synchronous collective; MPI ranks are
  separate processes with separate arbiters, so the cross-rank part is not the
  same problem. Within a rank, `PmapOptions.local_mode = LocalMode::Outer` runs
  local callbacks in the rank's pool, which the same ticket design covers.

Therefore the fan-out is already **scoped** (the caller joins before returning),
and a **borrowed** ticket is expressible: the closure may capture `&Ticket` and
the compiler enforces that no child outlives the parent. The Rust-idiomatic
"borrow first" shape is available for this consumer; an owned (`Arc`) ticket is
only needed for genuinely detached work, which this path does not need.

Remaining upstream decision (unchanged): children still need their own scratch
(work desk) and budget accounting, since the parent's engine resources are
borrowed as a single `&mut` for the whole callback
(`tenferro-cpu/src/backend.rs::run_backend_session_cached`).

Fix plan (option A, agreed with the maintainer):

1. Give the interpolation's contraction path an explicit-context form (the core
   crate already uses `*_in(&context, …)` forms, e.g. `IdxTensor::from_dense_in`;
   add the matching contraction entry if it is missing).
2. In `crates/tensor4all-partitionedtt/src/adaptive_interpolation.rs`, run the
   worker callbacks on a **per-worker** `CpuExecutionContext`
   (`CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?)`) instead of
   the process-global default, so each worker owns a backend and arbiter. This is
   also what `AGENTS.md` asks of canonical integrations: supply a configured
   `CpuBackend`, keep `with_default_*` for the legacy convenience surface.
3. Verify with the failing test, the `hataori_outer_matches_sequential_and_allows_nested_rayon`
   test, and the full workspace test command.

Correction to an earlier record: the `plain_tensor_retention` failure is *not*
reproducible locally. That claim came from running `cargo nextest run -q`, where
`-q` is not a nextest flag, so the test never ran; with the correct command the
original test passes 20/20 locally and also in the full workspace run. The test
file is untouched, and the earlier cross-thread "under-counting" theory is
withdrawn: the allocator subtracts for tagged blocks on any thread, so the CI
report is a separate, rarer effect to reproduce before changing anything.

### Upstream scope reduced: a reentrant permit already gets its own resources

Verified in the vendored revision: `tenferro-cpu/src/backend.rs:1660-1682`
(`with_execution_resources`) already branches on the permit:

    if permit.is_reentrant() {
        let mut resources = EngineResources::new(self.shared.buffer_limit.load(Relaxed));
        return op(&mut resources);
    }
    let mut resources = self.engine.resources.lock()...;

so a session admitted as **reentrant** (same owner) builds its **own**
`EngineResources` — its own buffer pool and caches, bounded by the configured
buffer limit — and never takes the shared `engine.resources` lock. The child
scratch question is therefore already answered by the existing design: inheriting
the owner is enough, and no workspace plumbing is needed.

The upstream change shrinks accordingly to an **API addition only**:

1. expose the active execution's owner as a *borrowed* ticket
   (`CpuBackend::current_ticket() -> Option<Ticket<'_>>`), and
2. let a child enter with it on the current thread
   (`Ticket::with_session(&self, f)`) so the arbiter admits it reentrant.

The arbiter policy (foreign owners are rejected, workers never park) is
unchanged, and the per-child resource cost is bounded by the buffer limit that
the child's `EngineResources::new` receives.

## Landing decision (this PR)

`adaptive-hataori-rayon` is no longer a default feature of
`tensor4all-partitionedtt`; the worker path is documented in `Cargo.toml` and in
this record as known-broken until tenferro-rs can hand a child ticket to workers
(the borrowed-ticket design above). The feature still compiles and its tests stay
under `#[cfg(feature = ...)]`, so they return with the fix. The alternative
(marking the single failing test `#[ignore]`) was rejected because it would keep
a deterministically failing runtime path enabled.

## Status

Local gates on the submitted commit: `cargo check --workspace --all-targets`,
`cargo check -p tensor4all-tensorbackend --no-default-features --features
explicit-context,tenferro-cpu-faer`, `--features tenferro-cuda`,
`cargo fmt --all -- --check`, clippy with
`-D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc`,
`cargo nextest run -p tensor4all-tensorbackend -p tensor4all-core` (1136 passed),
and doctests for both crates (494 passed). Hosted CI and the merge remain.
