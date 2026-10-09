# Worker fan-out on the explicit CPU execution context (issue #2004 follow-up)

Design note (revision 2, after independent review of revision 1) for restoring
`adaptive-hataori-rayon`.

## Context

PR #851 moved tensor4all-rs onto the tenferro-rs revision containing #2004
(merge `633fd264`). That revision arbitrates host CPU sets process-wide and
rejects a session entry a Rayon worker cannot wait for
(`SessionEntryError::Contended`). The interpolation's worker callbacks reach
tenferro through this crate's *process-global* convenience path, so every worker
session lands on the same default backend and arbiter and is rejected. #851
therefore removed `adaptive-hataori-rayon` from the default features of
`tensor4all-partitionedtt` and kept
`hataori_workers_finish_a_split_interpolation_issue830` under `#[cfg(feature = …)]`.

Upstream PR #2044 adds the missing capability: `CpuExecSession::child_execution()`
gives a borrowed handle whose `backend()` is admitted reentrant under the issuing
execution's owner, and each child session owns its N-ary contraction scratch.

## Revision 1 rejected two things (review findings)

1. **A child context per fan-out cannot work.** `CpuExecutionContext` caches one
   `Arc<EagerRuntime>` (`context.rs`), whose backend is a single-owner mutex on
   Rayon workers (`tenferro-ad/src/eager.rs`: competing entries `try_lock` and
   return `Contended`). Two workers sharing one child context would fail before
   arbiter admission is reached. → **one child context per worker callback**,
   reused for that callback's own operations.
2. **Value ownership must be explicit.** A child context has its own eager
   runtime identity, so parent-context tensors do not validate against it
   (`IdxTensor::validate_context`), and tensors created in a child runtime carry a
   backend whose issuing owner becomes stale once the fan-out ends. → callbacks
   work **only on values they build themselves in their own child context**, and
   the fan-out converts values at its boundary.

## Design

1. **Explicit context in.** `adaptiveinterpolate_in` takes an `ExecutionContext`
   parameter (change the signature in place; this repository keeps no
   compatibility shims and the review recommended the direct change). The caller
   supplies a configured context; nothing in the interpolation consults the
   process-global default any more.
2. **Fan-out inside one execution, child context per callback.** The fan-out
   (`hataori::map_in` / `hataori::pmap`) runs inside one entered session of that
   context. The issuing thread derives the child handle there and passes it (by
   reference, as a `Send + Sync` handle) into the closure. **Each callback builds
   its own child `CpuExecutionContext`** from `handle.backend()` and closes its
   work in it: operands are constructed from host values with the explicit
   `_in(&context, …)` constructors, the contraction runs through the explicit
   context, and the result is taken out as a plain value.
3. **Value boundary.** Only plain (non-tracked) host values cross the fan-out
   boundary: the callback returns extracted host data, and the issuing side
   rebuilds parent-context values from it. AD-tracked tensors never cross: if a
   future caller needs that, it is a separate design (explicit value × execution
   seam). The interpolation's tensors are plain today, so this costs an explicit
   copy at the boundary and nothing else.
4. **Contraction with an explicit context.** `contract_in(&ExecutionContext, …)`
   is added next to `contract`, implemented by giving the storage operations on
   the contraction path a context-taking seam instead of `with_default_session`.
   The seam stays crate-internal; the public shape follows the existing
   `factorize_in` / `norm_in` / `from_dense_in` pattern. A scoped thread-local
   default override is rejected: it would make the execution ambient again, which
   is the mechanism this migration removes. The matrix API path the review called
   out (`mat_mul_owned` → `dot_general_matrices` → `with_default_session`) is part
   of the same seam, not a separate mechanism.
5. **Restore the feature.** `adaptive-hataori-rayon` returns to the default
   features, the interpolation regression test runs again, and the tenferro pin
   moves to the merged upstream commit containing #2044.

## Acceptance criteria

- The interpolation regression test passes with the feature enabled, for one and
  two domain workers, with a context-aware scalar and batch callback, a known-value
  **N-ary** contraction in the target evaluation path, all expected patch outcomes
  and numerical results, and returned tensors usable under the original context
  after the fan-out has ended.
- Simultaneous child numerical work succeeds (no partial `Contended` results), and
  the binary, N-ary, mixed-dtype and structured paths in the callback are covered.
- Context mismatches are rejected before any clone/no-op shortcut; a callback
  error or panic releases admission and a later independent operation succeeds.
- `contract_in` matches `contract` numerically **and** the caller-side context
  identity is preserved (the result validates against the supplied context), which
  is the behavioural check that the global default was not used; a
  counter-based criterion (`default_context_hits()` unchanged) is explicitly
  rejected as insufficient.
- The workspace test command CI runs
  (`--cargo-profile ci --workspace --exclude tensor4all-hdf5`) reports zero
  failures with the feature enabled; clippy with the repository flags,
  `cargo fmt --all -- --check`, affected doctests, and the MPI feature compiles.

## Open questions

- Whether the callback's child context should also serve the callback's *batched*
  entry (`batched_f`) or whether that path keeps its own context.
- How much of the value-boundary copy can be avoided later without reintroducing
  ambient execution state.
