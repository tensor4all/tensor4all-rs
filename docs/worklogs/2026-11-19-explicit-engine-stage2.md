# Explicit engine stage 2: `tensor4all-core` off `global-defaults` (#1403)

Part of #1376. The durable record is the #1403 status note in
[docs/design/explicit-engine-migration.md](../design/explicit-engine-migration.md) §4.1.

## Decisions

- **One choke point with a runtime refusal, not a compile-time gate on the implicit
  surface.** Core's owner-less value adoption (F5) sits inside `IdxTensor`'s
  materialization cache, and the explicit `*_in` paths use implicit value methods on
  context-owned values internally. Gating everything that can reach the global runtime
  would have removed almost all of `IdxTensor` from an explicit build, and the explicit
  paths with it, so the flip would have required phases B2, B3 and D2 first. Instead every
  process-global touch in core goes through `core/src/implicit_route.rs`: with
  `global-defaults` its items *are* the tensorbackend global functions (re-exported, so
  the global build runs unchanged code); without it they refuse with
  `ExecutionContextEntryError::NoContext`. Rejected trade-off, stated in the PR: an
  explicit-only build compiles context-free entries that fail at run time with a typed
  error, rather than failing to compile. The design doc's bridges are compile-time; this
  is the reviewer decision to confirm.
- **Downstream library code uses the same module as a `#[doc(hidden)]` bridge**
  (`tensor4all_core::legacy`), not `tensor4all_tensorbackend` globals or a feature of its
  own, so a downstream crate and core cannot disagree about whether the global route
  exists. The `execute_reads` method and the gated `FullPivLuScalar` method get free
  function bridges.
- **The owner registry and `with_runtime_engine` compile under `explicit-context`.** Only
  the fallback to the global context is gated. This keeps implicit value methods working
  on values a caller-owned context created (same engine as today) and is what the
  explicit `_in` paths rely on. D2 still deletes the route.
- **Two explicit paths that reached the global route are threaded, not bridged**:
  `contract_structured_payloads_nary_in`'s host-operand route (native einsum and the
  result's runtime) and `src_error_estimate_in`'s host estimator
  (`src_error_estimate_general_in`). The CUDA-resident estimator readback has no caller
  CPU engine and stays on the bridge, like the treersi A3 arm.
- **The packed-site norm keeps its backend kernel in every build** (review of #1404).
  `tensor4all-itensorlike` chose between the backend einsum and a host reference loop by
  its own `backend-tenferro`, which the provider features used to enable; dropping that
  implication silently switched default builds to the reference loop. A first fix keyed
  the choice on `legacy::GLOBAL_DEFAULTS`, which still sent the default (explicit-only)
  build to the loop. Now the one kernel runs on an explicit `Session` over the CPU engine
  of the context that owns the chain (new `IdxTensor::with_owning_cpu_engine`), and only a
  chain with no CPU-owned value takes the implicit route (global context, or `NoContext`).
  The loop is a `cfg(test)` oracle. On the global context the session kernel is bitwise
  equal to the former graph einsum (30 f64/c64 chains, bond dimension up to 64; a
  throwaway comparison, not committed).
- **An untracked `AnyScalar` joins its operand's runtime.** Without `global-defaults` an
  `AnyScalar` cannot create its own rank-0 tensor, so `IdxTensor::scale` (and with it
  `TensorTrain::norm`) failed even on context-owned values. `scale` now places such a
  scalar in the payload's runtime; a scalar with its own tensor is used as before.
- **Explicit-only behaviour is tested in an isolated package**,
  `tests/feature-configurations/explicit-only` (its own workspace, so no dev-dependency
  can unify `global-defaults` back on). A `cfg(not(feature = "global-defaults"))` unit
  test inside core never compiled, because core's dev-dependency enables the feature.
- **Test, doctest and bench builds opt in through a dev-dependency** (a self
  dev-dependency in core and treersi). It is the only mechanism that also covers
  doctests; `required-features`/`cfg` gates would have dropped the 426 core doctests and
  most downstream tests from the default `cargo test`. `scripts/check-crate-boundaries.py`
  allows exactly these two feature-only self edges and fails if either crate's sources
  name it by its own extern path.

## Verification conclusions and constraints

- Library builds of tensorbackend, core and the twelve algorithm crates contain no
  `global-defaults` in the normal dependency graph and pass `clippy -D warnings`; CUDA
  builds of core, treetn and treersi check. `tenferro-cuda` implies `global-defaults`
  (review of #1404): CUDA execution keeps host steps with no caller CPU context.
- With `global-defaults` (every test build) the code path is unchanged except the two
  threaded explicit paths, which now run on the caller's engine (same kernel; thread count
  is the caller's). No parity harness was run for those two paths.
- The explicit-only build's run-time behaviour is covered by tensorbackend's
  explicit-surface test and the isolated `explicit-only` package (bridge refusals, typed
  `NoContext` on context-free construction and on a dropped context, context-owned
  contraction and `TensorTrain` norm). Not run locally: `tensor4all-hdf5`, `book-tests`
  (no libhdf5 on the host) and CUDA execution.
