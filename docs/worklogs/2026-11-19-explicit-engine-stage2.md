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
- **A crate-local feature that selected an implementation follows core's feature now.**
  `tensor4all-itensorlike` chose its packed-site norm (backend einsum versus a host
  reference loop) by its own `backend-tenferro`, which the provider features used to
  enable. Dropping that implication silently switched default builds to the reference
  loop, and a tiny-scale CG test broke down. The choice now reads
  `tensor4all_core::legacy::GLOBAL_DEFAULTS`, so each configuration keeps the
  implementation it had. No other workspace code keys an implementation on a feature
  whose default changed.
- **Test, doctest and bench builds opt in through a dev-dependency** (a self
  dev-dependency in core and treersi). It is the only mechanism that also covers
  doctests; `required-features`/`cfg` gates would have dropped the 426 core doctests and
  most downstream tests from the default `cargo test`. The explicit-only configuration is
  proven by library builds, a CI guard on the normal dependency graph, and tensorbackend's
  explicit-surface test.

## Verification conclusions and constraints

- Library builds of tensorbackend, core and the twelve algorithm crates contain no
  `global-defaults` in the normal dependency graph and pass `clippy -D warnings`; CUDA
  builds of core, treetn and treersi check in both configurations.
- With `global-defaults` (every test build) the code path is unchanged except the two
  threaded explicit paths, which now run on the caller's engine (same kernel; thread count
  is the caller's). No parity harness was run for those two paths.
- Unverified: run-time behaviour of the explicit-only build beyond tensorbackend's test
  and the refusal unit tests, because every test build enables `global-defaults`.
  `tensor4all-hdf5`, `book-tests` (no libhdf5 on the host) and CUDA execution were not
  run.
