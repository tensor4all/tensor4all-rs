# Core rejection of unsupported execution contexts (#869)

**Status:** implemented. Scope is the `tensor4all-core` side of
[tensor4all-rs#869](https://github.com/tensor4all/tensor4all-rs/issues/869). It does not
implement any part of [#859](https://github.com/tensor4all/tensor4all-rs/issues/859)
(the opt-in explicit/concrete frontend) and does not change feature forwarding anywhere.

## Problem

`tensor4all-core` gated the CUDA arms of its `ExecutionContext` matches on **its own**
`tenferro-cuda` feature, while the `ExecutionContext::Cuda` variant exists according to
**`tensor4all-tensorbackend`'s** `tenferro-cuda` feature. A crate cannot `#[cfg]` on a
dependency's feature, so enabling `tensor4all-tensorbackend/tenferro-cuda` without
`tensor4all-core/tenferro-cuda` left five matches non-exhaustive:

```text
cargo check -p tensor4all-core --no-default-features \
  --features tenferro-cpu-faer,tensor4all-tensorbackend/tenferro-cuda
error[E0004]: non-exhaustive patterns: `&ExecutionContext::Cuda(_)` not covered  (x5)
```

The reverse mix is unreachable: `tensor4all-core/tenferro-cuda` enables the backend
feature. `tensor4all-treetn` forwards both feature names together, so core was the only
crate with the gap.

Beyond the compile error, several context-aware entry points had a *silent* non-CPU path
in a build without CUDA support: `src_error_estimate_in` fell through to the host SRC
estimate, and `read_decision_data` fell through to the host decision values. Both were
unreachable in practice (no CUDA-resident tensor can be constructed without the CUDA
arms), but neither was expressed as a contract.

## Decision

Support the mixed configuration as far as a CPU context needs to go, and reject the
operations that need the missing feature with a typed error:

| core CUDA | backend CUDA | behaviour |
| --- | --- | --- |
| off | off | unchanged CPU behaviour |
| off | **on** | builds; CPU contexts work; a CUDA context is rejected by a typed error |
| on | on | unchanged CPU/CUDA behaviour |
| on | off | unreachable with the current feature forwarding |

Contract of the rejection:

- `IdxTensorError::UnsupportedExecutionContext { required_feature }` with
  `required_feature = "tensor4all-core/tenferro-cuda"`, so the cause is identifiable
  without string parsing; `FactorizeError::ComputationError` keeps it reachable through
  `downcast_ref` on the source chain.
- Raised before runtime initialisation, transfer, upload, reduction, or output mutation,
  and before any fallback to host storage or a process-global default context.
- No panic and no `unreachable!()`; the non-CPU arm is a variant-agnostic `_` pattern, so
  no non-CUDA code names the CUDA variant.

Implementation: `ensure_supported_execution_context` / `unsupported_execution_context_error`
/ `unsupported_factorize_context_error` in `crates/tensor4all-core/src/defaults/idx_tensor.rs`,
called at `from_dense_in`, `ones_in`, `context_scalar_in`, `read_resident_rank`, and
`validate_context` before its storage branch, plus a rejection arm on the five matches.
The `_` arms are unreachable in a build where the backend has no CUDA feature either, so
they carry a targeted `#[allow(unreachable_patterns)]` and an adjacent comment instead of
a crate-wide suppression.

The now-redundant `tensor4all-core/backend-tenferro` alias is retained: `treetn`, `aci`,
`hdf5` and `py` reference it. Retiring it is separate cleanup.

## Verification

- CUDA-free: `crates/tensor4all-core/tests/context_foundation.rs` and the
  `unsupported_context_*` unit tests cover the error classification, the `FactorizeError`
  source chain, the absence of eager-runtime initialisation in the support check, and the
  CPU known-value regressions (construction, decision readback, scaling, norm, f64 and
  `Complex64`).
- Mixed configuration: `tests/feature-configurations/core-backend-cuda` is an isolated
  consumer (separate workspace and lockfile, not a workspace member, no permanent CUDA
  dev-dependency) that pins `tensor4all-core` without CUDA and
  `tensor4all-tensorbackend` with `tenferro-cuda`, proves the effective feature set with
  `verify_features.py`, and exercises the rejection with a real CUDA context on a machine
  with the CUDA toolkit. It covers materialized and eager storage, checks after every
  rejection group that the CUDA eager runtime stayed uninitialised and the operand
  unchanged, and moving the guard below `validate_context`'s storage branch fails five of
  its seven tests. Its README records the exact commands and required environment.
- No CI job builds CUDA, so the mixed-configuration type check and the rejection behaviour
  are verified locally; CPU CI only covers regression safety.

### Unverifiable row of the table above

The "core CUDA on" configurations (`on/on`, and the unreachable `on/off`) cannot be
verified at this revision: `cargo check -p tensor4all-core --features tenferro-cuda` fails
with four pre-existing `E0599` errors on `&EagerTensor` extension methods (`transpose`,
`abs`, `triu`), identical before and after this change. Tracked separately; the
corresponding CUDA-enabled regression tests therefore cannot run against this revision.
