# Explicit CPU execution context

**Status:** Implemented for tensor4all-rs#663

> Updated 2026-10-21 for tenferro's caller-held engine model (tenferro-rs #2206).

## Goal

Make one caller-supplied `tensor4all_tensorbackend::CpuBackend` the root of plain
tensor, graph, and eager-AD execution without consulting process-global
defaults. Tensor4all does not define an executor abstraction; Rayon is used only
to recognize an entry thread that runs inside another pool.

## Public boundary

`tensor4all_tensorbackend::CpuExecutionContext::from_backend` consumes the
caller-selected backend. A `CpuBackend` owns one `tenferro_cpu::CpuEngine` and
is not `Clone`: the engine has exactly one owner at a time, and the context
becomes that owner. Construction never calls `CpuBackend::new`,
`CpuContext::from_env`, or placement resolution. The context owns:

- one `CpuEngine` behind a mutex, used for plain operations, graph execution,
  and session entry;
- a graph compiler, graph runtime, and prepared-plan cache, shared with the
  engine's session entries;
- an eager runtime, built lazily on first use. It is engine-independent: it
  holds no engine, and each eager operation takes the engine it runs on.

The context exposes explicit methods for borrowing the backend
(`with_backend(|&mut CpuBackend| ..)`), entering the engine
(`with_session(|&mut CpuEngine| ..)`), compiling and running graph programs, and
returning the shared eager runtime (`eager_runtime()`, an `Arc` handle). These
methods are the canonical explicit integration surface.

Eager tensors do not carry an engine. An implicit eager operation resolves the
engine from its operand's eager runtime through the owner table
(`with_runtime_engine`), which maps each runtime to the context that created it;
runtimes outside any explicit context resolve to the process-global default
context.

A session entry is closure-scoped: the engine borrow lasts for the whole
closure. A nested entry on the same thread into the same context returns the
typed `CpuExecutionContextError::Reentered` instead of waiting for the engine,
and `with_backend` also reports nested session entries that way. Entries from a
Rayon worker run inline on a serial engine that shares the context's plan cache,
instead of installing the supplied engine's pool inside the caller's Rayon pool;
entries from any other thread use the supplied engine. To run several lanes in
parallel, the caller holds one engine per lane (for example
`TenferroLanes::new(Vec<CpuEngine>)` in hataori); there is no shared engine
across lanes. Existing higher-level
`IdxTensor`, matrix, and bridge functions remain opt-in global convenience APIs;
an explicit host adapter must call the context surface (and may wrap values in
its own application type) rather than those convenience functions. Graph IR
and `tenferro::CompiledGraph` remain backend-neutral; backend-prepared
executables, workspaces, and caches stay private to the context-owned runtime.

A plain concrete session entered from a Rayon worker runs inline and
single-threaded on a context-local pool-less CPU backend instead of installing
the supplied backend's pool inside the caller's Rayon pool; entries from any
other thread use the supplied backend. See
[tensorbackend session entry](tensorbackend-session-entry.md) and
[tensor4all-rs#830](https://github.com/tensor4all/tensor4all-rs/issues/830).

Separately created contexts never share a tensor4all mutex or cache. Backend
identity checks remain tenferro's responsibility and errors are preserved as
typed sources.

## Global convenience path

The existing process-global helpers remain only behind the opt-in
`global-defaults` Cargo feature. The normal workspace enables that feature for
compatibility. An integration can depend on `tensor4all-tensorbackend` with
`default-features = false` and select a CPU provider feature without compiling
any `DEFAULT_*`, `from_env`, or global backend mutex path. The public context is
re-exported from `tensor4all-tensorbackend`; no C API change is made.

Explicit methods do not call global helpers. The global wrappers are thin
adapters over the same context implementation, avoiding a second operation
implementation.

### Feature-gating inventory

`explicit-context` compiles `context` without its default submodule plus the new
`logical_tensor` module. Its public surface is
`CpuExecutionContext`, its typed construction/execution errors,
`LogicalTensor`, `LogicalTensorData`, and their typed error. This is the small
integration API for plain backend borrowing, graph compile/run/cache, eager
runtime access, and target-context reconstruction.

`global-defaults` additionally compiles the legacy `any_scalar`, `backend`,
`matrix`, `memory`, `storage`, `tenferro_bridge`, and `tensor_element` modules
and their current re-exports. `backend-tenferro` remains a compatibility alias
for `global-defaults`, and the crate's default features keep enabling it. This
coarse module boundary is deliberate: it leaves no legacy operation that can
reference a default in an explicit-only build and avoids scattered per-function
cfg gates.

CI checks both isolated code and docs:

```text
cargo check -p tensor4all-tensorbackend --no-default-features \
  --features explicit-context,tenferro-cpu-faer
cargo doc -p tensor4all-tensorbackend --no-deps --no-default-features \
  --features explicit-context,tenferro-cpu-faer
```

A source-isolated build proves graph and reconstruction code cannot reference
the default module; a runtime hit-counter test separately covers explicit plain
and eager initialization. Other tests exercise graph/cache, reconstruction,
parallel-context, and drop/fresh-cache behavior. The cross-context test creates eager tensors in two
separate `EagerRuntime` values and calls `EagerTensor::add`, which tenferro
rejects specifically with `tenferro_ad::Error::ContextMismatch`; no plain host
tensor is claimed to be context-bound.

## Logical transfer

`LogicalTensor` is a new owned, canonical column-major host snapshot containing
only shape and dtype-preserving scalar data for all tenferro CPU dtypes (`f32`,
`f64`, `i32`, `i64`, `bool`, `Complex32`, and `Complex64`). It is distinct from
`Storage`, whose scalar contract intentionally covers only `f64` and
`Complex64` structured payloads. It contains no backend, executor, runtime,
cache, engine, pointer, or address identity. Reconstruction is a
method on `CpuExecutionContext`, making the receiving context mandatory; shape,
element-count, and dtype failures are typed bridge errors. The adapter owns
serialization and MPI transport.

## Errors and lifecycle

Construction and graph execution use crate-local typed errors with original
tenferro errors retained as sources. Dropping the last context owner drops its
engine, graph runtime/cache, and eager runtime; the engine was moved into the
context, so no other handle keeps it alive. A fresh context has a fresh
runtime/cache and cannot recover stale prepared entries. Tests cover both
properties.

## Non-goals

- Hataori, MPI, or serialization dependencies;
- a generic executor trait, TLS override, or process-global context registry;
- serializing compiled programs or provider workspaces;
- reintroducing a held session object or admission token: tenferro's
  caller-held engine model (tenferro-rs #2206) replaced the admission-based
  session boundary, and the caller owns the engine lifetime directly.
