# Dual-frontend coexistence in `tensor4all-tensorbackend` (#859 B0)

**Status:** baseline and specification only. This record delivers milestone B0 of
[tensor4all-rs#859](https://github.com/tensor4all/tensor4all-rs/issues/859):
"inventory existing public calls/features/representations and conversion
boundaries; record current remote revisions and workloads; specify the new opt-in
surface and legacy-preservation tests before implementation". It does **not**
implement the new frontend (B1), the GPU/transfer substrate (B2), the explicit AD
surface (B3), or any app follow-up (F1–F7), and it does not start the retirement
step (R).

Names in [§5](#5-specified-new-opt-in-surface-b1) are provisional. The issue
states that the new surface must be finalized with the backend API and must not be
a redefinition of an existing public method, so the exact spelling of the items
below is an open decision for the B1 change, not a commitment made here.

## 1. Recorded baseline

| Item | Recorded value |
| --- | --- |
| `tensor4all-rs` base | `origin/main` `e0a63a31a641fe13dc1b2dfe2b05e17c0e94a2c5` (2026-10-09) |
| `tenferro-rs` pin in the workspace `Cargo.toml` | `b3f47296244ff7b7c55ac0a75f782cb0835418c1` (2026-10-08) |
| Pin identification | merge of tenferro-rs PR #2026 `refactor/2004-cpu-boundaries`, i.e. the admission-based CPU session boundary landed by tenferro-rs #2004 |
| `tenferro-rs` upstream revision inspected for the dependency claims below | `origin/main` `763ba4c5034f952ef33601b14dd307ce2dc1ea8a` (2026-10-09) |
| Public API inventory | generated, never committed: `cargo run -p xtask --release -- api-dump` → `target/api-dump/tensor4all_tensorbackend.md` (978 lines at this revision) |
| Workspace consumers | 12 other workspace crates depend on `tensor4all-tensorbackend`: `aci`, `capi`, `core`, `interpolativeqtt`, `itensorlike`, `partitionedtt`, `quanticstci`, `simplett`, `tensorci`, `treeaci`, `treetci`, `treetn` |
| Hosted workspace test job | `cargo nextest run --locked --cargo-profile ci --workspace --timings -v` (`.github/workflows/CI_rs.yml`, "Run workspace tests") |
| Hosted coverage job (HDF5-excluded) | `cargo llvm-cov nextest --locked --release --no-clean --workspace --exclude tensor4all-hdf5 --json --output-path coverage.json` (same workflow) |

Two upstream facts constrain B1 and are recorded here so the dependency is
explicit rather than discovered mid-implementation:

1. **No held session exists upstream, at this pin or at the inspected upstream
   `origin/main`.** Every tenferro session type is lifetime-borrowed and only
   reachable from inside a callback: `BackendSessionHost::with_backend_session`
   returns `Result<R, SessionEntryError>` and yields `&mut dyn BackendSession`
   (`crates/tenferro-tensor/src/backend.rs`), with `CpuExecSession<'a>` /
   `EagerSession<'a>` as the concrete sessions. No public constructor returns a
   session object usable across calls (no `open_session`; the string `holdable`
   does not occur anywhere in tenferro-rs source). A session is therefore always
   bound to the callback that opened it.
2. **The upstream redesign record defers the evaluation-wide (A2) scope.**
   tenferro-rs `docs/design/tensor-session-redesign-1938.md` decision **D12**
   states "A2 is deferred, not solved … Use the existing borrowed session at
   named boundaries"; `docs/design/explicit-session-boundary.md` and
   `docs/design/exec-session.md` record the same deferral. D12 is the A2
   decision, not by itself a statement about every possible held-session API; the
   availability claim above rests on the inspected public surface, and the
   scheduling claim on tenferro-rs #1945 (holdable CPU/CUDA sessions), which is
   open and says it "is not scheduled work and does not block any current PR".

Consequently #859's long-term `held Session` contract — a `!Send + !Sync`
session object that outlives a callback and keeps admission/resources on the
opening thread — is **not implementable** at this pin. B1 can be delivered only
as the new frontend's *value/operation* surface over the existing borrowed
session, with the held-session lifetime deferred to the upstream U1 unit. The
issue's own boundary rules do not permit substituting a different session
lifetime, a global selector, or a second tenferro version to close that gap.

The pin also predates tenferro-rs #2044 (`CpuExecSession::child_execution()`,
merged 2026-10-09 as `1dccad7e7`), which #859's CPU child/phase execution and
#857 both require; those parts need a pin bump in their own change.

### 1.1 Named workloads (the B0 "workloads" record)

#859 assigns one representative workload per follow-up package. They are named
here so that the later measurements have fixed identities; no measurement is
made in B0:

| Package | Workload identity |
| --- | --- |
| F2 / #510 | Small-bond-density TT inner product over `simplett`: paired cold/warm, 1-thread and multi-thread, small chi, matched against the reference implementation |
| F3 / #670 | G0→Pi→W→Sigma SGW evaluation; the recorded R10 case is `T=0.1`, `mu=0.5`, `U=2`, `tolerance=1e-4`, `max bond 4096`, 30 initial pivots |
| F4 / #857 | Split-interpolation fan-out (the #830 regression): scalar and batch callbacks, known-value N-ary contraction, expected patch outcomes, 1/2/larger workers |
| F5 / #553 | The #675 TreeTN CUDA vertical slice, plus component-wise network transfer |
| F1 / F7 | Core/`IdxTensor`/`Matrix`/structured operation seam, then `tensorci`, `treetn`, `treetci`, `treeaci`, `partitionedtt`, `itensorlike`, `quanticstci` consumers |

## 2. Compatibility/eager frontend (retained)

This is the frontend that "existing callers continue using … No automatic
rerouting to a different value representation" applies to.

### 2.1 Feature graph

```text
default        = ["backend-tenferro", "tenferro-cpu-faer"]
backend-tenferro = ["global-defaults"]          # compatibility alias
global-defaults  = ["explicit-context"]         # legacy process-global operations
explicit-context = []                           # caller-supplied CPU context
tenferro-cuda    = ["explicit-context", "tenferro-cpu-faer", dep:tenferro-gpu, …]
tenferro-cpu-faer / tenferro-system-blas / tenferro-provider-inject = provider selection
einsum-dispatch-profile = []
```

`lib.rs` uses one coarse module boundary rather than per-function `cfg` gates:
`any_scalar`, `backend`, `incremental_qr`, `matrix`, `memory`, `storage`,
`tenferro_bridge` and `tensor_element` compile **only** under `global-defaults`;
`context` and `logical_tensor` compile under `explicit-context`; `cuda` under
`tenferro-cuda`. There is currently no module that compiles the concrete
operations *without* the process-global default.

### 2.2 Entry points

| Entry point | Visibility | Notes |
| --- | --- | --- |
| `with_default_backend(closure)` | `pub` (`global-defaults`) | Borrows the process-global `CpuBackend`; no session |
| `default_cpu_execution_context()` | `pub` (`global-defaults`) | `Arc<CpuExecutionContext>` over the process-global backend |
| `default_eager_ctx()` | `pub` (`global-defaults`) | Process-global eager runtime; typed `EagerContextError` |
| `with_default_session(closure)` | `pub(crate)` | Session-scoped concrete route; `tenferro_tensor::Error` on rejection |
| `with_default_graph_runtime(closure)` | `pub(crate)` | Graph compiler/runtime/backend triple |
| `default_engine_buffer_pool_stats()` | `pub(crate)` | Engine buffer-pool accounting |

### 2.3 Legacy entry-site inventory

The compatibility surface is reached from every consumer crate through its own
`backend-tenferro` feature, which forwards to
`tensor4all-tensorbackend/backend-tenferro`, or transitively through another
consumer's feature that does (for example `tensor4all-aci` and
`tensor4all-quanticstci` enable only provider features such as
`tensor4all-tensorbackend/tenferro-cpu-faer` and inherit `global-defaults` from
another workspace member). Inside `tensor4all-tensorbackend` the
session-entering **invocations** of the three named helpers are few and
centralized (symbol references such as imports and re-exports are excluded):

| Module | `with_default_session` | `with_default_graph_runtime` | `with_default_backend` |
| --- | ---:| ---:| ---:|
| `backend.rs` | 9 | 0 | 0 |
| `matrix.rs` | 1 | 0 | 0 |
| `tenferro_bridge.rs` | 6 | 1 | 0 |
| `context.rs`, `lib.rs` | 0 (definitions and re-exports only) | 0 | 0 |
| `any_scalar.rs`, `storage.rs`, `incremental_qr.rs`, `tensor_element.rs`, `memory.rs` | 0 | 0 | 0 |

This table is only the call-site inventory of those three helpers. It does not
count the other ways this crate reaches default state today: direct
`default_context()` use, the process-global eager runtime
(`default_eager_ctx` / `EagerRuntime`), or the direct `with_default_backend`
entry from consumers. Those are part of what R (retirement) has to migrate, and
the 12 consumer crates reach the same surface through their public convenience
APIs, so this is *not* a repository-wide zero-entry claim (see
[§6](#6-specified-legacy-preservation-tests)).

### 2.4 Session routing today

`CpuExecutionContext::with_session` is the single canonical entry:

```text
pub(crate) fn with_session<R: Send>(
    &self,
    f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
) -> Result<R, CpuExecutionContextError>
```

It asserts the thread-local `CanonicalSessionGuard` is inactive (**panic**, in
debug and release, on nested canonical entry), then enters the context backend
and installs the guard inside the closure. When called from a Rayon worker it
deliberately does not install into the context pool; it lazily creates and uses
an inline `CpuContext::with_threads(1)` backend (`inline_backend`) and leaves
parallelism to the enclosing pool. All production concrete operations route
through this or through `with_default_session`, which adapts the
process-global context and maps admission rejection into
`tenferro_tensor::Error`.

`CpuExecutionContext::with_backend` remains public for caller-managed low-level
integration and is *not* a session: entering a session from inside it, or
re-entering the default session from inside a canonical session, is a
programming error.

## 3. Explicit frontend as it exists today

Public under `explicit-context` (and therefore also under `global-defaults`):

| Item | Purpose |
| --- | --- |
| `ExecutionContext` | `Cpu(Arc<CpuExecutionContext>)`, `Cuda(Arc<CudaExecutionContext>)`; `is_global_default_cpu()` is the only legacy-recognition helper |
| `CpuExecutionContext` | `from_backend`, `with_backend`, `compile_graph`, `run_graph`, `eager_runtime`, `graph_cache_stats`, `graph_buffer_pool_stats`, `reset_graph_buffer_pool`, `reset_graph_runtime` |
| `CpuExecutionContextError` | `Initialization`, `SessionEntry`, `Graph`; tenferro diagnostics retained as `source` |
| `LogicalTensor`, `LogicalTensorData`, `LogicalTensorError` | Backend-free column-major snapshot over all tenferro CPU dtypes |
| `CudaExecutionContext`, `CudaExecutionContextError`, `CUDA_ORDINAL` | `tenferro-cuda` only: `upload_cuda`, `download`, placement validation, `synchronize` |

**The gap B1 has to close:** the explicit context exposes *lifecycle* (backend,
graph, eager runtime) and *transfer* (`LogicalTensor`, CUDA upload/download), but
no concrete operation route. Every numeric entry point is behind
`global-defaults` and routes through the default context, so an explicit caller
cannot run one primitive, einsum or linalg operation on its own context without
either reaching a `pub(crate)` helper or falling back to the process-global
route.

## 4. Conversion boundaries that already exist

| From → to | API | Feature | Boundary properties today |
| --- | --- | --- | --- |
| `Storage`/`StructuredStorage` → `Tensor` | `storage_to_native_tensor`, `storage_payload_native_read_input` | `global-defaults` | dtype/shape checked, `BridgeError`; read-input variant is a borrowed, non-materializing view |
| `Tensor` → `Storage` / dense vec / diagonal | `native_tensor_primal_to_storage`, `native_tensor_primal_to_dense_col_major`, `native_tensor_primal_to_diag` | `global-defaults` | dtype-checked materialization; `DType::External` scalars are rejected as unsupported. A native `Tensor` carries no tracking or laziness flag, so this is *not* an AD-detach check |
| `Matrix<T>` ↔ `TypedTensor<T>` | `Matrix::to_typed_tensor`, `into_typed_tensor`, `try_from_typed_tensor` | `global-defaults` | `MatrixTensorConversionError`; column-major contract preserved |
| flat column-major slice → `Tensor` | `dense_native_tensor_from_col_major[_owned]`, `diag_native_tensor_from_col_major`, `TensorElement` hooks | `global-defaults` | shape/element-count validated |
| `Tensor` → `LogicalTensor` | `LogicalTensor::from_native` | `explicit-context` | dtype/shape snapshot; no backend, executor or pointer identity |
| `LogicalTensor` → `Tensor` | `CpuExecutionContext::reconstruct(&LogicalTensor)` | `explicit-context` | receiving context is mandatory and entered for the call; typed dtype/shape/element-count errors |
| host `Tensor` → CUDA `Tensor` | `CudaExecutionContext::upload_cuda` | `tenferro-cuda` | visible-ordinal-0 only; host placement validated before entry. Not host-complete: the device write runs after the call returns (the pinned upstream upload path says so, citing tenferro-rs #2009), so `CudaExecutionContext::synchronize` is a separate, explicit step |
| CUDA `Tensor` → host `Tensor` | `CudaExecutionContext::download` | `tenferro-cuda` | Host-complete readback; never accepts a host tensor and never downs from a different allocation domain |
| eager/tracked → plain | tenferro-ad `EagerTensor::to_tensor`, `::tensor_read`, `::detach`, `::detach_into`, `EagerRuntime::with_eager_session` | `tenferro-ad`, reached through `explicit-context` | Explicit, caller-named extraction or materialization. Context identity is enforced at tenferro-ad's own eager entry points (`ContextMismatch` for a cross-context eager operation), *not* at native extraction |

The `global-defaults` bridges in this table take and return native `Tensor` and
`Storage` values, i.e. they sit *below* the tracking boundary: an already
detached value carries no identity for them to validate, and none of them rejects
a value because it was once tracked. Tracking, context identity and detach are
properties of the tenferro-ad entry points above. The new frontend
([§5](#5-specified-new-opt-in-surface-b1)) must therefore make its tracking and
identity decisions at bridges that accept eager values, and must not claim that a
native reader reveals a former AD identity.

## 5. Specified new opt-in surface (B1)

### 5.1 Namespace and entry

The new frontend is one tensorbackend-owned module — working name
`tensor4all_tensorbackend::explicit` — compiled under `explicit-context` and
independent of `global-defaults`. It is *not* a process-global selector: nothing
in it consults `DEFAULT_*`, `from_env` or the global context mutex.

The entry is the existing `CpuExecutionContext` (the `Context` of #859). The
session surface is:

```text
// Held form (needs upstream U1; see §8):
impl CpuExecutionContext {
    fn open_session(&self) -> Result<Session<'_>, SessionError>;
}
// Interim scoped form, usable with the pin in §1:
impl CpuExecutionContext {
    fn with_session<R>(&self, f: impl FnOnce(&mut Session<'_>) -> R) -> Result<R, SessionError>;
}
impl Session<'_> {
    fn context(&self) -> &CpuExecutionContext;
}
// The held form additionally owns admission, so it also needs an explicit
// `close(self) -> Result<(), SessionError>` next to its non-panicking `Drop`;
// a borrowed session has nothing to release early and gains no `close`.
```

`Session` is `!Send + !Sync`, rejects nesting with a typed error (no `assert!`
on a public route), and never opens a second root owner. Whether `Session` is a
held object or a borrow that lives only for the callback is the one part that
depends on upstream U1; the operation signatures below are identical either way,
so delivering them first is additive and does not redefine an existing public
method.

### 5.2 Operation routes

`Session`-parameterized counterparts of the operations that today enter
`with_default_session`, grouped by family. Each family gets read (`TensorRead`),
write/output-into (`&mut` destination, no hidden allocation) and
allocation-returning forms, with dtype, column-major layout, shape, provider and
structured representation preserved — never a silent materialization or
detach.

| Family | Operations |
| --- | --- |
| primitive | `axpby`/`scale`/`conj`/`sum`/`outer_product`, `permute`, `reshape`, dense/diagonal construction |
| einsum | binary `contract`, N-ary `einsum` over owned values and over borrowed `TensorRead` inputs, output-ids and cached plan reuse |
| linalg | `qr`, `svd`, `full_piv_lu`, `solve`, `triangular_solve`, `src_error_estimate`, `hermitian_eigendecomposition`, `hermitian_exponential_first_column`, `lowest_hermitian_eigenpair` |
| structured/matrix | `Storage`-level `contract`/`permute`/`axpby`/`to_dense`, `Matrix` `mat_mul`/`batched_mat_mul_same_shape`/`grouped_mat_mul_shared`/`submatrix`/`swap_rows`/`swap_cols`/`transpose` |

Two clarifications the pinned API forces on this table:

- **Output routes are per family, not uniform.** tenferro's prepared einsum has a
  real caller-written-output route, while the listed decompositions differ:
  `qr`/`svd` return allocated results and `solve` has a read-into form
  (`crates/tenferro-linalg/src/tensor_ext.rs` at the pin). B1 must state, per
  operation, which native read/write/plan route actually exists, which scratch
  the session owns, and must not implement an allocating result plus a copy and
  call it `_into`.
- **Not every legacy linalg wrapper ends at a concrete session.** In this crate
  `hermitian_eigendecomposition`, `hermitian_exponential_first_column` and
  `lowest_hermitian_eigenpair` currently enter the process-global *eager* runtime
  (`matrix.rs`, `default_eager_ctx` plus `EagerTensor` plus
  `with_eager_session`). Their default wrappers cannot become thin
  default-session adapters; they either keep their eager entry at the boundary or
  need a concrete linalg route that does not exist yet.

Requirement from #859: each body is **shared**, not copied. Where an operation
really is a concrete-session call, the implementation moves behind the session
parameter and the `global-defaults` wrapper becomes a thin adapter that passes
the default session. That is why the `global-defaults`-only module boundary in
[§2.1](#21-feature-graph) has to be re-cut in B1: the operation modules must
compile under `explicit-context`, with only the default-context adapters left
behind `global-defaults`.

### 5.3 Bridges

Narrow, explicitly named conversions at an algorithm/stage/batch boundary, with
the four #859 contracts made mechanical:

| Contract | Mechanism |
| --- | --- |
| 1. Preserve dtype, logical shape/layout, placement, provider ownership, structured representation | Every bridge validates and preserves these; the structured/diagonal payload is never materialized to dense implicitly. Index/prime/tag identity stays in the upper layers. |
| 2. Native ownership/aliasing for concrete↔eager; tracked inputs never silently become plain; explicit detach | Detach/materialize is a distinct, explicitly named operation; unsupported combinations return a typed error. `no_grad` on an eager value does not select the concrete representation. |
| 3. Legacy materialization/adoption happens before entering or after closing the session | The bridge API takes `&CpuExecutionContext`/`&mut Session` such that a legacy call cannot be reached from inside the session, including on error and fallback paths. |
| 4. Convert once per stage/batch, and account for it | Bridges report copies/allocations/registration; B1 tests measure them and report them separately from the backend-only number. |
| 5. No automatic fallback to eager/global/CPU | An unsupported concrete/GPU operation returns a typed error; the legacy caller stays on the legacy frontend. |

### 5.4 Errors and lifetime

`SessionError` distinguishes admission rejection (`SessionEntryError` source),
nesting, incompatible context/AD identity, and unsupported-route, all *before*
dispatch or partial output writes. No new process-global default backend, pool or
cache is introduced by the explicit route; the existing upstream arbitration and
thread-local reentry guards remain the safety mechanism.

## 5bis. B1 delivery: the `explicit` frontend (partial)

The first delivery of the new frontend shipped as `tensor4all_tensorbackend::explicit`
(module `crates/tensor4all-tensorbackend/src/explicit.rs`, compiled under
`explicit-context`), on the tenferro revision that provides the held concrete CPU
session (pin bump `b3f4729` -> `ff94aeded`, PR #873).

**This is a partial slice of B1, not the whole gate.** It delivers the session entry
and the allocation-returning primitive/einsum/linalg routes over native tensors; the
read/write/output routes, the compatibility bridges beyond `LogicalTensor`, the
private child-resource surface, the backend phase proof, the `Matrix`/`Storage`
families and the bridge-cost measurements are **not** in it. The table below states
exactly what is delivered, and §5c lists what is not.

| Element | Delivered |
| --- | --- |
| Context | the existing `CpuExecutionContext` |
| Entry | `CpuExecutionContext::with_concrete_session`: one session entry for the whole callback, on the caller's own backend. A caller inside a Rayon worker is rejected typed instead of being rerouted to the compatibility frontend's unrelated inline backend |
| Session | `explicit::Session`, a tensorbackend-owned view over the concrete `BackendSession`; neither the callback nor its value needs `Send`, because the tenferro entry at this pin runs it on the entering thread |
| Primitive routes | `reshape`, `permute`, `sum`, `conj` (allocation-returning) |
| Einsum routes | `contraction` (binary, by axes), `einsum` (N-ary, by integer labels) and `einsum_reads` (borrowed `TensorRead` operands), evaluated session-direct, all promoting heterogeneous operands to the same common dtype the compatibility frontend uses, plus `contraction_into` writing a caller-provided destination |
| Linalg routes | `qr`, `svd`, `solve`, `triangular_solve`, `full_piv_lu` (allocation-returning) |
| Matrix routes | `mat_mul` through the shared `Matrix` container and `grouped_mat_mul_shared`, both available where the compatibility frontend is (`global-defaults`) |
| Shared implementation | the label validation and the axis-to-label construction are one implementation (`src/einsum_ids.rs`) used by both frontends; the evaluation is tenferro-einsum's session-direct `einsum_subscripts`, which compiles no semantic graph and starts no runtime worker |
| No legacy entry | the module never names `with_default_session`, the default context or the eager runtime; the label helpers live outside the compatibility-only module so the explicit-only build compiles |
| No eager/AD | the routes reach only concrete `Tensor`/`BackendSession` operations; no `EagerTensor`, semantic node or gradient slot is constructed and no eager owner lock is taken |
| Errors | each route reports the backend's typed error; an invalid input, including a mismatched einsum label count, is rejected typed and never retried on the global or eager route. A nested canonical session entry is a typed `SessionEntryError::Reentered`, not a panic |
| Tests | route-by-route agreement with the compatibility frontend for six routes, one session serving a batch, typed rejection instead of fallback, a mixed-precision rejection, cross-context reuse with the input storage pointer unchanged, and the typed nested-entry rejection |

### 5c. B1 remaining after this slice

| Item | Notes |
| --- | --- |
| Read/write/output routes | the read route, the binary output-into route and the N-ary output route (`PreparedEinsum::execute_into`) are delivered |
| Remaining linalg operations | the Hermitian eigen routes and `src_error_estimate`; the Hermitian routes currently enter the process-global *eager* runtime, so they need the optional AD adapter rather than a concrete route, and `src_error_estimate` is session-free |
| Remaining `Matrix`-level linalg routes | `solve_matrix`, `full_piv_lu_matrix`, the typed `qr_backend`/`svd_backend` wrappers and `triangular_solve_matrix` still enter the compatibility session |
| Remaining batched GEMM route | `batched_mat_mul_same_shape` builds its jobs and then enters the compatibility session; `grouped_mat_mul_shared` now has a session entry (`Session::grouped_mat_mul_shared` via `grouped_mat_mul_shared_in`) |
| Session-free operations (no explicit route needed) | `scale`, `axpby`, `conj` on the structured paths, the `Storage`-level contraction/permutation kernels, and `Matrix` `submatrix`/`swap_rows`/`swap_cols`/`transpose` are pure host code that names no session, so an explicit caller uses them as they are. `outer_product` is already expressible as `contraction(lhs, &[], rhs, &[])` |
| Structured representation parity | structured/diagonal storage preservation on the explicit route (dtype promotion is delivered and shared) |
| Compatibility bridges | only `LogicalTensor` exists; explicit detach/lift and materialization bridges are still to come |
| Private child resources and the backend phase proof | the held session and the phase lease exist upstream; tensorbackend does not expose or prove them yet |
| Reusable plan surface | delivered as `explicit::PreparedEinsum`: prepare once over borrowed operands, then `execute` or `execute_into` on whichever session the caller holds |
| Measurements | bridge allocations/copies/registration are not measured, and no paired dispatch numbers are recorded for the explicit route |
| A held *object* form | see the note below |

Why the entry is callback-scoped here, and what the alternatives are: tenferro's held
session borrows a `&CpuBackend` for the caller's scope (`CpuBackend::open_session(&self)`),
while this context stores its backend in a `Mutex<CpuBackend>` because the legacy
`with_backend_session` needs `&mut`. What that rules out is the **single returned object
that owns both the guard and a session borrowing it** - the self-referential struct the
upstream design forbids. It does not rule out the two expressible alternatives:

```text
// 1. a lease that owns the guard, from which the held session is opened:
let lease = context.borrow_backend()?;          // owns the MutexGuard
let held = lease.open_session()?;              // borrows the lease, not the guard field

// 2. a cloned handle, using the documented clone semantics:
let backend = context.with_backend(|backend| backend.clone());
let held = backend.open_session()?;
```

Either would give this frontend a held object without an upstream change; neither is
taken here, because the object's ownership and the compatibility entry's worker policy
are decisions for the maintainer rather than consequences of the pin. What is delivered
is the callback-scoped entry plus the routes, which is what a stage needs today.

## 6. Specified legacy-preservation tests

To be added with the B1 implementation (they are intentionally not written here,
because until the operation modules are re-cut under `explicit-context` they
would only re-assert the current single-frontend build):

1. **Surface pin.** Compile-time assertions that every item in
   [§2.2](#22-entry-points) still exists with the same signature and feature
   gate, and that
   `with_default_backend`/`default_cpu_execution_context`/`default_eager_ctx`
   still behave as today. The call-site counts of [§2.3](#23-legacy-entry-site-inventory)
   are the inventory the pin is meant to keep honest, not a set of items to
   assert on.
2. **Numerical parity.** For a representative operation per family (primitive,
   einsum, linalg, structured/matrix), the legacy route and the new explicit route
   produce bitwise-identical results for identical inputs.
3. **No-eager-route property.** The new concrete route creates no
   eager/trace/value/gradient record and takes no eager owner lock; bridge
   copies/allocations/registration are counted and reported separately.
4. **Session lifetime and entry.** `Session` is `!Send + !Sync`; nested entry is a
   typed error rather than a panic on the public route; a closed/failed session
   releases admission and a later independent operation succeeds; a caught
   panic inside a session leaves the parent usable.
5. **No implicit fallback.** An unsupported concrete operation returns a typed
   error and does not silently execute on the global/eager/CPU route, including
   when the caller also holds the global context.
6. **Incompatible access and AD identity.** A value from a different context, a
   tracked value used as a plain one, and an incompatible provider all fail
   typed before any clone/no-op/single-operand shortcut.
7. **Consumer preservation.** The existing consumer crates build and pass
   without edits, i.e. the coexistence window holds.

## 7. Verification gates

This B0 record changes documentation only, so the applicable local tier is the
repository's "Documentation/API — prose only: check links and consistency"
(`CONTRIBUTING.md`), plus `cargo run -j 16 -p xtask --release -- api-dump` to
keep the generated inventory referenced in [§1](#1-recorded-baseline) verifiable.
The build, feature-matrix, test and audit checks are *not* an unconditional gate
for this record; they belong to the change that touches code.

For B1 those gates apply: `cargo fmt --all -- --check`; `cargo clippy -j 16
--workspace --all-targets -- -D warnings`; `cargo test -j 16 -p
tensor4all-tensorbackend`; the isolated explicit-only build and docs
(`cargo check -p tensor4all-tensorbackend --no-default-features --features
explicit-context,tenferro-cpu-faer` and the matching `cargo doc`) that
`docs/design/explicit-cpu-execution-context.md` already requires;
`python3 scripts/audit-library-panics.py`; the hosted workspace test job of
[§1](#1-recorded-baseline) with the new feature combination; the tests of
[§6](#6-specified-legacy-preservation-tests); and the app-layer measurements that
F2–F5 own.

## 8. Upstream dependencies and deferrals

| #859 requirement | Upstream dependency | State |
| --- | --- | --- |
| Held `Session` object, opening-thread admission, fallible close, non-panicking `Drop` | tenferro-rs #1945 unit U1 (explicit-executor CPU kernels → holdable session) | Open design discussion, explicitly not scheduled. tenferro-rs #1938 D12 defers A2 and instructs callers to reuse borrowed sessions. **Blocked, not implementable today.** |
| CPU child sessions and the phase scheduler | tenferro-rs #2044 `CpuExecSession::child_execution()` | Merged 2026-10-09, **not in the pin of [§1](#1-recorded-baseline)**; needs a pin bump in the owning change. |
| Real pending H2D/D2H, stream handoff, retirement, overlap | tenferro-rs #2009 / #1885, tensor4all/cubecl 0.10.2 | Partially reduced (borrowed upload 3→1 host copy, download without the extra copy); pinned staging and async transfer remain open. |
| Explicit AD surface (B3) | tenferro-rs U2-AD | Not started. |

B1's value/operation surface is *not* blocked by these; only its session
lifetime, child/phase execution, transfer substrate and AD variants are.

## 9. Non-goals

- No new process-global default backend, pool, cache or TLS override, no second
  tenferro revision and no duplicated numerical kernels in the explicit route
  ([§5.4](#54-errors-and-lifetime)).
- No workspace-wide API/storage rewrite, no core/algorithm/C-ABI/Julia changes
  and no mechanical workspace-wide renames in B1.
- No removal of the `global-defaults` surface, no default routing change, no
  deprecation: that is R, after the consumer inventory has migrated.
- No claim of a repository-wide zero-entry or zero-global audit while the 12
  consumer crates still use the compatibility frontend.
