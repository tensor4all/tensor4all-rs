# Explicit engine for every operation — inventory and migration plan

**Issue:** [tensor4all-rs#1376](https://github.com/tensor4all/tensor4all-rs/issues/1376)
(supersedes the narrower [#1375](https://github.com/tensor4all/tensor4all-rs/issues/1375))

**Status:** design and inventory only. Nothing in this document is implemented; no
public signature, feature default, or numeric path changes.

**Revision 3.** This revision follows the cross-repository policy in tensor4all-meta
[`core-boundary-and-execution-ownership.md`](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/core-boundary-and-execution-ownership.md),
whose scope includes tensor4all-rs, and tenferro #2206, which implements it. It replaces
two earlier models.

- **Revision 1 (#1388)** kept an *owner token*: a value resolved its engine from the
  context that created it. That breaks programs with several `CpuEngine`s, because any CPU
  engine can read any host value.
- **Revision 2** separated placement, engine and runtime, but still passed the engine
  wrapped in a locked `ExecutionContext` with `_in` / `_on_session` spellings. That
  re-creates the *session* that tenferro removed, with its per-entry lock, its re-entry
  error and its entry cost (§4.6).

Revision 3 has **one concept, the engine**. Every computing operation takes
`exec: &mut dyn ExecutionEngine` as an argument. There is no lock, no session and no
context wrapper. The eager runtime is separate and shared.

## 0. The one-line problem

`tensor4all-rs` runs numerical work through a caller-supplied execution context
(`CpuExecutionContext`, [explicit-cpu-execution-context.md](explicit-cpu-execution-context.md)),
and `tenferro` has already deleted its own implicit entry points. Three problems remain.

1. **The surface can still run without naming an engine.** There are eight distinct
   shapes of this (§3), and one silently substitutes the process-global engine.
2. **The context conflates the engine with the runtime.** Every context builds a private
   eager runtime, so a value is tied to the engine that created it (§4.5).
3. **The context is a session.** It wraps the engine in a `Mutex` with a re-entry guard,
   and when the caller is a Rayon worker it silently swaps in a fresh serial engine
   (§4.6). tenferro removed exactly this layer, because of its cost and its
   ownership problems.

The goal: every computing operation takes the engine by `&mut`, the caller owns its
engines, values are not tied to engines, and nothing chooses an engine on the caller's
behalf.

---

## 1. What the issue asks, and what already exists

Do **not** re-plan what has landed. The following are facts at the baseline:

| Already in place | Where (all paths from the repository root) |
| --- | --- |
| A caller-owned CPU context that owns one `CpuEngine`, a graph compiler/runtime, prepared-plan cache, and an eager runtime | `crates/tensor4all-tensorbackend/src/context.rs:1455` `CpuExecutionContext::from_backend`, `:1504` `with_backend`, `:1532` `with_session`, `:1544` `compile_graph`, `:1564` `run_graph`, `:1632` `eager_runtime` |
| A CPU/CUDA union used as the explicit parameter type | `crates/tensor4all-tensorbackend/src/context.rs:1061` `pub enum ExecutionContext { Cpu(Arc<CpuExecutionContext>), Cuda(..) }` |
| A tenferro-shaped concrete session over a borrowed engine, with 20 operation methods | `crates/tensor4all-tensorbackend/src/explicit.rs:95` `pub struct Session<'session> { engine: &'session mut dyn Engine }`; methods `reshape`, `permute`, `contraction`, `contraction_reads`, `contraction_into`, `einsum`, `einsum_reads`, `sum`, `conj`, `mat_mul`, `qr`, `svd`, `solve`, `triangular_solve`, `full_piv_lu`, `qr_backend`, `svd_backend`, `solve_matrix`, `triangular_solve_matrix`, `full_piv_lu_matrix` |
| `explicit-context` as a buildable route with the legacy modules compiled out | `crates/tensor4all-tensorbackend/src/lib.rs:13-57` gates nine modules behind `global-defaults`; `explicit-context` is a separate feature |
| `*_in` / `*_on_session` explicit twins on the higher crates | 41 `*_in` and 32 `*_on_session` public functions across ten crates (§3.2) |
| A recorded consumer/feature inventory | [859-f7-consumer-inventory.md](859-f7-consumer-inventory.md) — its §1/§2 crate and default-feature map predates the current tree (see the caveat below) |

**Caveat on the linked feature map.** [859-f7-consumer-inventory.md](859-f7-consumer-inventory.md)
§1 lists a crate `tensor4all-partitionedtt` that does not exist in `crates/` (the crate is
`tensor4all-partitionedtreetn`), and lists default features (`tenferro-cpu-faer`,
`simplett-bridge`) that the current `Cargo.toml` files no longer have: `treetn`,
`tensorci` and `treetci` now default to `tenferro-cpu-platform`, and `treetn` has no
`simplett-bridge` default. Use it for the *narrative* (which crate threads which route)
and read `crates/*/Cargo.toml` for the current feature facts. This document does not
repeat that table, so it cannot add a third copy of it.

What is missing is *removal*, not *addition*: the implicit twin is still the advertised
entry point everywhere, `tensor4all-core` cannot even be compiled against an
explicit-only backend, and one explicit path still borrows the global context (§4.3).

---

## 2. What "explicit" means here

The issue's contract is one sentence: **an operation cannot silently choose a backend.**
Five properties follow, and this document judges every call site by them.

- **E1 — the engine parameter is mandatory.** A public operation that executes numerical
  work takes `exec: &mut dyn ExecutionEngine`, or it cannot run. There is no exception for
  operator overloads or value methods. An `Option` engine fails E1 by construction.
- **E2 — nothing selects an engine for the caller.** That means no process-global context
  (`default_context()`, `default_eager_ctx()`, `default_cpu_execution_context()`) and no
  fallback. It also means no substitute engine, such as the fresh serial engine
  `CpuExecutionContext` builds for a Rayon worker (§4.6).
- **E3 — no silent adoption.** A value with no runtime joins a runtime only at an
  operation whose caller supplied one: the operands' runtime, or else the runtime passed
  with the engine (§6.3). It never joins a process-global runtime.
- **E4 — one shape.** The engine is a plain `&mut` argument, last in the parameter list
  as in tenferro's call grammar (`recv.op(operands…, exec)`). There is no session object,
  no lock and no second spelling for the same operation.
- **E5 — values are not bound to engines.** Any engine that can read an operand's
  placement may run the operation: a CPU engine reads any host value, and a CUDA engine
  reads values in its own allocation domain. Operands must share one runtime.

---

## 3. Inventory

### 3.1 Method

The surface splits into eight families that a single grep cannot see. The decisive
lesson from the first pass of this inventory (and from the plan review that caught it)
is that **grepping for `None` finds family F1 and nothing else.** Each family below
therefore has its own command.

### 3.2 F1 — a literal `None` context argument

`crates/tensor4all-core/src/defaults/eager_session.rs` defines three helpers that take
`context: Option<&ExecutionContext>` and route `None` to `with_runtime_engine`
(`:63`, `:97`, `:131`). Production code passes the literal `None` at **53 sites**. The count uses a multi-line-safe
query, because a single-line `grep` misses calls whose arguments wrap. Both give 53 at
`20b42b0d`, but only the `rg -U` form stays correct after `rustfmt` reflows a call:

```bash
rg -U --pcre2 -c 'with_(eager_session(_typed)?|engine|execution_engine)\(\s*[^,()]+(\([^()]*\))?,\s*None\s*,' \
  crates/*/src
```

| File | Sites |
| --- | --- |
| `crates/tensor4all-core/src/defaults/idx_tensor.rs` | 43 |
| `crates/tensor4all-core/src/any_scalar.rs` | 4 |
| `crates/tensor4all-core/src/defaults/contract.rs` | 3 |
| `crates/tensor4all-core/src/defaults/factorize.rs` | 2 |
| `crates/tensor4all-core/src/defaults/qr.rs` | 1 |

Verdict: **fails E1 and E2.** These are the operations reachable from
`IdxTensor` value methods (`conj`, `sum`, `permute`, `einsum`, `stack`, `select`,
`eigh`, `backward`, `detach`, arithmetic overloads), from `AnyScalar` elementwise
arithmetic, and from the context-free `contract`, `factorize` and `qr` entry points.

### 3.3 F2 — an `Option<&ExecutionContext>` parameter whose `None` is implicit

33 declarations, 28 of them outside `#[cfg(test)]`:

```bash
grep -rn "Option<&tensor4all_tensorbackend::CpuExecutionContext>\|Option<&CpuExecutionContext>\|Option<&ExecutionContext>" \
  --include=*.rs crates/*/src | awk -F: '{print $1}' | sort | uniq -c | sort -rn
```

| File | Declarations | Note |
| --- | --- | --- |
| `crates/tensor4all-aci/src/state.rs` | 10 | `LUCI` state and batched frame updates |
| `crates/tensor4all-core/src/defaults/idx_tensor.rs` | 5 | `:1903`, `:1973`, `:2018`, `:4488`, `:6712` |
| `crates/tensor4all-core/src/defaults/eager_session.rs` | 4 | the F1 helpers themselves |
| `crates/tensor4all-core/src/defaults/svd.rs` | 3 | |
| `crates/tensor4all-quanticstci/src/quantics_tci.rs` | 3 | `run_treetci_batch` `:168`, `context` parameter `:174`, `match context` `:214`, `None => optimize_with_proposer(..)` `:223`, second `match context` `:232`; also `:778` and `:1350` |
| `crates/tensor4all-core/src/defaults/qr.rs` | 2 | `:422`, `:625` |
| `crates/tensor4all-aci/src/elementwise.rs` | 1 | `:191` |
| `crates/tensor4all-treetn/src/treetn/tests/mod.rs`, `crates/tensor4all-treetn/src/treetn/contraction/outcome/tests/dispatch.rs` | 5 | test-only |

`crates/tensor4all-quanticstci/src/quantics_tci.rs:168-232` is the clearest case of the shape: a public batch entry threads
`Option<&CpuExecutionContext>` down to a `match context { Some(..) => optimize_with_proposer_in, None => optimize_with_proposer }`,
so the *same* public function selects between the explicit and the global route by a
`None`.

Verdict: **fails E1.** A `None` that means "use the global one" is the implicit path
wearing an explicit signature.

### 3.4 F3 — an implicit entry that builds the global context itself

Four production sites (plus one doc example) construct the global context inside a
function whose own signature has no context at all:

| Site | Enclosing public entry | Code |
| --- | --- | --- |
| `crates/tensor4all-treetn/src/treetn/contraction.rs:3151` | `contract` (`:3111`), `ContractionMethod::Src` arm | `let context = ExecutionContext::Cpu(default_cpu_execution_context());` |
| `crates/tensor4all-treetn/src/treetn/contraction.rs:3227` | `contract_src_with_rng` | same, with the comment "Legacy CPU-global compatibility boundary (see \`contract\` Src arm)" |
| `crates/tensor4all-treersi/src/api.rs:38` | `hadamard_many` (`:34`, `#[cfg(feature = "global-defaults")]`) | `let context = ExecutionContext::Cpu(tensor4all_tensorbackend::default_cpu_execution_context());` then delegates to `hadamard_many_in` |
| `crates/tensor4all-treersi/src/api.rs:113` | doc example | `ExecutionContext::Cpu(default_cpu_execution_context())` |
| `crates/tensor4all-interpolativeqtt/src/interpolation.rs:925` | `compress_train` (private, reached from the public QTT builder) | `default_cpu_execution_context().with_session(\|engine\| tree.compress_chain_on_session(..))` |

```bash
grep -rn "ExecutionContext::Cpu(default_cpu_execution_context())\|default_cpu_execution_context()" \
  --include=*.rs crates/*/src
```

`treetn` and `treersi` at least *have* an explicit twin (`contract_in`,
`hadamard_many_in`), and the implicit one is a thin adapter over it, so deleting it is
cheap. `treersi`'s adapter is behind `global-defaults`. `treetn`'s two sites
(`contract`, `contract_src_with_rng`) are **not** feature-gated, so a default build reaches
the global context through them. `interpolativeqtt` has no explicit twin at all.
`compress_train`'s three callers (`interpolate_single_scale_nd:146`,
`interpolate_single_scale_sparse_nd:499`, `build_refined_qtt:1042`) are themselves
context-free, so fixing that site needs `_in` twins for the public builders, not only a
context parameter on `compress_train`. (`interpolate_chebyshev_patches` and
`interpolate_patched_nd` already take a caller context and do not reach `compress_train`.)

Verdict: **fails E2.** These are the sites that make `global-defaults` unremovable
without an API change.

### 3.5 F3c — the global context reached from inside an explicit path

One site, and it is a removal blocker rather than an inventory row:

```rust
// crates/tensor4all-treersi/src/engine.rs:110-120
match exec {
    Exec::Session(session) => matrix_luci_row_interpolation_owned_on_session(matrix, Some(options), session),
    #[cfg(feature = "tenferro-cuda")]
    Exec::Cuda(_) => tensor4all_core::matrix_luci_row_interpolation_owned_in(
        matrix,
        Some(options),
        &tensor4all_tensorbackend::default_cpu_execution_context(),   // <-- :118
    ),
}
```

The CUDA arm of the **explicit** LUCI factorisation borrows the process-global CPU
context. Deleting `global-defaults` therefore breaks an `_in` entry point that is
supposed to be explicit, and it will break it at compile time only after the feature is
already gone. §7 stage A3 makes repairing this a prerequisite, not a follow-up.

Verdict: **fails E2** inside an entry that otherwise satisfies E1.

### 3.6 F4 — implicit public twins with no context parameter at all

These carry no `Option`, no `None`, and no `global-defaults` gate: they reach the
implicit route transitively, through `IdxTensor`'s F1 sites.

| Crate | Implicit entry | Explicit twin |
| --- | --- | --- |
| `treetn` | `contract` — `treetn/contraction.rs:3111` | `contract_in` — `:3050` |
| `itensorlike` | `contract` — `contract.rs:169` | `contract_in` — `:233` |
| `simplett` | `inner_product` — `contraction.rs:83` and `:344`; `contract` — `mpo/dispatch.rs:63` | `inner_product_with_session` — `contraction.rs:220` |
| `treetci` | `crossinterpolate2` — `api.rs:127` | `crossinterpolate2_in` — `:212`, `crossinterpolate2_with_rng_in` — `:295` |
| `tensorci` | `crossinterpolate2` — `tensorci2.rs:1759` | `crossinterpolate2_with_rng_on_session` |
| `treersi` | `hadamard_many` — `api.rs:34` (gated) | `hadamard_many_in`, `hadamard_many_with_rng_in` |

Verdict: **fails E1.** The twin exists, so removal is a deprecation-and-delete, not a
design question — but the implicit twin is what every example, doctest and binding calls
today.

### 3.7 F5 — owner-less value adoption into the global eager runtime

`IdxTensor`'s inner value is an `EagerTensor` that needs an `Arc<EagerRuntime>`
(`crates/tensor4all-core/src/defaults/idx_tensor.rs:1702` `untracked_inner(native, runtime)`). A tensor that has no runtime
yet — a plain `Storage::Materialized` — gets one from the global runtime at **10 sites**:

```bash
grep -n "default_eager_ctx()" crates/tensor4all-core/src/defaults/idx_tensor.rs
```

| Line | Context |
| --- | --- |
| `1066` | `scale_eager_payload` |
| `1726` | `compact_payload_inner` |
| `2167` | pairwise-contract payload adoption when operands do not share one runtime |
| `2969` | structured eager-cache seed |
| `3597` | structured-selector payload |
| `3633` | `from_native_with_axis_classes` |
| `3647` | `from_untracked_native_with_axis_classes` |
| `3987` | `enable_grad` |
| `4053` | gradient payload adoption |
| `9221` | `is_global_default_cpu`-style runtime identity check during explicit adoption |

This is the **E3** violation. A value's runtime is optional, and a missing runtime is
filled *at the value*, from the global runtime, rather than at the operation from the
context the caller named (§6.3). `crates/tensor4all-core/src/defaults/idx_tensor.rs:2167` is the sharpest case — the code picks the common
operand runtime when all operands agree, and falls back to the global runtime when they
do not. A mixed-origin contraction therefore runs on the global runtime and engine.
Under §6.3 it becomes a typed `RuntimeMismatch`.

### 3.8 F6 — the process-global conveniences

| Item | Site | Visibility |
| --- | --- | --- |
| `mod defaults`, `DEFAULT_CONTEXT: OnceLock<Arc<CpuExecutionContext>>` | `crates/tensor4all-tensorbackend/src/context.rs:1966`, `:1969` | private |
| `default_context()` | `crates/tensor4all-tensorbackend/src/context.rs:1981` | crate-private |
| `with_default_backend` | `crates/tensor4all-tensorbackend/src/context.rs:2012` | public (re-exported `crates/tensor4all-tensorbackend/src/lib.rs:71`) |
| `with_default_session` | `crates/tensor4all-tensorbackend/src/context.rs:2026` | crate-private |
| `with_default_graph_compiler` | `crates/tensor4all-tensorbackend/src/context.rs:2041` | crate-private |
| `default_engine_buffer_pool_stats`, `reset_default_engine_buffer_pool`, `reset_default_engine` | `crates/tensor4all-tensorbackend/src/context.rs:2046`, `:2052`, `:2058` | crate-private |
| `default_eager_ctx` | `crates/tensor4all-tensorbackend/src/context.rs:2080` | public |
| `default_cpu_execution_context` | `crates/tensor4all-tensorbackend/src/context.rs:2111` | public |
| `with_runtime_engine` | `crates/tensor4all-tensorbackend/src/context.rs:1940` | public |

`with_runtime_engine` is the fallback itself:

```rust
// crates/tensor4all-tensorbackend/src/context.rs:1940-1953
pub fn with_runtime_engine<R>(runtime: &EagerRuntime, f: impl FnOnce(&mut dyn ExecutionEngine) -> R)
    -> Result<R, ExecutionContextEntryError> {
    let owner = {
        let mut owners = eager_runtime_owners();
        match owners.get(&runtime.id()).and_then(RuntimeOwner::upgrade) {
            Some(owner) => owner,
            None => {
                owners.remove(&runtime.id());
                LiveRuntimeOwner::Cpu(Arc::clone(&defaults::default_context().owner))  // :1950
            }
        }
    };
    ..
}
```

A runtime whose creating context was dropped does not produce a typed error; it runs on
the global engine. This is issue #1375's finding 2, verbatim, and it is the single
behavioural change with the widest blast radius (§8).

### 3.9 F7 — the `global-defaults` modules in `tensor4all-tensorbackend`

Nine modules are compiled only on the legacy route (`crates/tensor4all-tensorbackend/src/lib.rs:13-57`); the first column gives the file under `crates/tensor4all-tensorbackend/src/`:

| Module | Public numerical content | Reaches the global engine? |
| --- | --- | --- |
| `backend.rs` | `svd_backend`, `qr_backend`, `solve_backend`, `triangular_solve_backend`, `solve_matrix{,_owned}`, `triangular_solve_matrix{,_owned}`, `full_piv_lu_backend`, `full_piv_lu_matrix{,_owned}`, `src_error_estimate{,_general}` | yes — **12** direct `with_default_session` call sites, at `:263` (`solve_right_full_piv_lu`), `:816` (`solve_matrix_direct`), `:837` (`solve_matrix_direct_owned`), `:898` (`triangular_solve_matrix_direct`), `:949` (`triangular_solve_matrix_direct_owned`), `:1113` (`triangular_solve_matrix_widened`), `:1467` (`convert_for_typed`), `:1520` (`svd_backend` → `svd_backend_in`), `:1566` (`qr_backend` → `qr_backend_in`), `:1619` (`solve_backend`), `:1659` (`triangular_solve_backend`), `:1802` (`full_piv_lu_tensor` → `full_piv_lu_tensor_in`). Each global twin delegates to its `_in` sibling or to a private `_in` helper |
| `matrix.rs` | `hermitian_eigendecomposition` (`:796`), `lowest_hermitian_eigenpair` (`:736`), `hermitian_exponential_first_column` (`:894`), `submatrix_argmax` (`:1484`), `mat_mul{,_owned}`, `batched_mat_mul_same_shape{,_owned}` | `hermitian_eigendecomposition` uses `default_eager_ctx()` `:812` **and** `default_context()` `:822`; `lowest_hermitian_eigenpair` and `hermitian_exponential_first_column` carry no direct call and delegate to it. The BLAS product path reaches the global engine through `dot_general_matrices` `with_default_session` at `:1563` and `blas_mat_mul` at `:1659`, so `mat_mul`/`mat_mul_owned`/`batched_mat_mul_same_shape{,_owned}` inherit it; `mat_mul` has `mat_mul_in`. **`submatrix_argmax` (`:1484`) is host-only** — no global call |
| `tenferro_bridge.rs` | 33 public functions | `with_default_session` call sites at `:1313` (`reshape_col_major_native_tensor`), `:1325` (`qr_native_tensor`), `:1339` (`svd_native_tensor`), `:1357` (`sum_native_tensor`), `:1967` (`permute_native_tensor`), and `:666` (`convert_tensor`, which every dtype conversion goes through). Every einsum/contraction path goes through `run_cached_native_einsum` (`default_context()` at `:757`) or `execute_native_einsum_reads` (`default_context()` at `:874`). `native_tensor_from_vec` (`:52`) is **not** a global path — it is a plain `NativeTensor::from_vec_col_major` plus error mapping |
| `incremental_qr.rs` | `IncrementalQr::{new:196, append:253, q_columns:465, error_estimate:534}` | via `factorize_backend` (`:545`), which calls the global twin `qr_backend` at `:551` |
| `capped_full_piv_lu.rs` | `capped_full_piv_lu:245` | **no** — `factor_with_faer:260` is pure host faer; this module only needs an ungated home, not a context |
| `storage.rs`, `any_scalar.rs`, `memory.rs`, `tensor_element.rs` | value types, elementwise arithmetic, allocator pressure relief | `any_scalar` arithmetic routes through `IdxTensor` (F1) |

`capped_full_piv_lu` is the useful counter-example: the module gate is coarse, so a
removal plan can move it to `explicit-context` unchanged.

### 3.10 F8 — bindings

| Binding | State | Evidence |
| --- | --- | --- |
| C (`tensor4all-capi`) | 87 `extern "C"` functions (`tensor.rs` 29, `treetn.rs` 32, `index.rs` 15, `quanticstransform.rs` 10, `lib.rs` 1). No context parameter, no context handle, no initialisation call. Every call runs on the global default. | `crates/tensor4all-capi/src/`; opaque handles only in `types.rs` |
| Python (`tensor4all-py`) | calls the implicit twins: `Tensor::contract` → context-free `contract` at `crates/tensor4all-py/src/tensor.rs:160`; `TreeTensorNetwork::contract` → `contraction::contract` at `crates/tensor4all-py/src/treetn.rs:157` | |

### 3.11 The explicit surface that already exists, and its shape problem

```bash
grep -rhoE "pub fn [a-z_0-9]+" --include=*.rs crates/*/src | sed 's/pub fn //' \
  | grep -E "_(in|on_session)$" | sed -E 's/.*_(in|on_session)$/_\1/' | sort | uniq -c
```

41 `*_in`, 32 `*_on_session`, and **both spellings exist for the same operation**:

| Operation | `_in` form | `_on_session` form |
| --- | --- | --- |
| factorization | `factorize_in(.., context: &ExecutionContext)` — `crates/tensor4all-core/src/defaults/factorize.rs:645` | `factorize_on_session(.., session: &mut explicit::Session<'_>)` — `:904` |
| QR | `qr_with_in(.., context: &ExecutionContext)` — `crates/tensor4all-core/src/defaults/qr.rs:385` | `qr_with_on_session(.., session: &mut explicit::Session<'_>)` — `:523` |

Two spellings of one explicit parameter is an E4 violation. Both spellings also enter
a locked context: the `_in` form directly, and the `_on_session` form through its
caller. That is the session shape §4.6 describes. §6.2 replaces both with one plain
`exec: &mut dyn ExecutionEngine` argument.

### 3.12 Summary of the eight families

| Family | Shape | Sites | Violates |
| --- | --- | --- | --- |
| F1 | literal `None` context argument | 53 | E1, E2 |
| F2 | `Option<&..Context>` parameter, `None` = implicit | 33 (28 non-test) | E1 |
| F3 | implicit entry constructs the global context | 4 production + 1 doc | E2 |
| F3c | explicit `_in` path borrows the global context | 1 | E2 |
| F4 | implicit public twin without any context parameter | 8 entries in 6 crates | E1 |
| F5 | owner-less value adopted into the global eager runtime | 10 | E3 |
| F6 | the process-global conveniences themselves | 9 items | E2 |
| F7 | nine `global-defaults`-only modules | 1 crate | E2 (mostly), E1 |
| F8 | bindings with no context concept | 87 C fns + all of `py` | E1 |

---

## 4. Structural blockers

### 4.1 `tensor4all-core` cannot be built without `global-defaults`

```toml
# crates/tensor4all-core/Cargo.toml:58-61
# `global-defaults` is required in every configuration: this crate re-exports
# `TensorElement` and uses `Matrix`/`ExecutionContext`/`mat_mul` in un-gated code.
tensor4all-tensorbackend = { path = "../tensor4all-tensorbackend", default-features = false, features = ["global-defaults"] }
```

The comment states the reason and it is accurate: `TensorElement` (`core/src/lib.rs:151`),
`Matrix`/`mat_mul` in un-gated helpers, and `ExecutionContext` are re-exported by
un-gated modules. It understates the dependency, though. Core also **calls a
`global-defaults`-only function from un-gated code**:
`core/src/defaults/eager_session.rs:15` imports `with_runtime_engine`, which is
`#[cfg(feature = "global-defaults")]` (`tensorbackend/src/context.rs:1939`), and the F1
helpers call it at `:63`, `:97`, `:131`. Core has no `global-defaults` cfg of its own.

Dropping the requirement therefore needs two things. Stage B2 moves the re-exports
and splits `mat_mul` (which already has `mat_mul_in`). Stage D2 deletes the `None` route,
and with it core's only use of `with_runtime_engine`. Only after both does stage D4 drop
the requirement.

**Status (#1377 stage 1).** The `explicit-context` feature of `tensor4all-tensorbackend` now
compiles `matrix`, `storage`, `tensor_element`, `backend`, `any_scalar` and `tenferro_bridge`,
and the six `explicit::Session` linalg methods; only the functions that resolve the
process-global context stay behind `global-defaults`. `tensorbackend`'s default features are
`explicit-context` plus the platform provider. The two internal global users are threaded:
`hermitian_eigendecomposition` and the bridge einsum core take a caller `CpuExecutionContext`
(`hermitian_eigendecomposition_in`, `einsum_native_tensors_in`, `einsum_native_tensors_owned_in`,
`einsum_native_tensor_reads_in`, `NativeEinsumPlan::execute_reads_in`). `core`'s
`global-defaults` pin is **deliberately kept**: dropping it was tried and fails to compile
because core's ungated code calls `mat_mul`, `solve_matrix`, `capped_full_piv_lu`,
`default_eager_ctx`, `with_runtime_engine`, `src_error_estimate` and
`native_tensor_primal_to_diag`, so it stays a D4 step after B2 and D2 as described above.

### 4.2 `treersi` enables `global-defaults` by default

`crates/tensor4all-treersi/Cargo.toml` has `default = ["tenferro-cpu-platform", "global-defaults"]`,
and `treersi/src/api.rs:34` gates `hadamard_many` on it. `treetn`, `treetci`, `tensorci`
and the nine other crates do *not*, so a default workspace build enables the legacy route
from three directions: `tensorbackend`'s own default, `core`'s hard requirement, and
`treersi`'s default.

### 4.3 The `Is_global_default_cpu` compatibility branch

`crates/tensor4all-tensorbackend/src/context.rs:1076` `ExecutionContext::is_global_default_cpu()` returns true when a context's
eager runtime *is* the global one, and callers use it to choose the "legacy host code path
(bitwise-identical numerics)". Removing the global route removes this branch. The numeric
equality that holds today rests partly on it, so phases C and D need parity evidence, not just
deletion (§8).

### 4.4 Crates that compile only through feature unification

`crates/tensor4all-interpolativeqtt/Cargo.toml:51` asks `tensorbackend` for
`explicit-context` only, yet `interpolation.rs:7` imports `default_cpu_execution_context`,
which `tensorbackend/src/lib.rs:69-71` gates on `global-defaults`. The crate compiles only
because `core` pins `global-defaults` (§4.1) and Cargo unifies features across the build.
When stage D4 drops that pin, every crate in this situation fails to compile, and the
manifests do not show it. Before that step, run an explicit-only `cargo check` per crate,
or `cargo hack check --each-feature`, so the list comes from the compiler rather than a
grep.

### 4.5 Every context builds its own runtime, which ties values to engines

`CpuExecutionContext` (`crates/tensor4all-tensorbackend/src/context.rs:1435-1439`) holds the
engine owner, the graph compiler, and a lazily built `EagerRuntime` that is private to
that context (`eager_runtime`, `:1632-1644`). `CudaExecutionContext`
(`crates/tensor4all-tensorbackend/src/cuda.rs:198-203`, `:492`) does the same. A value
created in context A therefore belongs to A's runtime, and two consequences follow.

- **Values from two CPU engines cannot be combined.** A contraction of a lane-1 value with
  a lane-2 value, both on the host, fails with tenferro's runtime-identity
  `ContextMismatch`.
- **Validation rejects valid engines.** `IdxTensor::validate_context`
  (`crates/tensor4all-core/src/defaults/idx_tensor.rs:9086-9120`) rejects any value whose
  runtime is not the context's runtime ("tensor belongs to a different CPU eager context",
  `:9106-9110`), even though the CPU engine could read it. `adopt_into` (`:9179`) exists to
  copy values across that artificial boundary.

The eager layer itself is already decoupled. `with_eager_session_typed(runtime,
Some(context), ..)` (`crates/tensor4all-core/src/defaults/eager_session.rs:58-63`) runs the
*value's* runtime on the *context's* engine. The coupling lives only in runtime
construction and in validation. That is why this program can fix it without a new
abstraction (§6.1, stage A2).

### 4.6 `CpuExecutionContext` is a session

`CpuExecutionContext::with_session` delegates to `EngineOwner::with_session`
(`crates/tensor4all-tensorbackend/src/context.rs:1369-1389`). On every call it:

- checks a per-thread **re-entry guard** and returns `CpuExecutionContextError::Reentered`
  if this thread is already inside (`:1342-1347`);
- **locks the engine** behind a `Mutex` (`begin_session`, `:1356-1362`);
- when the calling thread is a Rayon worker, **builds a fresh `CpuEngine::serial()`**
  instead, sharing only the plan cache (`:1373-1380`, tensor4all-rs#830). That silently
  runs on a different engine than the one the caller supplied, and it fails E2.

tenferro measured this kind of entry at about 5–8 µs of wrapper cost per single-op
session (`tenferro-rs/docs/design/cpu-session-open-cost.md`). It then removed the session
layer: arbiter, held session, re-entry errors and lease (tenferro #2206). The
tensor4all-meta policy states the replacement. Exclusivity comes from `&mut engine`. A
caller that shares an engine between threads wraps it in its own `Mutex`. Parallelism is
expressed by holding more engines, each task carrying its engine as an argument, because
finding an engine through a lock can deadlock when a waiting worker runs another task.

Any API shaped as "pass a context, the callee enters it" therefore re-creates the session.
The 41 `*_in(.., &ExecutionContext)` functions and the 32 `*_on_session` functions (§3.11)
both have this shape.

---

## 5. Comparison with tenferro

Anchored to the pinned revision
`tenferro* = { git = "https://github.com/tensor4all/tenferro-rs.git", rev = "32a4e98c12f9dc36ffa85b5273b9247b44936959" }`
(`Cargo.toml:114-121`), whose own design record is
`tenferro-rs/docs/design/execution-engine-ownership.md` (tenferro-rs #2206).

### 5.1 The engine trait and how it is obtained

```rust
// tenferro-rs/crates/tenferro-tensor/src/backend.rs:3606
pub trait Engine: TensorBackendOps + SessionCachedDot + TensorDeviceTransfer { .. }
```

`CpuEngine::new(pool: tenferro_cpu::Pool)` takes the caller's built pool by value
(`tenferro-rs/crates/tenferro-cpu/src/engine.rs:119`); `CpuEngine::serial()` runs on the calling thread.
There is no `CpuEngine::from_env`, no process-wide engine and no engine registry:

> "tenferro never subdivides a pool, keeps no process-wide engine registry and does not
> choose which engine runs a task."
> — `tenferro-rs/docs/design/execution-engine-ownership.md` §2

### 5.2 The parameter shape

Every operation takes the engine as a plain `&mut` argument on an extension trait, so the
call site reads as one fluent chain and there is no separate "session" object to build:

```rust
// tenferro-rs/crates/tenferro-einsum/src/concrete.rs:137
fn einsum(&self, subscripts: &str, session: &mut dyn Engine) -> Result<Tensor>;

// tenferro-rs/crates/tenferro-linalg/src/tensor_ext.rs:361 (doctest)
let (_u, s, _vt) = a.svd(&mut host)?;
```

Prepared programs are the same shape one level up: `ExecutionEngine: Engine`
(`tenferro-rs/crates/tenferro-runtime/src/runtime/engine.rs:150`) adds `target()`, extension-cache
borrowing and run retirement, and a run binds engine slot `i` to the `i`-th supplied
engine for that call only.

### 5.3 The eager/AD layer takes the engine too

```rust
// tenferro-rs/crates/tenferro-ad/src/eager.rs:3143
pub fn with_eager_session<T, E: From<Error>>(
    self: &Arc<Self>,
    engine: &mut dyn ExecutionEngine,
    f: impl FnOnce(&mut EagerSession<'_>) -> Result<T, E>,
) -> Result<T, E>
```

`EagerTensor::to_tensor(&self, engine: &mut dyn Engine)` (`tenferro-rs/crates/tenferro-ad/src/eager.rs:546`),
`detach(&self, engine: &mut dyn Engine)` (`:4822`). The runtime stores no engine; the
engine stores no graph.

### 5.4 How tenferro prevents silent selection

It removed the alternative rather than guarding it. From
`tenferro-rs/docs/design/explicit-session-boundary.md`:

> "The former `EagerTensor` operation methods and arithmetic overloads were removed, not
> replaced with implicit-entry shims; value/trace handles remain, as do named backward and
> runtime boundaries."

Cross-context misuse is a typed error, `tenferro_ad::Error::ContextMismatch`
(`tenferro-rs/crates/tenferro-ad/src/eager.rs:927` and elsewhere), not a silent re-home.

### 5.5 Where tensor4all differs, and why it matters

| Axis | tenferro | tensor4all today |
| --- | --- | --- |
| Who owns the engine | the caller, always | the caller for `_in`, a singleton for everything else |
| The parameter | `&mut dyn Engine` (concrete ops), `&mut dyn ExecutionEngine` (prepared/eager) | `&ExecutionContext` (a union) *and* `&mut explicit::Session<'_>` (a borrow) — two shapes |
| Eager value token | the `Arc<EagerRuntime>` the value belongs to; a foreign one is `ContextMismatch` | the same `Arc<EagerRuntime>`, but absence is filled from the global runtime |
| Implicit entry points | removed | 8 families still present |
| Bindings | n/a (Rust-first) | a C ABI with no context concept |

The two designs are closer than this table makes them look. `explicit::Session` *is* the
tenferro shape, and `EagerRuntime`-per-value *is* tenferro's value identity. Two things
are missing: neither is mandatory, and tensor4all builds one runtime per engine context
(§4.5) where tenferro keeps runtime and engine independent (§5.6).

### 5.6 Placement, engines and runtimes in tenferro

Read at tenferro-rs `origin/main` `8127d0d7`, which is newer than the pinned `32a4e98c`.
This section deliberately cites the newer revision, because it describes upstream's
current direction. That is an exception to §10 rule 4. The paths below are relative to
`tenferro-rs/crates/`, and every one of them also exists at the pin. The line numbers are
for `8127d0d7`.

- **Placement is a property of the tensor.** `Placement` (`tenferro-tensor/src/types.rs:359-372`)
  carries the memory kind and an optional `DeviceId` (CPU, or a GPU kind plus ordinal).
  Backend storage also carries an `AllocationDomainId` (`types.rs:394`). The engine knows
  its own target and remembers no tensor.
- **The engine is chosen per call.** The caller passes `&mut dyn Engine` or
  `&mut dyn ExecutionEngine`. `Engine` is object-safe, and no CPU/CUDA union type exists.
  Cargo features only decide which engine types are compiled.
- **A placement mismatch is an error, never a transfer.**
  - A host tensor given to a CUDA engine fails with `Error::Unsupported` ("would need an
    implicit transfer", `tenferro-gpu/src/cuda/structural.rs:960-963`) on the main read
    path, and with `RuntimeState` on others.
  - A device tensor given to a CPU engine fails with `RuntimeState`
    (`tenferro-cpu-basic/src/lib.rs:40-60`).
  - The error variant differs by path. That inconsistency is an upstream issue candidate.
- **Transfer is explicit.** It is done by `TensorDeviceTransfer::{upload_host_tensor,
  download_to_host}` on the engine (`tenferro-tensor/src/backend.rs:3489-3511`), or by a
  caller-registered `TransferProvider` in a prepared program. The library itself only
  uploads constants it builds and reads back control scalars.
- **The runtime pins no engine and no device.** `EagerRuntime` stores no engine
  (`tenferro-ad/src/eager.rs:825-843`). Each `with_eager_session` borrows the caller's
  engine for one closure, and `ensure_engine` (`:3150-3187`) registers the engine *kind*
  lazily. Every `CpuEngine` shares the canonical runtime engine id
  (`tenferro-cpu/src/engine.rs:171`), so one runtime's values can run on any number of
  CPU engine instances. `ContextMismatch` compares runtime identity only, not placement.
- **Multi-GPU is not supported upstream yet.** Every `CudaEngine` reports the fixed id
  `tenferro.cuda.default.v1` (`tenferro-gpu/src/cuda/engine.rs:48`), so one runtime holds
  at most one CUDA engine. This document does not design around that (§9).

### 5.7 The tensor4all-meta execution-ownership policy

[`core-boundary-and-execution-ownership.md`](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/core-boundary-and-execution-ownership.md)
fixes the cross-repository rules. Its scope names tensor4all-rs. The rules this program
depends on:

- **One concept, the engine.** Code that runs work takes `&mut dyn Engine` (or
  `&mut E: Engine`). The former session concept is removed, and a set of engines for one
  multi-engine run is an argument, "not a stored object, and it is not called a session".
- **The caller owns the threads and the engines.** `CpuEngine::new(pool)` takes its pool
  by value, and `CpuEngine::serial()` runs on the calling thread. tenferro never
  subdivides a pool and never picks an engine.
- **Runtimes hold what was prepared; engines hold what is mutable.** Graphs, compiled and
  prepared programs, AD rules, values and gradients belong to the runtime. Pools, buffer
  pools, scratch and plan caches belong to the engine.
- **The caller's scheduler gives each task its engine as an argument.** A worker of one
  pool may call an operation on another pool's engine.

---

## 6. Proposed tensor4all equivalent

### 6.1 Two objects the caller holds: engines and a runtime

| Object | Holds | Owned by | Passed as |
| --- | --- | --- | --- |
| Engine (`CpuEngine`, `CudaEngine`) | pool, buffer pool, scratch, plan cache, stream, allocation domain | the caller, one per concurrent computation | `exec: &mut dyn ExecutionEngine`, per operation |
| Runtime (tensor4all `Runtime`: tenferro `EagerRuntime` plus graph compiler and compiled-program cache) | AD rules, eager values, compiled and prepared programs | the caller, shared by `Arc` across engines | at value construction, and with the engine where an operation needs it (§6.3) |
| Placement | host, or a CUDA allocation domain | the value | — |

- **`CpuExecutionContext` / `CudaExecutionContext` / `ExecutionContext` are deleted**, and
  with them the engine lock, the re-entry guard, `Reentered` and the Rayon substitute
  engine. Their two halves become the engine (caller-owned, `&mut`) and the `Runtime`
  (`Arc`, no lock on the engine path).
- **Plan caches may be shared between engines.** `CpuEngine::with_plan_cache` already
  shares immutable prepared plans (`tenferro-cpu/src/engine.rs:151` at the pin).
- **CPU or GPU at run time** is the caller's choice of which `&mut dyn ExecutionEngine` to
  pass. tenferro's `Engine` is object-safe, so no CPU/CUDA union type is needed. A caller
  that wants a closed set can keep its own enum or `Box<dyn ExecutionEngine>`.
- **Layering.** `tensor4all-tensorbackend` re-exports `ExecutionEngine`, `CpuEngine` and
  `CudaEngine` (it already re-exports `CpuEngine`, `lib.rs:167`), so downstream crates do
  not take a direct `tenferro-*` dependency.

### 6.2 Parameter shape

```rust
// tenferro call grammar: operands first, options, then the engine.
pub fn svd_with(t: &IdxTensor, opts: &SvdOptions, exec: &mut dyn ExecutionEngine) -> Result<..>;
pub fn contract(a: &IdxTensor, b: &IdxTensor, exec: &mut dyn ExecutionEngine) -> Result<IdxTensor>;
let c = a.add(&b, exec)?;                      // value methods take the engine too
let grads = loss.backward_in(exec)?;           // as in the tensor4all-meta sketch
```

- **One spelling.** There is no `_on_session` form, because the engine *is* the borrowed
  resource. A loop that runs thousands of operations passes the same `exec` to each, at
  no entry cost.
- **Computing operator overloads are removed**, as tenferro removed them. `AnyScalar`'s
  `Add`/`Sub`/`Mul`/... (`core/src/any_scalar.rs:1225+`) become methods taking `exec`. Pure
  host arithmetic on an untracked scalar may stay an overload (open decision, §6.7).
- **Names (open decision, §6.7).** The end state follows tenferro: the base name takes
  `exec` last, with no suffix. During migration the implicit twin still holds the base
  name, so the engine form needs a temporary name. The recommendation is to reuse `_in`
  ("runs in this engine"), changing the 41 existing `_in(&ExecutionContext)` functions to
  take `exec`, and to rename back to the base name in phase D only if the maintainer
  wants tenferro's exact grammar.
- **Parallel work.** The scheduler (rayon, `threadteam`, the patch and lane fan-out) gives
  each task its own engine: `CpuEngine::serial()` for a task on a worker, or a pooled
  engine per lane. This replaces the #830 substitute engine with an explicit choice by
  the caller.
- Rejected: making every function generic over `E: ExecutionEngine`. It would
  monomorphise the whole surface per engine type, for no benefit over `dyn`, since
  engine calls are coarse-grained.

### 6.3 Runtime and validation

- **A value belongs to one `Runtime`.** Constructors that create values take the runtime
  explicitly, for example `IdxTensor::from_dense(&rt, ..)`.
- **Operations read the runtime from their operands.** All operands that have one must
  share it, else typed `IdxTensorError::RuntimeMismatch`.
- **The engine must be able to read every operand's placement.** A CPU engine reads host
  values, and a CUDA engine reads values in its own allocation domain. Otherwise the
  error is typed `IdxTensorError::PlacementMismatch`, and there is never an implicit
  transfer. Both checks run at operation entry, before any RNG draw or kernel.
- **Mapping errors at this boundary** also hides tenferro's per-path variant
  inconsistency (§5.6) from tensor4all's callers.
- **Runtime-less host values** (TCI materialisation, disk load) need an explicit runtime
  wherever no operand supplies one. Operations that create values from nothing (TCI from
  a function, `interpolativeqtt` builders, random initialisation) therefore take
  `rt: &Runtime` as well as `exec`.
- **Transfer.** `t.to_engine(exec)` copies a value to the engine's placement, explicitly,
  and is a no-op when the placement already matches. The same runtime is kept.

### 6.4 Engine, runtime and multiple engines in practice

```rust
let rt = Runtime::new()?;                                   // shared, Arc inside
let mut big = CpuEngine::new(Pool::builder().threads(8).build()?);
let mut lanes: Vec<CpuEngine> = (0..4).map(|_| CpuEngine::serial()).collect();
let a = IdxTensor::from_dense(&rt, idx, data)?;
let b = lane_results(&rt, &mut lanes)?;                     // each lane task carries its engine
let c = contract(&a, &b, &mut big)?;                        // same runtime, any CPU engine
#[cfg(feature = "tenferro-cuda")] {
    let mut gpu = CudaEngine::open(0)?;
    let a_dev = a.to_engine(&mut gpu)?;                     // explicit upload
    let d = contract(&a_dev, &a_dev, &mut gpu)?;
}
```

### 6.5 Error model

| New error | Replaces | At |
| --- | --- | --- |
| `IdxTensorError::RuntimeMismatch` | tenferro's `ContextMismatch` surfacing from inside a kernel | operation entry |
| `IdxTensorError::PlacementMismatch` | "tensor belongs to a different CPU eager context" and the upstream per-path variants | operation entry |
| `t4a_status_code::T4A_NO_ENGINE` | n/a | C boundary, stage E1 |

`CpuExecutionContextError::Reentered` and `CudaExecutionContextError::Reentered` are
deleted with the contexts. A `&mut` engine cannot be re-entered.

### 6.6 C API and bindings

```c
typedef struct t4a_runtime t4a_runtime;   /* shared; clone = same runtime */
typedef struct t4a_engine  t4a_engine;    /* exclusive; the caller must not use it from two threads at once */
t4a_status_code t4a_runtime_new(t4a_runtime **out);
t4a_status_code t4a_engine_new_cpu(size_t threads, t4a_engine **out);      /* 1 = serial */
void            t4a_engine_release(t4a_engine *);

t4a_status_code t4a_tensor_contract(const t4a_tensor *a, const t4a_tensor *b,
                                    t4a_engine *exec, t4a_tensor **out);
```

- A null engine or runtime is `T4A_NULL_POINTER`. There is no default.
- An engine handle is not cloneable: it owns its pool. Julia and Python keep one engine
  per concurrent task. A binding that shares one engine between threads wraps it in its
  own lock, which the policy assigns to the caller.
- This is a breaking ABI change, which the early-development policy
  ([AGENTS.md "Development Stage"](../../AGENTS.md)) permits. Python wraps `Runtime` and
  the engine in classes and calls the Rust API directly.

### 6.7 Open decisions

1. **Final names:** keep `_in` as the engine-taking suffix, or rename to tenferro's
   suffix-free grammar in phase D.
2. **Host-only scalar overloads:** keep `+`/`*` on untracked host `AnyScalar`, or remove
   every overload.
3. **CPU/CUDA shared runtime:** one `Runtime` serving both engine kinds, or separate
   runtimes with `to_engine` moving values between them. Stage A2 verifies feasibility.

---

## 7. Staged migration order

### 7.0 Constraints from concurrent work

Many PRs land on `main` in parallel with this program, and almost all of them touch the
surface it changes.

- **Add before migrate, migrate before remove.** Old and new forms coexist until phase D.
- **One crate per PR**, leaf-first.
- **Stop new debt first** with the A1 ratchet.
- **No `#[deprecated]`.** `-D warnings` would break every open PR.
- **Removal PRs are breaking and announced** on #1376 before they merge.
- **The one non-additive step before phase D** is changing the 41 `_in(&ExecutionContext)`
  signatures to take `exec`. Each such PR changes one crate's `_in` functions *and every
  workspace caller of them*, so it never leaves a caller broken.

### Phase A — groundwork

- **A0 — this document.**
- **A1 — ratchet** (#1392). Block *added*:
  - literal-`None` context arguments;
  - global accessors;
  - `Option<&..ExecutionContext>` parameters;
  - new `&ExecutionContext` parameters;
  - new `.with_session(` / `with_concrete_session(` calls outside `tensor4all-tensorbackend`.
- **A2 — `Runtime` and the engine-taking eager entry** (#1393). In `tensorbackend` and
  `core` only:
  - add the `Runtime` type: the eager runtime plus graph compiler and compiled-program
    cache, with no engine and no engine lock;
  - add `run_graph(&rt, exec, ..)` and an eager entry `with_eager(&rt, exec, f)` that
    takes `&mut dyn ExecutionEngine` directly;
  - add the `RuntimeMismatch` / `PlacementMismatch` validation;
  - let `CpuExecutionContext` build its runtime from a shared `Runtime`, so existing code
    keeps working on top of the new pieces.

  Verify at the pin that one graph runtime can serve several engines of the same kind
  (today the graph runtime is "derived from the engine" and lives under its lock,
  `context.rs:1241-1246`). Also verify whether one runtime can serve CPU and CUDA
  (§6.7-3).
- **A3 — `treersi/src/engine.rs:118`** (#1379). The CUDA LUCI arm uses
  `default_cpu_execution_context()`. Take a CPU engine from the caller, or return a typed
  refusal.

### Phase B — engine-taking surface, one crate per PR, leaf-first

1. **`tensorbackend`**:
   - engine-taking forms of the 10 engine-bearing bridge functions, the Hermitian
     helpers and `IncrementalQr::{new, append}`;
   - move host-only helpers and `capped_full_piv_lu` out of the `global-defaults` gate;
   - change `explicit::Session` users to plain `exec` arguments;
   - bridge/`Storage` visibility (#1380).
2. **`core`**:
   - move the `TensorElement` / `Matrix` re-exports (§4.1);
   - change the `_in` functions to take `exec`, together with their callers;
   - add `exec`-taking forms for every `IdxTensor` / `AnyScalar` method and computing
     operator that reaches a literal-`None` site;
   - make constructors take `&Runtime`.
3. **Algorithm crates, one PR each** (`simplett`, `treetn`, `itensorlike`, `tensorci`,
   `treetci`, `treersi`, `interpolativeqtt`, `aci`, `quanticstci`, others): change `_in`
   to take `exec` and `_on_session` to take `exec`, and add engine-taking twins where
   none exist. `interpolativeqtt`'s builders also take `&Runtime` (§3.4).

### Phase C — migrate callers, one crate per PR

- Internal callers, tests, examples, doctests and the tutorial chapter move to
  engine-taking forms.
- Parallel fan-out (`partitionedtreetn` patches, lanes, Rayon loops) gives each task its
  engine explicitly, replacing the #830 substitute.
- `Option<&..Context>` parameters become `exec` (ACI: #1382).
- The implicit public entries stay as thin adapters.

### Phase D — remove (breaking; announce on #1376 first)

1. **Per crate, leaf-last**: delete the implicit adapters, the F4 twins and the
   computing operator overloads.
2. **Delete the `None` route** (#1378): `with_runtime_engine`, the owner registry
   (`context.rs:1881-1893`) and the `None` arms in `eager_session.rs`.
3. **Delete the contexts**: `CpuExecutionContext`, `CudaExecutionContext`,
   `ExecutionContext`, `EngineOwner`, the re-entry guard and `explicit::Session` (if
   nothing still needs it).
4. **Delete the globals and flip the features** (#1377):
   - delete F6 and the F7 gates;
   - run an explicit-only `cargo check` per crate (§4.4);
   - flip the defaults and drop `core`'s `global-defaults` pin;
   - delete `backend-tenferro` / `global-defaults` and the A1 ratchet.

### Phase E — bindings and docs

1. **C API** `t4a_runtime` / `t4a_engine` (§6.6), plus the Tensor4all.jl PR. It can start
   once phase C reaches `treetn`.
2. **Python classes.**
3. **Remaining docs**: README, `llms.txt`, `skills/use-tensor4all-rs/`,
   `explicit-cpu-execution-context.md` (rewritten or retired) and
   `859-f7-consumer-inventory.md`.

### Mapping to the split issues

| Stage | Issue |
| --- | --- |
| A1 | #1392 |
| A2 | #1393 |
| A3, and the phase B/C/D steps for `treetn` adapters and `interpolativeqtt` builders | #1379 |
| D2 and the phase C F1 migration in `core` | #1378 (delete the route; no `NoLiveOwner`) |
| B2 re-exports and D4 | #1377 |
| phase C for `aci` | #1382 |
| B1 bridge visibility | #1380 |

---

## 8. Do not break numeric behavior

This is a **pure signature and selection** change. Every operation must produce
bit-identical results before and after each stage, and any stage that does not is a bug
in that stage.

Concretely:

- **Same kernel, different selection.** Phase C replaces *which* engine is used,
  not *what runs*. A migrated `IdxTensor::sum` must call the same
  `with_eager_session(.., Some(ctx), ..)` body it already had with `None`.
- **Runtime sharing must not change results.** Stage A2 changes which runtime records a
  value, not which kernel runs. The same engine with a private runtime and with a shared
  runtime must give bit-identical results.
- **The #830 substitute engine is a numeric path today.** A Rayon worker currently
  computes on a fresh serial engine. When phase C makes the caller pass that engine, the
  serial engine must stay serial, so that thread count and reduction order do not
  change.
- **The `is_global_default_cpu` branch is the one real numeric risk.** `context.rs:1076`
  selects the legacy host kernels for the global context, documented as
  "bitwise-identical numerics". Once the global route is gone, that branch is gone and
  every call takes the scoped primitives. Phases C and D must therefore run a parity harness —
  the same seeded operations through the old global route and through a
  `CpuExecutionContext::from_backend(CpuBackend::with_threads(n))` — and record max
  absolute deviation per operation. Existing precedent for this style of evidence:
  `859-f7-consumer-inventory.md` §6 records SRC exact parity, Zip-up tolerance parity, and
  the `itensorlike` plain-vs-context timings.
- **Thread count is part of the numeric contract.** The global context is built with
  `CpuBackend::new()`, so its thread count comes from placement resolution; a caller-built
  context pins it (`CpuBackend::with_threads`). Reduction order can change with thread
  count, so parity evidence must be taken at *matching* thread counts and state them.
- **RNG streams must not shift.** `treetn/src/treetn/contraction.rs:3151` and `:3227`
  construct the global context *before* RNG advancement, and the SRC contract says
  "Mixed host/CUDA inputs and foreign CUDA contexts fail validation **before RNG
  advancement**". Deleting the implicit adapter must not move a validation across an RNG
  draw.
- **Errors change, results do not.** Stages 1 and 3 legitimately change which calls
  fail: stage A2 turns some rejections into successes (a host value created on another
  CPU engine),
  and phase D turns silent successes into compile errors or typed errors. That is a behaviour change and must be stated in the PR body; it is
  not a numeric change. Any test that was passing *because* of the fallback should be
  rewritten to construct its context explicitly, not silenced.
- **`capped_full_piv_lu` must stay bit-identical across its un-gating**: it is pure faer
  (`crates/tensor4all-tensorbackend/src/capped_full_piv_lu.rs:260`) and moving its module must not change its algorithm.

Verification for stage A2 in particular: `cargo test -p tensor4all-tensorbackend`,
`cargo test -p tensor4all-core`, `cargo test --doc -p tensor4all-tensorbackend`,
`cargo test --doc -p tensor4all-core`, and
`cargo check --locked -p tensor4all-tensorbackend --no-default-features --features explicit-context,tenferro-cpu-faer`
with the matching `cargo doc` — the isolated-build pair already run by CI at
`.github/workflows/CI_rs.yml:39-42` ("Check explicit-context-only build"), documented in
[explicit-cpu-execution-context.md](explicit-cpu-execution-context.md).

---

## 9. What this document deliberately does not change

- **`GlobalDefault`** (`crates/tensor4all-core/src/global_default.rs`) is an `AtomicU64`
  tolerance default (`rtol`), used by `default_qr_rtol`/`set_default_qr_rtol`. It is not
  an engine, it has nothing to do with the `global-defaults` Cargo feature, and removing
  it is a separate API decision. Any reader who conflates the two should stop here.
- **Provider features.** `tenferro-cpu-platform`, `tenferro-cpu-faer`,
  `tenferro-system-blas`, `tenferro-provider-inject`, `tenferro-cuda` select *which*
  backend compiles. This program is about *who chooses* it. Folding provider selection
  into the engine parameter is out of scope.
- **The `#869` mixed configuration.** A `core` built without CUDA while `tensorbackend`
  has it keeps the `unreachable_patterns` arm returning
  `unsupported_execution_context_error()` (`core/src/defaults/eager_session.rs:65,102,138`).
  That arm is a typed refusal, not a default-selection path, and it stays.
- **`explicit::Session`'s frontend boundary.** It reaches only concrete tensor operations
  and never constructs eager/AD records (crates/tensor4all-tensorbackend/src/explicit.rs:1-40). It remains the concrete
  frontend's inner handle; eager operations keep the `EagerSession` shape.
- **Multiple GPUs.** tenferro supports at most one CUDA engine per runtime today (§5.6).
  Multi-GPU waits for upstream support and is a later program. This design only has to
  avoid ruling it out: placement carries the device and domain, and the context is chosen
  per call.
- **Performance.** Nothing here is a performance program. Where a stage could regress
  timings (the per-call context entry cost), the existing measurement convention applies:
  one measurement at a time, explicit 1-thread configuration for overhead numbers, and
  named CPUs.

---

## 10. Keeping this honest

Every table publishes the command that produced it. Two rules for the next reader:

1. **Read this file from `origin/main` or a fresh worktree.** A stale checkout has already
   made a merged entry point look absent during the #859 pass
   (`859-f7-consumer-inventory.md:158`); a verdict from the wrong tree is worse than none.
2. **Re-derive the family counts before acting.** If a count in §3 changed, the row is
   out of date, and the PR that changed it must say which row moved and why.

3. **Cite the line that contains the symbol, not the line that starts the enclosing
   function.** The first published revision of this document got this wrong in §3.9: it
   published function-declaration lines as if they were `with_default_session` call sites
   (`backend.rs` alone: eight published lines, twelve real sites), one row
   (`native_tensor_from_vec`) was a false positive from a `use` clause, and one `matrix.rs`
   row attributed `with_default_session` to `submatrix_argmax`, which is host-only. Check a
   citation with a symbol grep, never with a "which function does this line belong to"
   sweep:

   ```bash
   grep -n "with_default_session"  crates/tensor4all-tensorbackend/src/backend.rs   # 12 call sites + 1 import
   grep -nH "with_default_session" crates/tensor4all-tensorbackend/src/{matrix,tenferro_bridge}.rs
   grep -nH "default_context\|default_eager_ctx" crates/tensor4all-tensorbackend/src/{matrix,tenferro_bridge}.rs
   grep -n "qr_backend"            crates/tensor4all-tensorbackend/src/incremental_qr.rs
   ```

The `xtask api-dump` inventory
(`cargo run -p xtask --release -- api-dump`) covers all 18 public crates and is the
authoritative check that a public entry point exists or has been removed — it was used to
build §3.2, §3.6 and §3.11.

4. **An external tenferro citation must resolve at the pinned revision.** Check with
   `git -C <tenferro-rs> cat-file -e 32a4e98c:<path>`. Tenferro's sources live under
   `crates/`, so `tenferro-tensor/src/backend.rs` is a broken path and
   `crates/tenferro-tensor/src/backend.rs` is the real one; its design records are at
   `docs/design/` with no `crates/` prefix. The first revision got this wrong for eight
   citations.
