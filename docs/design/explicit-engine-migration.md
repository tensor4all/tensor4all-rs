# Explicit engine for every operation — inventory and migration plan

**Issue:** [tensor4all-rs#1376](https://github.com/tensor4all/tensor4all-rs/issues/1376)
(supersedes the narrower [#1375](https://github.com/tensor4all/tensor4all-rs/issues/1375))

**Status:** design and inventory only. Nothing in this document is implemented; no
public signature, feature default, or numeric path changes.

**Revision 2.** The first revision (#1388) kept an "owner token" model, in which a value
resolves its engine from the context that created it. That model is withdrawn. A host
value is readable by *every* CPU engine, so binding a value to one engine breaks
programs that run several `CpuEngine`s (parallel lanes over disjoint cores, a 1-thread
engine next to a many-thread one). This revision separates **where the data lives**
(the tensor), **which engine computes** (the caller, per call) and **which runtime the
value belongs to** (AD/eager identity), as tenferro does (§5, §6).

**Baseline:** `origin/main` at `20b42b0d`, read from a fresh worktree. Every table
below names the command that regenerates it; every `path:line` is checked with a symbol
grep at that revision (§10 rule 3). §3.9 carries a correction for the revision that cited
enclosing-function lines instead of call sites. Re-derive before acting on a row — see §10.

## 0. The one-line problem

`tensor4all-rs` already owns a caller-supplied execution context
(`CpuExecutionContext`, [explicit-cpu-execution-context.md](explicit-cpu-execution-context.md)),
and `tenferro` has already deleted its own implicit entry points. Two problems remain.

1. **The surface can still run without naming an engine.** There are eight distinct
   shapes of this (§3), and one of them silently substitutes the process-global engine
   *after* the caller passed a value that had an owner.
2. **The context conflates three things.** A tensor4all execution context is the engine,
   its caches, *and* the eager runtime its values belong to. As a result, a value is tied
   to the engine that created it (§4.5). That is why the code has both an ITensor-style
   "the tensor picks its engine" route and a tenferro-style "the caller passes the engine"
   route, and why the two disagree.

The goal is not a new context abstraction. It is to make every computing operation name
its engine, and to stop tying values to engines.

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
  work takes the execution context (engine), or it cannot run. There is no exception for
  operator overloads or value methods. `Option<&ExecutionContext>` with `None` meaning
  "implicit" fails E1 by construction.
- **E2 — no process-global resolution, not even as a fallback.** Reaching
  `default_context()`, `default_eager_ctx()`, or `default_cpu_execution_context()`
  anywhere inside an operation body fails E2.
- **E3 — no silent adoption.** A value with no runtime may join a runtime only at an
  operation whose caller named the context: the operands' runtime, or else the context's
  own. It never joins the process-global runtime.
- **E4 — one parameter shape per role.** There are exactly two roles. *Entry* (`_in`,
  `&ExecutionContext`) enters the context once. *Composition* (`_on_session`,
  `&mut explicit::Session`) runs inside a session that is already held. A third spelling,
  or two spellings for the same role, fails E4 (§3.11, §6.2).
- **E5 — values are not bound to engines.** Any engine that can read an operand's
  placement may run the operation: a CPU engine reads any host value, and a CUDA engine
  reads values in its own allocation domain. Operands must share one runtime. Neither
  rule depends on which context *created* the value.

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

Two spellings of one explicit parameter is the E4 violation, and it is why §5 has to
answer "which shape" rather than merely "add the parameter".

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
and with it core's only use of `with_runtime_engine`. Only after both does stage D3 drop
the requirement.

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
When stage D3 drops that pin, every crate in this situation fails to compile, and the
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

---

## 6. Proposed tensor4all equivalent

### 6.1 Three separate things: placement, engine context, runtime

| Concept | Owned by | Chosen by | tensor4all type |
| --- | --- | --- | --- |
| Placement (host, or a CUDA allocation domain) | the value | where the value was created or transferred | `IdxTensor` storage |
| Engine context (engine plus graph and plan caches) | the caller | the caller, **per operation** | `ExecutionContext` / `CpuExecutionContext` / `CudaExecutionContext` |
| Runtime (eager/AD identity of values) | shared by the values that may be combined | the caller, when contexts are built | `EagerRuntime`, held by `Arc` |

- **A context references a runtime; it does not have to own one.** Add
  `CpuExecutionContext::from_backend_with_runtime(backend, Arc<EagerRuntime>)`, and the
  CUDA equivalent. `from_backend` keeps creating a private runtime, which is right for a
  program with one engine.
- **Parallel lanes** build one runtime and N contexts that share it. Their values combine
  freely, and each operation runs on the context its caller passes.
- **The runtime is not a backend.** Sharing it chooses no engine, so a shared runtime does
  not conflict with E2.

### 6.2 Parameter shape

Every computing operation takes `&ExecutionContext`. Keep `ExecutionContext` as the
public type: it is the CPU/CUDA union that #869 needs and the C API maps it to one
handle; a bare `&mut dyn Engine` cannot carry the typed re-entry error
(`ExecutionContextEntryError`, `crates/tensor4all-tensorbackend/src/context.rs:1231`); and
it owns the graph and plan caches, which must outlive a session.

```rust
pub fn svd_in(t: &IdxTensor, opts: &SvdOptions, context: &ExecutionContext) -> Result<..>;
pub fn svd(t: &IdxTensor, opts: &SvdOptions) -> Result<..>;   // DELETE in phase D
```

- `_in` is the single *entry* spelling.
- `_on_session` stays as the single *composition* spelling. A caller that composes many
  operations under one session entry (the TCI/RSI sweeps; `treersi/src/engine.rs:110-120`)
  must not call an `_in` from inside it: that fails with
  `CpuExecutionContextError::Reentered`, and even where it cannot fail it pays one entry
  per operation.
- Third spellings are renamed to `_on_session` (`inner_product_with_session`,
  `simplett/src/contraction.rs:220`).
- **Operator overloads that compute are removed**, as tenferro removed them (§5.4).
  `AnyScalar`'s `Add`/`Sub`/`Mul`/... (`core/src/any_scalar.rs:1225+`) go through
  `try_*` → `with_eager_session_typed(.., None, ..)` and have no parameter slot. They
  become `add_in(&rhs, &context)` and so on.
  - Pure host arithmetic on an untracked scalar may stay an overload, because it calls
    no engine. Whether to keep that convenience is an open decision (§6.8).
- Rejected: making every function generic over `E: Engine`. It duplicates tenferro's
  trait and cannot express the union or the caches.

### 6.3 Validation: placement and runtime, not "created by"

`validate_context` is replaced by two checks at each operation boundary, made before any
RNG draw or kernel launch.

1. **Shared runtime.** Every operand with a runtime has the same one. A mismatch is a typed
   `RuntimeMismatch`, with the remedy "share one runtime between these contexts".
2. **Placement.** The context's engine can read every operand.
   - A CPU context reads host values, from whichever context created them.
   - A CUDA context reads values in its own allocation domain only.
   - A mismatch is a typed `PlacementMismatch`, with the remedy "transfer explicitly".

Mapping both checks to tensor4all's own error variants at this boundary also hides the
upstream variant inconsistency (§5.6) from tensor4all's callers.

**Runtime-less host values** (TCI materialisation, disk load, `from_dense_compact`) join
a runtime at the first operation that needs one: the operands' shared runtime if there is
one, else the passed context's. This replaces the 10 F5 adoptions into the global runtime
(§3.7). `adopt_into` is no longer needed for host-to-CPU use, and remains only as the
explicit host-to-device transfer.

### 6.4 Choosing CPU or GPU at run time

The caller picks the device by the context it passes, and data moves only when the caller
says so.

```rust
let cpu = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend_with_runtime(backend, rt.clone())));
let gpu = ExecutionContext::Cuda(Arc::new(CudaExecutionContext::with_runtime(device, rt.clone())?));
let a_dev = a.to_context(&gpu)?;          // explicit upload; same runtime, new placement
let c = contract_in(&a_dev, &b_dev, &gpu)?;
let c_host = c.to_context(&cpu)?;         // explicit download
```

- A host operand passed to a CUDA context is `PlacementMismatch`, never an implicit
  upload. This matches tenferro (§5.6).
- `to_context` is the single transfer verb. It is a copy across placements and a no-op
  when the placement already matches.
- A CPU/CUDA shared runtime needs the CUDA extension modules registered on that runtime.
  Stage A2 must verify this. Today each CUDA context builds its own runtime.

### 6.5 Error model

Typed, non-`anyhow` variants, one per boundary:

| New error | Replaces | At |
| --- | --- | --- |
| `IdxTensorError::RuntimeMismatch` | tenferro's `ContextMismatch` surfacing from inside a kernel | operation entry |
| `IdxTensorError::PlacementMismatch` | "tensor belongs to a different CPU eager context" and the per-path upstream variants | operation entry |
| `t4a_status_code::T4A_NO_CONTEXT` (new variant of `capi/src/lib.rs:53`) | n/a | C boundary, stage E1 |

No `NoLiveOwner` is needed. The owner registry and `with_runtime_engine` are deleted
outright (stage D2), because no operation resolves its engine from a value any more.

### 6.6 C API and bindings

The ABI needs an opaque context handle, not a new parameter type on 87 functions:

```c
/* proposed, stage E1 */
typedef struct t4a_context t4a_context;
t4a_status_code t4a_context_new(t4a_context **out);                                 /* CPU, own runtime */
t4a_status_code t4a_context_new_with_threads(size_t, t4a_context **out);
t4a_status_code t4a_context_new_sharing(const t4a_context *with, size_t threads, t4a_context **out); /* same runtime */
void            t4a_context_release(t4a_context *ctx);

t4a_status_code t4a_tensor_contract(const t4a_context *ctx, const t4a_tensor *a, ...);
```

- **A null context is `T4A_NULL_POINTER`, never "use the default".**
- **A cloned handle shares the same context.** `ExecutionContext::Cpu` holds
  `Arc<CpuExecutionContext>`, so cloning creates no second engine. Exposing
  `t4a_context_clone` is a policy choice, not a constraint.
- **No implicit global context remains reachable.** This is a breaking ABI change, which
  the early-development policy ([AGENTS.md "Development Stage"](../../AGENTS.md)) permits.
- **Python** wraps `Arc<CpuExecutionContext>` in a `Context` class and calls the Rust
  `_in` entries directly.

### 6.7 Bindings impact summary

| Consumer | Impact | Work |
| --- | --- | --- |
| `Tensor4all.jl` (external) | breaking: every numerical call gains a context argument | one `t4a_context` per Julia session (more for lanes) |
| Python (`tensor4all-py`) | breaking Python API | `Context(...)`; `Tensor.contract(other, ctx)` |
| mdBook guide + tutorials | every snippet | mechanically add a context |
| Doctests | ~all crate-level examples | build one context per example |

Doctest churn is the largest single line count in this program. It is mechanical, but it
is not free.

### 6.8 Open decisions

1. **Host-only scalar overloads.** Keep `+`/`*` on untracked host `AnyScalar` (no engine),
   or remove every overload for uniformity.
2. **Runtime ownership.** Is the runtime a separate public handle (`Runtime::new()`,
   passed to `from_backend_with_runtime`), or only reachable through
   `CpuExecutionContext::share_runtime()`? The second needs no new public type.
3. **CPU/CUDA shared runtime.** Allow it in stage A2, or keep CPU and CUDA runtimes separate
   and make `to_context` move a value between runtimes. Stage A2 verification decides.

---

## 7. Staged migration order

### 7.0 Constraints from concurrent work

Many PRs land on `main` in parallel with this program, and almost all of them touch the
surface it changes. The staging is built around that.

- **Add before migrate, migrate before remove.** Until the removal phase, the old and the
  new forms coexist, so an open PR that calls the old form keeps compiling.
- **One crate per migration PR.** Each PR is small, merges quickly, and conflicts with
  only the PRs that touch the same crate. Order is leaf-first, so no PR waits on a
  downstream crate.
- **Stop new debt first.** A deterministic check rejects *added* implicit sites from the
  first PR onward. Debt cannot grow while it is being paid down, and other authors find
  out in review, not after a rebase.
- **No `#[deprecated]`.** The workspace builds clippy with `-D warnings`, so a deprecation
  would break every open PR that still calls the old form. The ratchet does the
  deprecation's job without failing existing code.
- **Removal PRs are breaking and announced.** Each removal PR is posted on #1376 before it
  merges, so authors of open PRs can rebase onto it.

Each stage below is one or more separately reviewable PRs that leave the tree green.

### Phase A — groundwork (additive, low conflict)

**A0 — this document.** No code.

**A1 — ratchet against new implicit sites.** Add a delta-scoped, deterministic check to
`scripts/repository-rules-review.py`, built on `added_lines_with_text` the way the existing
`tenferro` dependency and doctest checks are. It blocks *added* lines that:
- pass a literal `None` context to `with_{eager_session,eager_session_typed,engine,execution_engine}`;
- call `default_context()`, `default_eager_ctx()` or `default_cpu_execution_context()`;
- declare a new `Option<&..ExecutionContext>` parameter.

Existing sites are not affected. The check needs a test in
`scripts/test-repository-rules-review.py` and a route in `REPOSITORY_RULES.md`. No Rust
code changes, so it conflicts with nothing.

**A2 — decouple runtime from engine.** This is §6.1 and §6.3, in `tensorbackend/src/context.rs`
and `core/src/defaults/idx_tensor.rs` (`validate_context`) only.
- Add `from_backend_with_runtime`.
- Replace the runtime-identity check with the shared-runtime and placement checks.
- Add `RuntimeMismatch` and `PlacementMismatch`.
- Adopt runtime-less host values at operations that already take a context.

Purely additive in behaviour: it turns some rejections into successes, and it fixes
multi-engine CPU programs. Tests:
- Two CPU contexts with different thread counts share a runtime, and a value from each
  contracts on either context.
- Values from unshared runtimes give `RuntimeMismatch`.
- A host value passed to a CUDA context gives `PlacementMismatch`.

**A3 — `treersi/engine.rs:118` (F3c).** The CUDA LUCI arm takes the caller's CPU context or
returns a typed refusal. One function.

### Phase B — complete the explicit surface (additive, per crate)

Add every missing `_in` / `_on_session` form next to the existing implicit one. Nothing
is removed, so open PRs are unaffected. One PR per crate, leaf-first:

1. **`tensor4all-tensorbackend`**: parity between `explicit::Session` and the 10
   engine-bearing bridge functions (§3.9). Add `_in` twins for the Hermitian helpers and
   `IncrementalQr::{new, append}`. Move host-only helpers and `capped_full_piv_lu` out of
   the `global-defaults` gate.
2. **`tensor4all-core`**:
   - Move the `TensorElement`, `Matrix` and `ExecutionContext` re-exports to
     `explicit-context` (§4.1).
   - Add `_in` forms for every `IdxTensor` / `AnyScalar` value method and computing
     operator that today reaches an F1 site. For example, `a.add_in(&b, &ctx)` coexists
     with `a + b`.
   - Add `_in` twins for the bridge-level owned and backend functions.
3. **Algorithm crates**, one PR each: `simplett`, `treetn`, `itensorlike`, `tensorci`,
   `treetci`, `treersi`, `interpolativeqtt` (builders, §3.4), `aci`, `quanticstci`, and
   the rest. Rename third spellings to `_on_session` here; they are internal names and
   conflict little.

### Phase C — migrate callers (behaviour-preserving, per crate)

Same crate order as phase B. In each crate, internal callers, tests, examples and doctests
move to the explicit forms, and the F2 `Option<&..Context>` parameters become
`&ExecutionContext`. The crate's implicit public entries **stay**, as thin adapters, so
downstream crates and open PRs still compile.

- Each PR lowers that crate's F1/F2/F3 counts (§3) and records the new numbers in its body.
- Numeric parity (§8) is checked per crate.
- Where a crate's tutorial chapter quotes the migrated API, the mdBook chapter moves in
  the same PR (AGENTS.md "Online Tutorial Synchronization").

### Phase D — remove (breaking, announced, per crate then global)

1. **Per crate, leaf-last**, once no workspace caller of a crate's implicit entries
   remains: delete its F3 adapters, F4 twins and computing operator overloads. Downstream
   crates were migrated in phase C, so each PR touches one crate.
2. **The `None` route**: delete `with_runtime_engine`, the owner registry
   (`context.rs:1881-1893`) and the `None` arms in `eager_session.rs`. This removes core's
   import of a `global-defaults`-only function (§4.1).
3. **Globals and features**:
   - Delete F6's global items and F7's module gate.
   - Run the explicit-only per-crate check (§4.4).
   - Flip the defaults (`default = ["tenferro-cpu-platform"]` in `tensorbackend` and
     `treersi`) and drop `core`'s `global-defaults` requirement.
   - Delete `backend-tenferro` and `global-defaults`.
   - Delete the A1 ratchet, which has nothing left to guard.

### Phase E — bindings and docs

1. **C API**: the `t4a_context` handle and a non-null context argument on every
   numerical entry (§6.6). Regenerate the header with `scripts/generate-capi-header.py`.
   This lands in one PR because the ABI changes together. Coordinate it with a
   `Tensor4all.jl` PR following the cross-repo procedure in AGENTS.md.
2. **Python**: a `Context` type.
3. **Remaining docs**: `README.md`, `llms.txt`, `skills/use-tensor4all-rs/`,
   [explicit-cpu-execution-context.md](explicit-cpu-execution-context.md) and
   [859-f7-consumer-inventory.md](859-f7-consumer-inventory.md), brought to their
   post-removal state.

Phase E1 can start once phase C has reached `tensor4all-treetn`. The C API only wraps
`core` and `treetn` entries, so it does not have to wait for phase D.

### Mapping to the split issues

| Stage | Issue |
| --- | --- |
| A1 (ratchet) | none yet: needs a new issue |
| A2 (runtime/engine decoupling, placement validation) | none yet: needs a new issue |
| A3 (`treersi/engine.rs:118`) and the phase B/C/D steps for `treetn` adapters and `interpolativeqtt` builders | #1379 |
| D2 (`None` route deleted) and the phase C F1 migration in `core` | #1378. Its fix becomes "migrate, then delete the route", not "add `NoLiveOwner`". |
| B2 re-exports and D3 (core's `global-defaults` requirement, tensorbackend's internal `default_context()` calls) | #1377 |
| ACI `Option<&CpuExecutionContext>` (F2), phase C for `aci` | #1382 |
| Bridge and `Storage` re-exports (F7), B1 | #1380 |

---

## 8. Do not break numeric behavior

This is a **pure signature and selection** change. Every operation must produce
bit-identical results before and after each stage, and any stage that does not is a bug
in that stage.

Concretely:

- **Same kernel, different selection.** Phase C replaces *which* context is used,
  not *what runs*. A migrated `IdxTensor::sum` must call the same
  `with_eager_session(.., Some(ctx), ..)` body it already had with `None`.
- **Runtime sharing must not change results.** Stage A2 changes which runtime records a
  value, not which kernel runs. A value computed on context A with a private runtime and
  on context A with a shared runtime must be bit-identical.
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
  fail: stage A2 turns some rejections into successes (another CPU context's host value),
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
