# Explicit engine for every operation — inventory and migration plan

**Issue:** [tensor4all-rs#1376](https://github.com/tensor4all/tensor4all-rs/issues/1376)
(supersedes the narrower [#1375](https://github.com/tensor4all/tensor4all-rs/issues/1375))

**Status:** design and inventory only. Nothing in this document is implemented; no
public signature, feature default, or numeric path changes.

**Baseline:** `origin/main` at `20b42b0d`, read from a fresh worktree. Every table
below names the command that regenerates it, and every `path:line` was read at that
revision. Re-derive before acting on a row — see §10.

## 0. The one-line problem

`tensor4all-rs` already owns a caller-supplied execution context
(`CpuExecutionContext`, [explicit-cpu-execution-context.md](explicit-cpu-execution-context.md)),
and `tenferro` has already deleted its own implicit entry points. What remains is that
the **whole public surface is still reachable without one**, in eight distinct shapes,
and one of them silently substitutes the process-global engine *after* the caller
passed a value that had an owner. The goal is not to build a context abstraction; it is
to stop the surface from being able to run without naming one.

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
Four properties follow, and this document judges every call site by them.

- **E1 — the parameter is mandatory.** A public numerical operation either takes the
  execution context/engine, or cannot run. `Option<&ExecutionContext>` with `None` =
  "implicit" fails E1 by construction. One exception is deliberate (§6.2): an operation
  with no parameter slot, such as an `AnyScalar`/`IdxTensor` operator overload or value
  method, may resolve the engine from its operands' **owner token**. That token was bound by
  an explicit context when the value was created, and it fails with a typed error when the
  token is absent or dead. It never falls back to a global context.
- **E2 — no process-global resolution, not even as a fallback.** Reaching
  `default_context()`, `default_eager_ctx()`, or `default_cpu_execution_context()`
  anywhere inside an operation body fails E2, including in the "the value lost its
  owner" branch.
- **E3 — no owner-less value adoption.** A value created without owning context must not
  be silently attached to one; the operation that needs an engine reports a typed
  "value has no execution context" error instead.
- **E4 — one parameter shape per role.** There are exactly two roles. *Entry* (`_in`,
  `&ExecutionContext`) enters the context once. *Composition* (`_on_session`,
  `&mut explicit::Session`) runs inside a session that is already held. A third spelling,
  or two spellings for the same role, fails E4 (§3.11, §6.1).

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
already gone. §6 stage 3 makes repairing this a prerequisite, not a follow-up.

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

This is the **E3** violation, and it is the reason `IdxTensor` is not merely a value: the
owner token already exists but is optional, and absence is resolved *at the value*, not
at the operation. `crates/tensor4all-core/src/defaults/idx_tensor.rs:2167` is the sharpest case — the code picks the common
operand runtime when all operands agree, and falls back to the global runtime when they
do not, i.e. a mixed-origin contraction silently runs on the global engine.

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
| `backend.rs` | `svd_backend`, `qr_backend`, `solve_backend`, `triangular_solve_backend`, `solve_matrix{,_owned}`, `triangular_solve_matrix{,_owned}`, `full_piv_lu_backend`, `full_piv_lu_matrix{,_owned}`, `src_error_estimate{,_general}` | yes — direct `with_default_session` at `:755, :874, :1008, :1509, :1555, :1598, :1634, :1781` |
| `matrix.rs` | `hermitian_eigendecomposition` (`:796`), `lowest_hermitian_eigenpair` (`:736`), `hermitian_exponential_first_column` (`:894`), `submatrix_argmax` (`:1484`), `mat_mul{,_owned}`, `batched_mat_mul_same_shape{,_owned}` | `hermitian_eigendecomposition` uses `default_eager_ctx()` **and** `default_context()`; `submatrix_argmax` uses `with_default_session`; `mat_mul` has `mat_mul_in` |
| `tenferro_bridge.rs` | 33 public functions | `reshape_col_major_native_tensor:1309`, `qr_native_tensor:1322`, `svd_native_tensor:1336`, `sum_native_tensor:1350`, `permute_native_tensor:1963` use `with_default_session`; `native_tensor_from_vec:52` and `convert_tensor:665` too, and every einsum/contraction path goes through `run_cached_native_einsum:744` / `execute_native_einsum_reads:861`, which use `default_context()` |
| `incremental_qr.rs` | `IncrementalQr::{new:196, append:253, q_columns:465, error_estimate:534}` | via `factorize_backend:545` → global |
| `capped_full_piv_lu.rs` | `capped_full_piv_lu:245` | **no** — `factor_with_faer:260` is pure host faer; this module only needs an ungated home, not a context |
| `storage.rs`, `any_scalar.rs`, `memory.rs`, `tensor_element.rs` | value types, elementwise arithmetic, allocator pressure relief | `any_scalar` arithmetic routes through `IdxTensor` (F1) |

`capped_full_piv_lu` is the useful counter-example: the module gate is coarse, so a
removal plan can move it to `explicit-context` unchanged.

### 3.10 F8 — bindings

| Binding | State | Evidence |
| --- | --- | --- |
| C (`tensor4all-capi`) | 87 `extern "C"` functions (`tensor.rs` 29, `treetn.rs` 32, `index.rs` 16, `quanticstransform.rs` 9, `lib.rs` 1). No context parameter, no context handle, no initialisation call. Every call runs on the global default. | `crates/tensor4all-capi/src/`; opaque handles only in `types.rs` |
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

Dropping the requirement therefore needs two things. Stage 2 step 2 moves the re-exports
and splits `mat_mul` (which already has `mat_mul_in`). Stage 1 plus §6.2 move
`with_runtime_engine`, without its global fallback, out of the `global-defaults` gate.
Only after both does stage 3 step 6 drop the requirement.

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
equality that holds today rests partly on it, so stage 3 needs parity evidence, not just
deletion (§8).

### 4.4 Crates that compile only through feature unification

`crates/tensor4all-interpolativeqtt/Cargo.toml:51` asks `tensorbackend` for
`explicit-context` only, yet `interpolation.rs:7` imports `default_cpu_execution_context`,
which `tensorbackend/src/lib.rs:69-71` gates on `global-defaults`. The crate compiles only
because `core` pins `global-defaults` (§4.1) and Cargo unifies features across the build.
When stage 3 step 6 drops that pin, every crate in this situation fails to compile, and the
manifests do not show it. Before that step, run an explicit-only `cargo check` per crate,
or `cargo hack check --each-feature`, so the list comes from the compiler rather than a
grep.

---

## 5. Comparison with tenferro

Anchored to the pinned revision
`tenferro* = { git = "https://github.com/tensor4all/tenferro-rs.git", rev = "32a4e98c12f9dc36ffa85b5273b9247b44936959" }`
(`Cargo.toml:114-121`), whose own design record is
`tenferro-rs/docs/design/execution-engine-ownership.md` (tenferro-rs #2206).

### 5.1 The engine trait and how it is obtained

```rust
// tenferro-rs/tenferro-tensor/src/backend.rs:3606
pub trait Engine: TensorBackendOps + SessionCachedDot + TensorDeviceTransfer { .. }
```

`CpuEngine::new(pool: tenferro_cpu::Pool)` takes the caller's built pool by value
(`tenferro-rs/tenferro-cpu/src/engine.rs:119`); `CpuEngine::serial()` runs on the calling thread.
There is no `CpuEngine::from_env`, no process-wide engine and no engine registry:

> "tenferro never subdivides a pool, keeps no process-wide engine registry and does not
> choose which engine runs a task."
> — `tenferro-rs/docs/design/execution-engine-ownership.md` §2

### 5.2 The parameter shape

Every operation takes the engine as a plain `&mut` argument on an extension trait, so the
call site reads as one fluent chain and there is no separate "session" object to build:

```rust
// tenferro-rs/tenferro-einsum/src/concrete.rs:137
fn einsum(&self, subscripts: &str, session: &mut dyn Engine) -> Result<Tensor>;

// tenferro-rs/tenferro-linalg/src/tensor_ext.rs:361 (doctest)
let (_u, s, _vt) = a.svd(&mut host)?;
```

Prepared programs are the same shape one level up: `ExecutionEngine: Engine`
(`tenferro-rs/tenferro-runtime/src/runtime/engine.rs:150`) adds `target()`, extension-cache
borrowing and run retirement, and a run binds engine slot `i` to the `i`-th supplied
engine for that call only.

### 5.3 The eager/AD layer takes the engine too

```rust
// tenferro-rs/tenferro-ad/src/eager.rs:3143
pub fn with_eager_session<T, E: From<Error>>(
    self: &Arc<Self>,
    engine: &mut dyn ExecutionEngine,
    f: impl FnOnce(&mut EagerSession<'_>) -> Result<T, E>,
) -> Result<T, E>
```

`EagerTensor::to_tensor(&self, engine: &mut dyn Engine)` (`tenferro-rs/tenferro-ad/src/eager.rs:546`),
`detach(&self, engine: &mut dyn Engine)` (`:4822`). The runtime stores no engine; the
engine stores no graph.

### 5.4 How tenferro prevents silent selection

It removed the alternative rather than guarding it. From
`tenferro-rs/docs/design/explicit-session-boundary.md`:

> "The former `EagerTensor` operation methods and arithmetic overloads were removed, not
> replaced with implicit-entry shims; value/trace handles remain, as do named backward and
> runtime boundaries."

Cross-context misuse is a typed error, `tenferro_ad::Error::ContextMismatch`
(`tenferro-rs/tenferro-ad/src/eager.rs:927` and elsewhere), not a silent re-home.

### 5.5 Where tensor4all differs, and why it matters

| Axis | tenferro | tensor4all today |
| --- | --- | --- |
| Who owns the engine | the caller, always | the caller for `_in`, a singleton for everything else |
| The parameter | `&mut dyn Engine` (concrete ops), `&mut dyn ExecutionEngine` (prepared/eager) | `&ExecutionContext` (a union) *and* `&mut explicit::Session<'_>` (a borrow) — two shapes |
| Eager value token | the `Arc<EagerRuntime>` the value belongs to; a foreign one is `ContextMismatch` | the same `Arc<EagerRuntime>`, but absence is filled from the global runtime |
| Implicit entry points | removed | 8 families still present |
| Bindings | n/a (Rust-first) | a C ABI with no context concept |

The two designs are closer than this table makes them look: `explicit::Session` *is* the
tenferro shape, and `EagerRuntime`-per-value *is* tenferro's context token. What is
missing is that neither is mandatory.

---

## 6. Proposed tensor4all equivalent

### 6.1 Parameter shape: one type, one spelling

Keep `ExecutionContext` as the public token and make it the **only** explicit parameter
type. Rationale:

- It is the CPU/CUDA union that #869 needs, and the C API will map it to one opaque
  handle; a bare `&mut dyn Engine` cannot carry the typed CUDA/CPU re-entry error
  (`ExecutionContextEntryError`, `crates/tensor4all-tensorbackend/src/context.rs:1231`) that the existing contract depends on.
- It already owns the graph cache and plan cache, which must outlive a session.
- `explicit::Session` stays as the *inner* handle — what `with_concrete_session` lends —
  so a stage pays one entry, not one per operation.

```rust
// proposed public shape (addition only)
pub fn svd_in(t: &IdxTensor, opts: &SvdOptions, context: &ExecutionContext) -> Result<..>;
pub fn svd(t: &IdxTensor, opts: &SvdOptions) -> Result<..>;   // DELETE at stage 3
```

`_in` becomes the single suffix for "takes `ExecutionContext`" (the *entry* role in E4).
The 32 `_on_session` entries are **not** folded wholesale. They exist so that a caller can
compose many operations under one session entry; `treersi/src/engine.rs:110-120` and the
TCI/RSI sweeps work this way. Calling an `_in` from inside a held session on the same
context fails with `CpuExecutionContextError::Reentered`, and even where that cannot happen
it pays one entry per operation. Stage 2 therefore:

- keeps `_on_session` as the single *composition* spelling;
- deletes an `_on_session` only where no caller holds a session around it;
- removes any third spelling (`_with_session`, e.g. `inner_product_with_session`) by
  renaming it to `_on_session`.

Rejected: making every function generic over `E: Engine`. It duplicates tenferro's trait
and cannot express the union or the caches (§9).

### 6.2 Context token on values

`IdxTensor` already carries the token (`EagerTensor` → `Arc<EagerRuntime>`). Make it
unable to be absent *or* silently filled:

- `IdxTensor` gains a typed owner state. A host-materialised payload keeps
  `Owner::Unbound`, meaning "no engine has been chosen for this value yet". It is **not**
  a promise to use the global one.
- An operation that needs an engine and finds `Owner::Unbound` returns a typed
  `IdxTensorError::NoExecutionContext`, with the value and the operation named.

  Exception, kept deliberately: an operation whose *only* parameter is a host snapshot
  and whose result is a fresh host value (elementwise arithmetic on `Scalar`, metadata
  reshapes) needs no engine and continues to work unbound — this is what keeps
  `IdxTensor::from_dense(vec![..])` (the crate's front-page doctest pattern) legitimate.
  Stage 1 must classify every F1 site into "needs an engine" and "host-only" before any
  signature changes; the 53 F1 sites are not all engine operations.
- `with_runtime_engine`'s `None` branch becomes `ExecutionContextEntryError::NoLiveOwner`.
  A runtime whose owner was dropped is an error, not a silent re-home.
- **The owner resolver is kept, ungated, and is not part of the global surface.** Operator
  overloads (`AnyScalar`'s `Add`/`Sub`/`Mul`/... at `core/src/any_scalar.rs:1225+`, through
  `try_*` → `with_eager_session_typed(.., None, ..)`) and context-free value methods have no
  parameter slot. For them, owner resolution *is* the explicit context: the one that created
  the value. Once its global fallback is gone, `with_runtime_engine` (renamed to
  `with_owner_engine` when it is ungated) is a pure value-token lookup and satisfies E1–E3.
  The alternative is to remove the arithmetic overloads, as tenferro did (§5.4). That is
  rejected here because it would break every `AnyScalar` expression in the workspace and
  the bindings for no gain in explicitness.

This is the minimum that satisfies E3. It also means **no** `IdxTensor` field changes
shape and **no** `EagerRuntime` reference changes — only the resolution decision moves
from value construction to operation entry.

### 6.3 Error model

Add one variant per boundary, typed and non-`anyhow`:

| New error | Replaces | At |
| --- | --- | --- |
| `IdxTensorError::NoExecutionContext { operation }` | `default_eager_ctx()?` at the 10 F5 sites | value-level |
| `ExecutionContextEntryError::NoLiveOwner { runtime: ContextId }` | `context.rs:1950` | entry-level |
| `t4a_status_code::T4A_NO_CONTEXT` (new variant of `capi/src/lib.rs:53`) | n/a | C boundary, stage 4 |

`ContextMismatch` already exists upstream and is preserved as a typed source.

### 6.4 C API and bindings

The ABI has no room for a `&ExecutionContext`, so it needs a new opaque handle, not a new
parameter on 87 functions:

```c
/* proposed, stage 4 — additive */
typedef struct t4a_context t4a_context;
t4a_status_code t4a_context_new(t4a_context **out);            /* owns one CpuExecutionContext */
t4a_status_code t4a_context_new_with_threads(size_t, t4a_context **out);
void            t4a_context_release(t4a_context *ctx);

/* operations gain an explicit, non-nullable context argument */
t4a_status_code t4a_tensor_contract(const t4a_context *ctx, const t4a_tensor *a, ...);
t4a_status_code t4a_treetn_contract(const t4a_context *ctx, const t4a_treetn *a, ...);
```

Rules this must follow, from [CAPI_DESIGN.md](../CAPI_DESIGN.md):

- **Opaque handle, `*_new(..., out)`, `*_release`, `*_clone` where meaningful.**
  `ExecutionContext::Cpu` holds `Arc<CpuExecutionContext>`, so `t4a_context_clone` would
  share the **same** context. It would not create a second engine. Exposing it is a policy
  choice, useful for handing a context to another binding object, not a constraint.
- **A null context is `T4A_NULL_POINTER`, never "use the default".** That is the whole
  point of the change; a nullable context would reintroduce F2 at the ABI.
- **No implicit global context remains reachable.** Once stage 4 lands, a binding that does
  not hold a `t4a_context` cannot execute numerics at all. Julia's `Tensor4all.jl` and the
  Python bindings both need one context created per session and threaded through — this is
  a breaking ABI change, which early-development policy
  ([AGENTS.md "Development Stage"](../../AGENTS.md)) permits.
- **Python** does not need a C-ABI detour: `tensor4all-py` can take a `PyContext` class
  wrapping `Arc<CpuExecutionContext>` and pass it to the Rust `_in` entries, which is
  cheaper than mirroring the C design.

### 6.5 Bindings impact summary

| Consumer | Impact | Work |
| --- | --- | --- |
| `Tensor4all.jl` (external) | breaking: every numerical call gains a context argument | one `t4a_context` per Julia session, created at init, released at finalize |
| Python (`tensor4all-py`) | breaking Python API | `Context(...)` object; `Tensor.contract(other, ctx)` |
| mdBook guide + tutorials | every snippet | mechanically add a context; the guide is the main discovery surface |
| Doctests | ~all crate-level examples | `let ctx = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?)?;` in each |

Doctest churn is the largest single line count in this program and is worth stating
up front: it is mechanical, but it is not free.

---

## 7. Staged migration order

Each stage is a separately reviewable PR, and each leaves the tree green on its own. The
order is chosen so that no stage has to both add and remove at once.

### Stage 0 — this document (no code)

Inventory, comparison, proposal. **Nothing else in this PR.**

### Stage 1 — the smallest first PR: make the silent fallback an error

**PR #1 (smallest).** Files: `crates/tensor4all-tensorbackend/src/context.rs`
(+ `context/tests/`), and whatever fixtures assert the current behaviour.

- Add `ExecutionContextEntryError::NoLiveOwner`.
- Replace `crates/tensor4all-tensorbackend/src/context.rs:1950`'s global substitution with that variant.
- Update the `with_runtime_engine` doc contract and add one regression test: drop a
  context, keep its eager runtime, call a `None`-context operation, assert the typed
  error. Plus one test that a live owner still resolves.

Why this one first: it is the single change that makes the contract *true* — a caller who
passed a value with an owner can no longer be silently moved to another engine — it
touches one function, and it is the only change in the program that can reveal a latent
bug rather than a compile error. It is also exactly #1375's finding 2, which #1376
supersedes.

Cost: an unknown number of existing tests and doctests currently rely on the fallback
(`core/src/defaults/idx_tensor.rs` alone has 43 F1 sites, and a test that drops a context
before using a tensor is the shape that breaks). Stage 1 must therefore *first* run the
suite with the variant in place and report how many sites depend on the fallback. If the
number is large, stage 1 is split: add the variant and a hit counter, then flip.

### Stage 2 — API addition: complete the explicit surface

Ordered by dependency, not by crate count.

1. **`tensor4all-tensorbackend`**: bring `explicit::Session` to parity with
   `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`. 25 of the 33 public bridge functions have no explicit twin
   today. Of those, the engine-bearing ones are `scale_native_tensor`,
   `axpby_native_tensor`, `contract_native_tensor`, `outer_product_native_tensor`,
   `einsum_native_tensor_reads`, `einsum_native_subscripts`,
   `contract_storage_native`, `outer_product_storage_native`, `scale_storage_native`,
   `axpby_storage_native`; the rest are host/dtype/metadata helpers that need no context
   and should move out of the `global-defaults` gate rather than gain a twin. Also add
   explicit twins for `hermitian_eigendecomposition` / `lowest_hermitian_eigenpair` /
   `hermitian_exponential_first_column` and `IncrementalQr::{new, append}`.
2. **`tensor4all-core`**: split the `global-defaults` requirement (§4.1). Move
   `TensorElement`, `Matrix` and `ExecutionContext` re-exports to `explicit-context`;
   give `mat_mul_owned`, `solve_backend`, `triangular_solve_backend`,
   `solve_matrix_owned`, `triangular_solve_matrix_owned`, `full_piv_lu_matrix_owned`,
   `src_error_estimate{,_general}`, `qr_backend`/`svd_backend` (bridge-level) their `_in`
   twins. Fold `_on_session` into `_in` (§6.1).
3. **`tensor4all-treetn` / `itensorlike` / `simplett` / `treetci` / `tensorci` / `treersi`**:
   they already have the twins; this step is only the `capped_full_piv_lu` un-gating and
   the `_on_session` → `_in` fold.
4. **`crates/tensor4all-treersi/src/engine.rs:118`** (F3c): thread the caller's CPU-side context into the
   CUDA LUCI arm, or return a typed "CUDA LUCI is unsupported" refusal. Must land before
   stage 3.

### Stage 3 — internal callers, then removal

1. Classify the 53 F1 sites three ways (§6.2) and migrate them:
   - **(a)** an enclosing function that has, or gains, a context parameter → `Some(ctx)`;
   - **(b)** an operator overload or value method with no parameter slot → keeps `None`,
     which now means owner-token resolution with a typed error, never the global context;
   - **(c)** host-only → no engine.

   This is the bulk of the mechanical work, and it is confined to `tensor4all-core`.
2. Migrate the 33 F2 `Option<&..Context>` parameters to `&ExecutionContext`.
3. Delete the F3 implicit adapters (`crates/tensor4all-treetn/src/treetn/contraction.rs:3151,3227`,
   `crates/tensor4all-treersi/src/api.rs:34`, `crates/tensor4all-interpolativeqtt/src/interpolation.rs:925`).
4. Delete the F4 implicit twins.
5. Delete F6's global items (`mod defaults`, `default_context`, `with_default_*`,
   `default_eager_ctx`, `default_cpu_execution_context`, the default-engine pool helpers)
   and F7's module gate, and move `capped_full_piv_lu` to `explicit-context`.
   `with_runtime_engine` is **not** deleted. It was already ungated and stripped of its
   fallback (stage 1, §6.2) and stays as the owner resolver.
6. Run the explicit-only per-crate check from §4.4 and fix every crate it finds. Then flip the default: `default = ["tenferro-cpu-platform"]` in `tensorbackend`,
   `treersi`, and drop `core`'s hard `global-defaults` requirement. `backend-tenferro`
   and `global-defaults` are then deleted, not deprecated (early-development policy).

### Stage 4 — bindings

C `t4a_context` handle + non-null context argument on every numerical entry (§6.4);
regenerate the header with `scripts/generate-capi-header.py`; Python `Context` type;
`Tensor4all.jl` follow-up in its own repo.

### Stage 5 — docs

Guide, tutorials, `README.md`, `llms.txt`, `skills/use-tensor4all-rs/`, every doctest.
Update [explicit-cpu-execution-context.md](explicit-cpu-execution-context.md) and
[859-f7-consumer-inventory.md](859-f7-consumer-inventory.md) to their post-removal state
rather than leaving them describing a deleted route.

### Mapping to the split issues

| Stage / item | Issue |
| --- | --- |
| Stage 1 (`NoLiveOwner`) and F1 classification | #1378 |
| §4.1 core's `global-defaults` requirement, tensorbackend's internal `default_context()` calls | #1377 (lands with or after #1378) |
| F3, F3c (`treetn` adapters, `interpolativeqtt::compress_train`, `treersi/engine.rs:118`) | #1379 |
| ACI `Option<&CpuExecutionContext>` (F2) | #1382 |
| Bridge and `Storage` re-exports (F7) | #1380 |

---

## 8. Do not break numeric behavior

This is a **pure signature and selection** change. Every operation must produce
bit-identical results before and after each stage, and any stage that does not is a bug
in that stage.

Concretely:

- **Same kernel, different selection.** Stage 3's step 1 replaces *which* context is used,
  not *what runs*. A migrated `IdxTensor::sum` must call the same
  `with_eager_session(.., Some(ctx), ..)` body it already had with `None`.
- **The `is_global_default_cpu` branch is the one real numeric risk.** `context.rs:1076`
  selects the legacy host kernels for the global context, documented as
  "bitwise-identical numerics". Once the global route is gone, that branch is gone and
  every call takes the scoped primitives. Stage 3 must therefore run a parity harness —
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
- **Errors change, results do not.** Stages 1 and 2 legitimately turn a silent success
  into a typed error. That is a behaviour change and must be stated in the PR body; it is
  not a numeric change. Any test that was passing *because* of the fallback should be
  rewritten to construct its context explicitly, not silenced.
- **`capped_full_piv_lu` must stay bit-identical across its un-gating**: it is pure faer
  (`crates/tensor4all-tensorbackend/src/capped_full_piv_lu.rs:260`) and moving its module must not change its algorithm.

Verification for stage 1 in particular: `cargo test -p tensor4all-tensorbackend`,
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

The `xtask api-dump` inventory
(`cargo run -p xtask --release -- api-dump`) covers all 18 public crates and is the
authoritative check that a public entry point exists or has been removed — it was used to
build §3.2, §3.6 and §3.11.
