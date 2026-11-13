# EagerTensor-Based Migration — Design Document (retired)

Status: retired 2026-11-11. This was the plan for moving tensor4all-rs onto the
2026-04 tenferro eager API; it was carried out by #411 (merged 2026-04-14). The
code has moved on since, and the plan no longer describes it. Do not follow it.
The original text is in git history:
`git show 928c567b:docs/design/symbolic-shape-and-tenferro-migration.md`.

Current design for the topics it covered:

- execution ownership (plain, graph and eager-AD): [explicit-cpu-execution-context.md](./explicit-cpu-execution-context.md)
- the core tensor type, renamed `TensorDynLen` -> `IdxTensor` in #612 (`a93f582c`): no design document describes it; its rustdoc in `crates/tensor4all-core/src/defaults/idx_tensor.rs` is the reference
- `AnyScalar` versus the backend scalar: [issue-659-public-type-disposition.md](./issue-659-public-type-disposition.md)
- `tensor4all-tcicore`, dissolved into `tensor4all-core`: [issue-639-dissolve-tcicore.md](./issue-639-dissolve-tcicore.md)

## Disposition by section

Each row says why the section is retired, with the code that contradicts it.
Paths are relative to `crates/`; line numbers are at the commit that retired
this document.

| Section | Disposition | What the code does instead |
|---|---|---|
| 1. Problem statement (crate table) | retired | The table says `tenferro-linalg` was merged into `tenferro-tensor`. It is still a separate crate and a direct dependency: `tensor4all-tensorbackend/Cargo.toml:81`. The symbol table maps to the 2026-04 API (`EagerContext`, `CpuContext`, `typed_eager_einsum`), which tensor4all-rs no longer uses (`typed_eager_einsum` has no reference in `crates/`). |
| 2. Design principles, item 4 (two-tier types) | retired | `TensorDynLen` does not wrap `EagerTensor` and no longer exists. `IdxTensor` holds indices, an authoritative `IdxTensorStorage` and a lazily built eager cache (`tensor4all-core/src/defaults/idx_tensor.rs:1364-1376`); the storage is representation-dependent: materialized `Storage`, an eager payload with `axis_classes`, or a compact structured payload (`idx_tensor.rs:636-643`). |
| 2. Design principles, item 5 (paired thread-local execution objects) | retired | There is no thread-local `CpuBackend` or eager context. A caller-owned `CpuExecutionContext` owns the engine, the graph compiler and the eager runtime (`tensor4all-tensorbackend/src/context.rs:1435-1439`). The remaining thread-locals in that file track context re-entry (`:1144-1150`). `with_default_backend` and `default_eager_ctx` exist only behind the `global-defaults` feature, as accessors to one process-global `CpuExecutionContext` (`tensor4all-tensorbackend/src/context.rs:1965-2080`), not a per-thread pair. |
| 2. Design principles, items 1-3, 6, 7 | retired | Process rules for the migration. The gradient lifecycle they describe (accumulating `backward`, explicit `clear_grad`) is documented on the methods themselves (`idx_tensor.rs:4078-4103`, `tensor4all-core/src/any_scalar.rs:591-622`). |
| 3.1 tensorbackend (Cargo.toml, backend.rs, bridge, context.rs) | retired | 3.1.1 and 3.1.4 remove `tenferro-linalg`; it is still a dependency (`tensor4all-tensorbackend/Cargo.toml:81`). 3.1.7 describes `default_eager_ctx` as the construction path; new code takes a caller-owned context (above). |
| 3.2 tcicore | retired | The crate no longer exists; see [issue-639-dissolve-tcicore.md](./issue-639-dissolve-tcicore.md). |
| 3.3.2 tensordynlen.rs (struct) | retired | The struct shown (`inner: EagerTensor<CpuBackend>`, `axis_classes`) is not the current one; see the row for item 4. |
| 3.3.3 core `AnyScalar` | retired | `AnyScalar` holds `Result<IdxTensor, AnyScalarTensorError>` (`tensor4all-core/src/any_scalar.rs:118-119`), not a rank-0 `TensorDynLen`. `enable_grad` and `detach` return `Result<Self, AnyScalarError>` (`tensor4all-core/src/any_scalar.rs:507`, `:653`), not `Self`. `try_from_tensor`, `as_tensor` and `into_tensor` do not exist on `AnyScalar`. |
| 3.3.5 AD surface on `TensorDynLen` | retired | The surface lives on `IdxTensor`, and `enable_grad` and `detach` return `Result<Self, IdxTensorError>` (`idx_tensor.rs:3963`, `:4125`), not `Self`. |
| 3.3.4, 3.3.6, 3.4.1, 3.4.3, 3.4.4, 3.5, 3.6 | retired | Per-file migration steps, done by #411; the files have changed since. |
| 3.4.2 simplett `einsum_helper.rs` | retired | The einsum adapter uses `TTScalar`, the scalar bound shared by simplett tensor types. |
| 4. Implementation order, 5. Testing strategy | retired | Completed plan. |
| 6. Risks | retired | Item 1 concerned layout at construction; the column-major convention is documented on `IdxTensor` (`idx_tensor.rs:1341-1345`). The others refer to the 2026-04 transition. |
