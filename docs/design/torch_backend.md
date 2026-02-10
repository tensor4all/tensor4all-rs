# Libtorch (tch-rs) Backend Integration

Design and implementation record for the Torch backend integration (Issue [#139](https://github.com/shinaoka/tensor4all-rs/issues/139), merged in PR [#143](https://github.com/tensor4all/tensor4all-rs/pull/143)).

## Goal

Feature-gated libtorch backend for **PyTorch tensor** operations with **autograd** support.

- **CPU only**
- **Default = mdarray**, Torch is **optional** (`backend-libtorch` feature)
- **Coexistence**: both mdarray and torch storage types can be used in the same build
- **Correctness first**, performance follow-ups are acceptable

## Implementation Status

| Phase | Status | PR |
|---|---|---|
| PR 1: Feature wiring + tch dependency | **Done** | #143 |
| PR 2: Storage coexistence + einsum facade | **Done** | #143 |
| PR 3: Wire tensor4all-core contraction routing | **Not started** | - |
| PR 4: Autograd surface on TorchStorage | **Done** | #143 |

## Dependency policy (important)

- **Do not depend on local clones** (e.g. `extern/tch-rs`) in the actual implementation.
- Use **crates.io** dependencies for Torch integration:
  - `tch = "..."`
- Keep Torch-related deps **feature-gated** so default builds do not pull libtorch tooling.

## Current architecture (relevant observations)

- `tensor4all-tensorbackend` currently exposes a **concrete mdarray-based** `Storage` and operations (`permute`, `contract`, etc.).
- `tensor4all-core`’s default contraction path already uses **`mdarray-einsum`** (hyperedge-aware optimizer).
- Issue #139 comment (2026-01-17) updates the direction:
  - **Avoid** integrating Torch via `MatMul<T>` (decomposing einsum into matmul loses native backend optimizations).
  - Prefer a backend abstraction at the **einsum** level:
    - mdarray backend uses `mdarray-einsum`
    - torch backend calls `torch.einsum()` directly (autograd + batch + native optimizations)

## Design decision: “Einsum-first” integration path

### Why

- `tensor4all-core` contraction is the most backend-sensitive hot path.
- PyTorch’s `einsum` provides:
  - autograd “for free”
  - contraction path optimization (opt-einsum integration)
  - supports complex tensors for many ops (with explicit failure for unsupported cases)

### What this means for v1

We integrate Torch by introducing a **minimal einsum backend interface** in `tensor4all-tensorbackend`, and then wiring `tensor4all-core` contraction to use it when `backend-libtorch` is enabled.

This lets us ship **autodiff-capable contraction** early, then expand parity for other ops incrementally.

## Concrete implementation plan (PR-sized)

### PR 1 — Feature wiring + Torch dependency (no behavior change by default) [DONE]

Files:
- `crates/tensor4all-tensorbackend/Cargo.toml`
- `crates/tensor4all-tensorbackend/src/lib.rs`

Changes:
- Add a new feature:
  - `backend-libtorch` (off by default)
- Add optional dependency from crates.io (feature-gated):
  - `tch = { version = "0.20", optional = true }`
- Keep existing defaults unchanged (`backend-faer` etc.).

Build notes:
- Prefer **system libtorch** via environment variables (no network):
  - `LIBTORCH=/path/to/libtorch`
  - `DYLD_LIBRARY_PATH=$LIBTORCH/lib` (macOS)
- Optionally allow `tch/download-libtorch` behind a *separate* opt-in feature if needed later.

Acceptance:
- `cargo test -p tensor4all-tensorbackend` passes with default features.

---

### PR 2 — Storage coexistence model + backend-neutral einsum facade [DONE]

Files (new + edits):
- **new** `crates/tensor4all-tensorbackend/src/einsum.rs`
- **new** `crates/tensor4all-tensorbackend/src/torch/` (module tree, feature-gated)
- `crates/tensor4all-tensorbackend/src/lib.rs`
- `crates/tensor4all-tensorbackend/src/storage.rs` (add a backend dimension)

Storage model (required by this issue)
- Keep mdarray storage always available (default).
- When `backend-libtorch` is enabled, add torch storage types and allow both to exist:
  - `MdarrayStorage` (current implementation)
  - `TorchStorage` (new)
- Expose a unified `Storage` surface that can represent either backend at runtime, e.g.:
  - `pub enum Storage { Mdarray(MdarrayStorage), #[cfg(feature="backend-libtorch")] Torch(TorchStorage) }`
  - (Exact naming is bikesheddable; the key is **runtime coexistence**.)

API (v1):
- Introduce a single public entry point for einsum over `Storage`:

  - **Public function** (bikesheddable name):
    - `pub fn einsum_storage(inputs: &[EinsumStorageInput], output_ids: &[u32]) -> anyhow::Result<Storage>`
  - `EinsumStorageInput` holds:
    - axis IDs (same ID semantics as `mdarray-einsum`)
    - a `&Storage` reference

Backend selection:
- Dispatch based on the input `Storage` backend(s):
  - If **all** operands are mdarray → use `mdarray-einsum` (status quo).
  - If **any** operand is torch → run via torch (`tch::Tensor::einsum`) and return torch storage.
    - Mixed-backend inputs (mdarray + torch) are allowed by converting mdarray operands to torch (v1 correctness-first).

Copy-minimization requirements (must)
- **Zero-copy fast path**:
  - If all operands are torch, `einsum_storage` must pass the underlying `tch::Tensor` references to `einsum` without cloning/copying data.
- **Mixed backend** (mdarray + torch):
  - Converting mdarray → torch may require a copy; in v1 we accept **at most one data copy per mdarray operand**.
  - Do not create additional temporary dense buffers beyond what is strictly required for the conversion.
  - Avoid extra `.contiguous()` / `.to_kind()` passes; do dtype promotion **once** at the boundary.
- **mdarray-only path**:
  - Keep the current `mdarray-einsum` API usage that operates on slices/views; do not materialize intermediate dense tensors unless required by existing behavior.
- **Diag**:
  - If v1 uses “densify diag in torch backend”, this is an explicit data expansion copy.
  - Ensure densification happens **only** when an operand is diag (not unconditionally), and only once per operand.

Follow-up (optional, for further copy reduction)
- Investigate a safe zero-copy bridge for mdarray → tch using `from_blob`-style APIs if available,
  with strict lifetime guarantees (likely requires unsafe + careful ownership design).

Key detail: equation construction
- Convert axis IDs (u32) to equation letters:
  - For v1, restrict to \( \le 52 \) unique labels (`a-zA-Z`).
  - If exceeded, return an error with guidance (future: multi-char labels or ellipsis packing).
- Map each input’s axes to a string like `"abC"`.
- Build equation `"abC,bd->aCd"` style.

Diag handling decision (v1):
- Choose **Option B: densify** in torch backend for now:
  - If operand is `Diag`, convert to dense before einsum.
  - Document the perf trade-off and add a follow-up issue to preserve diag semantics later.

Complex handling (v1):
- Use native PyTorch complex tensors for `Complex64`.
- If `tch`/PyTorch errors for unsupported ops/dtypes, return an explicit error:
  - include: operation name (`einsum`), dtype, equation, and an action item.

Acceptance:
- `cargo test -p tensor4all-tensorbackend` passes (default).
- `cargo test -p tensor4all-tensorbackend --features backend-libtorch` compiles (may be skipped in CI until libtorch is available, but should build locally).
 - In torch-enabled builds, add at least one targeted test/benchmark that asserts we do not introduce
   “double conversion copies” for mixed-backend einsum (e.g. via code-level invariants or profiling hooks).

---

### PR 3 — Wire `tensor4all-core` contraction to the einsum facade under libtorch [NOT STARTED]

Files:
- `crates/tensor4all-core/src/defaults/contract.rs`
- possibly small glue in `crates/tensor4all-tensorbackend/src/storage.rs` (re-exports)

Approach:
- When `backend-libtorch` is enabled, route “multi-tensor contraction / hyperedge cases” through:
  - `tensor4all_tensorbackend::einsum_storage`
- Keep the existing mdarray-einsum path intact for default builds.

Important: preserve current semantics
- Type promotion rules remain:
  - all f64 → f64 output
  - any Complex64 → Complex64 output
- Shape/rank behavior and index-ID collision rules stay consistent.

Acceptance:
- Default:
  - `cargo test -p tensor4all-core`
- Torch:
  - `cargo test -p tensor4all-core --features backend-libtorch` (best-effort; may be gated if libtorch not present in CI).

---

### PR 4 — Autograd surface (minimal, Dense f64 first) [DONE]

Files:
- `crates/tensor4all-tensorbackend/src/storage.rs` (torch-only additions behind cfg)
- **new** `crates/tensor4all-tensorbackend/tests/torch_autograd.rs` (cfg-gated)

API (implemented on `TorchStorage<T>`):
- `TorchStorage::requires_grad() -> bool`
- `TorchStorage::set_requires_grad(bool) -> &mut Self`
- `TorchStorage::grad() -> Option<TorchStorage<T>>`
- `TorchStorage::backward() -> Result<()>` (scalar output only)
- `TorchStorage::detach() -> Self`

Note: Autograd methods are on `TorchStorage` directly, not on the unified `Storage` enum.
Users must work with `TorchStorage` to use autograd.

Tests:
- `x` scalar: `y = x * x` gradient is `2x`
- small contraction gradient:
  - build `A` and `B` with requires_grad
  - `C = einsum_storage("ij,jk->ik")`
  - `loss = C.sum()`
  - backward and compare gradients against finite-diff or PyTorch reference
- **Double backward**: one tiny example if supported by the op chain.

Acceptance:
- `cargo test -p tensor4all-tensorbackend --features backend-libtorch` passes locally with libtorch installed.

## Implemented Files

```
crates/tensor4all-tensorbackend/
├── Cargo.toml                          # backend-libtorch feature, tch = "0.20"
├── src/
│   ├── lib.rs                          # re-exports torch module
│   ├── einsum.rs                       # backend-neutral einsum facade
│   ├── storage.rs                      # Storage enum with TorchF64/TorchC64 variants
│   └── torch/
│       ├── mod.rs                      # torch module
│       └── storage.rs                  # TorchStorage<T> + autograd + tests
```

## Non-goals (v1)

- GPU support
- Changing non-contraction ops to be fully torch-backed in v1 (focus is einsum/contraction + minimal autodiff)
- Full coverage of all tensor ops beyond what core contraction requires

## Risk register / mitigations

- **Toolchain friction (libtorch availability)**:
  - keep everything feature-gated
  - avoid `download-libtorch` by default
- **Equation label limit**:
  - enforce a clear error with next steps
- **Complex autograd gaps**:
  - fail fast with explicit error messages, add follow-up issues per missing op
- **Diag performance**:
  - densify for correctness; add follow-up for diag-preserving implementation

## Developer commands (recommended)

Default backend:

```bash
cargo test -p tensor4all-tensorbackend
cargo test -p tensor4all-core
```

Torch backend (CPU, assumes libtorch available locally):

```bash
export LIBTORCH=/path/to/libtorch
export DYLD_LIBRARY_PATH="$LIBTORCH/lib:$DYLD_LIBRARY_PATH"
cargo test -p tensor4all-tensorbackend --features backend-libtorch
cargo test -p tensor4all-core --features backend-libtorch
```

