# QuanticsTransform C-API Cleanup Design (Issue #428)

## Goal

Address design concerns raised in [#428](https://github.com/tensor4all/tensor4all-rs/issues/428) for the QuanticsTransform C API surface:

- Reduce redundancy between `affine`, `affine_pullback`, and `binaryop` entry points.
- Simplify the `t4a_qtt_layout_kind` enum by removing the currently-unused `Grouped` variant.
- Improve parameter documentation for affine materialization functions.

## Context

[#428](https://github.com/tensor4all/tensor4all-rs/issues/428) lists five concerns about the current C API:

1. Forward `affine_materialize` and `affine_pullback_materialize` look redundant.
2. The `_materialize` suffix is verbose.
3. `binaryop_materialize` appears redundant against `affine_pullback_materialize`.
4. The `Grouped` layout kind is questionable.
5. Documentation is scarce.

Investigation revealed the following facts that shape this design:

- **Forward vs pullback are not just direction inverses.** They have different site index permutations (`remap_affine_site_indices` vs `remap_affine_site_indices_pullback`), swap the roles of (M, N) in input/output dimensions, and serve distinct use cases (coordinate map vs function composition).
- **`_materialize` is consistent across the whole quanticstransform C API** (`fourier_materialize`, `shift_materialize`, `flip_materialize`, `cumsum_materialize`, `phase_rotation_materialize`, `binaryop_materialize`, `affine_materialize`, `affine_pullback_materialize`). Renaming affine alone would break consistency; renaming everything is pure churn.
- **`binaryop` is a historical artifact.** In Quantics.jl, `binaryop.jl` was introduced in 2022-11 for 2-variable matmul-style operations. `affine.jl` was added in 2024-09 as a general-purpose replacement. The specialized 2-variable interleaved carry structure saves only a constant factor (≈4× per site for M=N=1, independent of R) compared to unfusing an affine MPO. Scaling in R is linear for both paths.
- **`Grouped` layout is accepted by `t4a_qtt_layout_new` but rejected by every affine/binaryop materialization path.** It is a placeholder for future grid-based APIs whose shape is not yet committed.
- **Tensor4all.jl exports all three families**, but BubbleTeaCI does not use any of them directly yet. There are no external Rust crate consumers outside the capi tests.
- **The C API already exposes enough primitives for layout transformations outside the materialize path.** `t4a_treetn_new`, `t4a_treetn_tensor`, `t4a_treetn_set_tensor`, `t4a_tensor_svd`, `t4a_tensor_qr`, `t4a_tensor_contract`, and `t4a_tensor_new_dense_{f64,c64}` are sufficient for Julia to decompose a fused affine MPO into any layout via tensor-level operations.

## Non-Goals

- Do not rename `_materialize` functions — the convention is established and renaming is churn without user benefit.
- Do not remove `affine_operator` (forward). It expresses a mathematically distinct linear map from `affine_pullback_operator` and has legitimate applications (symmetry group actions, coordinate transforms).
- Do not add new C API functions. Existing tensor-level and TreeTN primitives are enough for Julia to replicate the binaryop workflow.
- Do not extend `materialize_affine_family` to additional layouts. Keep the Rust implementation Fused-only and push per-variable site splitting to the Julia layer.
- Do not redesign Tensor4all.jl wrappers in this change. That is a follow-up PR with its own compatibility path.
- Do not preserve backwards compatibility for `t4a_qtransform_binaryop_materialize` or `t4a_qtt_layout_kind::Grouped`. The repository is in early development and deprecated compatibility layers are removed immediately.

## Design Decisions

### Decision 1: Keep both forward and pullback affine (clarify docs)

Forward (`affine_operator`) and pullback (`affine_pullback_operator`) remain as separate entry points. Rustdoc and C header docs are updated to state explicitly:

- Forward: `y = A * x + b` where `x` is an `N`-variable quantics input and `y` is an `M`-variable quantics output. Used for representing a coordinate map as a linear operator.
- Pullback: `f(y) = g(A * y + b)` where `g` is an `M`-variable quantics function and `f` is the resulting `N`-variable quantics function. Used for composing a function with an affine change of coordinates.

The explanatory comment also clarifies that they share the same underlying MPO construction but differ in how site indices are permuted for the downstream linear-operator conversion.

### Decision 2: Keep `_materialize` naming

No rename. Retained for cross-operator consistency with `fourier_materialize`, `shift_materialize`, `flip_materialize`, `cumsum_materialize`, and `phase_rotation_materialize`.

### Decision 3: Remove binaryop; push layout transforms to Julia

`t4a_qtransform_binaryop_materialize` is removed. The Rust `materialize_affine_family` stays Fused-only and gains no new code paths. Two-variable interleaved affine maps become a Julia-level responsibility that reuses already-exposed C API primitives:

1. Julia calls `t4a_qtransform_affine_pullback_materialize` (or forward) with Fused layout to get a TreeTN of R nodes, each with fused input (dim `2^N`) and output (dim `2^M`) site indices.
2. Julia extracts each node's tensor via `t4a_treetn_tensor`, reshapes it into M+N dim-2 site indices (exact reshape, no SVD), and applies sequential `t4a_tensor_svd` calls to split it into M+N separate tensors.
3. Julia builds the new interleaved TreeTN via `t4a_treetn_new` on the resulting tensor list.

The following Rust-level items are deleted:

- `crates/tensor4all-quanticstransform/src/binaryop.rs`
- `crates/tensor4all-quanticstransform/src/binaryop/tests/mod.rs`
- Exports in `crates/tensor4all-quanticstransform/src/lib.rs` (`binaryop_operator`, `binaryop_single_operator`, `binaryop_single_mpo`, `BinaryCoeffs`).
- Any binaryop-specific integration tests in `crates/tensor4all-quanticstransform/tests/integration_test.rs`.
- C API: `t4a_qtransform_binaryop_materialize` and its test module entries.

`materialize_affine_family` continues to reject non-Fused layouts. It does not grow new conditional branches.

### Decision 4: Remove `Grouped` layout kind

`t4a_qtt_layout_kind::Grouped` is deleted along with:

- The corresponding C enum value in `tensor4all_capi.h`.
- `materialize_single_var_operator` branches that accept or reject `Grouped`.
- Any internal layout dispatch that mentions `Grouped`.
- Docstrings referencing the Grouped layout on other transforms.

If a future grid-aware API reintroduces a grouped concept, it is added at that time with a concrete, used design — not as a pre-committed placeholder.

### Decision 5: Improve affine documentation

The Rust function docs for `affine_operator`, `affine_pullback_operator`, and the C API entry points gain an `# Arguments` section explaining:

- `r`: bits per variable.
- `a_num`, `a_den`: numerators and denominators of the `M × N` matrix `A`, stored column-major, length `m * n`.
- `b_num`, `b_den`: numerators and denominators of the `M`-element translation vector `b`.
- `m`, `n`: output and input variable counts.
- `bc`: array of length `m` of boundary conditions for each output variable.

The C header reflects the same information on the `t4a_qtransform_affine_materialize` and `t4a_qtransform_affine_pullback_materialize` doc comments (generated by `cbindgen`).

## API Changes

### Rust API

- Remove: `binaryop_operator`, `binaryop_single_operator`, `binaryop_single_mpo`, `BinaryCoeffs`, `binaryop_mpo`, `binaryop_tensor_single` (all in the deleted `binaryop.rs`).
- Unchanged signatures: `affine_operator`, `affine_pullback_operator`.
- Documentation-only updates on `affine_operator`, `affine_pullback_operator`, and the `AffineParams` struct.

### C API

- Remove: `t4a_qtransform_binaryop_materialize`.
- Remove: `t4a_qtt_layout_kind::Grouped` enum value.
- No new functions added. No additional layout kinds accepted by `materialize_affine_family`.
- Regenerate `crates/tensor4all-capi/include/tensor4all_capi.h`.

### Downstream impact (Tensor4all.jl)

Handled in a separate PR after the tensor4all-rs pin hash is updated:

- Reimplement `binaryop_operator` and `binaryop_operator_multivar` at the Julia level using the affine pullback + tensor-level SVD workflow described in Decision 3.
- Delete the `_materialize_binaryop` helper (no longer backed by a C symbol).
- Keep the Julia public API (`binaryop_operator`, `binaryop_operator_multivar`) stable so existing downstream calls continue to work — only the implementation changes.
- Update `@testset "binaryop_operator"` to exercise the reimplemented Julia path.
- Remove references to Grouped layout in Julia wrappers and tests.

## Testing Strategy

### Rust unit tests

- Verify that `t4a_qtt_layout_kind::Grouped` is no longer constructible (the enum value does not exist).
- Verify that `t4a_qtransform_binaryop_materialize` no longer exists (capi tests that referenced it are deleted or replaced with affine coverage).

### C API tests

- Delete the binaryop test module entries that targeted `t4a_qtransform_binaryop_materialize`.
- No new C API surface to test — affine materialize coverage is unchanged.

### Documentation tests

- Ensure all new rustdoc examples are runnable with assertions, per project convention (no `ignore` / `no_run`).
- Run `cargo test --doc --release --workspace` and `./scripts/test-mdbook.sh`.

### Julia-side (follow-up PR)

- Reimplemented `binaryop_operator` must reproduce the old behavior on the existing `@testset "binaryop_operator"` fixtures.
- Compare dense evaluation on sample vectors to catch tolerance drift.

## Risks

### Tensor4all.jl breakage window

Between merging this PR and updating Tensor4all.jl, the Julia-side build will fail because `_materialize_binaryop` references a symbol that no longer exists. Mitigation: follow the documented cross-repo protocol — merge tensor4all-rs first, then update the pin, then open the Julia PR.

### Julia-side SVD cost for binaryop reimplementation

Reconstructing binaryop on top of affine + per-site SVD is a constant-factor slower than the specialized Rust implementation (≈4× per site for M=N=1, independent of R). Mitigation: document the trade-off. If a concrete use case surfaces, the specialized path can be reintroduced later as a Rust-internal helper, but the public API does not grow.

## Follow-Up

- Tensor4all.jl PR reimplementing binaryop on the affine + SVD workflow and removing Grouped layout references.
- Docs (`docs/book/src/guides/quantics.md`) updated to describe the affine-only approach.
- Close issue #428 once both PRs are merged.
