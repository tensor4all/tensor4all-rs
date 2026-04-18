# QuanticsTransform C-API Cleanup Implementation Plan (Issue #428)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the legacy `binaryop` operator family and the unused `Grouped` layout kind from the tensor4all-rs public Rust and C APIs, and improve docstring coverage for the remaining affine materialization entry points.

**Architecture:** Pure deletion plus documentation improvement. No new Rust or C API surface. `materialize_affine_family` stays Fused-only; Julia reimplements binaryop on top of existing primitives (`t4a_treetn_new`, `t4a_treetn_tensor`, `t4a_tensor_svd`) in a follow-up PR.

**Tech Stack:** Rust workspace crates (`tensor4all-quanticstransform`, `tensor4all-capi`), cbindgen for C header regeneration, cargo fmt/clippy/nextest, mdBook doctests.

**Design reference:** `docs/plans/2026-04-18-issue-428-qtransform-capi-cleanup-design.md`

---

### Task 1: Delete binaryop from the tensor4all-quanticstransform crate

**Files:**
- Delete: `crates/tensor4all-quanticstransform/src/binaryop.rs`
- Delete: `crates/tensor4all-quanticstransform/src/binaryop/tests/mod.rs`
- Modify: `crates/tensor4all-quanticstransform/src/lib.rs`
- Modify: `crates/tensor4all-quanticstransform/tests/integration_test.rs`

- [ ] **Step 1: Delete the binaryop source and tests**

```bash
rm crates/tensor4all-quanticstransform/src/binaryop.rs
rm -r crates/tensor4all-quanticstransform/src/binaryop
```

- [ ] **Step 2: Remove `mod binaryop;` and the re-exports from `lib.rs`**

In `crates/tensor4all-quanticstransform/src/lib.rs`, delete these lines:

```rust
mod binaryop;
```

and

```rust
pub use binaryop::{binaryop_operator, binaryop_single_operator, BinaryCoeffs};
```

- [ ] **Step 3: Remove binaryop tests from `integration_test.rs`**

In `crates/tensor4all-quanticstransform/tests/integration_test.rs`:

- Remove `binaryop_operator, binaryop_single_operator,` from the top-level `use` statement on line 23.
- Delete these test functions and their helpers (block around lines 2295–2700): `test_binaryop_identity_x`, `test_binaryop_sum`, `test_binaryop_difference`, `test_binaryop_single_numerical_correctness`, `test_binaryop_dual_output_numerical`, plus any interleaving helpers introduced only for these tests (block headed by the comment `// Bit interleaving helpers for binaryop tests`).
- Verify no remaining references with `grep -n "binaryop\|BinaryCoeffs" crates/tensor4all-quanticstransform/tests/integration_test.rs` — must return no matches.

- [ ] **Step 4: Build the crate to confirm no stale references**

Run: `cargo build -p tensor4all-quanticstransform --release`

Expected: build succeeds with no warnings about unused imports or dead code.

- [ ] **Step 5: Run the crate's tests**

Run: `cargo nextest run --release -p tensor4all-quanticstransform`

Expected: all remaining tests pass.

- [ ] **Step 6: Commit**

```bash
git add -A crates/tensor4all-quanticstransform
git commit -m "$(cat <<'EOF'
refactor(quanticstransform): remove legacy binaryop operator

binaryop is a historical artifact predating general affine support.
Removed ahead of Julia-side reimplementation on affine + tensor SVD.

Part of issue #428.
EOF
)"
```

---

### Task 2: Delete binaryop from the C API

**Files:**
- Modify: `crates/tensor4all-capi/src/quanticstransform.rs`
- Modify: `crates/tensor4all-capi/src/quanticstransform/tests/mod.rs`

- [ ] **Step 1: Remove `binaryop_operator` from the top-level `use` statement**

In `crates/tensor4all-capi/src/quanticstransform.rs` around line 15, remove `binaryop_operator,` from the `use tensor4all_quanticstransform::{...}` list. Expected result: line imports no longer mention binaryop.

- [ ] **Step 2: Delete the binaryop C entry point**

In `crates/tensor4all-capi/src/quanticstransform.rs`, delete the entire function starting with the doc comment `/// Materialize a binary operator directly as a chain-shaped TreeTN.` and spanning `pub extern "C" fn t4a_qtransform_binaryop_materialize(...) -> StatusCode { ... }` (approximately lines 889–979).

- [ ] **Step 3: Delete the binaryop helpers**

In the same file, delete:

- `fn fuse_binaryop_pairs(...)` and its helpers (around lines 502–543).
- `fn embed_two_var_fused(...)` (around lines 544–650) if it is only used by `t4a_qtransform_binaryop_materialize`. Confirm with `grep -n "embed_two_var_fused" crates/tensor4all-capi/src/quanticstransform.rs`; if the only remaining match is the definition, delete it.

- [ ] **Step 4: Delete binaryop tests from the capi test module**

In `crates/tensor4all-capi/src/quanticstransform/tests/mod.rs`, delete every `#[test]` and helper function that references `t4a_qtransform_binaryop_materialize` or `BinaryCoeffs`. Verify with `grep -n "binaryop\|BinaryCoeffs" crates/tensor4all-capi/src/quanticstransform/tests/mod.rs` — must return no matches.

- [ ] **Step 5: Build the capi crate and run tests**

Run: `cargo build -p tensor4all-capi --release` followed by `cargo nextest run --release -p tensor4all-capi`.

Expected: build succeeds, tests pass. Any orphaned imports surfaced as warnings must be cleaned up before moving on.

- [ ] **Step 6: Regenerate the C header**

Run from the repo root:

```bash
cbindgen crates/tensor4all-capi \
  --config crates/tensor4all-capi/cbindgen.toml \
  --output crates/tensor4all-capi/include/tensor4all_capi.h
```

Confirm the header no longer contains `t4a_qtransform_binaryop_materialize`:

```bash
grep -n "binaryop" crates/tensor4all-capi/include/tensor4all_capi.h
```

Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add -A crates/tensor4all-capi
git commit -m "$(cat <<'EOF'
refactor(capi): remove t4a_qtransform_binaryop_materialize

C API no longer exposes binaryop. Julia reimplements it on top of
existing primitives (t4a_treetn_new, t4a_treetn_tensor, t4a_tensor_svd).

Part of issue #428.
EOF
)"
```

---

### Task 3: Remove the `Grouped` layout kind

**Files:**
- Modify: `crates/tensor4all-capi/src/types.rs`
- Modify: `crates/tensor4all-capi/src/types/tests/mod.rs`
- Modify: `crates/tensor4all-capi/src/quanticstransform.rs`
- Modify: `crates/tensor4all-capi/include/tensor4all_capi.h` (regenerated)

- [ ] **Step 1: Remove the `Grouped` variant from the enum and renumber**

In `crates/tensor4all-capi/src/types.rs` around line 290, change:

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_qtt_layout_kind {
    Grouped = 0,
    Interleaved = 1,
    Fused = 2,
}
```

to:

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_qtt_layout_kind {
    Interleaved = 0,
    Fused = 1,
}
```

- [ ] **Step 2: Remove the `Grouped` branch from `InternalQttLayout::new`'s site-count computation**

In `crates/tensor4all-capi/src/types.rs` around line 328, remove the `t4a_qtt_layout_kind::Grouped => ...` match arm. The `Interleaved` and `Fused` arms stay unchanged.

- [ ] **Step 3: Delete the Grouped unit test**

In `crates/tensor4all-capi/src/types/tests/mod.rs` around line 56, delete the `let grouped = InternalQttLayout::new(t4a_qtt_layout_kind::Grouped, vec![3, 2]).unwrap();` test fixture and any assertions that depend on it.

- [ ] **Step 4: Remove the `Grouped` match arm from layout dispatch in quanticstransform**

In `crates/tensor4all-capi/src/quanticstransform.rs`:

- Around line 278, in `single_var_positions`, delete the `t4a_qtt_layout_kind::Grouped => { ... }` arm.
- Around line 441, in `materialize_single_var_operator`, replace `t4a_qtt_layout_kind::Grouped | t4a_qtt_layout_kind::Interleaved => { ... }` with just the `Interleaved` arm (keep its body unchanged).

- [ ] **Step 5: Build the capi crate and run tests**

Run: `cargo build -p tensor4all-capi --release` followed by `cargo nextest run --release -p tensor4all-capi`.

Expected: build succeeds, tests pass. Any `unreachable_patterns` warnings indicate missed matches — clean them up now.

- [ ] **Step 6: Regenerate the C header**

Run:

```bash
cbindgen crates/tensor4all-capi \
  --config crates/tensor4all-capi/cbindgen.toml \
  --output crates/tensor4all-capi/include/tensor4all_capi.h
```

Confirm:

```bash
grep -n "GROUPED\|Grouped" crates/tensor4all-capi/include/tensor4all_capi.h
```

Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add -A crates/tensor4all-capi
git commit -m "$(cat <<'EOF'
refactor(capi): remove unused t4a_qtt_layout_kind::Grouped variant

No C API function accepts Grouped today. Delete the enum value and
renumber Interleaved=0, Fused=1. If a future grid API needs a grouped
concept, reintroduce it with a concrete, used design.

Part of issue #428.
EOF
)"
```

---

### Task 4: Improve affine operator documentation

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine.rs`
- Modify: `crates/tensor4all-capi/src/quanticstransform.rs`
- Modify: `crates/tensor4all-capi/include/tensor4all_capi.h` (regenerated)

- [ ] **Step 1: Clarify the forward vs pullback split in `affine_operator`**

In `crates/tensor4all-quanticstransform/src/affine.rs` around line 310, replace the short docstring on `affine_operator` with:

```rust
/// Create the operator that realizes the coordinate map `y = A * x + b`.
///
/// This is the **forward** affine operator. It maps a quantics tensor train
/// representing an `N`-variable state `x` to the quantics tensor train of
/// the `M`-variable state `y = A * x + b`.
///
/// For the **pullback** direction (`f(y) = g(A * y + b)`, used to compose a
/// function with an affine change of coordinates), see
/// [`affine_pullback_operator`].
///
/// Forward and pullback share the same underlying MPO construction
/// (`affine_transform_mpo`) but apply different site-index permutations so
/// that the resulting `LinearOperator`'s input and output dimensions match
/// the intended direction of the map.
///
/// # Arguments
///
/// * `r` — bits per variable (number of sites in the output MPO).
/// * `params` — rational `M × N` matrix `A` and `M`-vector `b` describing
///   the affine map.
/// * `bc` — length `M` array of boundary conditions for each output variable.
///   `Periodic` wraps output coordinates modulo `2^r`; `Open` zeroes the
///   out-of-range contributions.
///
/// # Errors
///
/// Returns an error if `r == 0` or if `bc.len() != params.m`.
```

The existing `# Examples` section immediately below stays unchanged.

- [ ] **Step 2: Clarify `affine_pullback_operator`**

Similarly in `crates/tensor4all-quanticstransform/src/affine.rs` around line 390, replace the short docstring on `affine_pullback_operator` with:

```rust
/// Create the operator that realizes the pullback `f(y) = g(A * y + b)`.
///
/// This is the **pullback** (function-composition) operator. It maps a
/// quantics tensor train for an `M`-variable function `g` to the quantics
/// tensor train of the `N`-variable function `f = g ∘ (A y + b)`.
///
/// For the **forward** direction (the coordinate map `y = A * x + b`
/// itself), see [`affine_operator`].
///
/// Forward and pullback share the same underlying MPO construction
/// (`affine_transform_mpo`) but apply different site-index permutations so
/// that the resulting `LinearOperator`'s input and output dimensions match
/// the intended direction of the map.
///
/// # Arguments
///
/// * `r` — bits per variable.
/// * `params` — rational `M × N` matrix `A` and `M`-vector `b` describing
///   the affine map whose pullback is constructed.
/// * `bc` — length `M` array controlling how each source coordinate
///   `(A y + b)[i]` is treated when it leaves the representable interval
///   `[0, 2^r)`. `Periodic` wraps the coordinate; `Open` zero-extends.
///
/// # Errors
///
/// Returns an error if `r == 0` or if `bc.len() != params.m`.
```

The existing `# Examples` section immediately below stays unchanged.

- [ ] **Step 3: Document the C API entry points**

In `crates/tensor4all-capi/src/quanticstransform.rs`, replace the one-line doc comments on `t4a_qtransform_affine_materialize` and `t4a_qtransform_affine_pullback_materialize` with:

```rust
/// Materialize the forward affine operator `y = A * x + b` as a chain-shaped
/// TreeTN using the Fused QTT layout.
///
/// `a_num[i + k * m]` and `a_den[i + k * m]` hold the numerator and
/// denominator of `A[i, k]` (column-major, length `m * n`, where `i`
/// is the row index 0..m and `k` is the column index 0..n). `b_num[i]`
/// and `b_den[i]` describe the `i`-th component of `b` (length `m`).
/// `bc[i]` is the boundary condition applied to output coordinate `i`.
/// The resulting TreeTN has `layout->nsites()` nodes, each with fused
/// input and output site indices of dimensions `2^n` and `2^m`
/// respectively.
///
/// See also [`t4a_qtransform_affine_pullback_materialize`] for the pullback
/// direction `f(y) = g(A * y + b)`.
///
/// # Errors
///
/// Returns `T4A_INVALID_ARGUMENT` if `m == 0`, `n == 0`, `layout->kind()`
/// is not `Fused`, `b_den[i] == 0`, or `a_den[i + k * m] == 0`.
```

and

```rust
/// Materialize the pullback operator `f(y) = g(A * y + b)` as a chain-shaped
/// TreeTN using the Fused QTT layout.
///
/// Argument layout matches [`t4a_qtransform_affine_materialize`]; the
/// difference is in how the site indices of the resulting `LinearOperator`
/// are permuted so that the input encodes `g`'s `m`-variable quantics state
/// and the output encodes `f`'s `n`-variable quantics state.
///
/// `bc[i]` controls how source coordinate `i` is treated when
/// `(A * y + b)[i]` leaves the valid interval. `Periodic` wraps the
/// coordinate, while `Open` zero-extends it.
///
/// # Errors
///
/// Returns `T4A_INVALID_ARGUMENT` if `m == 0`, `n == 0`, `layout->kind()`
/// is not `Fused`, `b_den[i] == 0`, or `a_den[i + k * m] == 0`.
```

- [ ] **Step 4: Build and verify doctests compile**

Run: `cargo build -p tensor4all-quanticstransform --release` followed by `cargo test --doc --release -p tensor4all-quanticstransform -p tensor4all-capi`.

Expected: both doc builds succeed, all example assertions pass.

- [ ] **Step 5: Regenerate the C header**

Run:

```bash
cbindgen crates/tensor4all-capi \
  --config crates/tensor4all-capi/cbindgen.toml \
  --output crates/tensor4all-capi/include/tensor4all_capi.h
```

Verify the updated doc comments propagated into the header:

```bash
grep -n "forward affine operator\|pullback operator" crates/tensor4all-capi/include/tensor4all_capi.h
```

Expected: matches inside the two affine function doc blocks.

- [ ] **Step 6: Commit**

```bash
git add -A crates
git commit -m "$(cat <<'EOF'
docs(qtransform): clarify forward vs pullback affine operators

Rust and C API docs now distinguish the y = Ax + b direction from the
f(y) = g(Ay + b) pullback, document arguments (a_num/a_den/b_num/b_den,
m, n, bc), and point readers to the complementary function.

Part of issue #428.
EOF
)"
```

---

### Task 5: Update workspace documentation and run full verification

**Files:**
- Modify: `docs/CAPI_DESIGN.md`
- Modify: `docs/book/src/guides/quantics.md`
- Modify: `docs/design/quanticstransform_julia_comparison.md` (only if it still references `binaryop_operator` or Grouped layout after the above tasks)

- [ ] **Step 1: Scrub `docs/CAPI_DESIGN.md`**

Remove the bullet at line 156 that reads `- Grouped` and the bullet at line 167 that reads `- t4a_qtransform_binaryop_materialize`. Adjust surrounding prose so the lists still parse as valid Markdown (e.g., drop a leading "supports the following layouts:" caveat if the remaining list is self-evident).

- [ ] **Step 2: Scrub `docs/book/src/guides/quantics.md`**

Delete the `| Binary Operation | f(x, y), first variable -> a*x + b*y | binaryop_single_operator |` table row (around line 30) and the `BinaryCoeffs(-1, -1) for binaryop_single_operator` bullet (around line 45). Rewrite nearby prose so the guide still flows without the binaryop row.

- [ ] **Step 3: Scrub any remaining workspace references**

Run:

```bash
grep -rn "binaryop\|t4a_qtransform_binaryop_materialize\|t4a_qtt_layout_kind::Grouped\|T4A_QTT_LAYOUT_KIND_GROUPED" \
    crates docs README.md 2>/dev/null
```

Expected: the only matches are in `docs/plans/2026-04-18-issue-428-qtransform-capi-cleanup-design.md` (this PR's design doc) and `docs/plans/2026-04-18-issue-428-qtransform-capi-cleanup-plan.md` (this plan). Any other match must be removed or rewritten in this step.

- [ ] **Step 4: Run mdBook doctests**

Run: `./scripts/test-mdbook.sh`

Expected: PASS.

- [ ] **Step 5: Run the full workspace verification suite**

Run in order (stop and fix on any failure):

```bash
cargo fmt --all
cargo clippy --workspace
cargo nextest run --release --workspace
cargo test --doc --release --workspace
cargo doc --workspace --no-deps
```

Expected: all succeed.

- [ ] **Step 6: Commit**

```bash
git add -A docs
git commit -m "$(cat <<'EOF'
docs: remove binaryop and Grouped layout references

Cleanup after the Rust and C API deletions in this PR.

Part of issue #428.
EOF
)"
```

---

## Post-Implementation Steps (after tensor4all-rs PR merges)

These are deferred items — do **not** start them until the PR from tasks 1–5 is merged to main.

- [ ] Open a Tensor4all.jl issue titled "Follow-up: reimplement `binaryop_operator` on top of affine + tensor SVD (tracks tensor4all-rs #428)". Link the merged tensor4all-rs PR and summarise the required Julia-side changes:
  - Reimplement `binaryop_operator` / `binaryop_operator_multivar` using `affine_pullback_operator` + tensor-level SVD via the exposed C API primitives.
  - Drop `_materialize_binaryop`.
  - Remove Grouped layout references.
  - **Update layout enum constants in `src/TensorNetworks/backend/capi.jl`:** change `_T4A_QTT_LAYOUT_INTERLEAVED = Cint(1)` → `Cint(0)` and `_T4A_QTT_LAYOUT_FUSED = Cint(2)` → `Cint(1)`. The Rust-side enum was renumbered to close the gap left by removing `Grouped`.
  - Update tests.

- [ ] Reply to [#428](https://github.com/tensor4all/tensor4all-rs/issues/428) linking the merged PR. Summarise: forward/pullback kept (docs clarified), `_materialize` retained for consistency, `binaryop` removed with Julia-side follow-up issue linked, `Grouped` removed, docstrings expanded.

- [ ] Close issue #428.
