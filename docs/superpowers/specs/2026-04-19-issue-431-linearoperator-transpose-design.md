# Design: Issue #431 — LinearOperator::transpose & affine pullback unification

**Date**: 2026-04-19
**Issue**: [tensor4all-rs#431](https://github.com/tensor4all/tensor4all-rs/issues/431)
**Related Julia issue**: [Tensor4all.jl#58](https://github.com/tensor4all/Tensor4all.jl/issues/58)
**Related past work**: PR #427 (added `_pullback_materialize`), PR #428/#430 (cleanup), issue #426

## Motivation

The pullback of an affine operator is mathematically just the transpose of the
forward operator:

- Forward: `M_{y,x} = δ(y − Ax − b)`
- Pullback: `P_{y,x} = M_{x,y}`, i.e., `P = Mᵀ`

Today `affine_pullback_operator` (Rust) and `t4a_qtransform_affine_pullback_materialize`
(C API) are maintained as separate code paths from `affine_operator` /
`t4a_qtransform_affine_materialize`. Both rely on the same underlying MPO from
`affine_transform_mpo`; only the index remapping differs, and that remapping is
equivalent to swapping the `input_mapping` and `output_mapping` of the resulting
`LinearOperator`.

Following the workspace guideline

> Higher-level operations should be implemented in Julia using C API primitives,
> not as new C API functions.

we remove the duplicated pullback path and expose transpose as the single seam
for deriving pullback at the binding layer.

## Scope

### In scope

- Add `LinearOperator::transpose()` — O(1) swap of `input_mapping` and
  `output_mapping`; no MPO copy.
- Remove `affine_pullback_operator` (Rust function).
- Remove `remap_affine_site_indices_pullback` and any other pullback-only
  helpers in `crates/tensor4all-quanticstransform/src/affine.rs`.
- Remove `t4a_qtransform_affine_pullback_materialize` from the C API.
- Unfold the `materialize_affine_family` shared helper in
  `crates/tensor4all-capi/src/quanticstransform.rs` back into a single
  `t4a_qtransform_affine_materialize` implementation (the family abstraction
  exists only to share code with the pullback variant).
- Update tests: rewrite existing pullback tests to use
  `affine_operator(...)?.transpose()` so equivalence is verified before removal;
  delete tests that exercise only now-removed symbols.
- Update `docs/CAPI_DESIGN.md` and any rustdoc referring to the removed symbols;
  add a note to `affine_operator` rustdoc that pullback is obtained via
  `.transpose()`.

### Not in scope

- Adding a C API transpose function. `Tensor4all.jl`'s `LinearOperator` is pure
  Julia, so transpose is implemented in Julia without an FFI round-trip.
- Changes to other quantics transforms (`shift`, `fourier`, `phase_rotation`,
  `cumsum`, `flip`) or their C API entry points.
- Generalizing transpose to other operator-like types beyond `LinearOperator`.
- Retrofitting historical design docs under `docs/plans/`.

## Design

### 1. `LinearOperator::transpose()`

Location: `crates/tensor4all-treetn/src/operator/linear_operator.rs`

```rust
impl<T, V> LinearOperator<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Returns the transposed operator by swapping input and output mappings.
    ///
    /// O(1): no tensor data is cloned; only `input_mapping` and
    /// `output_mapping` are swapped. Semantically, if `op` maps input
    /// coordinates `x` to output coordinates `y`, then `op.transpose()`
    /// maps `y` to `x`, i.e., its matrix is the transpose.
    pub fn transpose(self) -> Self {
        Self {
            mpo: self.mpo,
            input_mapping: self.output_mapping,
            output_mapping: self.input_mapping,
        }
    }
}
```

Rationale for a consuming `fn transpose(self) -> Self`:

- Zero-copy with no allocation: fields are moved.
- Matches the mathematical identity `transpose . transpose = id`; calling
  `.transpose().transpose()` yields a value equivalent to the original.
- Callers who need to keep the original can `.clone()` before `.transpose()`.

### 2. Affine pullback removal

Location: `crates/tensor4all-quanticstransform/src/affine.rs`

Remove:

- `affine_pullback_operator(r, params, bc) -> Result<QuanticsOperator>`
- `remap_affine_site_indices_pullback(mpo, source_ndims, output_ndims, site_dim)`
- Any pullback-only unfused variant if one exists.

Keep and document:

- `affine_operator(...)` — add a line to rustdoc: "To obtain the pullback,
  call `.transpose()` on the returned operator."
- `remap_affine_site_indices(...)` — unchanged.
- `affine_transform_mpo(...)` — unchanged (internal helper).

### 3. C API changes

Location: `crates/tensor4all-capi/src/quanticstransform.rs`

Remove:

- `t4a_qtransform_affine_pullback_materialize`.
- The `materialize_affine_family` shared helper if it existed solely to
  deduplicate forward/pullback code. Inline the body into
  `t4a_qtransform_affine_materialize`.

Keep unchanged:

- `t4a_qtransform_affine_materialize`.
- All other `t4a_qtransform_*_materialize` functions.

### 4. Documentation

- `docs/CAPI_DESIGN.md`: remove the `affine_pullback_materialize` entry from the
  exported surface list; remove any pullback-specific notes in the affine
  section. Add a brief note that pullback is derived at the binding layer via
  transpose.
- `crates/tensor4all-quanticstransform/src/affine.rs` rustdoc for
  `affine_operator`: note pullback via `.transpose()`.
- `crates/tensor4all-treetn/src/operator/linear_operator.rs`: rustdoc for
  `transpose()` with a short, runnable example including an assertion
  (per project rules, doc examples must assert correctness).

## Testing

### Unit tests

Location: `crates/tensor4all-treetn/src/operator/linear_operator.rs` (in-module
`#[cfg(test)]`).

- `transpose_swaps_mappings`: build a `LinearOperator` with distinguishable
  input/output mappings; call `.transpose()`; assert `input_mapping` equals
  the original `output_mapping` and vice versa; assert `mpo` is unchanged
  (same internal TreeTN contents).
- `transpose_is_involutive`: `op.clone().transpose().transpose()` is equal to
  `op` (field-by-field).
- Cover both `V = usize` (chain) and a small tree `V` if a cheap fixture
  exists; otherwise the chain case is sufficient.

### Equivalence / regression tests

Location: `crates/tensor4all-quanticstransform/src/affine/tests/mod.rs` and
`crates/tensor4all-quanticstransform/tests/integration_test.rs`.

Before deleting `affine_pullback_operator`:

- Add a focused equivalence test: for a representative set of `(r, params, bc)`
  including the cases covered by the current pullback tests, construct
  `pullback_old = affine_pullback_operator(...)?` and
  `pullback_new = affine_operator(...)?.transpose()`, materialize both to
  dense matrices, and assert equality to floating-point tolerance. This
  proves the refactor is correct.
- Run the full test suite to confirm all existing pullback tests pass via
  the new path.

Then:

- Rewrite each existing pullback test to call
  `affine_operator(...)?.transpose()` instead of `affine_pullback_operator(...)`.
- Delete the equivalence test (it has served its purpose) OR keep it as a
  long-lived invariant — decision during implementation based on cost and
  clarity. Default: keep, renamed to
  `affine_pullback_via_transpose_matches_direct_dense_matrix` and use the
  reference dense matrix construction directly (no call to the deleted
  function).

### C API tests

- Remove tests in `crates/tensor4all-capi/src/quanticstransform/tests/mod.rs`
  that exercise `t4a_qtransform_affine_pullback_materialize`. No replacement
  in C API tests: Julia-side tests (separate repo) are the appropriate place
  to verify the `affine + transpose` composition.

### Coverage guard

Per `AGENTS.md`, run locally before pushing:

```bash
cargo llvm-cov --workspace --exclude tensor4all-hdf5 --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

If coverage drops below thresholds, add tests (do not lower thresholds).

### Pre-PR checks

```bash
cargo fmt --all
cargo clippy --workspace
cargo nextest run --release --workspace
cargo test --doc --release --workspace
cargo doc --workspace --no-deps
./scripts/test-mdbook.sh
```

## Cross-repo workflow

The `AGENTS.md` rule is: merge the tensor4all-rs PR first, then bump the pin
in Tensor4all.jl in a follow-up PR. For a removal (not an addition), this is
safe because Tensor4all.jl continues to use the old pin until the Julia PR
updates it, at which point the Julia code also drops the removed call.

Concrete order:

1. **tensor4all-rs PR (this design)**: add `LinearOperator::transpose()`,
   remove `affine_pullback_operator` and `t4a_qtransform_affine_pullback_materialize`,
   update tests and docs. Run all pre-PR checks. Merge.
2. **Tensor4all.jl PR**: bump `deps/build.jl` pin to the merged commit above,
   add pure Julia `transpose` on the operator wrapper, migrate the
   `affine_pullback` path to `affine + transpose`, drop the direct
   `_materialize_affine_pullback` call. Merge.

The Julia CI keeps passing between steps 1 and 2 because it pins the old
tensor4all-rs commit until step 2 bumps the pin simultaneously with dropping
the removed symbol.

(The wording in issue #431 says "Tensor4all.jl follow-up PR merged first"; this
contradicts `AGENTS.md` and the mechanics of pinning. We follow `AGENTS.md`.)

## Acceptance checklist

- [ ] `LinearOperator::transpose()` added with rustdoc and runnable assertion example.
- [ ] Unit tests for `transpose` (mapping swap, involution).
- [ ] `affine_pullback_operator` removed; existing pullback tests migrated to
      `affine_operator(...)?.transpose()` and all pass.
- [ ] `remap_affine_site_indices_pullback` removed (and any pullback-only helpers).
- [ ] `t4a_qtransform_affine_pullback_materialize` removed from C API.
- [ ] `materialize_affine_family` inlined into the single forward path.
- [ ] `docs/CAPI_DESIGN.md` updated.
- [ ] `affine_operator` rustdoc notes pullback via `.transpose()`.
- [ ] `cargo fmt --all`, `cargo clippy --workspace`, `cargo nextest run --release --workspace`, `cargo test --doc --release --workspace`, `cargo doc --workspace --no-deps`, `./scripts/test-mdbook.sh` all pass.
- [ ] Coverage check (`cargo llvm-cov` + `scripts/check-coverage.py`) passes at current thresholds.
- [ ] Tensor4all.jl follow-up PR opened with pin bump and Julia-side transpose.
