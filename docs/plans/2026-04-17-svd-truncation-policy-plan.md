# SVD Truncation Policy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `rtol/cutoff`-based SVD truncation APIs with an explicit `SvdTruncationPolicy`, remove LU/CI from truncate APIs, and propagate the new policy through core, TreeTN, and itensorlike interfaces.

**Architecture:** Introduce an SVD-only policy type in `tensor4all-core`, keep QR options independent, and make high-level truncation APIs explicitly SVD-based. Use `FactorizeOptions` as the only mixed-algorithm facade, with algorithm-specific validation to reject invalid field combinations.

**Tech Stack:** Rust workspace crates (`tensor4all-core`, `tensor4all-treetn`, `tensor4all-itensorlike`), cargo test/doc tooling, rustdoc examples.

---

### Task 1: Replace shared truncation params with `SvdTruncationPolicy`

**Files:**
- Modify: `crates/tensor4all-core/src/truncation.rs`
- Modify: `crates/tensor4all-core/src/lib.rs`
- Test: `crates/tensor4all-core/src/truncation/tests/mod.rs`

**Step 1: Write the failing tests**

Add tests covering:

```rust
#[test]
fn test_svd_truncation_policy_defaults() {
    let policy = SvdTruncationPolicy::new(1e-12);
    assert_eq!(policy.threshold, 1e-12);
    assert_eq!(policy.scale, ThresholdScale::Relative);
    assert_eq!(policy.measure, SingularValueMeasure::Value);
    assert_eq!(policy.rule, TruncationRule::PerValue);
}

#[test]
fn test_svd_truncation_policy_builders() {
    let policy = SvdTruncationPolicy::new(1e-8)
        .with_absolute()
        .with_squared_values()
        .with_discarded_tail_sum();
    assert_eq!(policy.scale, ThresholdScale::Absolute);
    assert_eq!(policy.measure, SingularValueMeasure::SquaredValue);
    assert_eq!(policy.rule, TruncationRule::DiscardedTailSum);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-core truncation`

Expected: FAIL with missing `SvdTruncationPolicy` / removed `TruncationParams` references.

**Step 3: Write minimal implementation**

- Remove `TruncationParams` and `HasTruncationParams`
- Keep `DecompositionAlg`
- Add `SvdTruncationPolicy`, `ThresholdScale`, `SingularValueMeasure`, `TruncationRule`
- Add validation helpers for threshold values
- Re-export the new types from `crates/tensor4all-core/src/lib.rs`

**Step 4: Run test to verify it passes**

Run: `cargo nextest run --release -p tensor4all-core truncation`

Expected: PASS

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/truncation.rs crates/tensor4all-core/src/truncation/tests/mod.rs crates/tensor4all-core/src/lib.rs
git commit -m "refactor(core): add explicit svd truncation policy"
```

### Task 2: Refactor `SvdOptions` and retained-rank logic

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/svd.rs`
- Test: `crates/tensor4all-core/src/defaults/svd/tests/mod.rs`
- Test: `crates/tensor4all-core/tests/linalg_svd.rs`

**Step 1: Write the failing tests**

Add tests that check:

```rust
#[test]
fn test_compute_retained_rank_relative_per_value() {
    let s = [10.0, 1.0, 0.1];
    let policy = SvdTruncationPolicy::new(0.2);
    assert_eq!(compute_retained_rank(&s, &policy), 2);
}

#[test]
fn test_compute_retained_rank_relative_squared_tail_sum() {
    let s = [4.0, 2.0, 1.0];
    let policy = SvdTruncationPolicy::new(0.05)
        .with_squared_values()
        .with_discarded_tail_sum();
    assert_eq!(compute_retained_rank(&s, &policy), 2);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-core svd`

Expected: FAIL because `SvdOptions` and retained-rank helpers still use `rtol`.

**Step 3: Write minimal implementation**

- Change `SvdOptions` to:

```rust
pub struct SvdOptions {
    pub max_rank: Option<usize>,
    pub policy: Option<SvdTruncationPolicy>,
}
```

- Replace `default_svd_rtol` with a default policy getter/setter
- Implement retained-rank helpers for:
  - relative/absolute
  - value/squared-value
  - per-value/discarded-tail-sum
- Preserve early-exit suffix accumulation for `DiscardedTailSum`

**Step 4: Run test to verify it passes**

Run: `cargo nextest run --release -p tensor4all-core svd`

Expected: PASS

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/svd.rs crates/tensor4all-core/src/defaults/svd/tests/mod.rs crates/tensor4all-core/tests/linalg_svd.rs
git commit -m "refactor(core): move svd truncation to explicit policy"
```

### Task 3: Decouple QR and validate mixed-algorithm factorization options

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/qr.rs`
- Modify: `crates/tensor4all-core/src/tensor_like.rs`
- Modify: `crates/tensor4all-core/src/defaults/factorize.rs`
- Test: `crates/tensor4all-core/src/defaults/qr/tests/mod.rs`
- Test: `crates/tensor4all-core/tests/linalg_factorize.rs`

**Step 1: Write the failing tests**

Add tests such as:

```rust
#[test]
fn test_factorize_options_reject_svd_policy_for_qr() {
    let opts = FactorizeOptions::qr()
        .with_svd_policy(SvdTruncationPolicy::new(1e-8));
    let err = validate_factorize_options(&opts).unwrap_err();
    assert!(err.to_string().contains("svd_policy"));
}

#[test]
fn test_factorize_options_reject_qr_rtol_for_lu() {
    let opts = FactorizeOptions::lu().with_qr_rtol(1e-8);
    let err = validate_factorize_options(&opts).unwrap_err();
    assert!(err.to_string().contains("qr_rtol"));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-core linalg_factorize qr`

Expected: FAIL because `FactorizeOptions` still only carries `rtol`.

**Step 3: Write minimal implementation**

- Refactor `QrOptions` to its own `rtol` field
- Refactor `FactorizeOptions` to:

```rust
pub struct FactorizeOptions {
    pub alg: FactorizeAlg,
    pub canonical: Canonical,
    pub max_rank: Option<usize>,
    pub svd_policy: Option<SvdTruncationPolicy>,
    pub qr_rtol: Option<f64>,
}
```

- Add validation before dispatch
- Update `factorize_svd`, `factorize_qr`, and LU/CI paths to consume the new fields

**Step 4: Run test to verify it passes**

Run: `cargo nextest run --release -p tensor4all-core linalg_factorize qr`

Expected: PASS

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/qr.rs crates/tensor4all-core/src/defaults/qr/tests/mod.rs crates/tensor4all-core/src/tensor_like.rs crates/tensor4all-core/src/defaults/factorize.rs crates/tensor4all-core/tests/linalg_factorize.rs
git commit -m "refactor(core): validate factorize options by algorithm"
```

### Task 4: Make TreeTN truncation explicitly SVD-based

**Files:**
- Modify: `crates/tensor4all-treetn/src/options.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/truncate.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/localupdate.rs`
- Test: `crates/tensor4all-treetn/src/options/tests/mod.rs`
- Test: `crates/tensor4all-treetn/tests/basic.rs`

**Step 1: Write the failing tests**

Add tests asserting:

```rust
#[test]
fn test_truncation_options_do_not_expose_form() {
    let opts = TruncationOptions::new().with_max_rank(8);
    assert_eq!(opts.max_rank(), Some(8));
}

#[test]
fn test_tree_truncate_uses_svd_policy() {
    let opts = TruncationOptions::new()
        .with_svd_policy(SvdTruncationPolicy::new(1e-10));
    assert_eq!(opts.svd_policy().unwrap().rule, TruncationRule::PerValue);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-treetn options basic`

Expected: FAIL because `TruncationOptions` still uses `form + rtol`.

**Step 3: Write minimal implementation**

- Remove `form` from `TruncationOptions`
- Add `svd_policy: Option<SvdTruncationPolicy>`
- Keep `CanonicalizationOptions` unchanged
- Thread `svd_policy` through `truncate_impl` into `TruncateUpdater`
- Replace `FactorizeOptions::svd().with_rtol(...)` in `TruncateUpdater` with `with_svd_policy(...)`

**Step 4: Run test to verify it passes**

Run: `cargo nextest run --release -p tensor4all-treetn options basic`

Expected: PASS

**Step 5: Commit**

```bash
git add crates/tensor4all-treetn/src/options.rs crates/tensor4all-treetn/src/options/tests/mod.rs crates/tensor4all-treetn/src/treetn/truncate.rs crates/tensor4all-treetn/src/treetn/localupdate.rs crates/tensor4all-treetn/tests/basic.rs
git commit -m "refactor(treetn): make truncate explicitly svd-based"
```

### Task 5: Propagate policy through TreeTN apply, contract, partial contraction, and linsolve

**Files:**
- Modify: `crates/tensor4all-treetn/src/operator/apply.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/contraction.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/partial_contraction.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/fit.rs`
- Modify: `crates/tensor4all-treetn/src/linsolve/common/options.rs`
- Test: `crates/tensor4all-treetn/src/operator/apply/tests/mod.rs`
- Test: `crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs`
- Test: `crates/tensor4all-treetn/src/linsolve/common/options/tests/mod.rs`

**Step 1: Write the failing tests**

Add tests that build high-level options with:

```rust
let policy = SvdTruncationPolicy::new(1e-10)
    .with_squared_values()
    .with_discarded_tail_sum();
```

and assert the policy survives conversion into lower-level factorization paths.

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-treetn apply contraction linsolve`

Expected: FAIL because these APIs still carry `rtol`.

**Step 3: Write minimal implementation**

- Replace `rtol` fields in SVD-based high-level options with `svd_policy`
- Keep `factorize_alg` for fit, but validate that:
  - SVD/RSVD accepts `svd_policy`
  - QR accepts `qr_rtol`
  - LU/CI reject `svd_policy`
- Update helper conversions like `factorize_options_from_contraction_options`

**Step 4: Run test to verify it passes**

Run: `cargo nextest run --release -p tensor4all-treetn apply contraction linsolve`

Expected: PASS

**Step 5: Commit**

```bash
git add crates/tensor4all-treetn/src/operator/apply.rs crates/tensor4all-treetn/src/operator/apply/tests/mod.rs crates/tensor4all-treetn/src/treetn/contraction.rs crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs crates/tensor4all-treetn/src/treetn/partial_contraction.rs crates/tensor4all-treetn/src/treetn/fit.rs crates/tensor4all-treetn/src/linsolve/common/options.rs crates/tensor4all-treetn/src/linsolve/common/options/tests/mod.rs
git commit -m "refactor(treetn): propagate svd truncation policy"
```

### Task 6: Remove legacy `rtol/cutoff` and LU/CI truncation from itensorlike

**Files:**
- Modify: `crates/tensor4all-itensorlike/src/options.rs`
- Modify: `crates/tensor4all-itensorlike/src/tensortrain.rs`
- Modify: `crates/tensor4all-itensorlike/src/contract.rs`
- Modify: `crates/tensor4all-itensorlike/src/linsolve.rs`
- Test: `crates/tensor4all-itensorlike/src/options/tests/mod.rs`
- Test: `crates/tensor4all-itensorlike/src/tensortrain/tests/mod.rs`
- Test: `crates/tensor4all-itensorlike/tests/fit_bond_capping.rs`
- Test: `crates/tensor4all-itensorlike/tests/bug_zipup_inflated_bonds.rs`
- Test: `crates/tensor4all-itensorlike/tests/linsolve_mpo.rs`

**Step 1: Write the failing tests**

Add tests that exercise:

```rust
let opts = TruncateOptions::svd()
    .with_svd_policy(SvdTruncationPolicy::new(1e-10))
    .with_max_rank(20);

assert_eq!(opts.alg(), TruncateAlg::SVD);
assert_eq!(opts.max_rank(), Some(20));
```

and update existing truncation/contract/linsolve tests to use explicit policy objects.

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-itensorlike`

Expected: FAIL because the crate still exposes `with_rtol`, `with_cutoff`, and `TruncateOptions::lu/ci`.

**Step 3: Write minimal implementation**

- Remove `TruncateOptions::lu()` and `TruncateOptions::ci()`
- Replace `TruncationParams` usage with:
  - `max_rank`
  - `svd_policy`
- Update validators and conversion code into `tensor4all_treetn`
- Rewrite examples and doctests to use explicit SVD policies

**Step 4: Run test to verify it passes**

Run: `cargo nextest run --release -p tensor4all-itensorlike`

Expected: PASS

**Step 5: Commit**

```bash
git add crates/tensor4all-itensorlike/src/options.rs crates/tensor4all-itensorlike/src/options/tests/mod.rs crates/tensor4all-itensorlike/src/tensortrain.rs crates/tensor4all-itensorlike/src/tensortrain/tests/mod.rs crates/tensor4all-itensorlike/src/contract.rs crates/tensor4all-itensorlike/src/linsolve.rs crates/tensor4all-itensorlike/tests/fit_bond_capping.rs crates/tensor4all-itensorlike/tests/bug_zipup_inflated_bonds.rs crates/tensor4all-itensorlike/tests/linsolve_mpo.rs
git commit -m "refactor(itensorlike): replace rtol and cutoff with svd policy"
```

### Task 7: Redesign the C API to expose explicit SVD truncation policies

**Files:**
- Modify: `crates/tensor4all-capi/src/types.rs`
- Modify: `crates/tensor4all-capi/src/tensor.rs`
- Modify: `crates/tensor4all-capi/src/treetn.rs`
- Modify: `crates/tensor4all-capi/src/tensor/tests/mod.rs`
- Modify: `crates/tensor4all-capi/src/treetn/tests/mod.rs`
- Modify: `crates/tensor4all-capi/include/tensor4all_capi.h`

**Step 1: Write the failing tests**

Add C API tests that:

```rust
let policy = t4a_svd_truncation_policy {
    threshold: 1e-10,
    scale: t4a_threshold_scale::Relative,
    measure: t4a_singular_value_measure::SquaredValue,
    rule: t4a_truncation_rule::DiscardedTailSum,
};
```

and verify:

- `t4a_treetn_truncate` accepts `&policy` and `maxdim`
- `t4a_treetn_add` accepts `&policy` and `maxdim`
- `t4a_treetn_contract` accepts `policy` plus QR-specific `qr_rtol`
- `t4a_treetn_linsolve` no longer takes `form`

**Step 2: Run test to verify it fails**

Run: `cargo nextest run --release -p tensor4all-capi treetn tensor`

Expected: FAIL because the TreeTN C entry points still expose legacy
`rtol/cutoff/form` signatures.

**Step 3: Write minimal implementation**

- Remove legacy `rtol/cutoff/form` parameters from SVD-based TreeTN C entry points
- Add `const t4a_svd_truncation_policy *policy` wherever SVD truncation is used
- Keep `form` only on canonicalization APIs
- Preserve separate `qr_rtol` inputs for QR-based paths
- Regenerate `crates/tensor4all-capi/include/tensor4all_capi.h`

**Step 4: Run test to verify it passes**

Run: `cargo nextest run --release -p tensor4all-capi treetn tensor`

Expected: PASS

**Step 5: Commit**

```bash
git add crates/tensor4all-capi/src/types.rs crates/tensor4all-capi/src/tensor.rs crates/tensor4all-capi/src/treetn.rs crates/tensor4all-capi/src/tensor/tests/mod.rs crates/tensor4all-capi/src/treetn/tests/mod.rs crates/tensor4all-capi/include/tensor4all_capi.h
git commit -m "refactor(capi): expose explicit svd truncation policies"
```

### Task 8: Refresh docs, rustdoc examples, and workspace verification

**Files:**
- Modify: `README.md`
- Modify: crate-level docs and rustdoc examples touched in prior tasks
- Modify: any affected API dump snapshots under `docs/api/` if intentionally regenerated

**Step 1: Write the failing checks**

No new tests are needed here. Treat the workspace checks as the failing signal.

**Step 2: Run checks to find remaining breakage**

Run:

```bash
cargo fmt --all
cargo clippy --workspace
cargo test --doc --release --workspace
./scripts/test-mdbook.sh
cargo nextest run --release --workspace
cbindgen --config crates/tensor4all-capi/cbindgen.toml --crate tensor4all-capi --output crates/tensor4all-capi/include/tensor4all_capi.h
```

Expected: at least one failure from stale docs/examples before cleanup.

**Step 3: Write minimal implementation**

- Update rustdoc examples to use `SvdTruncationPolicy`
- Remove all references to `with_rtol`, `with_cutoff`, `maxdim`, and LU/CI truncate builders
- Ensure README and crate docs describe `truncate` as SVD-based and `canonicalize` as the LU/CI entry point

**Step 4: Run checks to verify they pass**

Run:

```bash
cargo fmt --all
cargo clippy --workspace
cargo test --doc --release --workspace
./scripts/test-mdbook.sh
cargo nextest run --release --workspace
cbindgen --config crates/tensor4all-capi/cbindgen.toml --crate tensor4all-capi --output crates/tensor4all-capi/include/tensor4all_capi.h
```

Expected: PASS

**Step 5: Commit**

```bash
git add README.md docs/book crates
git commit -m "docs: update truncation APIs for explicit svd policy"
```
