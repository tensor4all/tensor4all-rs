# Affine Anti-Periodic Boundary and Difference Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add anti-periodic boundary support to affine quantics transforms and C API, then add an exact difference-kernel MPO built from an input QTT.

**Architecture:** Phase 1 adds `BoundaryCondition::AntiPeriodic` and replaces affine periodic/open booleans with final-carry boundary weights. Phase 2 constructs `A[x, x'] = f(x - x')` by contracting the affine delta tensor `z = x - x'` with the QTT cores of `f`.

**Tech Stack:** Rust workspace crates `tensor4all-quanticstransform` and `tensor4all-capi`, Julia wrapper `Tensor4all.jl`, mdBook/tutorial-code docs, `cargo test`, targeted Julia tests where available.

---

### Task 1: Rust Boundary Enum and Affine Anti-Periodic Tests

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/common.rs`
- Modify: `crates/tensor4all-quanticstransform/src/affine/tests/mod.rs`
- Modify: `crates/tensor4all-quanticstransform/src/shift.rs`
- Modify: `crates/tensor4all-quanticstransform/src/flip.rs`

- [ ] **Step 1: Add failing tests for anti-periodic affine high-bit shifts**

Add tests in `crates/tensor4all-quanticstransform/src/affine/tests/mod.rs`:

```rust
#[test]
fn test_affine_antiperiodic_full_cycle_shifts() {
    let r = 3;
    let n = 1usize << r;

    for (shift, expected_sign) in [(n as i64, -1.0), (-(n as i64), -1.0), (2 * n as i64, 1.0)] {
        let params = AffineParams::from_integers(vec![1], vec![shift], 1, 1).unwrap();
        let matrix =
            affine_transform_matrix(r, &params, &[BoundaryCondition::AntiPeriodic]).unwrap();

        for x in 0..n {
            for y in 0..n {
                let expected = if x == y { expected_sign } else { 0.0 };
                let actual = *matrix.get(y, x).unwrap_or(&0.0);
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "shift={shift}, x={x}, y={y}, expected={expected}, actual={actual}"
                );
            }
        }
    }
}

#[test]
fn test_affine_antiperiodic_full_cycle_plus_one_shift() {
    let r = 3;
    let n = 1usize << r;
    let params = AffineParams::from_integers(vec![1], vec![n as i64 + 1], 1, 1).unwrap();
    let matrix = affine_transform_matrix(r, &params, &[BoundaryCondition::AntiPeriodic]).unwrap();

    for x in 0..n {
        let y_expected = (x + 1) % n;
        for y in 0..n {
            let expected = if y == y_expected { -1.0 } else { 0.0 };
            let actual = *matrix.get(y, x).unwrap_or(&0.0);
            assert!(
                (actual - expected).abs() < 1e-12,
                "x={x}, y={y}, expected={expected}, actual={actual}"
            );
        }
    }
}
```

- [ ] **Step 2: Add failing test for `z = x - x'` anti-periodic signs**

Add this test in the same file:

```rust
#[test]
fn test_affine_antiperiodic_difference_delta_signs() {
    let r = 3;
    let n = 1usize << r;
    let params = AffineParams::from_integers(vec![1, -1], vec![0], 1, 2).unwrap();
    let matrix = affine_transform_matrix(r, &params, &[BoundaryCondition::AntiPeriodic]).unwrap();

    for x in 0..n {
        for xp in 0..n {
            let x_flat = x | (xp << r);
            let z = (x + n - xp) % n;
            let expected_sign = if x >= xp { 1.0 } else { -1.0 };
            for y in 0..n {
                let expected = if y == z { expected_sign } else { 0.0 };
                let actual = *matrix.get(y, x_flat).unwrap_or(&0.0);
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "x={x}, xp={xp}, z={z}, y={y}, expected={expected}, actual={actual}"
                );
            }
        }
    }
}
```

- [ ] **Step 3: Run tests and verify they fail because `AntiPeriodic` is missing**

Run:

```bash
cargo test -p tensor4all-quanticstransform affine_antiperiodic --offline
```

Expected: compile failure mentioning `BoundaryCondition::AntiPeriodic` is not found.

- [ ] **Step 4: Add `BoundaryCondition::AntiPeriodic` and temporary shift/flip handling**

Add the enum variant and docs in `common.rs`. Update `shift.rs` and `flip.rs` matches so `AntiPeriodic` maps to `-1` on one wrap and remains buildable while affine is implemented.

- [ ] **Step 5: Commit the failing-tests and enum scaffold**

Run:

```bash
git add crates/tensor4all-quanticstransform/src/common.rs \
        crates/tensor4all-quanticstransform/src/affine/tests/mod.rs \
        crates/tensor4all-quanticstransform/src/shift.rs \
        crates/tensor4all-quanticstransform/src/flip.rs
git commit -m "Add anti-periodic boundary tests"
```

### Task 2: Affine Boundary Weight Implementation

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine.rs`
- Modify: `crates/tensor4all-quanticstransform/src/affine/tests/mod.rs`

- [ ] **Step 1: Replace `bc_periodic` with boundary-weight helpers**

Add helpers in `affine.rs`:

```rust
fn affine_boundary_weight(carry: &[i64], bc: &[BoundaryCondition]) -> f64 {
    carry
        .iter()
        .zip(bc.iter())
        .map(|(&c, &boundary)| match boundary {
            BoundaryCondition::Periodic => 1.0,
            BoundaryCondition::AntiPeriodic => {
                if c.rem_euclid(2) == 0 { 1.0 } else { -1.0 }
            }
            BoundaryCondition::Open => {
                if c == 0 { 1.0 } else { 0.0 }
            }
        })
        .product()
}

fn affine_needs_extension(bc: &[BoundaryCondition], b_work: &[i64]) -> bool {
    b_work.iter().any(|&b| b > 0)
        && bc
            .iter()
            .any(|b| matches!(b, BoundaryCondition::Open | BoundaryCondition::AntiPeriodic))
}
```

- [ ] **Step 2: Update `affine_transform_matrix`**

For each `(x, y)`, compute `diff_i = v_i - scale * y_i`. A component is valid when:

```rust
Periodic | AntiPeriodic => diff_i.rem_euclid(modulus) == 0
Open => diff_i == 0
```

When valid, compute `carry_i = diff_i / modulus` for periodic and anti-periodic dimensions, and `0` for open dimensions. Store `affine_boundary_weight(&carry, bc)` instead of `1.0`, skipping exact zeros.

- [ ] **Step 3: Update `affine_transform_tensors`**

Pass `bc` into `affine_transform_tensors`. Use `affine_needs_extension(bc, &b_work)` for the extension path. Replace the final carry cap weight with `affine_boundary_weight(c, bc)`.

- [ ] **Step 4: Run affine anti-periodic tests**

Run:

```bash
cargo test -p tensor4all-quanticstransform affine_antiperiodic --offline
```

Expected: all anti-periodic affine tests pass.

- [ ] **Step 5: Run broader affine tests**

Run:

```bash
cargo test -p tensor4all-quanticstransform affine --offline
```

Expected: existing affine tests still pass.

- [ ] **Step 6: Commit affine implementation**

Run:

```bash
git add crates/tensor4all-quanticstransform/src/affine.rs \
        crates/tensor4all-quanticstransform/src/affine/tests/mod.rs
git commit -m "Support anti-periodic affine boundaries"
```

### Task 3: C API and Tensor4all.jl Boundary Plumbing

**Files:**
- Modify: `crates/tensor4all-capi/src/types.rs`
- Modify: `crates/tensor4all-capi/src/types/tests/mod.rs`
- Modify: `/Users/hiroshi/projects/tensor4all/Tensor4all.jl/src/QuanticsTransform/capi_helpers.jl`
- Modify: `/Users/hiroshi/projects/tensor4all/Tensor4all.jl/src/QuanticsTransform/operators.jl`

- [ ] **Step 1: Add C API failing round-trip coverage**

Extend the round-trip test in `crates/tensor4all-capi/src/types/tests/mod.rs` to include:

```rust
tensor4all_quanticstransform::BoundaryCondition::AntiPeriodic,
```

- [ ] **Step 2: Run C API round-trip test and verify failure**

Run:

```bash
cargo test -p tensor4all-capi test_boundary_condition_roundtrip --offline
```

Expected before implementation: compile failure or conversion failure for `AntiPeriodic`.

- [ ] **Step 3: Extend `t4a_boundary_condition`**

Add:

```rust
AntiPeriodic = 2,
```

and map it both directions in the `From` impls.

- [ ] **Step 4: Update Tensor4all.jl symbol mapping and docs**

In `_bc_code`, accept `:antiperiodic` and `:anti_periodic`. Update operator docstrings that currently say `:periodic` or `:open` to also mention `:antiperiodic`.

- [ ] **Step 5: Run C API test**

Run:

```bash
cargo test -p tensor4all-capi test_boundary_condition_roundtrip --offline
```

Expected: pass.

- [ ] **Step 6: Commit C API and Julia wrapper changes**

Run in `tensor4all-rs`:

```bash
git add crates/tensor4all-capi/src/types.rs crates/tensor4all-capi/src/types/tests/mod.rs
git commit -m "Expose anti-periodic boundary through C API"
```

Run in `Tensor4all.jl`:

```bash
git add src/QuanticsTransform/capi_helpers.jl src/QuanticsTransform/operators.jl
git commit -m "Accept anti-periodic quantics boundary"
```

### Task 4: Affine Tutorial Anti-Periodic Example

**Files:**
- Modify: `docs/book/src/tutorials/computations-with-qtt/affine-transformation.md`
- Modify: `docs/tutorial-code/src/bin/qtt_affine.rs`
- Modify: `docs/tutorial-code/src/qtt_affine_common.rs`
- Modify: `docs/tutorial-code/docs/plotting/qtt_affine_plot.jl`
- Modify: `docs/tutorial-code/docs/tutorials/qtt_affine_tutorial.md`

- [ ] **Step 1: Add anti-periodic boundary mode and reference**

Extend `AffineBoundaryMode` with `AntiPeriodic`. Update `transformed_reference` so anti-periodic returns `source_function((x + y) % n, y, n)` with sign `-1` when `x + y >= n`.

- [ ] **Step 2: Apply anti-periodic operator in the tutorial binary**

Build:

```rust
let antiperiodic_operator = affine_operator(
    config.bits,
    &affine_params,
    &[BoundaryCondition::AntiPeriodic, BoundaryCondition::Periodic],
)?
.transpose();
```

Align and apply it using the same path as periodic/open.

- [ ] **Step 3: Extend CSV schemas and plots**

Add anti-periodic exact, QTT, error, transformed-bond, and operator-bond columns. Update the Julia plotting script to display `source`, `periodic`, `anti-periodic`, and `open` value panels; include anti-periodic in error and bond plots.

- [ ] **Step 4: Update prose docs**

State that the anti-periodic example flips the sign only in the wrapped `u = x + y` region, using boundary conditions `[AntiPeriodic, Periodic]`.

- [ ] **Step 5: Run tutorial binary tests/build**

Run:

```bash
cargo test -p tensor4all-tutorial-code qtt_affine --offline
cargo run -p tensor4all-tutorial-code --bin qtt_affine --offline
```

Expected: tests pass and CSV files are generated.

- [ ] **Step 6: Commit tutorial changes**

Run:

```bash
git add docs/book/src/tutorials/computations-with-qtt/affine-transformation.md \
        docs/tutorial-code/src/bin/qtt_affine.rs \
        docs/tutorial-code/src/qtt_affine_common.rs \
        docs/tutorial-code/docs/plotting/qtt_affine_plot.jl \
        docs/tutorial-code/docs/tutorials/qtt_affine_tutorial.md
git commit -m "Add anti-periodic affine tutorial case"
```

### Task 5: Difference Kernel API and Tests

**Files:**
- Create: `crates/tensor4all-quanticstransform/src/difference_kernel.rs`
- Modify: `crates/tensor4all-quanticstransform/src/lib.rs`
- Modify: `crates/tensor4all-quanticstransform/tests/integration_test.rs`

- [ ] **Step 1: Add failing tests for difference kernel**

Add tests that construct a short exact QTT for `f(z)`, build the periodic and anti-periodic difference-kernel MPOs, contract them to dense matrices, and compare with:

```text
periodic:      A[x, x'] = f((x - x') mod N)
anti-periodic: A[x, x'] = sign(x, x') * f((x - x') mod N)
```

where `sign = 1` for `x >= x'` and `-1` otherwise.

- [ ] **Step 2: Run tests and verify unresolved symbols**

Run:

```bash
cargo test -p tensor4all-quanticstransform difference_kernel --offline
```

Expected: compile failure because the API does not exist.

- [ ] **Step 3: Implement `difference_kernel_mpo`**

Create a module that:

1. validates non-empty binary QTT cores;
2. rejects `BoundaryCondition::Open`;
3. builds `AffineParams::from_integers(vec![1, -1], vec![0], 1, 2)`;
4. obtains `affine_transform_tensors_unfused`;
5. contracts the `z` bit with each `f` core;
6. returns `TensorTrain<Complex64>` with site dimension `4`.

- [ ] **Step 4: Implement `difference_kernel_operator`**

Wrap the MPO with `tensortrain_to_linear_operator(&mpo, &[2; R])`.

- [ ] **Step 5: Export API**

Export:

```rust
pub fn difference_kernel_mpo(...)
pub fn difference_kernel_operator(...)
```

from `lib.rs`.

- [ ] **Step 6: Run difference-kernel tests**

Run:

```bash
cargo test -p tensor4all-quanticstransform difference_kernel --offline
```

Expected: pass.

- [ ] **Step 7: Commit difference-kernel API**

Run:

```bash
git add crates/tensor4all-quanticstransform/src/difference_kernel.rs \
        crates/tensor4all-quanticstransform/src/lib.rs \
        crates/tensor4all-quanticstransform/tests/integration_test.rs
git commit -m "Add difference kernel MPO"
```

### Task 6: Final Verification

**Files:**
- All files touched above.

- [ ] **Step 1: Format Rust**

Run:

```bash
cargo fmt
```

Expected: no errors.

- [ ] **Step 2: Run quantics transform tests**

Run:

```bash
cargo test -p tensor4all-quanticstransform --offline
```

Expected: pass.

- [ ] **Step 3: Run C API tests**

Run:

```bash
cargo test -p tensor4all-capi --offline
```

Expected: pass.

- [ ] **Step 4: Check repository status**

Run:

```bash
git status --short
```

Expected: only intentionally generated tutorial data/plots, or a clean tree if no generated files remain.
