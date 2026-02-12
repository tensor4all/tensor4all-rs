# Affine Transform Test Coverage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Strengthen affine transform test coverage to match Quantics.jl's test suite, closing gaps in negative b, Open BC, and MPO vs matrix verification.

**Architecture:** Add two reusable test helpers (`assert_affine_mpo_matches_matrix`, `assert_affine_matrix_correctness`), upgrade 4 existing partial tests to use them, and add 2 new parametric tests covering Julia's missing cases.

**Tech Stack:** Rust unit tests in `crates/tensor4all-quanticstransform/src/affine.rs` test module. Uses existing `mpo_to_dense_matrix` helper, `affine_transform_mpo`, `affine_transform_matrix`, `AffineParams`, `BoundaryCondition`.

---

### Task 1: Add `assert_affine_mpo_matches_matrix` helper

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine.rs` (test module, after `mpo_to_dense_matrix` ~line 1240)

**Step 1: Add the helper function**

Insert after the `mpo_to_dense_matrix` function (after line 1240), before the first MPO vs matrix test:

```rust
    /// Assert that the MPO representation matches the direct sparse matrix computation
    /// for all elements. This is the primary correctness check: two independent algorithms
    /// (carry-based MPO vs direct enumeration) must agree.
    fn assert_affine_mpo_matches_matrix(r: usize, params: &AffineParams, bc: &[BoundaryCondition]) {
        let m = params.m;
        let n = params.n;

        let matrix = affine_transform_matrix(r, params, bc).unwrap();
        let mpo = affine_transform_mpo(r, params, bc).unwrap();
        let mpo_matrix = mpo_to_dense_matrix(&mpo, m, n, r);

        let output_size = 1 << (m * r);
        let input_size = 1 << (n * r);

        for y in 0..output_size {
            for x in 0..input_size {
                let sparse_val = *matrix.get(y, x).unwrap_or(&0.0);
                let mpo_val = mpo_matrix[y][x].re;
                assert!(
                    (sparse_val - mpo_val).abs() < 1e-10,
                    "MPO vs matrix mismatch at ({}, {}): sparse={}, mpo={} \
                     [r={}, m={}, n={}, bc={:?}]",
                    y, x, sparse_val, mpo_val, r, m, n, bc
                );
            }
        }
    }
```

**Step 2: Verify existing tests still pass**

Run: `cargo test -p tensor4all-quanticstransform -- affine`
Expected: All existing tests PASS (helper is just added, not yet called from new places)

**Step 3: Commit**

```bash
git add crates/tensor4all-quanticstransform/src/affine.rs
git commit -m "test: add assert_affine_mpo_matches_matrix helper"
```

---

### Task 2: Add `assert_affine_matrix_correctness` helper

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine.rs` (test module, after `assert_affine_mpo_matches_matrix`)

**Step 1: Add the helper function**

This independently verifies `affine_transform_matrix` against direct Rational64 computation (no integer scaling). Equivalent to Julia's `test_affine_transform_matrix_multi_variables`.

```rust
    /// Assert that affine_transform_matrix produces correct results by independently
    /// computing y = A*x + b using Rational64 arithmetic (no integer scaling).
    /// Equivalent to Julia's test_affine_transform_matrix_multi_variables.
    fn assert_affine_matrix_correctness(
        r: usize,
        params: &AffineParams,
        bc: &[BoundaryCondition],
    ) {
        let m = params.m;
        let n = params.n;
        let modulus = 1i64 << r;

        let matrix = affine_transform_matrix(r, params, bc).unwrap();

        let input_size = 1usize << (r * n);
        let output_size = 1usize << (r * m);

        // Build expected matrix independently using Rational64
        for x_flat in 0..input_size {
            // Decode x_flat to N-dimensional vector
            let x_vals: Vec<i64> = (0..n)
                .map(|var| ((x_flat >> (var * r)) & ((1 << r) - 1)) as i64)
                .collect();

            // Compute y = A*x + b using Rational64 (independent of to_integer_scaled)
            let y_rational: Vec<Rational64> = (0..m)
                .map(|i| {
                    let mut val = params.b[i];
                    for j in 0..n {
                        val += params.a[i * n + j] * Rational64::from_integer(x_vals[j]);
                    }
                    val
                })
                .collect();

            // Check if all y values are integers
            if y_rational.iter().any(|y| !y.is_integer()) {
                // No valid output for this input - all entries in this column must be 0
                for y_flat in 0..output_size {
                    let val = *matrix.get(y_flat, x_flat).unwrap_or(&0.0);
                    assert!(
                        val.abs() < 1e-10,
                        "Expected zero at ({}, {}) for non-integer y, got {} [r={}, bc={:?}]",
                        y_flat, x_flat, val, r, bc
                    );
                }
                continue;
            }

            let y_int: Vec<i64> = y_rational.iter().map(|y| y.to_integer()).collect();

            // Apply boundary conditions
            let bc_periodic: Vec<bool> = bc
                .iter()
                .map(|b| matches!(b, BoundaryCondition::Periodic))
                .collect();

            let y_bounded: Vec<i64> = y_int
                .iter()
                .enumerate()
                .map(|(i, &yi)| {
                    if bc_periodic[i] {
                        ((yi % modulus) + modulus) % modulus
                    } else {
                        yi
                    }
                })
                .collect();

            let valid = y_bounded
                .iter()
                .enumerate()
                .all(|(i, &yi)| bc_periodic[i] || (yi >= 0 && yi < modulus));

            if valid {
                let y_flat: usize = y_bounded
                    .iter()
                    .enumerate()
                    .map(|(var, &yi)| (yi as usize) << (var * r))
                    .sum();

                // This (y_flat, x_flat) should be 1
                let val = *matrix.get(y_flat, x_flat).unwrap_or(&0.0);
                assert!(
                    (val - 1.0).abs() < 1e-10,
                    "Expected 1 at ({}, {}) but got {} [r={}, x={:?}, y={:?}, bc={:?}]",
                    y_flat, x_flat, val, r, x_vals, y_bounded, bc
                );
            }
            // For invalid outputs, the column should have no entries (already sparse)
        }
    }
```

**Step 2: Verify existing tests still pass**

Run: `cargo test -p tensor4all-quanticstransform -- affine`
Expected: All existing tests PASS

**Step 3: Commit**

```bash
git add crates/tensor4all-quanticstransform/src/affine.rs
git commit -m "test: add assert_affine_matrix_correctness helper"
```

---

### Task 3: Refactor existing MPO vs matrix tests to use helper

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine.rs` (existing tests at ~lines 1244-1351)

**Step 1: Refactor `test_affine_mpo_vs_matrix_1d_identity` (line 1244)**

Replace the body with:
```rust
    #[test]
    fn test_affine_mpo_vs_matrix_1d_identity() {
        let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];
        assert_affine_mpo_matches_matrix(3, &params, &bc);
    }
```

**Step 2: Refactor `test_affine_mpo_vs_matrix_1d_shift` (line 1273)**

Replace the body with:
```rust
    #[test]
    fn test_affine_mpo_vs_matrix_1d_shift() {
        let params = AffineParams::from_integers(vec![1], vec![3], 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];
        assert_affine_mpo_matches_matrix(3, &params, &bc);
    }
```

**Step 3: Refactor `test_affine_mpo_vs_matrix_simple` (line 1302)**

Replace the body with:
```rust
    #[test]
    fn test_affine_mpo_vs_matrix_simple() {
        let params = AffineParams::from_integers(vec![1, 0, 1, 1], vec![0, 0], 2, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic; 2];
        assert_affine_mpo_matches_matrix(3, &params, &bc);
    }
```

**Step 4: Refactor `test_affine_mpo_vs_matrix_r1` (line 1495)**

Replace the body with:
```rust
    #[test]
    fn test_affine_mpo_vs_matrix_r1() {
        // R=1 is a special case where is_msb and is_lsb are both true
        let bc = vec![BoundaryCondition::Periodic];

        // Identity R=1
        let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
        assert_affine_mpo_matches_matrix(1, &params, &bc);

        // Shift R=1 (y = x + 1 mod 2)
        let params = AffineParams::from_integers(vec![1], vec![1], 1, 1).unwrap();
        assert_affine_mpo_matches_matrix(1, &params, &bc);
    }
```

**Step 5: Run tests**

Run: `cargo test -p tensor4all-quanticstransform -- affine`
Expected: All tests PASS (same behavior, just refactored)

**Step 6: Commit**

```bash
git add crates/tensor4all-quanticstransform/src/affine.rs
git commit -m "refactor: use assert_affine_mpo_matches_matrix in existing tests"
```

---

### Task 4: Upgrade partial tests to full verification

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine.rs` (existing tests)

**Step 1: Upgrade `test_affine_matrix_3x3_hard` (line 1353)**

Add MPO vs matrix comparison after existing assertions:
```rust
    #[test]
    fn test_affine_matrix_3x3_hard() {
        // From Quantics.jl compare_hard test
        // A = [1 0 1; 1 2 -1; 0 1 1], b = [11; 23; -15]
        let r = 4;
        let a = vec![1i64, 0, 1, 1, 2, -1, 0, 1, 1];
        let b = vec![11i64, 23, -15];
        let params = AffineParams::from_integers(a, b, 3, 3).unwrap();
        let bc = vec![BoundaryCondition::Periodic; 3];

        // Verify MPO matches direct computation
        assert_affine_mpo_matches_matrix(r, &params, &bc);
    }
```

**Step 2: Upgrade `test_affine_matrix_rectangular` (line 1376)**

```rust
    #[test]
    fn test_affine_matrix_rectangular() {
        // From Quantics.jl compare_rect test
        // A = [1 0 1; 1 2 0] (2x3), b = [11; -3]
        let r = 4;
        let a = vec![1i64, 0, 1, 1, 2, 0];
        let b = vec![11i64, -3];
        let params = AffineParams::from_integers(a, b, 2, 3).unwrap();
        let bc = vec![BoundaryCondition::Periodic; 2];

        // Verify MPO matches direct computation
        assert_affine_mpo_matches_matrix(r, &params, &bc);
    }
```

**Step 3: Upgrade `test_affine_matrix_denom_odd` (line 1395)**

```rust
    #[test]
    fn test_affine_matrix_denom_odd() {
        // From Quantics.jl compare_denom_odd test
        // A = [1/3], b = [0]
        for r in [1, 3, 6] {
            for bc in [BoundaryCondition::Periodic, BoundaryCondition::Open] {
                let a = vec![Rational64::new(1, 3)];
                let b = vec![Rational64::from_integer(0)];
                let params = AffineParams::new(a, b, 1, 1).unwrap();
                let bcs = vec![bc];

                // Verify MPO matches direct computation
                assert_affine_mpo_matches_matrix(r, &params, &bcs);
            }
        }
    }
```

**Step 4: Upgrade `test_affine_matrix_light_cone` (line 1425)**

```rust
    #[test]
    fn test_affine_matrix_light_cone() {
        // From Quantics.jl compare_light_cone test
        // Light cone transformation: A = 1/2 * [[1, 1], [1, -1]], b = [2, 3]
        for r in [3, 4] {
            for bc in [BoundaryCondition::Periodic, BoundaryCondition::Open] {
                let a = vec![
                    Rational64::new(1, 2),
                    Rational64::new(1, 2),
                    Rational64::new(1, 2),
                    Rational64::new(-1, 2),
                ];
                let b = vec![Rational64::from_integer(2), Rational64::from_integer(3)];
                let params = AffineParams::new(a, b, 2, 2).unwrap();
                let bcs = vec![bc; 2];

                // Verify both correctness and MPO consistency
                assert_affine_matrix_correctness(r, &params, &bcs);
                assert_affine_mpo_matches_matrix(r, &params, &bcs);
            }
        }
    }
```

**Step 5: Run tests**

Run: `cargo test -p tensor4all-quanticstransform -- affine`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add crates/tensor4all-quanticstransform/src/affine.rs
git commit -m "test: upgrade partial affine tests to full MPO vs matrix verification"
```

---

### Task 5: Add parametric full test (Julia "full R=...")

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine.rs` (test module, add new test)

**Step 1: Write the test**

Add after the unfused tests section:

```rust
    // ========== Parametric tests from Quantics.jl ==========

    #[test]
    fn test_affine_parametric_full() {
        // From Quantics.jl "full R=$R, boundary=$boundary, M=$M, N=$N" test
        // Tests all combinations from Julia's testtests dictionary
        struct TestCase {
            a: Vec<i64>,
            b: Vec<i64>,
            m: usize,
            n: usize,
        }

        let cases = vec![
            // (1,1): A=[1], b=[1]
            TestCase { a: vec![1], b: vec![1], m: 1, n: 1 },
            // (1,2): A=[1 0], b=[0]
            TestCase { a: vec![1, 0], b: vec![0], m: 1, n: 2 },
            // (1,2): A=[2 -1], b=[1]
            TestCase { a: vec![2, -1], b: vec![1], m: 1, n: 2 },
            // (2,1): A=[1; 0], b=[0, 0]
            TestCase { a: vec![1, 0], b: vec![0, 0], m: 2, n: 1 },
            // (2,1): A=[2; -1], b=[1, -1]
            TestCase { a: vec![2, -1], b: vec![1, -1], m: 2, n: 1 },
            // (2,2): A=[1 0; 1 1], b=[0, 1]
            TestCase { a: vec![1, 0, 1, 1], b: vec![0, 1], m: 2, n: 2 },
            // (2,2): A=[2 0; 4 1], b=[100, -1]
            TestCase { a: vec![2, 0, 4, 1], b: vec![100, -1], m: 2, n: 2 },
        ];

        for r in [1, 2] {
            for bc_type in [BoundaryCondition::Open, BoundaryCondition::Periodic] {
                for case in &cases {
                    let params = AffineParams::from_integers(
                        case.a.clone(), case.b.clone(), case.m, case.n,
                    ).unwrap();
                    let bc = vec![bc_type; case.m];

                    assert_affine_matrix_correctness(r, &params, &bc);
                    assert_affine_mpo_matches_matrix(r, &params, &bc);
                }
            }
        }
    }
```

**Step 2: Run tests**

Run: `cargo test -p tensor4all-quanticstransform -- test_affine_parametric_full`
Expected: PASS (this tests the most critical gaps: negative b, Open BC, MPO correctness)

**Step 3: Commit**

```bash
git add crates/tensor4all-quanticstransform/src/affine.rs
git commit -m "test: add parametric affine test matching Quantics.jl full suite"
```

---

### Task 6: Add denom_even test (Julia "compare_denom_even")

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine.rs` (test module)

**Step 1: Write the test**

```rust
    #[test]
    fn test_affine_denom_even() {
        // From Quantics.jl compare_denom_even test
        // A = [1/2], b in {3, 5, -3, -5}, R in {3, 5}, BC = Periodic
        let a = vec![Rational64::new(1, 2)];

        for b_val in [3i64, 5, -3, -5] {
            let b = vec![Rational64::from_integer(b_val)];
            let params = AffineParams::new(a.clone(), b, 1, 1).unwrap();
            let bc = vec![BoundaryCondition::Periodic];

            for r in [3, 5] {
                assert_affine_mpo_matches_matrix(r, &params, &bc);
            }
        }
    }
```

**Step 2: Run tests**

Run: `cargo test -p tensor4all-quanticstransform -- test_affine_denom_even`
Expected: PASS

**Step 3: Run full test suite**

Run: `cargo test -p tensor4all-quanticstransform -- affine`
Expected: All affine tests PASS

**Step 4: Commit**

```bash
git add crates/tensor4all-quanticstransform/src/affine.rs
git commit -m "test: add denom_even affine test with negative b offsets"
```

---

## Verification

After all tasks:

```bash
cargo test -p tensor4all-quanticstransform -- affine
cargo clippy -p tensor4all-quanticstransform
cargo fmt --all
```

All tests should pass. The coverage gaps identified in the design doc should now be closed:
- Negative b: covered by parametric_full and denom_even
- Open BC numerical verification: covered by parametric_full
- MPO vs matrix for all Julia cases: covered by upgraded existing tests
- Rational + negative b: covered by denom_even
