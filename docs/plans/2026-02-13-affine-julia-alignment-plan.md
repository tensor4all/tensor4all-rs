# Affine Transform Julia Algorithm Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align `affine_transform_tensors` and `affine_transform_core` with Julia's Quantics.jl algorithm, adding the extension loop for `abs(b) >= 2^R` with Open BC.

**Architecture:** Replace bit-position indexing with Julia's iterative shift approach. Add `activebit` parameter to `affine_transform_core`. Implement extension loop that folds extra carry-only tensors into the MSB tensor. Add comprehensive test coverage.

**Tech Stack:** Rust, `crates/tensor4all-quanticstransform/src/affine.rs`. Uses `DTensor`, `Complex64`, `HashMap`, `TensorTrain`.

---

### Task 1: Add `activebit` parameter to `affine_transform_core`

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine.rs:684-783`

**Step 1: Add `activebit` parameter**

Change function signature from:
```rust
fn affine_transform_core(
    a_int: &[i64],
    b_curr: &[i64],
    scale: i64,
    m: usize,
    n: usize,
    carries_in: &[Vec<i64>],
) -> Result<AffineCoreData> {
```
to:
```rust
fn affine_transform_core(
    a_int: &[i64],
    b_curr: &[i64],
    scale: i64,
    m: usize,
    n: usize,
    carries_in: &[Vec<i64>],
    activebit: bool,
) -> Result<AffineCoreData> {
```

**Step 2: Adjust site dimensions based on `activebit`**

When `activebit=false`, x and y can only be 0 (Julia: `bitrange = range(0, 0)`). Replace the fixed `site_dim = 1 << (m + n)` with:

```rust
    let x_range = if activebit { 1 << n } else { 1 };
    let y_range = if activebit { 1 << m } else { 1 };
    let site_dim = x_range * y_range;
```

And update all `0..(1 << n)` to `0..x_range`, and `0..(1 << m)` to `0..y_range`.

**Step 3: Add `activebit=false` skip for odd scale (Julia PR #45 fix)**

In the `scale % 2 == 1` branch, after computing `y`, add:

```rust
            if scale % 2 == 1 {
                let y: Vec<i64> = z.iter().map(|&zi| zi & 1).collect();

                // When bits are inactive, y must be zero (Julia PR #45 fix)
                if !activebit && y.iter().any(|&yi| yi != 0) {
                    continue;
                }

                // ... rest unchanged
```

**Step 4: Update the call site in `affine_transform_tensors`**

Change the existing call from:
```rust
let core_data = affine_transform_core(a_int, &b_curr, scale, m, n, &carries)?;
```
to:
```rust
let core_data = affine_transform_core(a_int, &b_curr, scale, m, n, &carries, true)?;
```

**Step 5: Run tests**

Run: `cargo test -p tensor4all-quanticstransform --lib`
Expected: All existing tests PASS (behavior unchanged since `activebit=true` everywhere)

**Step 6: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-quanticstransform/src/affine.rs
git commit -m "refactor: add activebit parameter to affine_transform_core"
```

---

### Task 2: Rewrite `affine_transform_tensors` with Julia algorithm

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine.rs:505-664`

**Reference:** Julia `Quantics.jl/src/affine.jl:104-169`

**Step 1: Replace bit extraction loop with iterative shift**

Replace the main loop (lines ~531-549):

```rust
    // Track sign separately and work with absolute value
    // so that right-shifting always terminates (Julia PR #45 approach)
    let bsign: Vec<i64> = b_int.iter().map(|&b| if b >= 0 { 1 } else { -1 }).collect();
    let mut b_work: Vec<i64> = b_int.iter().map(|&b| b.abs()).collect();

    // Process from LSB (site R-1) to MSB (site 0)
    let mut carries: Vec<Vec<i64>> = vec![vec![0i64; m]];
    let mut core_data_list: Vec<AffineCoreData> = Vec::with_capacity(r);

    for _site in (0..r).rev() {
        // Extract current bit: (b_work & 1) * bsign
        let b_curr: Vec<i64> = b_work
            .iter()
            .zip(bsign.iter())
            .map(|(&b, &s)| (b & 1) * s)
            .collect();

        let core_data = affine_transform_core(a_int, &b_curr, scale, m, n, &carries, true)?;
        carries = core_data.carries_out.clone();
        core_data_list.push(core_data);

        // Shift right
        b_work.iter_mut().for_each(|b| *b >>= 1);
    }
```

**Step 2: Add extension loop for Open BC**

After the main loop, before the tensor building section:

```rust
    // Extension loop: handle remaining bits of b for Open BC
    // When abs(b) >= 2^R, high bits of b contribute to carries that affect validity.
    // Extension tensors have site_dim=1 (activebit=false: only x=0, y=0).
    // We fold them into the MSB tensor as a "cap matrix".
    let cap_matrix: Option<Vec<Vec<f64>>> =
        if !bc_periodic.iter().all(|&p| p) && b_work.iter().any(|&b| b > 0) {
            // Collect extension core data (MSB to outermost)
            let mut ext_data_list: Vec<AffineCoreData> = Vec::new();
            while b_work.iter().any(|&b| b > 0) {
                let b_curr: Vec<i64> = b_work
                    .iter()
                    .zip(bsign.iter())
                    .map(|(&b, &s)| (b & 1) * s)
                    .collect();

                let core_data =
                    affine_transform_core(a_int, &b_curr, scale, m, n, &carries, false)?;
                carries = core_data.carries_out.clone();
                ext_data_list.push(core_data);

                b_work.iter_mut().for_each(|b| *b >>= 1);
            }

            // Build cap matrix by contracting extension tensors with BC weights.
            // Extension tensors have site_dim=1, so they are carry transition matrices:
            //   ext_matrix[cout_idx, cin_idx] = core_data.tensor[[cout_idx, cin_idx, 0]]
            //
            // Process: outermost (last computed) gets BC weights applied,
            // then multiply inward toward the main tensor chain.

            // Start with BC weights on the final carries
            let num_final = carries.len();
            let mut bc_weights: Vec<f64> = carries
                .iter()
                .map(|c| {
                    if c.iter().all(|&ci| ci == 0) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();

            // Contract extension tensors from outermost to innermost
            // ext_data_list is [innermost, ..., outermost] (order of computation)
            // We process from outermost to innermost
            let mut current_weights = bc_weights;
            for ext_data in ext_data_list.iter().rev() {
                let num_cin = ext_data.tensor.shape()[1];
                let mut new_weights = vec![0.0; num_cin];
                for cin_idx in 0..num_cin {
                    for (cout_idx, w) in current_weights.iter().enumerate() {
                        if *w != 0.0 && ext_data.tensor[[cout_idx, cin_idx, 0]] {
                            new_weights[cin_idx] += w;
                        }
                    }
                }
                current_weights = new_weights;
            }

            // current_weights now maps: MSB carry_out index -> effective BC weight
            Some(vec![current_weights])
        } else {
            None
        };
```

**Step 3: Update tensor building to use cap_matrix**

In the MSB tensor building section, replace the simple `bc_weight` computation with the cap matrix weights. The `is_msb` and `is_lsb && is_msb` cases need updating.

For the `is_msb` case (and `is_lsb && is_msb`), replace:
```rust
let bc_weight = if bc_periodic.iter().all(|&p| p) {
    Complex64::one()
} else {
    if carry.iter().all(|&c| c == 0) { Complex64::one() } else { Complex64::new(0.0, 0.0) }
};
```

with a function that uses cap_matrix when available:

```rust
    // Helper: compute BC weight for a carry-out index
    let compute_bc_weight = |cout_idx: usize, core_data: &AffineCoreData| -> Complex64 {
        if bc_periodic.iter().all(|&p| p) {
            Complex64::one()
        } else if let Some(ref cap) = cap_matrix {
            // Extension loop was used: weight comes from cap matrix
            Complex64::new(cap[0][cout_idx], 0.0)
        } else {
            // No extension: weight is 1 if carry is zero, 0 otherwise
            let carry = &core_data.carries_out[cout_idx];
            if carry.iter().all(|&c| c == 0) {
                Complex64::one()
            } else {
                Complex64::new(0.0, 0.0)
            }
        }
    };
```

Then in the MSB/LSB+MSB tensor building, replace `bc_weight` with `compute_bc_weight(cout_idx, core_data)`.

**Step 4: Run tests**

Run: `cargo test -p tensor4all-quanticstransform --lib`
Expected: All existing tests PASS

**Step 5: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-quanticstransform/src/affine.rs
git commit -m "refactor: align affine_transform_tensors with Julia algorithm

Replace bit-position indexing with iterative shift.
Add extension loop for Open BC when abs(b) >= 2^R."
```

---

### Task 3: Add test coverage

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine.rs` (test module)

**Step 1: Add `test_affine_parametric_full`**

```rust
    #[test]
    fn test_affine_parametric_full() {
        // From Quantics.jl "full R=$R, boundary=$boundary, M=$M, N=$N" test
        struct TestCase {
            a: Vec<i64>,
            b: Vec<i64>,
            m: usize,
            n: usize,
        }

        let cases = vec![
            TestCase { a: vec![1], b: vec![1], m: 1, n: 1 },
            TestCase { a: vec![1, 0], b: vec![0], m: 1, n: 2 },
            TestCase { a: vec![2, -1], b: vec![1], m: 1, n: 2 },
            TestCase { a: vec![1, 0], b: vec![0, 0], m: 2, n: 1 },
            TestCase { a: vec![2, -1], b: vec![1, -1], m: 2, n: 1 },
            TestCase { a: vec![1, 0, 1, 1], b: vec![0, 1], m: 2, n: 2 },
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

**Step 2: Add `test_affine_denom_even`**

```rust
    #[test]
    fn test_affine_denom_even() {
        // From Quantics.jl compare_denom_even test
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

**Step 3: Add `test_affine_extension_loop`**

```rust
    #[test]
    fn test_affine_extension_loop() {
        // Test abs(b) >= 2^R with Open BC (requires extension loop)
        // b=[-32, 32] with R=6, identity matrix (Quantics.jl issue #44 case)
        let r = 5; // Use R=5 so abs(32)=2^5=2^R triggers extension at exact boundary
        let params = AffineParams::from_integers(
            vec![1, 0, 0, 1], vec![-32, 32], 2, 2,
        ).unwrap();
        let bc = vec![BoundaryCondition::Open; 2];
        assert_affine_mpo_matches_matrix(r, &params, &bc);
        assert_affine_matrix_correctness(r, &params, &bc);

        // abs(b) clearly exceeds 2^R
        let r = 4; // 2^4=16, abs(b)=32 > 16
        assert_affine_mpo_matches_matrix(r, &params, &bc);
        assert_affine_matrix_correctness(r, &params, &bc);

        // 1D case: y = x + 64 with R=6, Open BC
        let r = 6;
        let params = AffineParams::from_integers(vec![1], vec![64], 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Open];
        assert_affine_mpo_matches_matrix(r, &params, &bc);
        assert_affine_matrix_correctness(r, &params, &bc);
    }
```

**Step 4: Run all tests**

Run: `cargo test -p tensor4all-quanticstransform --lib`
Expected: All tests PASS

**Step 5: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-quanticstransform/src/affine.rs
git commit -m "test: add parametric, denom_even, and extension loop affine tests"
```

---

### Task 4: Final verification and cleanup

**Step 1: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests PASS

**Step 2: Run lints**

Run: `cargo clippy --workspace`
Expected: No warnings

**Step 3: Format**

Run: `cargo fmt --all`

**Step 4: Remove `assert_affine_matrix_correctness` unused import warning if any**

Check `cargo clippy` output. The `Rational64` import may need `use num_integer::Integer;` for `is_integer()`.

**Step 5: Final commit if needed**

```bash
git add -A
git commit -m "chore: cleanup lint warnings"
```

## Verification

```bash
cargo test -p tensor4all-quanticstransform --lib  # All affine tests
cargo test --workspace                              # Full suite
cargo clippy --workspace                            # No warnings
```
