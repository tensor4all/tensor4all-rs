# Simplify Test Assertions Using isapprox/maxabs (Issue #244)

## Scope

Replace element-wise comparison loops on TensorDynLen with `isapprox`/`maxabs`. Scalar comparisons are out of scope.

## Target pattern

Before:
```rust
let data_a = a.to_vec_f64().unwrap();
let data_b = b.to_vec_f64().unwrap();
for (i, (&x, &y)) in data_a.iter().zip(data_b.iter()).enumerate() {
    assert!((x - y).abs() < 1e-10, "...", i, x, y);
}
```

After:
```rust
assert!(a.isapprox(&b, 1e-10, 0.0));
```

## Files to update

1. `crates/tensor4all-itensorlike/src/contract.rs` — `assert_matches_naive` helper
2. `crates/tensor4all-treetn/src/treetn/addition.rs` — `test_add_verifies_with_contraction`
3. `crates/tensor4all-treetn/tests/ops.rs` — tensor comparison loops
4. `crates/tensor4all-core/tests/linalg_factorize.rs` — `assert_tensors_approx_equal` helper
5. `crates/tensor4all-core/tests/linalg_svd.rs` — reconstruction loops
6. `crates/tensor4all-core/tests/linalg_qr.rs` — reconstruction loops
7. `crates/tensor4all-core/tests/tensor_basic.rs` — `.iter().zip()` comparisons

## Rules

- Preserve existing tolerance values (don't change `1e-10` to `1e-8` etc.)
- Only replace tensor-to-tensor comparisons, not scalar result checks
- Use `isapprox(other, atol, 0.0)` for absolute tolerance patterns
