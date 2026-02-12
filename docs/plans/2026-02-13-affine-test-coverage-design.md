# Affine Transform Test Coverage Strengthening

## Context

Quantics.jl PR #45 fixed 3 bugs in `affine_transform` (negative b handling). The Rust implementation avoids these bugs structurally (bit-position indexing vs iterative shifting), but test coverage analysis revealed significant gaps compared to Quantics.jl's test suite:

- **No numerical verification for Open BC** (only `is_ok()` or `nnz`)
- **No numerical verification for negative b** (only `nnz` or dimension checks)
- **MPO vs matrix comparison missing** for hard/rect/denom/light-cone cases
- **Julia's parametric "full R=..." tests** largely uncovered (16 parameter combos)

## Approach: Parametric Helpers + Gap Filling

Create reusable helper functions and use them to both upgrade existing partial tests and add missing Julia test cases.

## Helper Functions

### `assert_affine_mpo_matches_matrix`

Compares MPO (TensorTrain) dense reconstruction against `affine_transform_matrix` for all elements. Refactors existing `mpo_to_dense_matrix` + comparison loop pattern used in 4 existing tests.

```rust
fn assert_affine_mpo_matches_matrix(r: usize, params: &AffineParams, bc: &[BoundaryCondition])
```

### `assert_affine_matrix_correctness`

Exhaustively checks `affine_transform_matrix` output against direct computation of `y = A*x + b` for all inputs. Equivalent to Julia's `test_affine_transform_matrix_multi_variables`.

```rust
fn assert_affine_matrix_correctness(r: usize, params: &AffineParams, bc: &[BoundaryCondition])
```

## Test Changes

### Existing Test Upgrades

| Test | Current | After |
|------|---------|-------|
| `test_affine_matrix_3x3_hard` | nnz + dimensions | + `assert_affine_mpo_matches_matrix` |
| `test_affine_matrix_rectangular` | dimensions only | + `assert_affine_mpo_matches_matrix` |
| `test_affine_matrix_denom_odd` | nnz only | + `assert_affine_mpo_matches_matrix` |
| `test_affine_matrix_light_cone` | `is_ok()` only | + `assert_affine_mpo_matches_matrix` + `assert_affine_matrix_correctness` |

### New Tests

#### `test_affine_parametric_full` (Julia "full R=...")

Parametric test covering all combinations from Julia's `testtests` dictionary:

| (M,N) | A | b |
|-------|---|---|
| (1,1) | [1] | [1] |
| (1,2) | [1 0] | [0] |
| (1,2) | [2 -1] | [1] |
| (2,1) | [1; 0] | [0, 0] |
| (2,1) | [2; -1] | [1, -1] |
| (2,2) | [1 0; 1 1] | [0, 1] |
| (2,2) | [2 0; 4 1] | [100, -1] |

For each case: R in {1, 2}, BC in {Open, Periodic}. Both `assert_affine_mpo_matches_matrix` and `assert_affine_matrix_correctness`.

#### `test_affine_denom_even` (Julia "compare_denom_even")

- A = [1/2], b in {[3], [5], [-3], [-5]}, R in {3, 5}, BC = Periodic
- `assert_affine_mpo_matches_matrix`

#### Key Coverage Additions Summary

| Gap | Covered By |
|-----|-----------|
| Negative b numerical verification | parametric_full (b=[1,-1], b=[100,-1]), denom_even (b=[-3], b=[-5]) |
| Open BC numerical verification | parametric_full (all cases x Open BC) |
| Negative b + Open BC | parametric_full |
| MPO vs matrix for hard/rect cases | Upgraded existing tests |
| Rational + negative b | denom_even |

## Files to Modify

- `crates/tensor4all-quanticstransform/src/affine.rs` (test module only)

## Verification

```bash
cargo test -p tensor4all-quanticstransform -- affine
```
