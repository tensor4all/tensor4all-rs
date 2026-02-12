# Affine Transform Julia Algorithm Alignment

## Context

During test coverage work, we discovered two bugs in `affine_transform_matrix` (not MPO):
- Periodic BC + scale > 1: missed modular-equivalent y values
- Root cause: exact division check instead of Julia's `equiv` (modular equivalence)

Additionally, the MPO construction (`affine_transform_tensors`) uses a different algorithm than Julia (bit-position indexing vs iterative shifting). While functionally equivalent for the main loop, the Rust version lacks Julia's extension loop for `abs(b) >= 2^R` with Open BC. Since there's no advantage to the divergent approach, we align both functions with Julia.

## Changes

### 1. `affine_transform_matrix` (already fixed)

Iterate over all (x, y) pairs and check `equiv(A*x+b, scale*y, R, bc)` per component, matching Julia's approach.

### 2. `affine_transform_tensors` → Julia iterative shift

Replace bit-position indexing with Julia's iterative shifting:
- Separate `b` into `bsign = sign.(b)` and `b_work = abs.(b)`
- Main loop (R sites): extract `b_curr = (b_work .& 1) .* bsign`, then `b_work >>= 1`
- Extension loop (Open BC only): while `max(b_work) > 0`, create extra carry-only tensors with `activebit=false`
- Extension tensors prepended at MSB side; BC applied to leftmost tensor

### 3. `affine_transform_core` + `activebit` parameter

Add `activebit: bool` parameter (default `true`):
- When `activebit=false` and scale is odd: skip carry paths where `y != 0` (Julia PR #45 fix)
- When `activebit=false` and scale is even: only `y = 0` is valid (z must be even, carry_out = z/2)

### 4. Test coverage (remaining from previous plan)

- `test_affine_parametric_full` — Julia "full R=..." parametric test
- `test_affine_denom_even` — rational + negative b
- `test_affine_extension_loop` — `abs(b) >= 2^R` + Open BC

## Files

- `crates/tensor4all-quanticstransform/src/affine.rs` (implementation + tests)

## Verification

```bash
cargo test -p tensor4all-quanticstransform --lib
cargo clippy -p tensor4all-quanticstransform
cargo fmt --all
```
