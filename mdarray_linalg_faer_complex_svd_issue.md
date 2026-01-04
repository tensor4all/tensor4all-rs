# mdarray-linalg-faer: Complex SVD returns V^T instead of V^H

## Summary

When performing SVD on complex matrices, `mdarray-linalg-faer` returns `V^T` (plain transpose) instead of the expected `V^H` (Hermitian conjugate / conjugate transpose).

## Expected Behavior

For the SVD decomposition `A = U * Σ * V^H`:
- The output `vt` should represent `V^H = conj(V)^T`
- This is the standard convention used by LAPACK, NumPy, and other linear algebra libraries

## Actual Behavior

- `mdarray-linalg-faer` returns `vt = V^T` (plain transpose, no conjugation)
- This causes incorrect reconstruction for complex matrices

## Root Cause Analysis

The `faer` library itself correctly computes `V` (right singular vectors). The issue is in the `mdarray-linalg-faer` wrapper:

```rust
// In mdarray-linalg-faer/src/svd/simple.rs
let vt_faer = into_faer_mut_transpose(y);
```

The wrapper transposes the output buffer before passing to `faer`. Since `faer` writes `V` into this transposed buffer, the result is `V^T`, not `V^H`.

## Reproduction

```rust
use num_complex::Complex64;
use mdarray::DTensor;
use mdarray_linalg::svd::SVD;
use mdarray_linalg_faer::Faer;

fn main() {
    // Create a 3x2 complex matrix
    let mut matrix: DTensor<Complex64, 2> = DTensor::from_elem([3, 2], Complex64::new(0.0, 0.0));
    matrix[[0, 0]] = Complex64::new(1.0, 0.5);
    matrix[[0, 1]] = Complex64::new(2.0, -0.3);
    matrix[[1, 0]] = Complex64::new(3.0, 0.2);
    matrix[[1, 1]] = Complex64::new(4.0, 0.4);
    matrix[[2, 0]] = Complex64::new(5.0, -0.1);
    matrix[[2, 1]] = Complex64::new(6.0, 0.6);

    let original = matrix.clone();

    let bd = Faer;
    let svd_result = bd.svd(matrix.as_mut()).unwrap();

    let u = &svd_result.u;
    let s = &svd_result.s;
    let vt = &svd_result.vt;

    // Reconstruct: A = U * diag(S) * Vt
    let m = 3;
    let n = 2;
    let k = 2;

    // Using vt directly (FAILS for complex)
    let mut max_error_direct = 0.0_f64;
    for i in 0..m {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for l in 0..k {
                sum = sum + u[[i, l]] * s[[0, l]] * vt[[l, j]];
            }
            let err = (sum - original[[i, j]]).norm();
            max_error_direct = max_error_direct.max(err);
        }
    }

    // Using conj(vt) (WORKS)
    let mut max_error_conj = 0.0_f64;
    for i in 0..m {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for l in 0..k {
                sum = sum + u[[i, l]] * s[[0, l]] * vt[[l, j]].conj();
            }
            let err = (sum - original[[i, j]]).norm();
            max_error_conj = max_error_conj.max(err);
        }
    }

    println!("Max error using vt directly: {}", max_error_direct);   // ~0.74 (WRONG)
    println!("Max error using conj(vt):    {}", max_error_conj);     // ~1e-15 (CORRECT)
}
```

## Proposed Fix

In `mdarray-linalg-faer/src/svd/simple.rs`, after receiving `V` from `faer`, apply conjugation for complex types:

```rust
// After faer::linalg::svd::svd() call, for complex types:
// Apply conjugation to convert V^T to V^H
if T is complex {
    for each element in vt {
        vt[i, j] = conj(vt[i, j]);
    }
}
```

Or alternatively, use `ComplexFloat::conj()` which is a no-op for real types:

```rust
for i in 0..vt.nrows() {
    for j in 0..vt.ncols() {
        vt[(i, j)] = vt[(i, j)].conj();
    }
}
```

## Environment

- `mdarray-linalg-faer`: v0.2.0 (commit b8199991)
- `faer`: v0.23.2
- Rust: stable

## Impact

Any code using `mdarray-linalg-faer` for complex matrix SVD will get incorrect results when reconstructing the original matrix or using the right singular vectors.

## Workaround

Until this is fixed upstream, users can manually conjugate `vt` elements:

```rust
let corrected_vt_ij = vt[[i, j]].conj();
```
