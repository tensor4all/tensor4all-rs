//! Krylov subspace methods for solving linear equations with abstract tensors.
//!
//! This module provides iterative solvers that work with any type implementing [`TensorLike`],
//! enabling their use in tensor network algorithms without requiring dense vector representations.
//!
//! # Solvers
//!
//! - [`gmres`]: Generalized Minimal Residual Method (GMRES) for non-symmetric systems
//!
//! # Future Extensions
//!
//! - CG (Conjugate Gradient) for symmetric positive definite systems
//! - BiCGSTAB for non-symmetric systems with better convergence properties
//!
//! # Example
//!
//! ```ignore
//! use tensor4all_core::krylov::{gmres, GmresOptions};
//!
//! // Define a linear operator as a closure
//! let apply_operator = |x: &T| -> Result<T> {
//!     // Apply your linear operator to x
//!     operator.apply(x)
//! };
//!
//! let result = gmres(&apply_operator, &rhs, &initial_guess, &GmresOptions::default())?;
//! ```

use crate::any_scalar::AnyScalar;
use crate::TensorLike;
use anyhow::Result;

/// Options for GMRES solver.
#[derive(Debug, Clone)]
pub struct GmresOptions {
    /// Maximum number of iterations (restart cycle length).
    /// Default: 100
    pub max_iter: usize,

    /// Convergence tolerance for relative residual norm.
    /// The solver stops when `||r|| / ||b|| < rtol`.
    /// Default: 1e-10
    pub rtol: f64,

    /// Maximum number of restarts.
    /// Total iterations = max_iter * max_restarts.
    /// Default: 10
    pub max_restarts: usize,

    /// Whether to print convergence information.
    /// Default: false
    pub verbose: bool,
}

impl Default for GmresOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            rtol: 1e-10,
            max_restarts: 10,
            verbose: false,
        }
    }
}

/// Result of GMRES solver.
#[derive(Debug, Clone)]
pub struct GmresResult<T> {
    /// The solution vector.
    pub solution: T,

    /// Number of iterations performed.
    pub iterations: usize,

    /// Final relative residual norm.
    pub residual_norm: f64,

    /// Whether the solver converged.
    pub converged: bool,
}

/// Solve `A x = b` using GMRES (Generalized Minimal Residual Method).
///
/// This implements the restarted GMRES algorithm that works with abstract tensor types
/// through the [`TensorLike`] trait's vector space operations.
///
/// # Algorithm
///
/// GMRES builds an orthonormal basis for the Krylov subspace
/// `K_m = span{r_0, A r_0, A^2 r_0, ..., A^{m-1} r_0}` and finds the
/// solution that minimizes `||b - A x||` over this subspace.
///
/// # Type Parameters
///
/// * `T` - A tensor type implementing `TensorLike`
/// * `F` - A function that applies the linear operator: `F(x) = A x`
///
/// # Arguments
///
/// * `apply_a` - Function that applies the linear operator A to a tensor
/// * `b` - Right-hand side tensor
/// * `x0` - Initial guess
/// * `options` - Solver options
///
/// # Returns
///
/// A `GmresResult` containing the solution and convergence information.
///
/// # Errors
///
/// Returns an error if:
/// - Vector space operations (add, sub, scale, inner_product) fail
/// - The linear operator application fails
pub fn gmres<T, F>(apply_a: F, b: &T, x0: &T, options: &GmresOptions) -> Result<GmresResult<T>>
where
    T: TensorLike,
    F: Fn(&T) -> Result<T>,
{
    let b_norm = b.norm();
    if b_norm < 1e-15 {
        // b is effectively zero, return x0
        return Ok(GmresResult {
            solution: x0.clone(),
            iterations: 0,
            residual_norm: 0.0,
            converged: true,
        });
    }

    let mut x = x0.clone();
    let mut total_iters = 0;

    for _restart in 0..options.max_restarts {
        // Compute initial residual: r = b - A*x
        let ax = apply_a(&x)?;
        // r = 1.0 * b + (-1.0) * ax
        let r = b.axpby(AnyScalar::F64(1.0), &ax, AnyScalar::F64(-1.0))?;
        let r_norm = r.norm();
        let rel_res = r_norm / b_norm;

        if options.verbose {
            eprintln!(
                "GMRES restart {}: initial residual = {:.6e}",
                _restart, rel_res
            );
        }

        if rel_res < options.rtol {
            return Ok(GmresResult {
                solution: x,
                iterations: total_iters,
                residual_norm: rel_res,
                converged: true,
            });
        }

        // Arnoldi process with modified Gram-Schmidt
        let mut v_basis: Vec<T> = Vec::with_capacity(options.max_iter + 1);
        let mut h_matrix: Vec<Vec<AnyScalar>> = Vec::with_capacity(options.max_iter);

        // v_0 = r / ||r||
        let v0 = r.scale(AnyScalar::F64(1.0 / r_norm))?;
        v_basis.push(v0);

        // Initialize Givens rotation storage
        let mut cs: Vec<AnyScalar> = Vec::with_capacity(options.max_iter);
        let mut sn: Vec<AnyScalar> = Vec::with_capacity(options.max_iter);
        let mut g: Vec<AnyScalar> = vec![AnyScalar::F64(r_norm)]; // residual in upper Hessenberg space

        for j in 0..options.max_iter {
            total_iters += 1;

            // w = A * v_j
            let w = apply_a(&v_basis[j])?;

            // Modified Gram-Schmidt orthogonalization
            let mut h_col: Vec<AnyScalar> = Vec::with_capacity(j + 2);
            let mut w_orth = w;

            for v_i in v_basis.iter().take(j + 1) {
                let h_ij = v_i.inner_product(&w_orth)?;
                h_col.push(h_ij.clone());
                // w_orth = w_orth - h_ij * v_i = 1.0 * w_orth + (-h_ij) * v_i
                let neg_h_ij = AnyScalar::F64(0.0) - h_ij;
                w_orth = w_orth.axpby(AnyScalar::F64(1.0), v_i, neg_h_ij)?;
            }

            let h_jp1_j_real = w_orth.norm();
            let h_jp1_j = AnyScalar::F64(h_jp1_j_real);
            h_col.push(h_jp1_j);

            // Apply previous Givens rotations to new column
            #[allow(clippy::needless_range_loop)]
            for i in 0..j {
                let h_i = h_col[i].clone();
                let h_ip1 = h_col[i + 1].clone();
                let (new_hi, new_hip1) = apply_givens_rotation(&cs[i], &sn[i], &h_i, &h_ip1);
                h_col[i] = new_hi;
                h_col[i + 1] = new_hip1;
            }

            // Compute new Givens rotation for h_col[j] and h_col[j+1]
            let (c_j, s_j) = compute_givens_rotation(&h_col[j], &h_col[j + 1]);
            cs.push(c_j.clone());
            sn.push(s_j.clone());

            // Apply new rotation to eliminate h_col[j+1]
            let (new_hj, _) = apply_givens_rotation(&c_j, &s_j, &h_col[j], &h_col[j + 1]);
            h_col[j] = new_hj;
            h_col[j + 1] = AnyScalar::F64(0.0);

            // Apply rotation to g
            let g_j = g[j].clone();
            let g_jp1 = AnyScalar::F64(0.0);
            let (new_gj, new_gjp1) = apply_givens_rotation(&c_j, &s_j, &g_j, &g_jp1);
            g[j] = new_gj;
            let res_norm = new_gjp1.abs();
            g.push(new_gjp1);

            h_matrix.push(h_col);

            // Check convergence
            let rel_res = res_norm / b_norm;

            if options.verbose && (j + 1) % 10 == 0 {
                eprintln!("GMRES iter {}: residual = {:.6e}", j + 1, rel_res);
            }

            if rel_res < options.rtol {
                // Solve upper triangular system and update x
                let y = solve_upper_triangular(&h_matrix, &g[..=j])?;
                x = update_solution(&x, &v_basis[..=j], &y)?;
                return Ok(GmresResult {
                    solution: x,
                    iterations: total_iters,
                    residual_norm: rel_res,
                    converged: true,
                });
            }

            // Add new basis vector (if not converged and h_jp1_j is not too small)
            if h_jp1_j_real > 1e-14 {
                let v_jp1 = w_orth.scale(AnyScalar::F64(1.0 / h_jp1_j_real))?;
                v_basis.push(v_jp1);
            } else {
                // Lucky breakdown - we've found the exact solution in the Krylov subspace
                let y = solve_upper_triangular(&h_matrix, &g[..=j])?;
                x = update_solution(&x, &v_basis[..=j], &y)?;
                let ax_final = apply_a(&x)?;
                let r_final = b.axpby(AnyScalar::F64(1.0), &ax_final, AnyScalar::F64(-1.0))?;
                let final_res = r_final.norm() / b_norm;
                return Ok(GmresResult {
                    solution: x,
                    iterations: total_iters,
                    residual_norm: final_res,
                    converged: final_res < options.rtol,
                });
            }
        }

        // End of restart cycle - update x with current solution
        let y = solve_upper_triangular(&h_matrix, &g[..options.max_iter])?;
        x = update_solution(&x, &v_basis[..options.max_iter], &y)?;
    }

    // Compute final residual
    let ax_final = apply_a(&x)?;
    let r_final = b.axpby(AnyScalar::F64(1.0), &ax_final, AnyScalar::F64(-1.0))?;
    let final_res = r_final.norm() / b_norm;

    Ok(GmresResult {
        solution: x,
        iterations: total_iters,
        residual_norm: final_res,
        converged: final_res < options.rtol,
    })
}

/// Compute Givens rotation coefficients to eliminate b in (a, b).
///
/// This function works with any AnyScalar variant by converting to Complex64
/// for computation, then returning results in the appropriate type.
fn compute_givens_rotation(a: &AnyScalar, b: &AnyScalar) -> (AnyScalar, AnyScalar) {
    use num_complex::Complex64;

    // Handle the simple f64-only case without conversion
    if !a.is_complex() && !b.is_complex() {
        let a_val = a.real();
        let b_val = b.real();
        let r = (a_val * a_val + b_val * b_val).sqrt();
        if r < 1e-15 {
            return (AnyScalar::F64(1.0), AnyScalar::F64(0.0));
        }
        return (AnyScalar::F64(a_val / r), AnyScalar::F64(b_val / r));
    }

    // For complex or mixed cases, convert to Complex64
    let a_c: Complex64 = a.clone().into();
    let b_c: Complex64 = b.clone().into();
    let r = (a_c.norm_sqr() + b_c.norm_sqr()).sqrt();
    if r < 1e-15 {
        (
            AnyScalar::C64(Complex64::new(1.0, 0.0)),
            AnyScalar::C64(Complex64::new(0.0, 0.0)),
        )
    } else {
        (AnyScalar::C64(a_c / r), AnyScalar::C64(b_c / r))
    }
}

/// Apply Givens rotation: (c, s) @ (x, y) -> (c*x + s*y, -conj(s)*x + c*y) for complex
/// or (c*x + s*y, -s*x + c*y) for real.
///
/// This function works with any AnyScalar variant by converting to Complex64
/// for computation when needed.
fn apply_givens_rotation(
    c: &AnyScalar,
    s: &AnyScalar,
    x: &AnyScalar,
    y: &AnyScalar,
) -> (AnyScalar, AnyScalar) {
    use num_complex::Complex64;

    // Handle the simple f64-only case without conversion
    if !c.is_complex() && !s.is_complex() && !x.is_complex() && !y.is_complex() {
        let c_val = c.real();
        let s_val = s.real();
        let x_val = x.real();
        let y_val = y.real();
        let new_x = c_val * x_val + s_val * y_val;
        let new_y = -s_val * x_val + c_val * y_val;
        return (AnyScalar::F64(new_x), AnyScalar::F64(new_y));
    }

    // For complex or mixed cases, convert to Complex64
    let c_c: Complex64 = c.clone().into();
    let s_c: Complex64 = s.clone().into();
    let x_c: Complex64 = x.clone().into();
    let y_c: Complex64 = y.clone().into();

    let new_x = c_c * x_c + s_c * y_c;
    let new_y = -s_c.conj() * x_c + c_c * y_c;
    (AnyScalar::C64(new_x), AnyScalar::C64(new_y))
}

/// Solve upper triangular system R y = g using back substitution.
fn solve_upper_triangular(h: &[Vec<AnyScalar>], g: &[AnyScalar]) -> Result<Vec<AnyScalar>> {
    let n = g.len();
    if n == 0 {
        return Ok(vec![]);
    }

    let mut y = vec![AnyScalar::F64(0.0); n];

    for i in (0..n).rev() {
        let mut sum = g[i].clone();

        for j in (i + 1)..n {
            // sum = sum - h[j][i] * y[j]
            let prod = h[j][i].clone() * y[j].clone();
            sum = sum - prod;
        }

        let h_ii = &h[i][i];
        if h_ii.abs() < 1e-15 {
            return Err(anyhow::anyhow!(
                "Near-singular upper triangular matrix in GMRES"
            ));
        }

        y[i] = sum / h_ii.clone();
    }

    Ok(y)
}

/// Update solution: x_new = x + sum_i y_i * v_i
fn update_solution<T: TensorLike>(x: &T, v_basis: &[T], y: &[AnyScalar]) -> Result<T> {
    let mut result = x.clone();

    for (vi, yi) in v_basis.iter().zip(y.iter()) {
        let scaled_vi = vi.scale(yi.clone())?;
        // result = result + scaled_vi = 1.0 * result + 1.0 * scaled_vi
        result = result.axpby(AnyScalar::F64(1.0), &scaled_vi, AnyScalar::F64(1.0))?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::tensordynlen::TensorDynLen;
    use crate::defaults::DynIndex;
    use crate::storage::{DenseStorageF64, Storage};
    use std::sync::Arc;

    /// Helper to create a 1D tensor (vector) with given data and shared index.
    fn make_vector_with_index(data: Vec<f64>, idx: &DynIndex) -> TensorDynLen {
        let n = data.len();
        TensorDynLen::new(
            vec![idx.clone()],
            Arc::new(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                data,
                &[n],
            ))),
        )
    }

    #[test]
    fn test_givens_rotation_real() {
        let a = AnyScalar::new_real(3.0);
        let b = AnyScalar::new_real(4.0);
        let (c, s) = compute_givens_rotation(&a, &b);

        // c = 3/5 = 0.6, s = 4/5 = 0.8
        assert!(!c.is_complex());
        assert!(!s.is_complex());
        assert!((c.real() - 0.6).abs() < 1e-10);
        assert!((s.real() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_apply_givens_rotation_real() {
        let c = AnyScalar::new_real(0.6);
        let s = AnyScalar::new_real(0.8);
        let x = AnyScalar::new_real(3.0);
        let y = AnyScalar::new_real(4.0);

        let (new_x, new_y) = apply_givens_rotation(&c, &s, &x, &y);

        // new_x = 0.6*3 + 0.8*4 = 1.8 + 3.2 = 5.0
        // new_y = -0.8*3 + 0.6*4 = -2.4 + 2.4 = 0.0
        assert!(!new_x.is_complex());
        assert!(!new_y.is_complex());
        assert!((new_x.real() - 5.0).abs() < 1e-10);
        assert!(new_y.real().abs() < 1e-10);
    }

    #[test]
    fn test_gmres_identity_operator() {
        // Solve A x = b where A = I (identity)
        // Solution: x = b
        let idx = DynIndex::new_dyn(3);
        let b = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);
        let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);

        // Identity operator: A x = x
        let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };

        let options = GmresOptions {
            max_iter: 10,
            rtol: 1e-10,
            max_restarts: 1,
            verbose: false,
        };

        let result = gmres(apply_a, &b, &x0, &options).unwrap();

        assert!(result.converged, "GMRES should converge for identity");
        assert!(
            result.residual_norm < 1e-10,
            "Residual should be small: {}",
            result.residual_norm
        );

        // Check solution matches b
        let diff = result
            .solution
            .axpby(AnyScalar::F64(1.0), &b, AnyScalar::F64(-1.0))
            .unwrap();
        assert!(diff.norm() < 1e-10, "Solution should equal b");
    }

    #[test]
    fn test_gmres_diagonal_matrix() {
        // Solve A x = b where A = diag(2, 3, 4)
        // b = [2, 6, 12] → x = [1, 2, 3]
        let idx = DynIndex::new_dyn(3);
        let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
        let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);
        let expected_x = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

        // Diagonal scaling operator
        let diag = [2.0, 3.0, 4.0];
        let apply_a = move |x: &TensorDynLen| -> Result<TensorDynLen> {
            // Element-wise multiply by diagonal
            let x_data = match x.storage().as_ref() {
                Storage::DenseF64(d) => d.as_slice().to_vec(),
                _ => panic!("Expected DenseF64"),
            };
            let result_data: Vec<f64> = x_data
                .iter()
                .zip(diag.iter())
                .map(|(&xi, &di)| xi * di)
                .collect();
            let dims = x.dims();
            Ok(TensorDynLen::new(
                x.indices.clone(),
                Arc::new(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                    result_data,
                    &dims,
                ))),
            ))
        };

        let options = GmresOptions {
            max_iter: 10,
            rtol: 1e-10,
            max_restarts: 1,
            verbose: false,
        };

        let result = gmres(apply_a, &b, &x0, &options).unwrap();

        assert!(result.converged, "GMRES should converge");
        assert!(
            result.residual_norm < 1e-10,
            "Residual should be small: {}",
            result.residual_norm
        );

        // Check solution
        let diff = result
            .solution
            .axpby(AnyScalar::F64(1.0), &expected_x, AnyScalar::F64(-1.0))
            .unwrap();
        assert!(
            diff.norm() < 1e-8,
            "Solution error too large: {}",
            diff.norm()
        );
    }

    #[test]
    fn test_gmres_nonsymmetric_matrix() {
        // Solve A x = b where A is a 2x2 non-symmetric matrix
        // A = [[2, 1], [0, 3]]
        // A x = [2*1 + 1*2, 0*1 + 3*2] = [4, 6]
        // So b = [4, 6] → x = [1, 2]
        let idx = DynIndex::new_dyn(2);
        let b = make_vector_with_index(vec![4.0, 6.0], &idx);
        let x0 = make_vector_with_index(vec![0.0, 0.0], &idx);
        let expected_x = make_vector_with_index(vec![1.0, 2.0], &idx);

        // Matrix A (stored as row-major)
        let a_data = [2.0, 1.0, 0.0, 3.0];

        let apply_a = move |x: &TensorDynLen| -> Result<TensorDynLen> {
            let x_data = match x.storage().as_ref() {
                Storage::DenseF64(d) => d.as_slice().to_vec(),
                _ => panic!("Expected DenseF64"),
            };
            // Matrix-vector multiply (2x2)
            let result_data = vec![
                a_data[0] * x_data[0] + a_data[1] * x_data[1],
                a_data[2] * x_data[0] + a_data[3] * x_data[1],
            ];
            let dims = x.dims();
            Ok(TensorDynLen::new(
                x.indices.clone(),
                Arc::new(Storage::DenseF64(DenseStorageF64::from_vec_with_shape(
                    result_data,
                    &dims,
                ))),
            ))
        };

        let options = GmresOptions {
            max_iter: 10,
            rtol: 1e-10,
            max_restarts: 1,
            verbose: false,
        };

        let result = gmres(apply_a, &b, &x0, &options).unwrap();

        assert!(result.converged, "GMRES should converge");

        // Check solution
        let diff = result
            .solution
            .axpby(AnyScalar::F64(1.0), &expected_x, AnyScalar::F64(-1.0))
            .unwrap();
        assert!(
            diff.norm() < 1e-8,
            "Solution error too large: {}",
            diff.norm()
        );
    }

    #[test]
    fn test_gmres_with_good_initial_guess() {
        // If initial guess is already the solution, should converge immediately
        let idx = DynIndex::new_dyn(3);
        let b = make_vector_with_index(vec![2.0, 4.0, 6.0], &idx);
        let x0 = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx); // Already the solution for A=diag(2,2,2)

        let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> {
            // A = 2*I
            x.scale(AnyScalar::F64(2.0))
        };

        let options = GmresOptions {
            max_iter: 10,
            rtol: 1e-10,
            max_restarts: 1,
            verbose: false,
        };

        let result = gmres(apply_a, &b, &x0, &options).unwrap();

        assert!(result.converged);
        assert_eq!(result.iterations, 0, "Should converge with 0 iterations");
    }

    /// Helper to create a 1D complex tensor (vector) with given data and shared index.
    fn make_vector_c64_with_index(
        data: Vec<num_complex::Complex64>,
        idx: &DynIndex,
    ) -> TensorDynLen {
        TensorDynLen::from_dense_c64(vec![idx.clone()], data)
    }

    #[test]
    fn test_gmres_identity_operator_c64() {
        // Solve A x = b where A = I (identity), b is complex
        // Solution: x = b
        use num_complex::Complex64;

        let idx = DynIndex::new_dyn(4);
        let b = make_vector_c64_with_index(
            vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(-3.0, 0.5),
                Complex64::new(0.0, -1.0),
                Complex64::new(2.5, 3.5),
            ],
            &idx,
        );
        let x0 = make_vector_c64_with_index(vec![Complex64::new(0.0, 0.0); 4], &idx);

        // Identity operator: A x = x
        let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };

        let options = GmresOptions {
            max_iter: 20,
            rtol: 1e-10,
            max_restarts: 1,
            verbose: false,
        };

        let result = gmres(apply_a, &b, &x0, &options).unwrap();

        // Check solution matches b
        let diff = result
            .solution
            .axpby(AnyScalar::new_real(1.0), &b, AnyScalar::new_real(-1.0))
            .unwrap();
        let err = diff.norm();

        assert!(result.converged, "GMRES should converge for identity (c64)");
        assert!(
            result.residual_norm < 1e-10,
            "Residual should be small: {}",
            result.residual_norm
        );
        assert!(err < 1e-8, "Solution should equal b, error: {}", err);
    }

    #[test]
    fn test_gmres_diagonal_c64() {
        // Solve A x = b where A = diag(2+i, 3-i, 1+2i, 4)
        use num_complex::Complex64;

        let idx = DynIndex::new_dyn(4);
        let diag = [
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(1.0, 2.0),
            Complex64::new(4.0, 0.0),
        ];
        let x_true = [
            Complex64::new(1.0, -1.0),
            Complex64::new(0.5, 2.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 1.0),
        ];
        let b_data: Vec<Complex64> = diag.iter().zip(x_true.iter()).map(|(d, x)| d * x).collect();

        let b = make_vector_c64_with_index(b_data, &idx);
        let x0 = make_vector_c64_with_index(vec![Complex64::new(0.0, 0.0); 4], &idx);
        let expected = make_vector_c64_with_index(x_true.to_vec(), &idx);

        let apply_a = move |x: &TensorDynLen| -> Result<TensorDynLen> {
            let x_data = x.to_vec_c64()?;
            let result_data: Vec<Complex64> = x_data
                .iter()
                .zip(diag.iter())
                .map(|(&xi, &di)| di * xi)
                .collect();
            Ok(TensorDynLen::from_dense_c64(x.indices.clone(), result_data))
        };

        let options = GmresOptions {
            max_iter: 20,
            rtol: 1e-10,
            max_restarts: 1,
            verbose: false,
        };

        let result = gmres(apply_a, &b, &x0, &options).unwrap();

        let diff = result
            .solution
            .axpby(
                AnyScalar::new_real(1.0),
                &expected,
                AnyScalar::new_real(-1.0),
            )
            .unwrap();
        let err = diff.norm();

        assert!(result.converged, "GMRES should converge for diagonal (c64)");
        assert!(err < 1e-8, "Solution error too large: {}", err);
    }

    #[test]
    fn test_gmres_zero_rhs() {
        // Solve A x = 0 → x = 0 (for any invertible A)
        let idx = DynIndex::new_dyn(3);
        let b = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);
        let x0 = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

        let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };

        let options = GmresOptions::default();

        let result = gmres(apply_a, &b, &x0, &options).unwrap();

        // With zero RHS, should return initial guess immediately
        assert!(result.converged);
    }
}
