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

    /// When true, verify convergence by computing the true residual `||b - A*x|| / ||b||`
    /// before declaring convergence. This prevents false convergence caused by
    /// truncation corrupting the Krylov basis orthogonality (see Issue #207).
    /// Costs one additional `apply_a` call when convergence is detected.
    /// Default: false
    pub check_true_residual: bool,
}

impl Default for GmresOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            rtol: 1e-10,
            max_restarts: 10,
            verbose: false,
            check_true_residual: false,
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
    // Validate structural consistency of inputs
    b.validate()?;
    x0.validate()?;

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
        // Validate operator output on first restart
        if _restart == 0 {
            ax.validate()?;
        }
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

            if options.verbose {
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

/// Solve `A x = b` using GMRES with optional truncation after each iteration.
///
/// This is an extension of [`gmres`] that allows truncating Krylov basis vectors
/// to control bond dimension growth in tensor network representations.
///
/// # Type Parameters
///
/// * `T` - A tensor type implementing `TensorLike`
/// * `F` - A function that applies the linear operator: `F(x) = A x`
/// * `Tr` - A function that truncates a tensor in-place: `Tr(&mut x)`
///
/// # Arguments
///
/// * `apply_a` - Function that applies the linear operator A to a tensor
/// * `b` - Right-hand side tensor
/// * `x0` - Initial guess
/// * `options` - Solver options
/// * `truncate` - Function that truncates a tensor to control bond dimension
///
/// # Note
///
/// Truncation is applied after each Gram-Schmidt orthogonalization step
/// and after the final solution update. This helps control the bond dimension
/// growth that would otherwise occur in MPS/MPO representations.
pub fn gmres_with_truncation<T, F, Tr>(
    apply_a: F,
    b: &T,
    x0: &T,
    options: &GmresOptions,
    truncate: Tr,
) -> Result<GmresResult<T>>
where
    T: TensorLike,
    F: Fn(&T) -> Result<T>,
    Tr: Fn(&mut T) -> Result<()>,
{
    // Validate structural consistency of inputs
    b.validate()?;
    x0.validate()?;

    let b_norm = b.norm();
    if b_norm < 1e-15 {
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
        let ax = apply_a(&x)?;
        // Validate operator output on first restart
        if _restart == 0 {
            ax.validate()?;
        }
        let mut r = b.axpby(AnyScalar::F64(1.0), &ax, AnyScalar::F64(-1.0))?;
        truncate(&mut r)?;
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

        let mut v_basis: Vec<T> = Vec::with_capacity(options.max_iter + 1);
        let mut h_matrix: Vec<Vec<AnyScalar>> = Vec::with_capacity(options.max_iter);

        let mut v0 = r.scale(AnyScalar::F64(1.0 / r_norm))?;
        truncate(&mut v0)?;
        // After truncation, v0 might not be unit norm and might point in a different direction.
        // We need to:
        // 1. Renormalize v0 to unit norm for numerical stability
        // 2. Recompute g[0] = <r, v0> to maintain the correct relationship
        let v0_norm = v0.norm();
        let effective_g0 = if v0_norm > 1e-15 {
            v0 = v0.scale(AnyScalar::F64(1.0 / v0_norm))?;
            // g[0] should be the component of r in the direction of v0
            // Since r was truncated and v0 = truncate(r/||r||)/||truncate(r/||r||)||,
            // g[0] = <r, v0> ≈ ||r|| * ||truncate(r/||r||)|| = r_norm * v0_norm
            r_norm * v0_norm
        } else {
            r_norm
        };
        v_basis.push(v0);

        let mut cs: Vec<AnyScalar> = Vec::with_capacity(options.max_iter);
        let mut sn: Vec<AnyScalar> = Vec::with_capacity(options.max_iter);
        let mut g: Vec<AnyScalar> = vec![AnyScalar::F64(effective_g0)];
        let mut solution_already_updated = false;

        for j in 0..options.max_iter {
            total_iters += 1;

            let w = apply_a(&v_basis[j])?;

            let mut h_col: Vec<AnyScalar> = Vec::with_capacity(j + 2);
            let mut w_orth = w;

            for v_i in v_basis.iter().take(j + 1) {
                let h_ij = v_i.inner_product(&w_orth)?;
                h_col.push(h_ij.clone());
                let neg_h_ij = AnyScalar::F64(0.0) - h_ij;
                w_orth = w_orth.axpby(AnyScalar::F64(1.0), v_i, neg_h_ij)?;
            }

            // Iterative reorthogonalization with truncation
            // Truncation can change the direction of w_orth, breaking orthogonality.
            // We iterate until all corrections are below a threshold to ensure
            // the Krylov basis remains orthogonal despite truncation.
            const REORTH_THRESHOLD: f64 = 1e-12;
            const MAX_REORTH_ITERS: usize = 10;

            let mut reorth_iter_count = 0;
            for reorth_iter in 0..MAX_REORTH_ITERS {
                reorth_iter_count = reorth_iter + 1;
                let norm_before_truncate = w_orth.norm();
                truncate(&mut w_orth)?;
                let norm_after_truncate = w_orth.norm();

                let mut max_correction = 0.0;
                for (i, v_i) in v_basis.iter().enumerate() {
                    let correction = v_i.inner_product(&w_orth)?;
                    let correction_abs = correction.abs();
                    if correction_abs > max_correction {
                        max_correction = correction_abs;
                    }
                    if correction_abs > REORTH_THRESHOLD {
                        let neg_correction = AnyScalar::F64(0.0) - correction.clone();
                        w_orth = w_orth.axpby(AnyScalar::F64(1.0), v_i, neg_correction)?;
                        // Update Hessenberg matrix entry to include correction
                        h_col[i] = h_col[i].clone() + correction;
                    }
                }

                if options.verbose {
                    eprintln!(
                        "  reorth iter {}: norm {:.6e} -> {:.6e}, max_correction = {:.6e}",
                        reorth_iter, norm_before_truncate, norm_after_truncate, max_correction
                    );
                }

                // If all corrections are small enough, we're done
                if max_correction < REORTH_THRESHOLD {
                    break;
                }
            }

            if options.verbose && reorth_iter_count > 1 {
                eprintln!("  (needed {} reorth iterations)", reorth_iter_count);
            }

            let h_jp1_j_real = w_orth.norm();
            let h_jp1_j = AnyScalar::F64(h_jp1_j_real);
            h_col.push(h_jp1_j);

            #[allow(clippy::needless_range_loop)]
            for i in 0..j {
                let h_i = h_col[i].clone();
                let h_ip1 = h_col[i + 1].clone();
                let (new_hi, new_hip1) = apply_givens_rotation(&cs[i], &sn[i], &h_i, &h_ip1);
                h_col[i] = new_hi;
                h_col[i + 1] = new_hip1;
            }

            let (c_j, s_j) = compute_givens_rotation(&h_col[j], &h_col[j + 1]);
            cs.push(c_j.clone());
            sn.push(s_j.clone());

            let (new_hj, _) = apply_givens_rotation(&c_j, &s_j, &h_col[j], &h_col[j + 1]);
            h_col[j] = new_hj;
            h_col[j + 1] = AnyScalar::F64(0.0);

            let g_j = g[j].clone();
            let g_jp1 = AnyScalar::F64(0.0);
            let (new_gj, new_gjp1) = apply_givens_rotation(&c_j, &s_j, &g_j, &g_jp1);
            g[j] = new_gj;
            let res_norm = new_gjp1.abs();
            g.push(new_gjp1);

            h_matrix.push(h_col);

            let rel_res = res_norm / b_norm;

            if options.verbose {
                eprintln!("GMRES iter {}: residual = {:.6e}", j + 1, rel_res);
            }

            if rel_res < options.rtol {
                let y = solve_upper_triangular(&h_matrix, &g[..=j])?;
                x = update_solution_truncated(&x, &v_basis[..=j], &y, &truncate)?;

                if options.check_true_residual {
                    // Verify with true residual to prevent false convergence
                    let ax_check = apply_a(&x)?;
                    let mut r_check =
                        b.axpby(AnyScalar::F64(1.0), &ax_check, AnyScalar::F64(-1.0))?;
                    truncate(&mut r_check)?;
                    let true_rel_res = r_check.norm() / b_norm;

                    if options.verbose {
                        eprintln!(
                            "GMRES true residual check: hessenberg={:.6e}, checked={:.6e}",
                            rel_res, true_rel_res
                        );
                    }

                    if true_rel_res < options.rtol {
                        return Ok(GmresResult {
                            solution: x,
                            iterations: total_iters,
                            residual_norm: true_rel_res,
                            converged: true,
                        });
                    }
                    // False convergence detected: x is already updated above,
                    // so skip the end-of-cycle update and go to next restart
                    solution_already_updated = true;
                    break;
                } else {
                    return Ok(GmresResult {
                        solution: x,
                        iterations: total_iters,
                        residual_norm: rel_res,
                        converged: true,
                    });
                }
            }

            if h_jp1_j_real > 1e-14 {
                // Create v_{j+1} = w_orth / ||w_orth||
                // w_orth has already been truncated twice (after orthogonalization and after reorthogonalization)
                // so we don't need to truncate again. Scale doesn't increase bond dimensions.
                let v_jp1 = w_orth.scale(AnyScalar::F64(1.0 / h_jp1_j_real))?;
                // v_jp1 should have norm ~1.0 by construction
                // The Arnoldi relation h_{j+1,j} * v_{j+1} = w_orth is maintained exactly
                v_basis.push(v_jp1);
            } else {
                let y = solve_upper_triangular(&h_matrix, &g[..=j])?;
                x = update_solution_truncated(&x, &v_basis[..=j], &y, &truncate)?;
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

        if !solution_already_updated {
            let actual_iters = v_basis.len().min(options.max_iter);
            let y = solve_upper_triangular(&h_matrix, &g[..actual_iters])?;
            x = update_solution_truncated(&x, &v_basis[..actual_iters], &y, &truncate)?;
        }
    }

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

/// Options for restarted GMRES with truncation.
///
/// This is used by [`restart_gmres_with_truncation`] which wraps the standard GMRES
/// with an outer loop that recomputes the true residual at each restart.
#[derive(Debug, Clone)]
pub struct RestartGmresOptions {
    /// Maximum number of outer restart iterations.
    /// Default: 20
    pub max_outer_iters: usize,

    /// Convergence tolerance for relative residual norm (based on true residual).
    /// The solver stops when `||b - A*x|| / ||b|| < rtol`.
    /// Default: 1e-10
    pub rtol: f64,

    /// Maximum iterations per inner GMRES cycle.
    /// Default: 10
    pub inner_max_iter: usize,

    /// Number of restarts within each inner GMRES (usually 0).
    /// Default: 0
    pub inner_max_restarts: usize,

    /// Stagnation detection threshold.
    /// If the residual reduction ratio exceeds this value (i.e., residual doesn't decrease enough),
    /// the solver considers it stagnated.
    /// For example, 0.99 means stagnation is detected when residual decreases by less than 1%.
    /// Default: None (no stagnation detection)
    pub min_reduction: Option<f64>,

    /// Inner GMRES relative tolerance.
    /// If None, uses 0.1 (solve inner problem loosely).
    /// Default: None
    pub inner_rtol: Option<f64>,

    /// Whether to print convergence information.
    /// Default: false
    pub verbose: bool,
}

impl Default for RestartGmresOptions {
    fn default() -> Self {
        Self {
            max_outer_iters: 20,
            rtol: 1e-10,
            inner_max_iter: 10,
            inner_max_restarts: 0,
            min_reduction: None,
            inner_rtol: None,
            verbose: false,
        }
    }
}

impl RestartGmresOptions {
    /// Create new options with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum number of outer iterations.
    pub fn with_max_outer_iters(mut self, max_outer_iters: usize) -> Self {
        self.max_outer_iters = max_outer_iters;
        self
    }

    /// Set convergence tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol;
        self
    }

    /// Set maximum iterations per inner GMRES cycle.
    pub fn with_inner_max_iter(mut self, inner_max_iter: usize) -> Self {
        self.inner_max_iter = inner_max_iter;
        self
    }

    /// Set number of restarts within each inner GMRES.
    pub fn with_inner_max_restarts(mut self, inner_max_restarts: usize) -> Self {
        self.inner_max_restarts = inner_max_restarts;
        self
    }

    /// Set stagnation detection threshold.
    pub fn with_min_reduction(mut self, min_reduction: f64) -> Self {
        self.min_reduction = Some(min_reduction);
        self
    }

    /// Set inner GMRES relative tolerance.
    pub fn with_inner_rtol(mut self, inner_rtol: f64) -> Self {
        self.inner_rtol = Some(inner_rtol);
        self
    }

    /// Enable verbose output.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// Result of restarted GMRES solver.
#[derive(Debug, Clone)]
pub struct RestartGmresResult<T> {
    /// The solution vector.
    pub solution: T,

    /// Total number of inner GMRES iterations performed.
    pub iterations: usize,

    /// Number of outer restart iterations performed.
    pub outer_iterations: usize,

    /// Final relative residual norm (true residual).
    pub residual_norm: f64,

    /// Whether the solver converged.
    pub converged: bool,
}

/// Solve `A x = b` using restarted GMRES with truncation.
///
/// This wraps [`gmres_with_truncation`] with an outer loop that recomputes the true residual
/// at each restart. This is particularly useful for MPS/MPO computations where truncation
/// can cause the inner GMRES residual to be inaccurate.
///
/// # Algorithm
///
/// ```text
/// for outer_iter in 0..max_outer_iters:
///     r = b - A*x0          // Compute true residual
///     r = truncate(r)
///     if ||r|| / ||b|| < rtol:
///         return x0         // Converged
///     x' = gmres_with_truncation(A, r, 0, inner_options, truncate)
///     x0 = truncate(x0 + x')
/// ```
///
/// # Type Parameters
///
/// * `T` - A tensor type implementing `TensorLike`
/// * `F` - A function that applies the linear operator: `F(x) = A x`
/// * `Tr` - A function that truncates a tensor in-place: `Tr(&mut x)`
///
/// # Arguments
///
/// * `apply_a` - Function that applies the linear operator A to a tensor
/// * `b` - Right-hand side tensor
/// * `x0` - Initial guess (if None, starts from zero)
/// * `options` - Solver options
/// * `truncate` - Function that truncates a tensor to control bond dimension
///
/// # Returns
///
/// A `RestartGmresResult` containing the solution and convergence information.
pub fn restart_gmres_with_truncation<T, F, Tr>(
    apply_a: F,
    b: &T,
    x0: Option<&T>,
    options: &RestartGmresOptions,
    truncate: Tr,
) -> Result<RestartGmresResult<T>>
where
    T: TensorLike,
    F: Fn(&T) -> Result<T>,
    Tr: Fn(&mut T) -> Result<()>,
{
    // Validate structural consistency of inputs
    b.validate()?;
    if let Some(x) = x0 {
        x.validate()?;
    }

    let b_norm = b.norm();
    if b_norm < 1e-15 {
        // b is effectively zero, return x0 or zero
        let solution = match x0 {
            Some(x) => x.clone(),
            None => b.scale(AnyScalar::F64(0.0))?,
        };
        return Ok(RestartGmresResult {
            solution,
            iterations: 0,
            outer_iterations: 0,
            residual_norm: 0.0,
            converged: true,
        });
    }

    // Initialize x: use x0 if provided, otherwise start from zero
    let mut x = match x0 {
        Some(x) => x.clone(),
        None => b.scale(AnyScalar::F64(0.0))?,
    };

    let mut total_inner_iters = 0;
    let mut prev_residual_norm = f64::INFINITY;

    // Inner GMRES options
    let inner_options = GmresOptions {
        max_iter: options.inner_max_iter,
        rtol: options.inner_rtol.unwrap_or(0.1), // Solve loosely by default
        max_restarts: options.inner_max_restarts + 1, // +1 because max_restarts=0 means 1 cycle
        verbose: options.verbose,
        check_true_residual: true, // Always check in restart context to avoid false convergence
    };

    for outer_iter in 0..options.max_outer_iters {
        // Compute true residual: r = b - A*x
        let ax = apply_a(&x)?;
        // Validate operator output on first outer iteration
        if outer_iter == 0 {
            ax.validate()?;
        }
        let mut r = b.axpby(AnyScalar::F64(1.0), &ax, AnyScalar::F64(-1.0))?;
        truncate(&mut r)?;

        let r_norm = r.norm();
        let rel_res = r_norm / b_norm;

        if options.verbose {
            eprintln!(
                "Restart GMRES outer iter {}: true residual = {:.6e}",
                outer_iter, rel_res
            );
        }

        // Check convergence
        if rel_res < options.rtol {
            return Ok(RestartGmresResult {
                solution: x,
                iterations: total_inner_iters,
                outer_iterations: outer_iter,
                residual_norm: rel_res,
                converged: true,
            });
        }

        // Check stagnation
        if let Some(min_reduction) = options.min_reduction {
            if outer_iter > 0 && rel_res > prev_residual_norm * min_reduction {
                if options.verbose {
                    eprintln!(
                        "Restart GMRES stagnated: residual ratio = {:.6e} > {:.6e}",
                        rel_res / prev_residual_norm,
                        min_reduction
                    );
                }
                return Ok(RestartGmresResult {
                    solution: x,
                    iterations: total_inner_iters,
                    outer_iterations: outer_iter,
                    residual_norm: rel_res,
                    converged: false,
                });
            }
        }
        prev_residual_norm = rel_res;

        // Solve A*x' = r using inner GMRES with zero initial guess
        // The zero initial guess is created by scaling r by 0
        let zero = r.scale(AnyScalar::F64(0.0))?;
        let inner_result = gmres_with_truncation(&apply_a, &r, &zero, &inner_options, &truncate)?;

        total_inner_iters += inner_result.iterations;

        if options.verbose {
            eprintln!(
                "  Inner GMRES: {} iterations, residual = {:.6e}, converged = {}",
                inner_result.iterations, inner_result.residual_norm, inner_result.converged
            );
        }

        // Update solution: x = x + x'
        x = x.axpby(
            AnyScalar::F64(1.0),
            &inner_result.solution,
            AnyScalar::F64(1.0),
        )?;
        truncate(&mut x)?;
    }

    // Did not converge within max_outer_iters
    // Compute final residual
    let ax = apply_a(&x)?;
    let mut r = b.axpby(AnyScalar::F64(1.0), &ax, AnyScalar::F64(-1.0))?;
    truncate(&mut r)?;
    let final_rel_res = r.norm() / b_norm;

    Ok(RestartGmresResult {
        solution: x,
        iterations: total_inner_iters,
        outer_iterations: options.max_outer_iters,
        residual_norm: final_rel_res,
        converged: false,
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

/// Update solution with truncation: x_new = truncate(x + sum_i y_i * v_i)
fn update_solution_truncated<T, Tr>(
    x: &T,
    v_basis: &[T],
    y: &[AnyScalar],
    truncate: &Tr,
) -> Result<T>
where
    T: TensorLike,
    Tr: Fn(&mut T) -> Result<()>,
{
    let mut result = x.clone();

    for (vi, yi) in v_basis.iter().zip(y.iter()) {
        let scaled_vi = vi.scale(yi.clone())?;
        result = result.axpby(AnyScalar::F64(1.0), &scaled_vi, AnyScalar::F64(1.0))?;
        // Truncate after each addition to control bond dimension growth
        truncate(&mut result)?;
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
            check_true_residual: false,
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
            check_true_residual: false,
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
            check_true_residual: false,
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
            check_true_residual: false,
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
            check_true_residual: false,
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
            check_true_residual: false,
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

    // ==========================================================================
    // Tests for restart_gmres_with_truncation
    // ==========================================================================

    #[test]
    fn test_restart_gmres_identity_operator() {
        // Solve A x = b where A = I (identity)
        let idx = DynIndex::new_dyn(3);
        let b = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

        let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };

        // No-op truncation
        let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

        let options = RestartGmresOptions::default();

        let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();

        assert!(
            result.converged,
            "Restart GMRES should converge for identity"
        );
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
    fn test_restart_gmres_diagonal_matrix() {
        // Solve A x = b where A = diag(2, 3, 4)
        // b = [2, 6, 12] → x = [1, 2, 3]
        let idx = DynIndex::new_dyn(3);
        let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
        let expected_x = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

        let diag = [2.0, 3.0, 4.0];
        let apply_a = move |x: &TensorDynLen| -> Result<TensorDynLen> {
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

        // No-op truncation
        let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

        let options = RestartGmresOptions {
            max_outer_iters: 10,
            rtol: 1e-10,
            inner_max_iter: 5,
            inner_max_restarts: 0,
            min_reduction: None,
            inner_rtol: None,
            verbose: false,
        };

        let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();

        assert!(result.converged, "Restart GMRES should converge");
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
    fn test_restart_gmres_with_initial_guess() {
        // Solve A x = b with a good initial guess
        let idx = DynIndex::new_dyn(3);
        let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
        let x0 = make_vector_with_index(vec![0.9, 1.9, 2.9], &idx); // Close to [1, 2, 3]
        let expected_x = make_vector_with_index(vec![1.0, 2.0, 3.0], &idx);

        let diag = [2.0, 3.0, 4.0];
        let apply_a = move |x: &TensorDynLen| -> Result<TensorDynLen> {
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

        let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

        let options = RestartGmresOptions::default();

        let result =
            restart_gmres_with_truncation(apply_a, &b, Some(&x0), &options, truncate).unwrap();

        assert!(result.converged, "Should converge with good initial guess");

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
    fn test_restart_gmres_outer_iterations_tracked() {
        // Verify that outer_iterations is tracked correctly
        let idx = DynIndex::new_dyn(3);
        let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);

        let diag = [2.0, 3.0, 4.0];
        let apply_a = move |x: &TensorDynLen| -> Result<TensorDynLen> {
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

        let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

        // Use small inner_max_iter to encourage multiple outer iterations
        let options = RestartGmresOptions {
            max_outer_iters: 20,
            rtol: 1e-10,
            inner_max_iter: 2,
            inner_max_restarts: 0,
            min_reduction: None,
            inner_rtol: Some(0.1),
            verbose: false,
        };

        let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();

        assert!(result.converged, "Should converge");
        // Verify iteration counts are reasonable
        assert!(
            result.iterations >= 1,
            "Should have at least 1 inner iteration"
        );
        // outer_iterations can be 0 if converged at first check
        assert!(
            result.outer_iterations <= options.max_outer_iters,
            "outer_iterations should not exceed max_outer_iters"
        );
    }

    #[test]
    fn test_restart_gmres_zero_rhs() {
        // Solve A x = 0 → x = 0
        let idx = DynIndex::new_dyn(3);
        let b = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);

        let apply_a = |x: &TensorDynLen| -> Result<TensorDynLen> { Ok(x.clone()) };
        let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

        let options = RestartGmresOptions::default();

        let result = restart_gmres_with_truncation(apply_a, &b, None, &options, truncate).unwrap();

        assert!(result.converged, "Should converge for zero RHS");
        assert_eq!(result.iterations, 0, "Should converge immediately");
        assert_eq!(result.outer_iterations, 0, "Should have 0 outer iterations");
    }

    #[test]
    fn test_restart_gmres_options_builder() {
        let options = RestartGmresOptions::new()
            .with_max_outer_iters(30)
            .with_rtol(1e-12)
            .with_inner_max_iter(15)
            .with_inner_max_restarts(1)
            .with_min_reduction(0.95)
            .with_inner_rtol(0.01)
            .with_verbose(true);

        assert_eq!(options.max_outer_iters, 30);
        assert!((options.rtol - 1e-12).abs() < 1e-15);
        assert_eq!(options.inner_max_iter, 15);
        assert_eq!(options.inner_max_restarts, 1);
        assert_eq!(options.min_reduction, Some(0.95));
        assert_eq!(options.inner_rtol, Some(0.01));
        assert!(options.verbose);
    }

    #[test]
    fn test_gmres_with_truncation_check_true_residual_safe() {
        // When check_true_residual is enabled and convergence is reported,
        // the residual_norm should reflect the checked residual (not the
        // potentially inaccurate Hessenberg estimate).
        let idx = DynIndex::new_dyn(3);
        let b = make_vector_with_index(vec![2.0, 6.0, 12.0], &idx);
        let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0], &idx);

        let diag = [2.0, 3.0, 4.0];
        let apply_a = move |x: &TensorDynLen| -> Result<TensorDynLen> {
            let x_data = match x.storage().as_ref() {
                Storage::DenseF64(d) => d.as_slice().to_vec(),
                _ => anyhow::bail!("Expected DenseF64"),
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

        // No-op truncation: convergence should work normally with check enabled
        let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

        let options = GmresOptions {
            max_iter: 10,
            rtol: 1e-10,
            max_restarts: 3,
            verbose: false,
            check_true_residual: true,
        };

        let result = gmres_with_truncation(apply_a, &b, &x0, &options, truncate).unwrap();

        assert!(
            result.converged,
            "Should converge with true residual check and no-op truncation"
        );

        // Verify the reported residual is actually the checked residual
        let ax = apply_a(&result.solution).unwrap();
        let r = ax
            .axpby(AnyScalar::F64(1.0), &b, AnyScalar::F64(-1.0))
            .unwrap();
        let true_rel_res = r.norm() / b.norm();
        assert!(
            true_rel_res < 1e-8,
            "True residual should be small: {}",
            true_rel_res
        );
    }

    #[test]
    #[allow(clippy::needless_borrows_for_generic_args)]
    fn test_gmres_with_truncation_check_true_residual_consistency() {
        // Test that when check_true_residual is enabled, the reported residual_norm
        // is consistent with the actual checked (truncated) residual.
        let idx = DynIndex::new_dyn(4);
        let b = make_vector_with_index(vec![1.0, 2.0, 3.0, 4.0], &idx);
        let x0 = make_vector_with_index(vec![0.0, 0.0, 0.0, 0.0], &idx);

        // A = diag(1, 2, 3, 4)
        let diag = [1.0, 2.0, 3.0, 4.0];
        let apply_a = move |x: &TensorDynLen| -> Result<TensorDynLen> {
            let x_data = match x.storage().as_ref() {
                Storage::DenseF64(d) => d.as_slice().to_vec(),
                _ => anyhow::bail!("Expected DenseF64"),
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

        // No-op truncation: the solver should converge normally
        let truncate = |_x: &mut TensorDynLen| -> Result<()> { Ok(()) };

        // Without check: converged with Hessenberg residual
        let options_no_check = GmresOptions {
            max_iter: 10,
            rtol: 1e-10,
            max_restarts: 3,
            verbose: false,
            check_true_residual: false,
        };
        let result_no_check =
            gmres_with_truncation(&apply_a, &b, &x0, &options_no_check, &truncate).unwrap();

        // With check: converged with verified residual
        let options_check = GmresOptions {
            max_iter: 10,
            rtol: 1e-10,
            max_restarts: 3,
            verbose: false,
            check_true_residual: true,
        };
        let result_check =
            gmres_with_truncation(&apply_a, &b, &x0, &options_check, &truncate).unwrap();

        // Both should converge for this simple problem
        assert!(result_no_check.converged, "No-check should converge");
        assert!(result_check.converged, "With-check should converge");

        // With check, the reported residual should be the true residual
        let ax = apply_a(&result_check.solution).unwrap();
        let r = ax
            .axpby(AnyScalar::F64(1.0), &b, AnyScalar::F64(-1.0))
            .unwrap();
        let true_rel_res = r.norm() / b.norm();

        // The reported residual should match the true residual closely
        assert!(
            (result_check.residual_norm - true_rel_res).abs() < 1e-8,
            "Reported residual ({:.6e}) should match true residual ({:.6e})",
            result_check.residual_norm,
            true_rel_res
        );
    }
}
