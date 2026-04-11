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
//! ```
//! use tensor4all_core::{
//!     krylov::{gmres, GmresOptions},
//!     DynIndex, TensorDynLen, TensorLike,
//! };
//!
//! # fn main() -> anyhow::Result<()> {
//! let i = DynIndex::new_dyn(2);
//! let rhs = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, -1.0])?;
//! let initial_guess = TensorDynLen::from_dense(vec![i.clone()], vec![0.0, 0.0])?;
//!
//! let apply_operator = |x: &TensorDynLen| Ok(x.clone());
//! let result = gmres(apply_operator, &rhs, &initial_guess, &GmresOptions::default())?;
//!
//! assert!(result.converged);
//! assert!(result.solution.sub(&rhs)?.maxabs() < 1e-12);
//! # Ok(())
//! # }
//! ```

use crate::any_scalar::AnyScalar;
use crate::TensorLike;
use anyhow::Result;

/// Options for GMRES solver.
///
/// # Examples
///
/// ```
/// use tensor4all_core::krylov::GmresOptions;
///
/// let opts = GmresOptions {
///     max_iter: 50,
///     rtol: 1e-8,
///     max_restarts: 5,
///     verbose: false,
///     check_true_residual: true,
/// };
/// assert_eq!(opts.max_iter, 50);
/// assert_eq!(opts.rtol, 1e-8);
/// ```
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
///
/// Contains the solution, iteration count, final residual norm, and
/// convergence status.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
/// use tensor4all_core::krylov::{gmres, GmresOptions};
///
/// let i = DynIndex::new_dyn(2);
/// let b = TensorDynLen::from_dense(vec![i.clone()], vec![3.0, 7.0]).unwrap();
/// let x0 = TensorDynLen::from_dense(vec![i.clone()], vec![0.0, 0.0]).unwrap();
///
/// let result = gmres(|x: &TensorDynLen| Ok(x.clone()), &b, &x0, &GmresOptions::default()).unwrap();
/// assert!(result.converged);
/// assert!(result.residual_norm < 1e-10);
/// ```
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
        let r = b.axpby(AnyScalar::new_real(1.0), &ax, AnyScalar::new_real(-1.0))?;
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
        let v0 = r.scale(AnyScalar::new_real(1.0 / r_norm))?;
        v_basis.push(v0);

        // Initialize Givens rotation storage
        let mut cs: Vec<AnyScalar> = Vec::with_capacity(options.max_iter);
        let mut sn: Vec<AnyScalar> = Vec::with_capacity(options.max_iter);
        let mut g: Vec<AnyScalar> = vec![AnyScalar::new_real(r_norm)]; // residual in upper Hessenberg space

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
                let neg_h_ij = AnyScalar::new_real(0.0) - h_ij;
                w_orth = w_orth.axpby(AnyScalar::new_real(1.0), v_i, neg_h_ij)?;
            }

            let h_jp1_j_real = w_orth.norm();
            let h_jp1_j = AnyScalar::new_real(h_jp1_j_real);
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
            h_col[j + 1] = AnyScalar::new_real(0.0);

            // Apply rotation to g
            let g_j = g[j].clone();
            let g_jp1 = AnyScalar::new_real(0.0);
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
                let v_jp1 = w_orth.scale(AnyScalar::new_real(1.0 / h_jp1_j_real))?;
                v_basis.push(v_jp1);
            } else {
                // Lucky breakdown - we've found the exact solution in the Krylov subspace
                let y = solve_upper_triangular(&h_matrix, &g[..=j])?;
                x = update_solution(&x, &v_basis[..=j], &y)?;
                let ax_final = apply_a(&x)?;
                let r_final = b.axpby(
                    AnyScalar::new_real(1.0),
                    &ax_final,
                    AnyScalar::new_real(-1.0),
                )?;
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
    let r_final = b.axpby(
        AnyScalar::new_real(1.0),
        &ax_final,
        AnyScalar::new_real(-1.0),
    )?;
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
        let mut r = b.axpby(AnyScalar::new_real(1.0), &ax, AnyScalar::new_real(-1.0))?;
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

        let mut v0 = r.scale(AnyScalar::new_real(1.0 / r_norm))?;
        truncate(&mut v0)?;
        // After truncation, v0 might not be unit norm and might point in a different direction.
        // We need to:
        // 1. Renormalize v0 to unit norm for numerical stability
        // 2. Recompute g[0] = <r, v0> to maintain the correct relationship
        let v0_norm = v0.norm();
        let effective_g0 = if v0_norm > 1e-15 {
            v0 = v0.scale(AnyScalar::new_real(1.0 / v0_norm))?;
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
        let mut g: Vec<AnyScalar> = vec![AnyScalar::new_real(effective_g0)];
        let mut solution_already_updated = false;

        for j in 0..options.max_iter {
            total_iters += 1;

            let w = apply_a(&v_basis[j])?;

            let mut h_col: Vec<AnyScalar> = Vec::with_capacity(j + 2);
            let mut w_orth = w;

            for v_i in v_basis.iter().take(j + 1) {
                let h_ij = v_i.inner_product(&w_orth)?;
                h_col.push(h_ij.clone());
                let neg_h_ij = AnyScalar::new_real(0.0) - h_ij;
                w_orth = w_orth.axpby(AnyScalar::new_real(1.0), v_i, neg_h_ij)?;
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
                        let neg_correction = AnyScalar::new_real(0.0) - correction.clone();
                        w_orth = w_orth.axpby(AnyScalar::new_real(1.0), v_i, neg_correction)?;
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
            let h_jp1_j = AnyScalar::new_real(h_jp1_j_real);
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
            h_col[j + 1] = AnyScalar::new_real(0.0);

            let g_j = g[j].clone();
            let g_jp1 = AnyScalar::new_real(0.0);
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
                    let mut r_check = b.axpby(
                        AnyScalar::new_real(1.0),
                        &ax_check,
                        AnyScalar::new_real(-1.0),
                    )?;
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
                let v_jp1 = w_orth.scale(AnyScalar::new_real(1.0 / h_jp1_j_real))?;
                // v_jp1 should have norm ~1.0 by construction
                // The Arnoldi relation h_{j+1,j} * v_{j+1} = w_orth is maintained exactly
                v_basis.push(v_jp1);
            } else {
                let y = solve_upper_triangular(&h_matrix, &g[..=j])?;
                x = update_solution_truncated(&x, &v_basis[..=j], &y, &truncate)?;
                let ax_final = apply_a(&x)?;
                let r_final = b.axpby(
                    AnyScalar::new_real(1.0),
                    &ax_final,
                    AnyScalar::new_real(-1.0),
                )?;
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
    let r_final = b.axpby(
        AnyScalar::new_real(1.0),
        &ax_final,
        AnyScalar::new_real(-1.0),
    )?;
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
            None => b.scale(AnyScalar::new_real(0.0))?,
        };
        return Ok(RestartGmresResult {
            solution,
            iterations: 0,
            outer_iterations: 0,
            residual_norm: 0.0,
            converged: true,
        });
    }

    // Initialize x: use x0 if provided, otherwise start from zero.
    // Track whether x is zero to avoid unnecessary bond dimension doubling
    // when adding the first correction via axpby.
    let mut x_is_zero = x0.is_none();
    let mut x = match x0 {
        Some(x) => x.clone(),
        None => b.scale(AnyScalar::new_real(0.0))?,
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
        let mut r = b.axpby(AnyScalar::new_real(1.0), &ax, AnyScalar::new_real(-1.0))?;
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
        let zero = r.scale(AnyScalar::new_real(0.0))?;
        let inner_result = gmres_with_truncation(&apply_a, &r, &zero, &inner_options, &truncate)?;

        total_inner_iters += inner_result.iterations;

        if options.verbose {
            eprintln!(
                "  Inner GMRES: {} iterations, residual = {:.6e}, converged = {}",
                inner_result.iterations, inner_result.residual_norm, inner_result.converged
            );
        }

        // Update solution: x = x + x'
        // When x is zero (first iteration with no initial guess), use x' directly
        // to avoid bond dimension doubling from axpby with a zero tensor.
        if x_is_zero {
            x = inner_result.solution;
            x_is_zero = false;
        } else {
            x = x.axpby(
                AnyScalar::new_real(1.0),
                &inner_result.solution,
                AnyScalar::new_real(1.0),
            )?;
        }
        truncate(&mut x)?;
    }

    // Did not converge within max_outer_iters
    // Compute final residual
    let ax = apply_a(&x)?;
    let mut r = b.axpby(AnyScalar::new_real(1.0), &ax, AnyScalar::new_real(-1.0))?;
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
/// This function keeps computation in `AnyScalar` space to preserve AD metadata
/// as much as possible.
fn compute_givens_rotation(a: &AnyScalar, b: &AnyScalar) -> (AnyScalar, AnyScalar) {
    // r^2 = conj(a)*a + conj(b)*b (works for both real and complex)
    let norm2 = a.clone().conj() * a.clone() + b.clone().conj() * b.clone();
    let r = norm2.sqrt();
    if r.abs() < 1e-15 {
        (AnyScalar::new_real(1.0), AnyScalar::new_real(0.0))
    } else {
        (a.clone() / r.clone(), b.clone() / r)
    }
}

/// Apply Givens rotation: (c, s) @ (x, y) -> (c*x + s*y, -conj(s)*x + c*y) for complex
/// or (c*x + s*y, -s*x + c*y) for real.
///
/// This function keeps computation in `AnyScalar` space to preserve AD metadata
/// as much as possible.
fn apply_givens_rotation(
    c: &AnyScalar,
    s: &AnyScalar,
    x: &AnyScalar,
    y: &AnyScalar,
) -> (AnyScalar, AnyScalar) {
    let new_x = c.clone() * x.clone() + s.clone() * y.clone();
    let new_y = -(s.clone().conj() * x.clone()) + c.clone() * y.clone();
    (new_x, new_y)
}

/// Solve upper triangular system R y = g using back substitution.
fn solve_upper_triangular(h: &[Vec<AnyScalar>], g: &[AnyScalar]) -> Result<Vec<AnyScalar>> {
    let n = g.len();
    if n == 0 {
        return Ok(vec![]);
    }

    let mut y = vec![AnyScalar::new_real(0.0); n];

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
        result = result.axpby(
            AnyScalar::new_real(1.0),
            &scaled_vi,
            AnyScalar::new_real(1.0),
        )?;
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
    // Detect if x is effectively zero.
    // When x is created via scale(0.0), it preserves the original bond structure
    // (e.g., bond dim 4), causing axpby to double bond dimensions unnecessarily.
    // By detecting zero, we can use scaled_vi directly, avoiding the doubling.
    let mut result_is_zero = x.norm() == 0.0;

    for (vi, yi) in v_basis.iter().zip(y.iter()) {
        let scaled_vi = vi.scale(yi.clone())?;
        if result_is_zero {
            result = scaled_vi;
            result_is_zero = false;
        } else {
            result = result.axpby(
                AnyScalar::new_real(1.0),
                &scaled_vi,
                AnyScalar::new_real(1.0),
            )?;
        }
        // Truncate after each addition to control bond dimension growth
        truncate(&mut result)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests;
