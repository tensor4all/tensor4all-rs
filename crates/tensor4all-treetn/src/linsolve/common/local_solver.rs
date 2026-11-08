//! Selection and dispatch for projected local linear equations.

use anyhow::Result;
use tensor4all_core::krylov::{
    bicgstab, cg, gmres_affine, gmres_affine_with_absolute_tolerance, BiCgStabOptions, CgOptions,
    GmresOptions,
};
use tensor4all_core::{AnyScalar, TensorVectorSpace};

use super::options::{DEFAULT_GMRES_MAX_RESTARTS, DEFAULT_GMRES_RESTART_DIM, DEFAULT_GMRES_TOL};
use super::{GmresToleranceMode, LinsolveOptions};
use crate::error::TreeTNOperationError;

/// Local Krylov solver used by one-site ALS and two-site MALS sweeps.
///
/// GMRES is the default and uses the sweep options' `gmres_*` controls. CG and
/// BiCGSTAB use their native option types, including absolute/relative tolerances
/// and iteration budgets; with them the `gmres_*` controls must keep their
/// defaults. No automatic selection or fallback is performed.
///
/// # Examples
/// ```
/// use tensor4all_core::krylov::CgOptions;
/// use tensor4all_treetn::{LinsolveOptions, LocalSolver};
/// let options = LinsolveOptions::new(3)
///     .with_local_solver(LocalSolver::Cg(CgOptions { max_iter: 20, ..Default::default() }));
/// assert!(matches!(options.local_solver, LocalSolver::Cg(ref cg) if cg.max_iter == 20));
/// assert!(matches!(LocalSolver::default(), LocalSolver::Gmres));
/// ```
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub enum LocalSolver {
    /// Restarted GMRES for general real/complex systems (default).
    #[default]
    Gmres,
    /// Conjugate gradients; the effective operator `a0 I + a1 A` must be
    /// Hermitian positive-definite. Start with [`CgOptions::default`].
    ///
    /// The core CG checks only the directions it visits and does not certify
    /// the whole operator. Its Hermitian-consistency check measures the
    /// imaginary part of the directional curvature, which is zero for real
    /// scalars, so a real non-symmetric operator is not detected as such. CG
    /// then returns a
    /// [`KrylovError::NonPositiveCurvature`](tensor4all_core::krylov::KrylovError::NonPositiveCurvature)
    /// error when a visited curvature is not positive; otherwise it may still
    /// converge (for example when the right-hand side spans an invariant
    /// subspace) or it exhausts its budget, which the sweep reports through
    /// its true residual rather than as an error.
    Cg(CgOptions),
    /// BiCGSTAB for general real/complex systems using a fixed number of vectors.
    /// Start with [`BiCgStabOptions::default`]. Breakdown is returned as an error.
    BiCgStab(BiCgStabOptions),
}

impl LocalSolver {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Gmres => "gmres",
            Self::Cg(_) => "cg",
            Self::BiCgStab(_) => "bicgstab",
        }
    }
}

impl LinsolveOptions {
    /// Check the affine coefficients and the selected local solver's controls.
    ///
    /// [`crate::square_linsolve`] runs this check before any sweep or shortcut;
    /// call it directly to validate options without building a problem. Invalid
    /// options are always reported as errors; no value is clamped or replaced.
    ///
    /// # Errors
    /// Returns [`TreeTNOperationError`] when
    /// - `a0` and `a1` are both zero;
    /// - GMRES is selected and `gmres_tol` is not finite and positive (GMRES
    ///   stops when the residual is strictly below `gmres_tol`, so zero can never
    ///   be met), `gmres_restart_dim` is zero or `usize::MAX`, or
    ///   `gmres_max_restarts` is zero;
    /// - CG or BiCGSTAB is selected and a `gmres_*` control (including
    ///   `gmres_tolerance_mode`) differs from its default;
    /// - the native CG/BiCGSTAB options are invalid (a negative or non-finite
    ///   tolerance); the error source chain then contains
    ///   [`tensor4all_core::krylov::KrylovError::InvalidOptions`].
    ///
    /// Conditions that depend on the problem, such as a `gmres_restart_dim` whose
    /// Krylov workspace cannot be allocated or an RHS-dependent CG/BiCGSTAB
    /// threshold that overflows, are detected by the local solve and reported by
    /// [`crate::square_linsolve`].
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::krylov::CgOptions;
    /// use tensor4all_treetn::{LinsolveOptions, LocalSolver};
    /// let cg = LinsolveOptions::new(2).with_local_solver(LocalSolver::Cg(CgOptions::default()));
    /// assert!(cg.validate_local_solver().is_ok());
    /// // GMRES controls do not apply to CG, so changing one is an error.
    /// assert!(cg.clone().with_gmres_tol(1e-12).validate_local_solver().is_err());
    /// assert!(LinsolveOptions::new(2).with_gmres_restart_dim(0).validate_local_solver().is_err());
    /// ```
    pub fn validate_local_solver(&self) -> std::result::Result<(), TreeTNOperationError> {
        Ok(validate_local_options(self)?)
    }
}

pub(crate) struct LocalSolveResult<T> {
    pub(crate) solution: T,
    pub(crate) iterations: usize,
    pub(crate) residual_norm: f64,
    pub(crate) residual_is_relative: bool,
    pub(crate) converged: bool,
}

pub(crate) fn validate_local_options(options: &LinsolveOptions) -> Result<()> {
    // CG/BiCGSTAB would otherwise see a zero operator and report a curvature
    // or breakdown error, or "solve" 0 x = 0 when the RHS is zero.
    anyhow::ensure!(
        !(options.a0.is_zero() && options.a1.is_zero()),
        "LinsolveOptions: a0 and a1 are both zero; at least one affine coefficient must be nonzero"
    );
    match &options.local_solver {
        LocalSolver::Gmres => {
            local_gmres_options(options)?;
        }
        LocalSolver::Cg(native) => {
            ensure_default_gmres_controls(options)?;
            native.validate()?;
        }
        LocalSolver::BiCgStab(native) => {
            ensure_default_gmres_controls(options)?;
            native.validate()?;
        }
    }
    Ok(())
}

fn ensure_default_gmres_controls(options: &LinsolveOptions) -> Result<()> {
    let solver = options.local_solver.name();
    // A NaN `gmres_tol` compares unequal to the default and is rejected too.
    for (name, is_default) in [
        ("gmres_tol", options.gmres_tol == DEFAULT_GMRES_TOL),
        (
            "gmres_tolerance_mode",
            options.gmres_tolerance_mode == GmresToleranceMode::Relative,
        ),
        (
            "gmres_max_restarts",
            options.gmres_max_restarts == DEFAULT_GMRES_MAX_RESTARTS,
        ),
        (
            "gmres_restart_dim",
            options.gmres_restart_dim == DEFAULT_GMRES_RESTART_DIM,
        ),
    ] {
        anyhow::ensure!(
            is_default,
            "LinsolveOptions::{name} configures LocalSolver::Gmres only, but the selected local \
             solver is {solver}; set the {solver} options instead or keep {name} at its default"
        );
    }
    Ok(())
}

fn local_gmres_options(options: &LinsolveOptions) -> Result<GmresOptions> {
    anyhow::ensure!(
        options.gmres_tol.is_finite() && options.gmres_tol > 0.0,
        "LinsolveOptions::gmres_tol must be finite and greater than zero"
    );
    anyhow::ensure!(
        options.gmres_restart_dim > 0,
        "LinsolveOptions::gmres_restart_dim must be greater than zero"
    );
    anyhow::ensure!(
        options.gmres_restart_dim.checked_add(1).is_some(),
        "LinsolveOptions::gmres_restart_dim is too large"
    );
    anyhow::ensure!(
        options.gmres_max_restarts > 0,
        "LinsolveOptions::gmres_max_restarts must be greater than zero"
    );
    Ok(GmresOptions {
        max_iter: options.gmres_restart_dim,
        rtol: options.gmres_tol,
        max_restarts: options.gmres_max_restarts,
        verbose: false,
        check_true_residual: true,
    })
}

fn is_exactly_one(value: &AnyScalar) -> bool {
    value.real() == 1.0 && value.imag() == 0.0
}

pub(crate) fn solve_local_system<T, F>(
    options: &LinsolveOptions,
    apply_a: F,
    b: &T,
    init: &T,
    verbose_gmres: bool,
) -> Result<LocalSolveResult<T>>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    validate_local_options(options)?;
    let apply_affine = |x: &T| -> Result<T> {
        if options.a1.is_zero() {
            return Ok(x.scale(options.a0.clone())?);
        }
        let ax = apply_a(x)?;
        if options.a0.is_zero() {
            // `a1 = 1`, real or complex, needs no scaling copy of `A x`.
            if is_exactly_one(&options.a1) {
                Ok(ax)
            } else {
                Ok(ax.scale(options.a1.clone())?)
            }
        } else {
            Ok(x.axpby(options.a0.clone(), &ax, options.a1.clone())?)
        }
    };
    match &options.local_solver {
        LocalSolver::Gmres => {
            let mut native = local_gmres_options(options)?;
            native.verbose = verbose_gmres;
            let result = match options.gmres_tolerance_mode {
                GmresToleranceMode::Relative => gmres_affine(
                    &apply_a,
                    b,
                    init,
                    options.a0.clone(),
                    options.a1.clone(),
                    &native,
                )?,
                GmresToleranceMode::Absolute => gmres_affine_with_absolute_tolerance(
                    &apply_a,
                    b,
                    init,
                    options.a0.clone(),
                    options.a1.clone(),
                    &native,
                    options.gmres_tol,
                )?,
            };
            Ok(LocalSolveResult {
                solution: result.solution,
                iterations: result.iterations,
                residual_norm: result.residual_norm,
                converged: result.converged,
                residual_is_relative: options.gmres_tolerance_mode == GmresToleranceMode::Relative,
            })
        }
        LocalSolver::Cg(native) => {
            let result = cg(apply_affine, b, init, native)?;
            Ok(LocalSolveResult {
                solution: result.solution,
                iterations: result.iterations,
                residual_norm: result.residual_norm,
                converged: result.converged,
                residual_is_relative: false,
            })
        }
        LocalSolver::BiCgStab(native) => {
            let result = bicgstab(apply_affine, b, init, native)?;
            Ok(LocalSolveResult {
                solution: result.solution,
                iterations: result.iterations,
                residual_norm: result.residual_norm,
                converged: result.converged,
                residual_is_relative: false,
            })
        }
    }
}

#[cfg(test)]
mod tests;
