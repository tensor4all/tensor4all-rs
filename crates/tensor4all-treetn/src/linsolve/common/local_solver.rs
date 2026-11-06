//! Selection and dispatch for projected local linear equations.

use anyhow::Result;
use tensor4all_core::krylov::{
    bicgstab, cg, gmres_affine, gmres_affine_with_absolute_tolerance, BiCgStabOptions, CgOptions,
    GmresOptions,
};
use tensor4all_core::TensorVectorSpace;

use super::{GmresToleranceMode, LinsolveOptions};

/// Local Krylov solver used by one-site ALS and two-site MALS sweeps.
///
/// GMRES is the default and uses the sweep options' `gmres_*` controls. CG and
/// BiCGSTAB use their native option types, including absolute/relative tolerances
/// and iteration budgets. No automatic selection or fallback is performed.
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
pub enum LocalSolver {
    /// Restarted GMRES for general real/complex systems (default).
    #[default]
    Gmres,
    /// Conjugate gradients; the effective operator `a0 I + a1 A` must be
    /// Hermitian positive-definite. Core directional checks are retained, but
    /// do not certify the full operator. Start with [`CgOptions::default`].
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
        LocalSolver::Cg(native) => native.validate()?,
        LocalSolver::BiCgStab(native) => native.validate()?,
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
            Ok(ax.scale(options.a1.clone())?)
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
