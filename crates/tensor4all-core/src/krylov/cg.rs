//! Matrix-free conjugate gradients for Hermitian positive-definite operators.
//!
//! Derived from KrylovKit.jl's `src/linsolve/cg.jl` at revision
//! d51fa25adfa4cda5530d8e3d42eedfb288cca447, including residual replacement
//! and recurrence restart after an unsuccessful true-residual check.
//! <https://github.com/Jutho/KrylovKit.jl/blob/d51fa25adfa4cda5530d8e3d42eedfb288cca447/src/linsolve/cg.jl>
//!
//! Copyright (c) 2017: Jutho. MIT license; see ../../LICENSE-KrylovKit.md.

use super::KrylovError;
use crate::{AnyScalar, TensorVectorSpace};
use anyhow::Result;

/// Convergence and iteration controls for [`cg`].
/// Unlike [`super::GmresOptions`], CG needs no restart dimension: it keeps a
/// fixed number of vectors. Start with the defaults for double precision;
/// use looser tolerances (for example `rtol = hermitian_tol = 1e-5`) for
/// single precision.
///
/// # Examples
/// ```
/// use tensor4all_core::{krylov::{cg, CgOptions}, DynIndex, IdxTensor};
/// # fn main() -> anyhow::Result<()> {
/// let b = IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![1.0_f64, 2.0])?;
/// let options = CgOptions { max_iter: 2, ..CgOptions::default() };
/// let x0 = IdxTensor::zeros::<f64>(b.external_indices())?;
/// # use tensor4all_core::TensorIndex;
/// let result = cg(|x: &IdxTensor| Ok(x.clone()), &b, &x0, &options)?;
/// assert_eq!(result.solution.to_vec::<f64>()?, vec![1.0, 2.0]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct CgOptions {
    /// Maximum completed CG steps; zero only evaluates the initial residual.
    /// Default: `100`. Increase this when the true residual has not converged.
    pub max_iter: usize,
    /// Nonnegative, finite residual tolerance relative to `||b||`.
    /// Default: `1e-10`. The stopping threshold is `max(atol, rtol * ||b||)`.
    pub rtol: f64,
    /// Nonnegative, finite absolute residual tolerance; default: `0.0`.
    /// Set a positive value when a very small RHS needs an absolute error
    /// target. With both tolerances zero, only an exact residual passes.
    pub atol: f64,
    /// Nonnegative, finite allowance for the imaginary part of the directional
    /// Rayleigh quotient `q = p^H A p / p^H p`:
    /// `|Im(q)| <= hermitian_tol * max(1, |q|)`. Default: `1e-10`.
    /// The absolute floor admits cancellation roundoff, matching the projected
    /// Hermitian solvers. This checks visited directions, not the full operator,
    /// and does not relax the positive-definiteness requirement.
    pub hermitian_tol: f64,
}

impl Default for CgOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            rtol: 1e-10,
            atol: 0.0,
            hermitian_tol: 1e-10,
        }
    }
}

/// Solution and true-residual diagnostics returned by [`cg`].
/// As with [`super::HermitianLanczosResult`], `residual_norm` is an absolute
/// norm; [`super::GmresResult`] instead reports a relative norm by default.
///
/// # Examples
/// ```
/// use tensor4all_core::{krylov::{cg, CgOptions}, DynIndex, IdxTensor, TensorIndex};
/// # fn main() -> anyhow::Result<()> {
/// let b = IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![3.0_f64, -1.0])?;
/// let x0 = IdxTensor::zeros::<f64>(b.external_indices())?;
/// let result = cg(|x: &IdxTensor| Ok(x.clone()), &b, &x0, &CgOptions::default())?;
/// assert!(result.converged);
/// assert_eq!(result.solution.to_vec::<f64>()?, vec![3.0, -1.0]);
/// assert_eq!(result.residual_norm, 0.0);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct CgResult<T> {
    /// Approximate solution, including the initial guess's contribution.
    pub solution: T,
    /// Number of completed CG steps, bounded by [`CgOptions::max_iter`].
    pub iterations: usize,
    /// Absolute true residual norm `||b - A*x||`, also on iteration exhaustion.
    pub residual_norm: f64,
    /// Whether the true residual met `max(atol, rtol * ||b||)`.
    pub converged: bool,
}

/// Solve `A x = b` with matrix-free conjugate gradients.
///
/// # Arguments
/// * `apply_a` - Fixed linear operator `x -> A*x`, preserving the vector space.
///   The caller must supply a Hermitian positive-definite operator; checking
///   visited curvatures cannot certify the full matrix-free operator.
/// * `b` - Right-hand side, in the same vector space as `x0`.
/// * `x0` - Initial guess, which may be zero. A zero-norm RHS returns the zero
///   vector, the exact solution, after the initial residual confirms that `x0`
///   and `b` share a vector space.
/// * `options` - Iteration budget and finite, nonnegative tolerances. In doubt,
///   start with [`CgOptions::default`] for double precision.
///
/// # Returns
/// The approximate solution, completed step count, absolute true residual norm,
/// and convergence flag. Iteration exhaustion returns `converged = false`.
/// Working vectors are rescaled by exact powers of two before inner products
/// to keep CG reductions in range.
/// The true residual is recomputed before declaring convergence. If the
/// recursive residual was optimistic, CG restarts from the true residual
/// within the original iteration budget, following KrylovKit.
///
/// Vector and recurrence arithmetic stays in `TensorVectorSpace`/`AnyScalar`
/// operations. Norm, curvature, and stopping decisions are scalar control-flow
/// boundaries; this API does not promise a differentiable solve. No dense
/// operator, preconditioner, or tensor-network truncation is introduced.
///
/// # Errors
/// Returns [`KrylovError::InvalidOptions`] for invalid tolerances or an
/// overflowing stopping threshold, [`KrylovError::NonHermitian`] for a visited
/// curvature with an excessive imaginary part,
/// [`KrylovError::NonPositiveCurvature`] for zero or negative curvature,
/// [`KrylovError::NumericalBreakdown`] for nonfinite or unusable recurrence
/// quantities, and [`KrylovError::Operation`] for incompatible vector spaces,
/// invalid tensor structures, operator errors, or failed tensor operations.
/// Underlying operation errors retain their source chains; an error returned by
/// `apply_a`, including a `KrylovError` from a nested solve, is always wrapped in
/// [`KrylovError::Operation`].
///
/// # Examples
/// ```
/// use tensor4all_core::{krylov::{cg, CgOptions}, DynIndex, IdxTensor, TensorVectorSpace};
/// # fn main() -> anyhow::Result<()> {
/// let i = DynIndex::new_dyn(2);
/// let b = IdxTensor::from_dense(vec![i.clone()], vec![2.0_f64, 4.0])?;
/// let x0 = IdxTensor::zeros::<f64>(vec![i])?;
/// let result = cg(
///     |x: &IdxTensor| Ok(x.scale(2.0.into())?), &b, &x0, &CgOptions::default(),
/// )?;
/// assert!(result.converged);
/// assert_eq!(result.solution.to_vec::<f64>()?, vec![1.0, 2.0]);
/// # Ok(())
/// # }
/// ```
pub fn cg<T, F>(
    apply_a: F,
    b: &T,
    x0: &T,
    options: &CgOptions,
) -> std::result::Result<CgResult<T>, KrylovError>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    cg_impl(apply_a, b, x0, options).map_err(|source| match source.downcast::<KrylovError>() {
        Ok(classified) => classified,
        Err(source) => KrylovError::Operation {
            solver: "cg",
            source,
        },
    })
}

fn cg_impl<T, F>(apply_a: F, b: &T, x0: &T, options: &CgOptions) -> Result<CgResult<T>>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    for (name, tolerance) in [
        ("rtol", options.rtol),
        ("atol", options.atol),
        ("hermitian_tol", options.hermitian_tol),
    ] {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(KrylovError::InvalidOptions {
                solver: "cg",
                reason: format!("{name} must be finite and nonnegative"),
            }
            .into());
        }
    }
    b.validate()?;
    x0.validate()?;
    let b_norm = finite_norm(b, 0, "right-hand-side norm")?;
    let threshold = options.atol.max(options.rtol * b_norm);
    if !threshold.is_finite() {
        return Err(KrylovError::InvalidOptions {
            solver: "cg",
            reason: "rtol * ||b|| must be finite".to_string(),
        }
        .into());
    }
    // A KrylovError from the callback belongs to the callback, not to this
    // solver; wrapping it keeps it from being reported as CG's own failure.
    let apply_a = |x: &T| {
        apply_a(x).map_err(|source| {
            anyhow::Error::new(KrylovError::Operation {
                solver: "cg",
                source,
            })
        })
    };
    let mut x = x0.clone();
    let mut r = true_residual(&apply_a, b, &x)?;
    let mut residual_norm = finite_norm(&r, 0, "initial residual norm")?;
    let mut iterations = 0;
    if b_norm == 0.0 {
        return Ok(CgResult {
            solution: b.scale(AnyScalar::new_real(0.0))?,
            iterations,
            residual_norm: 0.0,
            converged: true,
        });
    }
    if residual_norm <= threshold || options.max_iter == 0 {
        return Ok(CgResult {
            solution: x,
            iterations,
            residual_norm,
            converged: residual_norm <= threshold,
        });
    }
    // INVARIANT: r represents the physical residual divided by 2^residual_exponent.
    // Common power-of-two rescaling preserves the CG recurrence exactly in real
    // arithmetic, while keeping self-inner-products in the scalar dtype's range.
    let mut residual_exponent = normalization_exponent(residual_norm);
    r = scale_pow2(r, -residual_exponent)?;
    let mut p = r.clone();
    let mut rho = residual_inner_product(&r, 0)?;
    for iteration in 1..=options.max_iter {
        let p_norm = finite_norm(&p, iteration, "search direction norm")?;
        if p_norm == 0.0 {
            return Err(KrylovError::NumericalBreakdown {
                solver: "cg",
                iteration,
                quantity: "search direction norm",
            }
            .into());
        }
        let direction_exponent = normalization_exponent(p_norm);
        p = scale_pow2(p, -direction_exponent)?;
        let ap = apply_a(&p)?;
        ap.validate()?;
        let ap_norm = finite_norm(&ap, iteration, "operator image norm")?;
        if ap_norm == 0.0 {
            return Err(KrylovError::NonPositiveCurvature {
                solver: "cg",
                iteration,
                curvature: 0.0,
            }
            .into());
        }
        let image_exponent = normalization_exponent(ap_norm);
        let ap = scale_pow2(ap, -image_exponent)?;
        let curvature = p.inner_product(&ap)?;
        finite_scalar(&curvature, iteration, "directional curvature")?;
        let normalized_p_norm = scale_real_pow2(p_norm, -direction_exponent);
        let rayleigh_imaginary = scale_real_pow2(
            curvature.imag().abs() / normalized_p_norm.powi(2),
            image_exponent,
        );
        if curvature.imag().abs() > options.hermitian_tol * curvature.abs()
            && rayleigh_imaginary > options.hermitian_tol
        {
            return Err(KrylovError::NonHermitian {
                solver: "cg",
                source: anyhow::anyhow!(
                    "directional Rayleigh quotient = {} + {}i at iteration {iteration}",
                    scale_real_pow2(curvature.real() / normalized_p_norm.powi(2), image_exponent),
                    scale_real_pow2(curvature.imag() / normalized_p_norm.powi(2), image_exponent),
                ),
            }
            .into());
        }
        if curvature.real() <= 0.0 {
            return Err(KrylovError::NonPositiveCurvature {
                solver: "cg",
                iteration,
                curvature: scale_real_pow2(curvature.real(), image_exponent),
            }
            .into());
        }
        // HPD CG coefficients are mathematically real. Use the backend real
        // projection, preserving scalar metadata instead of extracting f64.
        let alpha = &rho / curvature.real_part();
        finite_scalar(&alpha, iteration, "step length")?;
        // Undo the residual/direction/image scales on vectors, not on scalar
        // coefficients: a representable update must not need an overflowing ratio.
        let step = scale_pow2(
            p.scale(alpha.clone())?,
            residual_exponent - direction_exponent - image_exponent,
        )?;
        x = x.axpby(AnyScalar::new_real(1.0), &step, AnyScalar::new_real(1.0))?;
        let image = scale_pow2(ap, -direction_exponent)?;
        r = r.axpby(AnyScalar::new_real(1.0), &image, -alpha)?;
        iterations = iteration;
        let working_norm = finite_norm(&r, iteration, "recursive residual norm")?;
        residual_norm = scale_real_pow2(working_norm, residual_exponent);
        if !residual_norm.is_finite() {
            return Err(KrylovError::NumericalBreakdown {
                solver: "cg",
                iteration,
                quantity: "physical residual norm",
            }
            .into());
        }
        if residual_norm <= threshold || iteration == options.max_iter {
            r = true_residual(&apply_a, b, &x)?;
            residual_norm = finite_norm(&r, iteration, "true residual norm")?;
            if residual_norm <= threshold || iteration == options.max_iter {
                break;
            }
            // Replacing r invalidates the old conjugacy recurrence.
            residual_exponent = normalization_exponent(residual_norm);
            r = scale_pow2(r, -residual_exponent)?;
            p = r.clone();
            rho = residual_inner_product(&r, iteration)?;
            continue;
        }
        let next_exponent = normalization_exponent(working_norm);
        r = scale_pow2(r, -next_exponent)?;
        let next_rho = residual_inner_product(&r, iteration)?;
        let beta = &next_rho / &rho;
        finite_scalar(&beta, iteration, "direction coefficient")?;
        // If r_new = 2^next_exponent * r_normalized, the usual squared-residual
        // beta becomes this ratio times 2^(next_exponent + direction_exponent)
        // in the newly scaled vector space.
        p = scale_pow2(p, next_exponent + direction_exponent)?;
        p = r.axpby(AnyScalar::new_real(1.0), &p, beta)?;
        rho = next_rho;
        // INVARIANT: the physical residual on this path is finite and positive;
        // its rescaled exponent stays within the finite f64 range. All exponent
        // sums/differences above are therefore bounded by a few thousand.
        residual_exponent += next_exponent;
    }
    Ok(CgResult {
        solution: x,
        iterations,
        residual_norm,
        converged: residual_norm <= threshold,
    })
}

// INVARIANT: callers supply a positive finite f64 norm; its binary exponent
// lies in [-1074, 1024], including subnormals and log2 rounding at f64::MAX.
fn normalization_exponent(norm: f64) -> i32 {
    norm.log2().floor() as i32
}

// Each factor is a normal, finite power of two even in f32/Complex32.
const SCALE_CHUNK: i32 = 126;

fn scale_pow2<T: TensorVectorSpace>(mut value: T, mut exponent: i32) -> Result<T> {
    while exponent != 0 {
        let chunk = exponent.clamp(-SCALE_CHUNK, SCALE_CHUNK);
        value = value.scale(AnyScalar::new_real(2.0_f64.powi(chunk)))?;
        exponent -= chunk;
    }
    Ok(value)
}

fn scale_real_pow2(mut value: f64, mut exponent: i32) -> f64 {
    while exponent != 0 {
        let chunk = exponent.clamp(-SCALE_CHUNK, SCALE_CHUNK);
        value *= 2.0_f64.powi(chunk);
        exponent -= chunk;
    }
    value
}

fn true_residual<T, F>(apply_a: &F, b: &T, x: &T) -> Result<T>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    let ax = apply_a(x)?;
    ax.validate()?;
    Ok(b.sub(&ax)?)
}

fn finite_norm<T: TensorVectorSpace>(
    x: &T,
    iteration: usize,
    quantity: &'static str,
) -> Result<f64> {
    let norm = x.norm()?;
    if !norm.is_finite() || norm < 0.0 {
        return Err(KrylovError::NumericalBreakdown {
            solver: "cg",
            iteration,
            quantity,
        }
        .into());
    }
    Ok(norm)
}

fn finite_scalar(value: &AnyScalar, iteration: usize, quantity: &'static str) -> Result<()> {
    if !value.real().is_finite() || !value.imag().is_finite() {
        return Err(KrylovError::NumericalBreakdown {
            solver: "cg",
            iteration,
            quantity,
        }
        .into());
    }
    Ok(())
}

fn residual_inner_product<T: TensorVectorSpace>(r: &T, iteration: usize) -> Result<AnyScalar> {
    let rho = r.inner_product(r)?.real_part();
    finite_scalar(&rho, iteration, "residual inner product")?;
    if rho.real() <= 0.0 {
        return Err(KrylovError::NumericalBreakdown {
            solver: "cg",
            iteration,
            quantity: "residual inner product",
        }
        .into());
    }
    Ok(rho)
}
