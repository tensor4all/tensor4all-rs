//! Matrix-free BiCGSTAB for general real and complex linear operators.
//!
//! Derived from KrylovKit.jl's `src/linsolve/bicgstab.jl` at revision
//! d51fa25adfa4cda5530d8e3d42eedfb288cca447.
//! <https://github.com/Jutho/KrylovKit.jl/blob/d51fa25adfa4cda5530d8e3d42eedfb288cca447/src/linsolve/bicgstab.jl>
//!
//! Copyright (c) 2017: Jutho. MIT license; see ../../LICENSE-KrylovKit.md.

use super::vector_ops::{
    finite_norm, normalization_exponent, scale_pow2, scale_real_pow2, true_residual,
};
use super::KrylovError;
use crate::{AnyScalar, TensorVectorSpace};
use anyhow::Result;

/// Convergence controls for [`bicgstab`], which retains a fixed number of vectors.
/// Start with the defaults for double precision and use `rtol = 1e-5` for
/// single precision.
///
/// # Examples
/// ```
/// use tensor4all_core::{krylov::{bicgstab, BiCgStabOptions}, DynIndex, IdxTensor};
/// # fn main() -> anyhow::Result<()> {
/// let i = DynIndex::new_dyn(2);
/// let b = IdxTensor::from_dense(vec![i.clone()], vec![1.0_f64, -2.0])?;
/// let x0 = IdxTensor::zeros::<f64>(vec![i])?;
/// let options = BiCgStabOptions { max_iter: 1, ..BiCgStabOptions::default() };
/// let result = bicgstab(|x: &IdxTensor| Ok(x.clone()), &b, &x0, &options)?;
/// assert_eq!(result.solution.to_vec::<f64>()?, vec![1.0, -2.0]);
/// assert_eq!(result.iterations, 1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct BiCgStabOptions {
    /// Maximum BiCGSTAB cycles, including cycles restarted after residual
    /// replacement. Default: `100`. Zero checks the initial residual only.
    pub max_iter: usize,
    /// Finite, nonnegative relative tolerance. Default: `1e-10`.
    /// The stopping threshold is `max(atol, rtol * ||b||)`.
    pub rtol: f64,
    /// Finite, nonnegative absolute tolerance. Default: `0.0`.
    /// Set a positive value for an absolute target with a tiny RHS.
    /// Both tolerances zero require an exactly zero true residual.
    pub atol: f64,
}

impl Default for BiCgStabOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            rtol: 1e-10,
            atol: 0.0,
        }
    }
}

/// Solution and true-residual diagnostics returned by [`bicgstab`].
/// Like [`super::CgResult`], the residual is absolute; [`super::GmresResult`]
/// reports a relative residual by default.
///
/// # Examples
/// ```
/// use tensor4all_core::{krylov::{bicgstab, BiCgStabOptions}, DynIndex, IdxTensor, TensorVectorSpace};
/// # fn main() -> anyhow::Result<()> {
/// let i = DynIndex::new_dyn(2);
/// let b = IdxTensor::from_dense(vec![i.clone()], vec![2.0_f64, 4.0])?;
/// let x0 = IdxTensor::zeros::<f64>(vec![i])?;
/// let result = bicgstab(|x: &IdxTensor| Ok(x.scale(2.0.into())?), &b, &x0, &BiCgStabOptions::default())?;
/// assert!(result.converged);
/// assert_eq!(result.solution.to_vec::<f64>()?, vec![1.0, 2.0]);
/// assert_eq!(result.residual_norm, 0.0);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct BiCgStabResult<T> {
    /// Approximate solution, including the initial guess's contribution.
    pub solution: T,
    /// Consumed cycles, bounded by [`BiCgStabOptions::max_iter`].
    /// Convergence at the intermediate BiCG step counts as one cycle.
    pub iterations: usize,
    /// Absolute true residual norm `||b - A*x||`, also on budget exhaustion.
    pub residual_norm: f64,
    /// Whether the true residual met `max(atol, rtol * ||b||)`.
    pub converged: bool,
}

/// Solve `A x = b` with matrix-free BiCGSTAB.
///
/// # Arguments
/// * `apply_a` - Fixed linear operator preserving the vector space. It may be
///   nonsymmetric or complex non-Hermitian.
/// * `b` - Right-hand side in the same vector space as `x0`.
/// * `x0` - Initial guess. A zero-norm RHS returns the zero vector, the exact
///   solution, after checking that `x0` and `b` share a vector space.
/// * `options` - Cycle budget and tolerances; start with the defaults for
///   double precision.
///
/// # Returns
/// The approximate solution, consumed cycles, absolute true residual, and
/// convergence flag. True residuals are checked at intermediate and full-step
/// convergence and at budget exhaustion. Residual replacement restarts the
/// recurrence within the original budget.
/// Working vectors and operator images are normalized by powers of two before
/// inner products. Intermediate arithmetic remains in `TensorVectorSpace` and
/// `AnyScalar`; norm, range, and stopping decisions cross scalar control-flow boundaries.
/// This solver does not promise a differentiable solve. The caller's vector
/// space must provide a norm preserving the required dynamic range.
///
/// BiCGSTAB may break down even for an invertible operator. In that case,
/// [`super::gmres`] provides an alternative with different memory requirements.
///
/// # Errors
/// Returns [`KrylovError::InvalidOptions`] for invalid tolerances or an
/// overflowing stopping threshold, [`KrylovError::NumericalBreakdown`] for
/// unusable recurrence quantities, and [`KrylovError::Operation`] for invalid
/// or incompatible vector spaces, callback failures, or tensor errors.
/// Underlying operation errors retain their source chains.
///
/// # Examples
/// ```
/// use tensor4all_core::{contract, krylov::{bicgstab, BiCgStabOptions}, DynIndex, IdxTensor, TensorIndex};
/// # fn main() -> anyhow::Result<()> {
/// let i = DynIndex::new_dyn(2);
/// let input = i.prime();
/// // Column-major [[1, 1], [0, 2]], with solution [-0.5, 0.5].
/// let a = IdxTensor::from_dense(vec![i.clone(), input.clone()], vec![1.0_f64, 0.0, 1.0, 2.0])?;
/// let b = IdxTensor::from_dense(vec![i.clone()], vec![0.0_f64, 1.0])?;
/// let x0 = IdxTensor::zeros::<f64>(vec![i.clone()])?;
/// let result = bicgstab(
///     |x: &IdxTensor| Ok(contract(&[&a, &x.replaceind(&i, &input)?])?),
///     &b, &x0, &BiCgStabOptions::default(),
/// )?;
/// assert!(result.converged);
/// assert_eq!(result.solution.to_vec::<f64>()?, vec![-0.5, 0.5]);
/// assert_eq!(result.residual_norm, 0.0);
/// # Ok(())
/// # }
/// ```
pub fn bicgstab<T, F>(
    apply_a: F,
    b: &T,
    x0: &T,
    options: &BiCgStabOptions,
) -> std::result::Result<BiCgStabResult<T>, KrylovError>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    bicgstab_impl(apply_a, b, x0, options).map_err(|source| {
        match source.downcast::<KrylovError>() {
            Ok(classified) => classified,
            Err(source) => KrylovError::Operation {
                solver: "bicgstab",
                source,
            },
        }
    })
}

fn bicgstab_impl<T, F>(
    apply_a: F,
    b: &T,
    x0: &T,
    options: &BiCgStabOptions,
) -> Result<BiCgStabResult<T>>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    for (name, tolerance) in [("rtol", options.rtol), ("atol", options.atol)] {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(KrylovError::InvalidOptions {
                solver: "bicgstab",
                reason: format!("{name} must be finite and nonnegative"),
            }
            .into());
        }
    }
    b.validate()?;
    x0.validate()?;
    let b_norm = finite_norm(b, "bicgstab", 0, "right-hand-side norm")?;
    let threshold = options.atol.max(options.rtol * b_norm);
    if !threshold.is_finite() {
        return Err(KrylovError::InvalidOptions {
            solver: "bicgstab",
            reason: "rtol * ||b|| must be finite".into(),
        }
        .into());
    }
    // Callback failures keep their origin, including nested Krylov failures;
    // only this solver's own numerical breakdowns may enter recovery.
    let apply_a = |x: &T| {
        apply_a(x).map_err(|source| {
            anyhow::Error::new(KrylovError::Operation {
                solver: "bicgstab",
                source,
            })
        })
    };
    let mut x = x0.clone();
    let mut r = true_residual(&apply_a, b, &x)?;
    let mut residual_norm = finite_norm(&r, "bicgstab", 0, "initial residual norm")?;
    if b_norm == 0.0 {
        return Ok(BiCgStabResult {
            solution: b.scale(AnyScalar::new_real(0.0))?,
            iterations: 0,
            residual_norm: 0.0,
            converged: true,
        });
    }
    if residual_norm <= threshold || options.max_iter == 0 {
        return Ok(BiCgStabResult {
            solution: x,
            iterations: 0,
            residual_norm,
            converged: residual_norm <= threshold,
        });
    }
    let mut residual_exponent = normalization_exponent(residual_norm);
    r = scale_pow2(r, -residual_exponent)?;
    let mut shadow = r.clone();
    let mut p = r.clone();
    let mut rho = shadow.inner_product(&r)?;
    let mut iteration = 0;
    let run = (|| -> Result<()> {
        loop {
            // INVARIANT: the initial path returned for zero budget, and the final
            // cycle returns at its intermediate or full residual check.
            iteration += 1;
            if !usable(&rho) {
                return Err(breakdown(iteration, "shadow residual inner product"));
            }
            let p_norm = finite_norm(&p, "bicgstab", iteration, "search direction norm")?;
            if p_norm == 0.0 {
                return Err(breakdown(iteration, "search direction norm"));
            }
            p = scale_pow2(p, -normalization_exponent(p_norm))?;
            let v = apply_a(&p)?;
            v.validate()?;
            let v_norm = finite_norm(&v, "bicgstab", iteration, "first operator image norm")?;
            if v_norm == 0.0 {
                return Err(breakdown(iteration, "alpha denominator"));
            }
            let image_exponent = normalization_exponent(v_norm);
            let v = scale_pow2(v, -image_exponent)?;
            let sigma = shadow.inner_product(&v)?;
            if !usable(&sigma) {
                return Err(breakdown(iteration, "alpha denominator"));
            }
            let alpha = &rho / sigma;
            if !usable(&alpha) {
                return Err(breakdown(iteration, "alpha"));
            }
            let s = r.axpby(1.0.into(), &v, -alpha.clone())?;
            let step = scale_pow2(p.scale(alpha.clone())?, residual_exponent - image_exponent)?;
            x = x.axpby(1.0.into(), &step, 1.0.into())?;
            let s_norm = finite_norm(&s, "bicgstab", iteration, "intermediate residual norm")?;
            let estimate = scale_real_pow2(s_norm, residual_exponent);
            if !estimate.is_finite() {
                return Err(breakdown(iteration, "physical intermediate residual norm"));
            }
            if estimate <= threshold {
                r = true_residual(&apply_a, b, &x)?;
                residual_norm = finite_norm(&r, "bicgstab", iteration, "true residual norm")?;
                if residual_norm <= threshold || iteration == options.max_iter {
                    return Ok(());
                }
                residual_exponent = normalization_exponent(residual_norm);
                r = scale_pow2(r, -residual_exponent)?;
                shadow = r.clone();
                p = r.clone();
                rho = shadow.inner_product(&r)?;
                continue;
            }
            // R = 2^er r, A p = 2^ev v; S = 2^(er+es) s after this normalization.
            let intermediate_exponent = normalization_exponent(s_norm);
            let s = scale_pow2(s, -intermediate_exponent)?;
            let t = apply_a(&s)?;
            t.validate()?;
            let t_norm = finite_norm(&t, "bicgstab", iteration, "second operator image norm")?;
            if t_norm == 0.0 {
                return Err(breakdown(iteration, "stabilization denominator"));
            }
            let stabilization_exponent = normalization_exponent(t_norm);
            let t = scale_pow2(t, -stabilization_exponent)?;
            let denominator = t.inner_product(&t)?.real_part();
            if !usable(&denominator) {
                return Err(breakdown(iteration, "stabilization denominator"));
            }
            let omega = t.inner_product(&s)? / denominator;
            if !usable(&omega) {
                return Err(breakdown(iteration, "omega"));
            }
            let step = scale_pow2(
                s.scale(omega.clone())?,
                residual_exponent + intermediate_exponent - stabilization_exponent,
            )?;
            x = x.axpby(1.0.into(), &step, 1.0.into())?;
            r = s.axpby(1.0.into(), &t, -omega.clone())?;
            residual_exponent += intermediate_exponent;
            let r_norm = finite_norm(&r, "bicgstab", iteration, "recursive residual norm")?;
            let estimate = scale_real_pow2(r_norm, residual_exponent);
            if !estimate.is_finite() {
                return Err(breakdown(iteration, "physical residual norm"));
            }
            if estimate <= threshold || iteration == options.max_iter {
                r = true_residual(&apply_a, b, &x)?;
                residual_norm = finite_norm(&r, "bicgstab", iteration, "true residual norm")?;
                if residual_norm <= threshold || iteration == options.max_iter {
                    return Ok(());
                }
                residual_exponent = normalization_exponent(residual_norm);
                r = scale_pow2(r, -residual_exponent)?;
                shadow = r.clone();
                p = r.clone();
                rho = shadow.inner_product(&r)?;
                continue;
            }
            let next_exponent = normalization_exponent(r_norm);
            r = scale_pow2(r, -next_exponent)?;
            residual_exponent += next_exponent;
            let next_rho = shadow.inner_product(&r)?;
            if !usable(&next_rho) {
                return Err(breakdown(iteration, "shadow residual inner product"));
            }
            let gamma = scaled_mul_div(&next_rho, &alpha, &rho);
            let beta = &gamma / omega;
            if !usable(&gamma) || !usable(&beta) {
                return Err(breakdown(iteration, "direction coefficient"));
            }
            // P_new / 2^er = r + 2^(et-ev) beta p - gamma v. A common
            // scale makes all power-of-two factors nonpositive before adding.
            let shift = stabilization_exponent - image_exponent;
            let common = shift.max(0);
            let direction =
                scale_pow2(p, shift - common)?.axpby(beta, &scale_pow2(v, -common)?, -gamma)?;
            p = scale_pow2(r.clone(), -common)?.axpby(1.0.into(), &direction, 1.0.into())?;
            rho = next_rho;
        }
    })();
    if let Err(source) = run {
        match source.downcast::<KrylovError>() {
            Ok(
                error @ KrylovError::NumericalBreakdown {
                    solver: "bicgstab", ..
                },
            ) => {
                // A rounded recurrence can break down after reaching a true
                // solution. Only a callback failure outranks the breakdown; a
                // residual that cannot be evaluated reports the breakdown.
                let check = true_residual(&apply_a, b, &x)
                    .and_then(|r| finite_norm(&r, "bicgstab", iteration, "true residual norm"));
                match check {
                    Ok(norm) if norm <= threshold => residual_norm = norm,
                    Ok(_) => return Err(error.into()),
                    Err(check) => {
                        return Err(match check.downcast::<KrylovError>() {
                            Ok(callback @ KrylovError::Operation { .. }) => callback.into(),
                            _ => error.into(),
                        })
                    }
                }
            }
            Ok(error) => return Err(error.into()),
            Err(source) => return Err(source),
        }
    }
    Ok(BiCgStabResult {
        solution: x,
        iterations: iteration,
        residual_norm,
        converged: residual_norm <= threshold,
    })
}

fn usable(value: &AnyScalar) -> bool {
    value.real().is_finite() && value.imag().is_finite() && !value.is_zero()
}

fn scaled_mul_div(x: &AnyScalar, y: &AnyScalar, z: &AnyScalar) -> AnyScalar {
    // INVARIANT: the recurrence checked that all three operands are finite
    // and nonzero. Their component exponents are in [-1074, 1024].
    let exponent = |v: &AnyScalar| normalization_exponent(v.real().abs().max(v.imag().abs()));
    let (xe, ye, ze) = (exponent(x), exponent(y), exponent(z));
    // Four exponent bits of headroom cover normalized complex products/sums;
    // retain the usual evaluation order when both intermediates are in range.
    if (-1020..=1020).contains(&(xe - ze)) && (-1020..=1020).contains(&(xe + ye - ze)) {
        return (x / z) * y;
    }
    let normalized = (scale_scalar_pow2(x.clone(), -xe) * scale_scalar_pow2(y.clone(), -ye))
        / scale_scalar_pow2(z.clone(), -ze);
    scale_scalar_pow2(normalized, xe + ye - ze)
}

fn scale_scalar_pow2(mut value: AnyScalar, mut exponent: i32) -> AnyScalar {
    while exponent != 0 {
        let chunk = exponent.clamp(-126, 126);
        value = value * AnyScalar::new_real(2.0_f64.powi(chunk));
        exponent -= chunk;
    }
    value
}

fn breakdown(iteration: usize, quantity: &'static str) -> anyhow::Error {
    KrylovError::NumericalBreakdown {
        solver: "bicgstab",
        iteration,
        quantity,
    }
    .into()
}

#[cfg(test)]
mod tests;
