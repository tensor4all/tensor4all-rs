use super::KrylovError;
use crate::{AnyScalar, TensorVectorSpace};
use anyhow::Result;

// INVARIANT: callers supply a positive finite f64 norm; its binary exponent
// lies in [-1074, 1024], including subnormals and log2 rounding at f64::MAX.
pub(super) fn normalization_exponent(norm: f64) -> i32 {
    norm.log2().floor() as i32
}

// Each factor is a normal, finite power of two even in f32/Complex32.
const SCALE_CHUNK: i32 = 126;

pub(super) fn scale_pow2<T: TensorVectorSpace>(mut value: T, mut exponent: i32) -> Result<T> {
    while exponent != 0 {
        let chunk = exponent.clamp(-SCALE_CHUNK, SCALE_CHUNK);
        value = value.scale(AnyScalar::new_real(2.0_f64.powi(chunk)))?;
        exponent -= chunk;
    }
    Ok(value)
}

pub(super) fn scale_real_pow2(mut value: f64, mut exponent: i32) -> f64 {
    while exponent != 0 {
        let chunk = exponent.clamp(-SCALE_CHUNK, SCALE_CHUNK);
        value *= 2.0_f64.powi(chunk);
        exponent -= chunk;
    }
    value
}

/// `value / norm` for a positive finite `norm`, without forming a reciprocal
/// that overflows: when `1 / norm` is not a normal `f32`, `value` is first
/// scaled by `2^-e` with `norm * 2^-e` in `[1, 2)`. Every factor is normal in
/// `f32` and `f64`. Returns the result and the number of `scale` calls.
pub(super) fn divide_by_norm<T: TensorVectorSpace>(value: &T, norm: f64) -> Result<(T, usize)> {
    let inverse = AnyScalar::new_real(1.0 / norm);
    if inverse.is_in_single_precision_range() {
        return Ok((value.scale(inverse)?, 1));
    }
    let exponent = normalization_exponent(norm);
    let mantissa = scale_real_pow2(norm, -exponent);
    let calls = (exponent.unsigned_abs() as usize).div_ceil(SCALE_CHUNK as usize) + 1;
    let scaled = scale_pow2(value.clone(), -exponent)?;
    Ok((scaled.scale(AnyScalar::new_real(1.0 / mantissa))?, calls))
}

pub(super) fn true_residual<T, F>(apply_a: &F, b: &T, x: &T) -> Result<T>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    let ax = apply_a(x)?;
    ax.validate()?;
    Ok(b.sub(&ax)?)
}

pub(super) fn finite_norm<T: TensorVectorSpace>(
    x: &T,
    solver: &'static str,
    iteration: usize,
    quantity: &'static str,
) -> Result<f64> {
    let norm = x.norm()?;
    if !norm.is_finite() || norm < 0.0 {
        return Err(KrylovError::NumericalBreakdown {
            solver,
            iteration,
            quantity,
        }
        .into());
    }
    Ok(norm)
}
