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
