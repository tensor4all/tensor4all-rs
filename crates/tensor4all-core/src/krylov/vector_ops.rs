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

/// Number of `scale` calls that `scale_pow2` makes for `exponent`.
pub(super) fn scale_pow2_calls(exponent: i32) -> usize {
    (exponent.unsigned_abs() as usize).div_ceil(SCALE_CHUNK as usize)
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
/// `f32` and `f64`. Returns the result and the number of `scale` calls, or an
/// error for a `norm` that is zero, negative or not finite.
pub(super) fn divide_by_norm<T: TensorVectorSpace>(value: &T, norm: f64) -> Result<(T, usize)> {
    if !(norm.is_finite() && norm > 0.0) {
        anyhow::bail!("normalization needs a positive finite norm, got {norm}");
    }
    let inverse = AnyScalar::new_real(1.0 / norm);
    if inverse.is_in_single_precision_range() {
        return Ok((value.scale(inverse)?, 1));
    }
    let exponent = normalization_exponent(norm);
    let mantissa = scale_real_pow2(norm, -exponent);
    let calls = scale_pow2_calls(exponent) + 1;
    let scaled = scale_pow2(value.clone(), -exponent)?;
    Ok((scaled.scale(AnyScalar::new_real(1.0 / mantissa))?, calls))
}

/// Largest `s` for which `scale_pow2(x, -s)` keeps a nonzero component, or
/// `None` for `x = 0`.
///
/// Where that ends depends on the storage precision (`2^-1074` for `f64`
/// entries, `2^-149` for `f32`) and on the mantissa of the largest component
/// (`1.5 * 2^-1075` still rounds up to `2^-1074`), so it is found on the scaled
/// vector itself, with the same `scale_pow2` the retry applies: bisection
/// between `0` (nonzero) and an exponent past the `f64` range, which zeroes
/// every entry. Being nonzero is monotone in `s`, since a larger scaling
/// never makes a rounded-to-zero entry nonzero again. About 11 `scale_pow2`
/// calls and no `apply_a` call.
pub(super) fn largest_informative_exponent<T: TensorVectorSpace>(x: &T) -> Result<Option<i32>> {
    if x.maxabs()? == 0.0 {
        return Ok(None);
    }
    // Every component is below 2^1024 and every nonzero entry is at least
    // 2^-1074, so 2^-(1024 + 1076) takes all of them to zero.
    let mut nonzero = 0;
    let mut zero = 1024 + 1076;
    while zero - nonzero > 1 {
        let middle = nonzero + (zero - nonzero) / 2;
        if scale_pow2(x.clone(), -middle)?.maxabs()? > 0.0 {
            nonzero = middle;
        } else {
            zero = middle;
        }
    }
    Ok(Some(nonzero))
}

/// The smallest `s` in `1..=limit` for which `A(x 2^-s)` has only finite
/// components, with that image, or `None` when no `s` in that range gives one.
/// `s` is found by doubling (1, 2, 4, ..., with the last step clamped to
/// `limit`, so every exponent up to `limit` is covered) and then bisecting
/// between the last non-finite and the first finite exponent, which assumes
/// that the image stays finite once it is (true for a linear `A` in floating
/// point). A NaN image, as `inf - inf` inside `A` forms, counts as non-finite.
/// Also returns the number of `apply_a` calls.
pub(super) fn smallest_finite_scaling<T, F>(
    apply_a: &F,
    x: &T,
    limit: i32,
) -> Result<(Option<(i32, T)>, usize)>
where
    T: TensorVectorSpace,
    F: Fn(&T) -> Result<T>,
{
    let mut calls = 0;
    let mut probe = |exponent: i32| -> Result<Option<T>> {
        calls += 1;
        let image = apply_a(&scale_pow2(x.clone(), -exponent)?)?;
        Ok(image.is_all_finite()?.then_some(image))
    };
    let mut non_finite = 0;
    let mut exponent = 1;
    let mut found = None;
    if limit < 1 {
        return Ok((None, 0));
    }
    // The last doubling step is clamped to `limit`, so every exponent up to
    // it is covered.
    loop {
        if let Some(image) = probe(exponent)? {
            found = Some((exponent, image));
            break;
        }
        non_finite = exponent;
        if exponent >= limit {
            break;
        }
        exponent = (exponent * 2).min(limit);
    }
    if let Some((mut upper, mut image)) = found {
        while upper - non_finite > 1 {
            let middle = non_finite + (upper - non_finite) / 2;
            match probe(middle)? {
                Some(finite) => {
                    upper = middle;
                    image = finite;
                }
                None => non_finite = middle,
            }
        }
        found = Some((upper, image));
    }
    Ok((found, calls))
}

/// `divisor * 2^exponent`, applied in power-of-two steps that move
/// monotonically from `divisor` toward the result, so a real divisor is
/// scaled exactly when the result is a normal number.
fn scale_scalar_pow2(divisor: &AnyScalar, mut exponent: i32) -> AnyScalar {
    let mut value = divisor.clone();
    while exponent != 0 {
        let chunk = exponent.clamp(-SCALE_CHUNK, SCALE_CHUNK);
        value = value * AnyScalar::new_real(2.0_f64.powi(chunk));
        exponent -= chunk;
    }
    value
}

/// `value * 2^exponent / divisor` for a finite nonzero `divisor`.
///
/// The part of `2^exponent` that points against the division (`exponent > 0`
/// with `|divisor| > 2`, or `exponent < 0` with `|divisor| < 1`) is folded
/// into the divisor first, as `divisor * 2^-c` with `c` the smaller of the two
/// exponents in magnitude, so that `divisor * 2^-c` moves toward 1. What is
/// left, the division by `divisor * 2^-c` through
/// [`AnyScalar::reciprocal_factors`] and the remaining `2^(exponent - c)`,
/// scales in one direction only. For a real divisor every intermediate
/// component then lies between the input component and the result component,
/// so a component that is normal in both is never flushed or overflowed on
/// the way; the result is rounded at most twice. A complex divisor has the
/// guarantees of `reciprocal_factors` for the scaled divisor, whose smaller
/// component can lose bits when it is more than `2^1022` below the larger.
///
/// Errors when single-precision entries need a factor outside the `f32`
/// range. Returns the result and the number of `scale` calls.
pub(super) fn scaled_quotient<T: TensorVectorSpace>(
    value: T,
    exponent: i32,
    divisor: &AnyScalar,
    solver: &'static str,
) -> Result<(T, usize)> {
    let divisor_exponent = normalization_exponent(divisor.real().abs().max(divisor.imag().abs()));
    let cancel = if exponent > 0 && divisor_exponent > 0 {
        exponent.min(divisor_exponent)
    } else if exponent < 0 && divisor_exponent < 0 {
        exponent.max(divisor_exponent)
    } else {
        0
    };
    let factors = scale_scalar_pow2(divisor, -cancel).reciprocal_factors()?;
    if value.is_single_precision() {
        if let Some(factor) = factors.iter().find(|f| !f.is_in_single_precision_range()) {
            return Err(KrylovError::InvalidOptions {
                solver,
                reason: format!(
                    "dividing single-precision entries by {divisor:?} needs the factor \
                     {factor:?}, outside the f32 range"
                ),
            }
            .into());
        }
    }
    let quotient = factors
        .iter()
        .try_fold(value, |value, factor| value.scale(factor.clone()))?;
    let rest = exponent - cancel;
    let scale_calls = factors.len() + scale_pow2_calls(rest);
    Ok((scale_pow2(quotient, rest)?, scale_calls))
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
