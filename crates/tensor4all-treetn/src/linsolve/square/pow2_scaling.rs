//! Exact power-of-two rescaling for residual enrichment.
//!
//! Multiplying by a power of two is exact unless a result leaves the normal
//! range, so the enrichment step rescales the residual by one `2^k` chosen
//! from its largest entry, after checking that every nonzero entry stays a
//! normal `f64` (see `ResidualEnrichment::MAX_RESIDUAL_SPREAD`).

use anyhow::Result;
use tensor4all_core::{AnyScalar, TensorLike};

/// Largest step of one multiplication, so that every factor `2^k` is a
/// normal `f64`.
const MAX_STEP: i32 = 1000;

/// `2^k` for `|k| <= MAX_STEP`.
fn pow2(k: i32) -> f64 {
    debug_assert!(k.abs() <= MAX_STEP);
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// `floor(log2 |x|)` for a finite nonzero `x`, exact also for subnormal `x`
/// (where `x.log2()` may round across an integer).
pub(super) fn exponent(x: f64) -> i32 {
    debug_assert!(x.is_finite() && x != 0.0);
    let bits = x.to_bits() & !(1 << 63);
    let biased = (bits >> 52) as i32;
    if biased == 0 {
        // Subnormal: x = mantissa * 2^-1074.
        let mantissa = bits & ((1 << 52) - 1);
        -1074 + (63 - mantissa.leading_zeros() as i32)
    } else {
        biased - 1023
    }
}

/// `x * 2^k`, in steps so that no factor overflows or underflows on its own.
pub(super) fn ldexp(mut x: f64, mut k: i32) -> f64 {
    while k != 0 {
        let step = k.clamp(-MAX_STEP, MAX_STEP);
        x *= pow2(step);
        k -= step;
    }
    x
}

/// `t * 2^k`, in exact power-of-two steps.
pub(super) fn scale_pow2<T: TensorLike>(t: &T, mut k: i32) -> Result<T> {
    let mut t = t.clone();
    while k != 0 {
        let step = k.clamp(-MAX_STEP, MAX_STEP);
        t = t.scale(AnyScalar::new_real(pow2(step)))?;
        k -= step;
    }
    Ok(t)
}

/// `floor(log2 maxabs(t))` for a nonzero `t` whose entries are finite. The
/// largest modulus itself overflows for a complex entry whose finite parts
/// give a modulus above `f64::MAX`; it is then read from `t / 2`, which is
/// only inspected, not used further.
///
/// # Errors
///
/// Returns an error when `t` has a non-finite entry or cannot be read.
pub(super) fn largest_exponent<T: TensorLike>(t: &T) -> Result<i32> {
    let largest = t.maxabs()?;
    if largest.is_finite() {
        anyhow::ensure!(largest > 0.0, "largest_exponent of a zero tensor");
        return Ok(exponent(largest));
    }
    let half = scale_pow2(t, -1)?.maxabs()?;
    anyhow::ensure!(
        half.is_finite(),
        "residual enrichment: the local RHS or residual has a non-finite entry"
    );
    Ok(exponent(half) + 1)
}

/// `coefficient * norm * 2^k` without overflowing the product before the
/// exponent is applied: `coefficient` is split into a mantissa in `[1, 2)`
/// and an exponent first. Returns infinity only when the result itself
/// exceeds `f64::MAX`.
pub(super) fn scaled_product(coefficient: f64, norm: f64, k: i32) -> f64 {
    debug_assert!(coefficient.is_finite() && coefficient > 0.0 && norm.is_finite());
    let coefficient_exp = exponent(coefficient);
    let mantissa = ldexp(coefficient, -coefficient_exp);
    ldexp(mantissa * norm, coefficient_exp.saturating_add(k))
}

#[cfg(test)]
mod tests;
