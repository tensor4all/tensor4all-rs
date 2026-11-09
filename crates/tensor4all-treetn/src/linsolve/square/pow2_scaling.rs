//! Exact power-of-two rescaling for residual enrichment.
//!
//! A tensor `t` with exponent `e` stands for the value `t * 2^e`. Multiplying
//! by a power of two is exact unless a result leaves the normal range, so
//! these helpers move the largest entry where the next operation needs it
//! while keeping every other entry that the target range can hold.

use anyhow::Result;
use tensor4all_core::{AnyScalar, IndexLike, TensorLike};

/// Largest step of one multiplication, so that every factor `2^k` is a
/// normal `f64`.
const MAX_STEP: i32 = 1000;

/// `2^k` for `|k| <= MAX_STEP`.
fn pow2(k: i32) -> f64 {
    debug_assert!(k.abs() <= MAX_STEP);
    f64::from_bits(((k + 1023) as u64) << 52)
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

/// The largest entry modulus of `t` when every entry is finite, and `None`
/// when an entry is infinite or NaN.
pub(super) fn finite_maxabs<T: TensorLike>(t: &T) -> Option<f64> {
    t.maxabs().ok().filter(|value| value.is_finite())
}

/// `(t', e)` with `t = t' * 2^e` and the largest modulus of `t'` finite: `t`
/// itself unless its largest modulus overflows, which a complex entry whose
/// finite parts give a modulus above `f64::MAX` can do; then `t / 2`.
///
/// # Errors
///
/// Returns an error when `t` has a non-finite entry.
pub(super) fn finite_modulus<T: TensorLike>(t: T) -> Result<(T, i32)> {
    if finite_maxabs(&t).is_some() {
        return Ok((t, 0));
    }
    let half = scale_pow2(&t, -1)?;
    anyhow::ensure!(
        finite_maxabs(&half).is_some(),
        "residual enrichment: the local RHS or residual has a non-finite entry"
    );
    Ok((half, 1))
}

/// `(t', e)` with `t = t' * 2^e` and the largest modulus of `t'` in about
/// `[1, 2)`, or `None` when `t` is zero. Scaling up is exact; scaling down
/// only drops entries more than `2^1074` below the largest one.
///
/// # Errors
///
/// Returns an error when `t` has a non-finite entry.
pub(super) fn normalize<T: TensorLike>(t: T) -> Result<Option<(T, i32)>> {
    let (t, offset) = finite_modulus(t)?;
    let max = t.maxabs()?;
    if max == 0.0 {
        return Ok(None);
    }
    let exponent = max.log2().floor() as i32;
    Ok(Some((scale_pow2(&t, -exponent)?, exponent + offset)))
}

/// `r - b b^dagger r` for `b` with orthonormal columns along `bond`, with the
/// exponent of the result: `r * 2^e` projected is `p * 2^e'`. The projection
/// is computed in the units of `r` when that stays finite, so no entry is
/// rescaled; only when the overlap or the product overflows is `r` shrunk, by
/// the smallest power of two that bounds every intermediate.
///
/// # Errors
///
/// Returns an error when a contraction fails.
pub(super) fn project_out<T>(b: &T, r: &T, e: i32) -> Result<(T, i32)>
where
    T: TensorLike,
    T::Index: IndexLike,
{
    let project = |r: &T| -> Result<T> {
        let overlap = T::contract(&[&b.conj(), r])?;
        Ok(r.sub(&T::contract(&[b, &overlap])?)?)
    };
    let projected = project(r)?;
    if finite_maxabs(&projected).is_some() {
        return Ok((projected, e));
    }
    // Every intermediate is at most `len * max` with `len` the number of
    // entries of `r` (which bounds the bond and src dimensions): the overlap
    // by `sqrt(src) * max`, `b * overlap` by `sqrt(bond)` times that.
    let len: usize = r
        .external_indices()
        .iter()
        .map(|index| index.dim())
        .product();
    let max = r.maxabs()?;
    let headroom = (len as f64).log2().ceil() as i32 + 2;
    let shift = (1023 - headroom) - max.log2().ceil() as i32;
    let shift = shift.min(-1);
    let projected = project(&scale_pow2(r, shift)?)?;
    anyhow::ensure!(
        finite_maxabs(&projected).is_some(),
        "residual enrichment: projecting the residual produced a non-finite entry"
    );
    Ok((projected, e - shift))
}
