use anyhow::anyhow;

use super::{AnyScalar, AnyScalarError};

/// Largest exponent magnitude of a real divisor for which `1 / divisor` is
/// formed directly.
const DIRECT_RECIPROCAL_EXPONENT: i32 = 500;

/// Magnitude exponent of the scaled divisor `divisor * 2^j` for a large
/// divisor.
const LARGE_DIVISOR_TARGET_EXPONENT: i32 = 64;

/// Exponent above which a complex divisor is scaled to the large target, where
/// `|quotient| * 2^130` cannot overflow.
const COMPLEX_LARGE_DIVISOR_EXPONENT: i32 = 200;

/// Exact powers of two whose product is `2^k`, each a normal `f64`.
fn pow2_factors(mut k: i32) -> impl Iterator<Item = f64> {
    const STEP: i32 = 1000;
    std::iter::from_fn(move || {
        if k == 0 {
            return None;
        }
        let j = k.clamp(-STEP, STEP);
        k -= j;
        Some(f64::from_bits(((1023 + j) as u64) << 52))
    })
}

fn times_pow2(value: &AnyScalar, k: i32) -> AnyScalar {
    pow2_factors(k).fold(value.clone(), |acc, factor| {
        acc * AnyScalar::new_real(factor)
    })
}

impl AnyScalar {
    /// Returns scalar factors whose product is `1 / self`, ordered so that
    /// multiplying a value by each in turn divides it by `self` with every
    /// intermediate finite when the quotient components are, and without
    /// forming a component of `1 / self` that could underflow.
    ///
    /// Use this instead of multiplying by `1 / self` when `self` may be tiny or
    /// huge: for `self = 3 * 2^-1070`, `1 / self` overflows although
    /// `2^-1000 / self` is finite. Apply the factors in order.
    ///
    /// - A real divisor whose magnitude is in `[2^-500, 2^500]` gives the single
    ///   factor `1 / self`.
    /// - Another real divisor gives exact powers of two `2^j` followed by
    ///   `1 / (self * 2^j)`, where `|self * 2^j|` is in `[1/4, 1/2)` for a
    ///   small divisor and in `[2^64, 2^65)` for a large one. Each component
    ///   of the result is then `fl(x * fl(1 / self))`, with the reciprocal
    ///   rounded as if the exponent range were unbounded, also when the
    ///   product is subnormal.
    /// - A purely imaginary divisor `b i` gives `-i`, which is exact, followed
    ///   by the factors of the real `b`, so the real-divisor guarantees apply
    ///   to each component.
    /// - Another complex divisor gives powers of two `2^j`, then
    ///   `conj(self * 2^j)`, then the real `1 / |self * 2^j|^2`, with
    ///   `|self * 2^j|` in `[1/4, 1/2)` when `|self| < 1/2`, in `[1/2, 1)` up
    ///   to `2^201`, and in `[2^64, 2^65)` above. No component of `1 / self`
    ///   is formed on its own, so a quotient component is not lost because the
    ///   matching component of the reciprocal would underflow, and an
    ///   intermediate overflows only when a quotient component does. For a
    ///   real value each component has a relative error of a few units in the
    ///   last place; in the subnormal range the intermediate `q * |self * 2^j|^2`
    ///   is rounded at subnormal granularity, so the absolute error is up to
    ///   `9 * 2^-1074` when `|self| < 1/2` and up to `4 * 2^-1074` otherwise,
    ///   and a component that small can round to zero. For a complex value the
    ///   error of each component is a few units in the last place of
    ///   `|quotient|`.
    ///
    /// A gradient-tracked divisor yields tracked factors.
    ///
    /// # Returns
    ///
    /// One to four factors, to be applied in order.
    ///
    /// # Errors
    ///
    /// Returns [`AnyScalarError`] when `self` is zero or has a NaN or infinite
    /// component.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let divide = |value: AnyScalar, divisor: &AnyScalar| {
    ///     divisor
    ///         .reciprocal_factors()
    ///         .unwrap()
    ///         .into_iter()
    ///         .fold(value, |acc, factor| acc * factor)
    /// };
    ///
    /// // 3 * 2^-1070 is subnormal and its reciprocal overflows.
    /// let tiny = AnyScalar::new_real(f64::from_bits(48));
    /// assert!((1.0 / tiny.real()).is_infinite());
    /// let quotient = divide(AnyScalar::new_real(2.0_f64.powi(-1000)), &tiny);
    /// assert_eq!(quotient.real(), 2.0_f64.powi(70) / 3.0);
    ///
    /// // (2^600 + 2^-420 i): the imaginary part of the reciprocal is below
    /// // the smallest subnormal, but that of 2^600 / divisor is -2^-1020.
    /// let divisor = AnyScalar::new_complex(2.0_f64.powi(600), 2.0_f64.powi(-420));
    /// let quotient = divide(AnyScalar::new_real(2.0_f64.powi(600)), &divisor);
    /// assert_eq!(quotient.real(), 1.0);
    /// assert_eq!(quotient.imag(), -2.0_f64.powi(-1020));
    ///
    /// assert!(AnyScalar::new_real(0.0).reciprocal_factors().is_err());
    /// ```
    pub fn reciprocal_factors(&self) -> Result<Vec<AnyScalar>, AnyScalarError> {
        // `f64::max` drops a NaN operand, so check each component.
        let (re, im) = (self.real(), self.imag());
        if !(re.is_finite() && im.is_finite()) || self.is_zero() {
            return Err(anyhow!("cannot divide by {self:?}").into());
        }
        if re == 0.0 && im != 0.0 {
            // 1 / (b i) = -i / b, and multiplying by -i is exact.
            let mut factors = vec![AnyScalar::new_complex(0.0, -1.0)];
            factors.extend(self.imag_part().reciprocal_factors()?);
            return Ok(factors);
        }
        let largest = re.abs().max(im.abs());
        let complex = im != 0.0;
        let exponent = largest.log2().floor() as i32;
        if !complex && exponent.abs() <= DIRECT_RECIPROCAL_EXPONENT {
            return Ok(vec![AnyScalar::new_real(1.0) / self.clone()]);
        }
        // `|self| * 2^k` lies in [1, 2) after these loops; each step is exact.
        let mut k = -exponent;
        let mut magnitude = times_pow2(self, k).abs();
        while magnitude >= 2.0 {
            magnitude *= 0.5;
            k -= 1;
        }
        while magnitude < 1.0 {
            magnitude *= 2.0;
            k += 1;
        }
        // With `u = |self * 2^j|`, a value is first scaled to `q * u` and, for
        // a complex divisor, then multiplied to `q * u^2`, where `q` is the
        // quotient. `u <= 1/2` keeps both at or below `q` when `j > 0`, so they
        // overflow only when the quotient does; `j <= 0` cannot overflow the
        // first step, so `u` in `[1/2, 1)` suffices there. The large target
        // `[2^64, 2^65)` loses bits of a scaled value only for a quotient below
        // `2^-1086`, and those bits stay below `2^-1139` after the reciprocal.
        let j = if complex {
            if k >= 2 {
                k - 2
            } else if k >= -COMPLEX_LARGE_DIVISOR_EXPONENT {
                k - 1
            } else {
                k + LARGE_DIVISOR_TARGET_EXPONENT
            }
        } else if k > 0 {
            k - 2
        } else {
            k + LARGE_DIVISOR_TARGET_EXPONENT
        };
        let scaled = times_pow2(self, j);
        let mut factors: Vec<AnyScalar> = pow2_factors(j).map(AnyScalar::new_real).collect();
        if complex {
            let conj = scaled.conj();
            let norm_sqr = (&scaled * &conj).real_part();
            factors.push(conj);
            factors.push(AnyScalar::new_real(1.0) / norm_sqr);
        } else {
            factors.push(AnyScalar::new_real(1.0) / scaled);
        }
        Ok(factors)
    }
}

#[cfg(test)]
mod tests;
