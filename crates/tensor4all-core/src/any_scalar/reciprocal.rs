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
    /// Returns whether every nonzero component converts to a normal `f32`, so
    /// that converting the scalar to single precision only rounds it to 24
    /// significant bits.
    ///
    /// Use it on the factors of [`reciprocal_factors`](Self::reciprocal_factors)
    /// before applying them to single-precision entries: a factor outside
    /// this range would become zero or infinite, or lose a component.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// assert!(AnyScalar::new_real(0.25).is_in_single_precision_range());
    /// assert!(AnyScalar::new_complex(3.0, 0.0).is_in_single_precision_range());
    /// assert!(!AnyScalar::new_real(2.0_f64.powi(-199)).is_in_single_precision_range());
    /// assert!(!AnyScalar::new_complex(1.0, 1e300).is_in_single_precision_range());
    /// assert!(!AnyScalar::new_real(f64::NAN).is_in_single_precision_range());
    /// ```
    pub fn is_in_single_precision_range(&self) -> bool {
        [self.real(), self.imag()]
            .into_iter()
            .all(|component| component == 0.0 || (component as f32).is_normal())
    }

    /// Returns scalar factors whose product is `1 / self`, ordered so that
    /// multiplying a value by each in turn divides it by `self` with every
    /// intermediate before the last product finite when the quotient
    /// components are, and without forming a component of `1 / self` that
    /// could underflow. The last product multiplies by a rounded factor, so a
    /// quotient component within a few units in the last place of `f64::MAX`
    /// can round to infinity.
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
    /// - An untracked purely imaginary divisor `b i` gives `-i`, which is
    ///   exact, followed by the factors of the real `b`, so the real-divisor
    ///   guarantees apply to each component.
    /// - Another complex divisor gives a real factor last, built from
    ///   `1 / |self * 2^j|^2` with `|self * 2^j|` in `[1/4, 1/2)` up to
    ///   `|self| = 2^201` and in `[2^64, 2^65)` above. When `j >= 0`, `2^j`
    ///   and `conj(self * 2^j)` come first. When `j < 0` and `|self|` is below
    ///   `2^201`, the factors are `conj(self * 2^j)` and
    ///   `2^j / |self * 2^j|^2` if scaling `self` down is exact; if it would
    ///   flush a component, `conj(self)` is applied unscaled instead, as
    ///   `2^j`, `conj(self)`, `2^j / |self * 2^j|^2`. Above `2^201`,
    ///   `conj(self)` comes between two powers of two whose product is
    ///   `2^(2j)`, followed by `1 / |self * 2^j|^2`. No component of `1 / self` or of a
    ///   scaled-down `self` is formed on its own, and no intermediate
    ///   component exceeds the largest component of the value or of the
    ///   quotient, so none overflows unless a quotient component does. The
    ///   error of each component is a few units in the last place of
    ///   `|quotient|`. For a real value with a subnormal quotient, the
    ///   intermediate before the last factor is rounded at subnormal
    ///   granularity, so the absolute error is up to `9 * 2^-1074` when
    ///   `|self| < 1/2` and up to `7 * 2^-1074` otherwise, and a component
    ///   that small can round to zero.
    ///
    /// The guarantees are for `f64` and `Complex64` arithmetic. Entries stored
    /// in single precision convert each factor to `f32`, which keeps the
    /// guarantees only when every factor passes
    /// [`is_in_single_precision_range`](Self::is_in_single_precision_range).
    ///
    /// A gradient-tracked divisor yields tracked factors.
    ///
    /// # Returns
    ///
    /// One to five factors, to be applied in order.
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
        // A tracked divisor takes the general path, whose recorded expression
        // depends on both components.
        if re == 0.0 && im != 0.0 && !self.tracks_grad() {
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
        // With `u = |self * 2^j|` and quotient `q`, the last intermediate
        // before the real reciprocal is `q * u` (real divisor) or `q * u^2`
        // (complex). For a real divisor and `j > 0` a value is scaled to
        // `q * u` first, and `u <= 1/2` keeps every intermediate at or below
        // `q`; for `j <= 0` no step exceeds the value itself, so `u` in
        // `[1/2, 1)` suffices. A complex factor `w` can grow a component of a
        // complex value to `|x| |w|`, up to `sqrt(2)` times its largest
        // component, so a complex divisor keeps `|w| <= 1/2` with `u` in
        // `[1/4, 1/2)`. The large target `[2^64, 2^65)` loses bits of a scaled
        // value only for a quotient below `2^-1086`, and those bits stay below
        // `2^-1139`.
        let j = if complex {
            if k >= -COMPLEX_LARGE_DIVISOR_EXPONENT {
                k - 2
            } else {
                k + LARGE_DIVISOR_TARGET_EXPONENT
            }
        } else if k > 0 {
            k - 2
        } else {
            k + LARGE_DIVISOR_TARGET_EXPONENT
        };
        let scaled = times_pow2(self, j);
        let pow2 = |k: i32| pow2_factors(k).map(AnyScalar::new_real);
        let mut factors: Vec<AnyScalar> = Vec::new();
        if !complex {
            factors.extend(pow2(j));
            factors.push(AnyScalar::new_real(1.0) / scaled);
            return Ok(factors);
        }
        // `|self * 2^j|^2` needs only relative precision, so a component of
        // `self * 2^j` flushed by scaling down does not matter there.
        let norm_sqr = (&scaled * &scaled.conj()).real_part();
        if j >= 0 {
            // Scaling the divisor up is exact.
            factors.extend(pow2(j));
            factors.push(scaled.conj());
        } else {
            let back = times_pow2(&scaled, -j);
            if back.real() == re && back.imag() == im && k >= -COMPLEX_LARGE_DIVISOR_EXPONENT {
                // Scaling the divisor down was exact. `|self * 2^j| < 1/2`
                // bounds the first intermediate and `2^j / |self * 2^j|^2`
                // stays normal, so the quotient is rounded once after a
                // single complex product.
                factors.push(scaled.conj());
                factors.push(times_pow2(&(AnyScalar::new_real(1.0) / norm_sqr), j));
                return Ok(factors);
            }
            // Scaling the divisor down can flush a component whose product
            // with the value is representable, so `conj(self)` is applied
            // unscaled after `2^(k-2)`, which keeps `|2^(k-2) self|` below
            // 1/2. The remaining `2^(2j-k+2)` is folded into the real
            // reciprocal when that stays normal (`j = k - 2`), so the quotient
            // is rounded once at the end.
            factors.extend(pow2(k - 2));
            factors.push(self.conj());
            let remaining = 2 * j - k + 2;
            if k >= -COMPLEX_LARGE_DIVISOR_EXPONENT {
                factors.push(times_pow2(
                    &(AnyScalar::new_real(1.0) / norm_sqr),
                    remaining,
                ));
                return Ok(factors);
            }
            factors.extend(pow2(remaining));
        }
        factors.push(AnyScalar::new_real(1.0) / norm_sqr);
        Ok(factors)
    }
}

#[cfg(test)]
mod tests;
