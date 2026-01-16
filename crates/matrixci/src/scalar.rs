//! Common scalar trait for matrix and tensor operations.
//!
//! This module provides a unified scalar trait that can be used across
//! tensor4all crates, reducing code duplication.

use num_complex::{Complex32, Complex64};
use num_traits::{Float, One, Zero};

/// Common scalar trait for matrix and tensor operations.
///
/// This trait defines the minimal requirements for scalar types used in
/// matrix cross interpolation and tensor train operations.
pub trait Scalar:
    Clone
    + Copy
    + Zero
    + One
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + Default
    + Send
    + Sync
    + 'static
{
    /// Complex conjugate of the value.
    fn conj(self) -> Self;

    /// Square of the absolute value (for complex numbers, |z|^2).
    fn abs_sq(self) -> f64;

    /// Absolute value as Self type.
    ///
    /// For real types, this returns the absolute value.
    /// For complex types, this returns a real-valued complex (re=|z|, im=0).
    fn abs(self) -> Self;

    /// Absolute value as f64.
    fn abs_val(self) -> f64 {
        self.abs_sq().sqrt()
    }

    /// Create from f64 value.
    fn from_f64(val: f64) -> Self;

    /// Check if value is NaN.
    fn is_nan(self) -> bool;

    /// Small epsilon value for numerical comparisons.
    fn epsilon() -> f64 {
        1e-30
    }
}

impl Scalar for f64 {
    #[inline]
    fn conj(self) -> Self {
        self
    }

    #[inline]
    fn abs_sq(self) -> f64 {
        self * self
    }

    #[inline]
    fn abs(self) -> Self {
        Float::abs(self)
    }

    #[inline]
    fn abs_val(self) -> f64 {
        Float::abs(self)
    }

    #[inline]
    fn from_f64(val: f64) -> Self {
        val
    }

    #[inline]
    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }
}

impl Scalar for f32 {
    #[inline]
    fn conj(self) -> Self {
        self
    }

    #[inline]
    fn abs_sq(self) -> f64 {
        (self * self) as f64
    }

    #[inline]
    fn abs(self) -> Self {
        Float::abs(self)
    }

    #[inline]
    fn abs_val(self) -> f64 {
        Float::abs(self) as f64
    }

    #[inline]
    fn from_f64(val: f64) -> Self {
        val as f32
    }

    #[inline]
    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }
}

impl Scalar for Complex64 {
    #[inline]
    fn conj(self) -> Self {
        Complex64::conj(&self)
    }

    #[inline]
    fn abs_sq(self) -> f64 {
        self.norm_sqr()
    }

    #[inline]
    fn abs(self) -> Self {
        Complex64::new(self.norm(), 0.0)
    }

    #[inline]
    fn abs_val(self) -> f64 {
        self.norm()
    }

    #[inline]
    fn from_f64(val: f64) -> Self {
        Complex64::new(val, 0.0)
    }

    #[inline]
    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

impl Scalar for Complex32 {
    #[inline]
    fn conj(self) -> Self {
        Complex32::conj(&self)
    }

    #[inline]
    fn abs_sq(self) -> f64 {
        self.norm_sqr() as f64
    }

    #[inline]
    fn abs(self) -> Self {
        Complex32::new(self.norm(), 0.0)
    }

    #[inline]
    fn abs_val(self) -> f64 {
        self.norm() as f64
    }

    #[inline]
    fn from_f64(val: f64) -> Self {
        Complex32::new(val as f32, 0.0)
    }

    #[inline]
    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

/// Macro to generate f64 and Complex64 test variants from a generic test function.
///
/// # Example
///
/// ```ignore
/// fn test_operation_generic<T: Scalar>() {
///     // test implementation
/// }
///
/// matrixci::scalar_tests!(test_operation, test_operation_generic);
/// // Generates:
/// // #[test] fn test_operation_f64() { test_operation_generic::<f64>(); }
/// // #[test] fn test_operation_c64() { test_operation_generic::<Complex64>(); }
/// ```
#[macro_export]
macro_rules! scalar_tests {
    ($name:ident, $test_fn:ident) => {
        paste::paste! {
            #[test]
            fn [<$name _f64>]() {
                $test_fn::<f64>();
            }

            #[test]
            fn [<$name _c64>]() {
                $test_fn::<num_complex::Complex64>();
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_scalar_generic<T: Scalar>() {
        let one = T::from_f64(1.0);
        let two = T::from_f64(2.0);

        // Basic arithmetic
        let sum = one + one;
        assert!((sum.abs_sq() - 4.0).abs() < 1e-10);

        // Conjugate (for real, conj is identity)
        let conj_two = two.conj();
        assert!((conj_two.abs_sq() - 4.0).abs() < 1e-10);

        // NaN check
        assert!(!one.is_nan());
    }

    #[test]
    fn test_scalar_f64() {
        test_scalar_generic::<f64>();
    }

    #[test]
    fn test_scalar_f32() {
        test_scalar_generic::<f32>();
    }

    #[test]
    fn test_scalar_c64() {
        test_scalar_generic::<Complex64>();

        // Complex-specific test
        let z = Complex64::new(3.0, 4.0);
        assert!((z.abs_sq() - 25.0).abs() < 1e-10);
        assert!((z.abs_val() - 5.0).abs() < 1e-10);

        let z_conj = z.conj();
        assert!((z_conj.re - 3.0).abs() < 1e-10);
        assert!((z_conj.im - (-4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_scalar_c32() {
        test_scalar_generic::<Complex32>();
    }
}
