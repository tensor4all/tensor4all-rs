//! Scalar trait shared by matrixluci implementations.

use num_complex::{Complex32, Complex64};
use num_traits::{Float, One, Zero};

/// Common scalar trait for matrix LUCI operations.
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
    /// Complex conjugate.
    fn conj(self) -> Self;

    /// Squared absolute value.
    fn abs_sq(self) -> f64;

    /// Absolute value as same scalar type.
    fn abs(self) -> Self;

    /// Absolute value as f64.
    fn abs_val(self) -> f64 {
        self.abs_sq().sqrt()
    }

    /// Construct from f64.
    fn from_f64(val: f64) -> Self;

    /// Whether value is NaN.
    fn is_nan(self) -> bool;

    /// Machine epsilon.
    fn epsilon() -> f64 {
        f64::EPSILON
    }
}

impl Scalar for f64 {
    fn conj(self) -> Self {
        self
    }

    fn abs_sq(self) -> f64 {
        self * self
    }

    fn abs(self) -> Self {
        Float::abs(self)
    }

    fn abs_val(self) -> f64 {
        Float::abs(self)
    }

    fn from_f64(val: f64) -> Self {
        val
    }

    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }
}

impl Scalar for f32 {
    fn conj(self) -> Self {
        self
    }

    fn abs_sq(self) -> f64 {
        (self * self) as f64
    }

    fn abs(self) -> Self {
        Float::abs(self)
    }

    fn abs_val(self) -> f64 {
        Float::abs(self) as f64
    }

    fn from_f64(val: f64) -> Self {
        val as f32
    }

    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }
}

impl Scalar for Complex64 {
    fn conj(self) -> Self {
        Complex64::conj(&self)
    }

    fn abs_sq(self) -> f64 {
        self.norm_sqr()
    }

    fn abs(self) -> Self {
        Complex64::new(self.norm(), 0.0)
    }

    fn abs_val(self) -> f64 {
        self.norm()
    }

    fn from_f64(val: f64) -> Self {
        Complex64::new(val, 0.0)
    }

    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

impl Scalar for Complex32 {
    fn conj(self) -> Self {
        Complex32::conj(&self)
    }

    fn abs_sq(self) -> f64 {
        self.norm_sqr() as f64
    }

    fn abs(self) -> Self {
        Complex32::new(self.norm(), 0.0)
    }

    fn abs_val(self) -> f64 {
        self.norm() as f64
    }

    fn from_f64(val: f64) -> Self {
        Complex32::new(val as f32, 0.0)
    }

    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}
