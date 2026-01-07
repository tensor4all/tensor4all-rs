//! Scalar trait for generic block operations.
//!
//! This module defines the `Scalar` trait that abstracts over f64 and Complex64
//! for blocked array operations.

use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

use faer_traits::ComplexField;
use num_complex::{Complex64, ComplexFloat};
use num_traits::{MulAdd, One, Zero};

/// Trait for scalar types used in blocked array operations.
///
/// This trait provides the minimal interface needed for block operations including
/// matrix multiplication via the Faer backend.
pub trait Scalar:
    Clone
    + Copy
    + Debug
    + Default
    + PartialEq
    + Zero
    + One
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + ComplexFloat
    + ComplexField
    + MulAdd<Output = Self>
    + Send
    + Sync
    + 'static
{
    /// Create a scalar from f64.
    fn from_f64(val: f64) -> Self;

    /// Get the real part as f64.
    fn real_f64(&self) -> f64;

    /// Check if this type is complex.
    fn is_complex_type() -> bool;
}

impl Scalar for f64 {
    fn from_f64(val: f64) -> Self {
        val
    }

    fn real_f64(&self) -> f64 {
        *self
    }

    fn is_complex_type() -> bool {
        false
    }
}

impl Scalar for Complex64 {
    fn from_f64(val: f64) -> Self {
        Complex64::new(val, 0.0)
    }

    fn real_f64(&self) -> f64 {
        self.re
    }

    fn is_complex_type() -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_f64() {
        let x: f64 = Scalar::from_f64(3.0);
        assert_eq!(x, 3.0);
        assert_eq!(x.real_f64(), 3.0);
        assert!(!f64::is_complex_type());
    }

    #[test]
    fn test_scalar_complex64() {
        let z: Complex64 = Scalar::from_f64(3.0);
        assert_eq!(z, Complex64::new(3.0, 0.0));
        assert_eq!(z.real_f64(), 3.0);
        assert!(Complex64::is_complex_type());
    }
}
