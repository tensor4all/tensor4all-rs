//! Global default values with atomic access.
//!
//! This module provides a type-safe way to manage global default values
//! that can be accessed and modified atomically.

use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

/// Error for invalid tolerance values.
#[derive(Debug, Error, Clone, Copy)]
#[error("Invalid rtol value: {0}. rtol must be finite and non-negative.")]
pub struct InvalidRtolError(pub f64);

/// A global default f64 value with atomic access.
///
/// This type provides thread-safe access to a global default value,
/// typically used for tolerance parameters like `rtol`.
///
/// # Example
///
/// ```
/// use tensor4all_core::GlobalDefault;
///
/// static MY_DEFAULT: GlobalDefault = GlobalDefault::new(1e-12);
///
/// // Get the default value
/// let rtol = MY_DEFAULT.get();
///
/// // Set a new default value
/// MY_DEFAULT.set(1e-10).unwrap();
/// ```
pub struct GlobalDefault {
    value: AtomicU64,
}

impl GlobalDefault {
    /// Create a new global default with the given initial value.
    ///
    /// This is a const fn, so it can be used in static declarations.
    #[must_use]
    pub const fn new(initial: f64) -> Self {
        Self {
            value: AtomicU64::new(initial.to_bits()),
        }
    }

    /// Get the current default value.
    #[must_use]
    pub fn get(&self) -> f64 {
        f64::from_bits(self.value.load(Ordering::Relaxed))
    }

    /// Set a new default value.
    ///
    /// # Errors
    ///
    /// Returns `InvalidRtolError` if the value is not finite or is negative.
    pub fn set(&self, value: f64) -> Result<(), InvalidRtolError> {
        if !value.is_finite() || value < 0.0 {
            return Err(InvalidRtolError(value));
        }
        self.value.store(value.to_bits(), Ordering::Relaxed);
        Ok(())
    }

    /// Set a new default value without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure the value is valid (finite and non-negative).
    pub fn set_unchecked(&self, value: f64) {
        self.value.store(value.to_bits(), Ordering::Relaxed);
    }
}

// Safety: AtomicU64 is Sync, so GlobalDefault is too
unsafe impl Sync for GlobalDefault {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_default() {
        static TEST_DEFAULT: GlobalDefault = GlobalDefault::new(1e-12);

        assert!((TEST_DEFAULT.get() - 1e-12).abs() < 1e-20);

        TEST_DEFAULT.set(1e-10).unwrap();
        assert!((TEST_DEFAULT.get() - 1e-10).abs() < 1e-20);
    }

    #[test]
    fn test_invalid_values() {
        static TEST_DEFAULT: GlobalDefault = GlobalDefault::new(1e-12);

        assert!(TEST_DEFAULT.set(f64::NAN).is_err());
        assert!(TEST_DEFAULT.set(f64::INFINITY).is_err());
        assert!(TEST_DEFAULT.set(-1.0).is_err());
    }

    #[test]
    fn test_set_unchecked() {
        static TEST_DEFAULT: GlobalDefault = GlobalDefault::new(1e-12);

        TEST_DEFAULT.set_unchecked(1e-8);
        assert!((TEST_DEFAULT.get() - 1e-8).abs() < 1e-20);
    }

    #[test]
    fn test_error_display() {
        let err = InvalidRtolError(-1.0);
        let msg = format!("{}", err);
        assert!(msg.contains("-1"));
        assert!(msg.contains("rtol"));
    }
}
