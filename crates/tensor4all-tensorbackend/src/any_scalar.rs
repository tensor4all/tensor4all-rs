use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use anyhow::{anyhow, Result};
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tenferro::{ScalarType, Tensor as NativeTensor};

use crate::storage::{Storage, SumFromStorage};
use crate::tenferro_bridge::with_default_runtime;
use crate::tensor_element::TensorElement;

#[derive(Clone, Copy, Debug, PartialEq)]
enum ScalarValue {
    F32(f32),
    F64(f64),
    C32(Complex32),
    C64(Complex64),
}

impl ScalarValue {
    fn real(self) -> f64 {
        match self {
            Self::F32(value) => value as f64,
            Self::F64(value) => value,
            Self::C32(value) => value.re as f64,
            Self::C64(value) => value.re,
        }
    }

    fn imag(self) -> f64 {
        match self {
            Self::F32(_) | Self::F64(_) => 0.0,
            Self::C32(value) => value.im as f64,
            Self::C64(value) => value.im,
        }
    }

    fn abs(self) -> f64 {
        match self {
            Self::F32(value) => value.abs() as f64,
            Self::F64(value) => value.abs(),
            Self::C32(value) => value.norm() as f64,
            Self::C64(value) => value.norm(),
        }
    }

    fn is_complex(self) -> bool {
        matches!(self, Self::C32(_) | Self::C64(_))
    }

    fn is_zero(self) -> bool {
        match self {
            Self::F32(value) => value == 0.0,
            Self::F64(value) => value == 0.0,
            Self::C32(value) => value == Complex32::new(0.0, 0.0),
            Self::C64(value) => value == Complex64::new(0.0, 0.0),
        }
    }

    fn into_complex(self) -> Complex64 {
        match self {
            Self::F32(value) => Complex64::new(value as f64, 0.0),
            Self::F64(value) => Complex64::new(value, 0.0),
            Self::C32(value) => Complex64::new(value.re as f64, value.im as f64),
            Self::C64(value) => value,
        }
    }
}

fn rank0_real_tensor(value: f64) -> NativeTensor {
    NativeTensor::from_slice(&[value], &[])
        .unwrap_or_else(|e| panic!("failed to build rank-0 real tensor: {e}"))
}

fn scalar_value_from_storage(storage: &Storage) -> ScalarValue {
    if storage.is_f64() {
        ScalarValue::F64(f64::sum_from_storage(storage))
    } else {
        ScalarValue::C64(Complex64::sum_from_storage(storage))
    }
}

fn scalar_value_from_native(native: &NativeTensor) -> ScalarValue {
    match native.scalar_type() {
        ScalarType::F32 => ScalarValue::F32(
            native
                .try_get::<f32>(&[])
                .unwrap_or_else(|e| panic!("failed to read f32 scalar tensor value: {e}")),
        ),
        ScalarType::F64 => ScalarValue::F64(
            native
                .try_get::<f64>(&[])
                .unwrap_or_else(|e| panic!("failed to read f64 scalar tensor value: {e}")),
        ),
        ScalarType::C32 => ScalarValue::C32(
            native
                .try_get::<Complex32>(&[])
                .unwrap_or_else(|e| panic!("failed to read c32 scalar tensor value: {e}")),
        ),
        ScalarType::C64 => ScalarValue::C64(
            native
                .try_get::<Complex64>(&[])
                .unwrap_or_else(|e| panic!("failed to read c64 scalar tensor value: {e}")),
        ),
    }
}

fn scalar_tensor_result(op: &'static str, native: Result<NativeTensor>) -> Scalar {
    match native {
        Ok(native) => Scalar::wrap_native(native)
            .unwrap_or_else(|e| panic!("Scalar::{op} returned a non-scalar tensor: {e}")),
        Err(err) => panic!("Scalar::{op} failed: {err}"),
    }
}

fn neg_native(native: &NativeTensor) -> Result<NativeTensor> {
    Ok(match scalar_value_from_native(native) {
        ScalarValue::F32(value) => Scalar::from_value(-value).native,
        ScalarValue::F64(value) => Scalar::from_value(-value).native,
        ScalarValue::C32(value) => Scalar::from_value(-value).native,
        ScalarValue::C64(value) => Scalar::from_value(-value).native,
    })
}

pub(crate) fn promote_scalar_native(
    native: &NativeTensor,
    target: ScalarType,
) -> Result<NativeTensor> {
    let promoted = match (scalar_value_from_native(native), target) {
        (ScalarValue::F32(value), ScalarType::F32) => Scalar::from_value(value),
        (ScalarValue::F32(value), ScalarType::F64) => Scalar::from_value(value as f64),
        (ScalarValue::F32(value), ScalarType::C32) => {
            Scalar::from_value(Complex32::new(value, 0.0))
        }
        (ScalarValue::F32(value), ScalarType::C64) => {
            Scalar::from_value(Complex64::new(value as f64, 0.0))
        }
        (ScalarValue::F64(value), ScalarType::F32) => Scalar::from_value(value as f32),
        (ScalarValue::F64(value), ScalarType::F64) => Scalar::from_value(value),
        (ScalarValue::F64(value), ScalarType::C32) => {
            Scalar::from_value(Complex32::new(value as f32, 0.0))
        }
        (ScalarValue::F64(value), ScalarType::C64) => {
            Scalar::from_value(Complex64::new(value, 0.0))
        }
        (ScalarValue::C32(value), ScalarType::F32) => Scalar::from_value(value.re),
        (ScalarValue::C32(value), ScalarType::F64) => Scalar::from_value(value.re as f64),
        (ScalarValue::C32(value), ScalarType::C32) => Scalar::from_value(value),
        (ScalarValue::C32(value), ScalarType::C64) => {
            Scalar::from_value(Complex64::new(value.re as f64, value.im as f64))
        }
        (ScalarValue::C64(value), ScalarType::F32) => Scalar::from_value(value.re as f32),
        (ScalarValue::C64(value), ScalarType::F64) => Scalar::from_value(value.re),
        (ScalarValue::C64(value), ScalarType::C32) => {
            Scalar::from_value(Complex32::new(value.re as f32, value.im as f32))
        }
        (ScalarValue::C64(value), ScalarType::C64) => Scalar::from_value(value),
    };
    Ok(promoted.native)
}

/// Dynamic scalar used across tensor4all backends.
///
/// This is a tensor4all-owned rank-0 wrapper over tenferro's dynamic tensor.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::AnyScalar;
///
/// // Real scalar
/// let a = AnyScalar::new_real(3.14);
/// assert!((a.real() - 3.14).abs() < 1e-10);
/// assert_eq!(a.imag(), 0.0);
/// assert!(a.is_real());
/// assert!(!a.is_complex());
///
/// // Complex scalar
/// let b = AnyScalar::new_complex(1.0, 2.0);
/// assert!((b.real() - 1.0).abs() < 1e-10);
/// assert!((b.imag() - 2.0).abs() < 1e-10);
/// assert!(b.is_complex());
///
/// // Arithmetic
/// let c = AnyScalar::new_real(2.0);
/// let d = a + c;
/// assert!((d.real() - 5.14).abs() < 1e-10);
/// ```
pub struct Scalar {
    native: NativeTensor,
}

/// Backward-compatible scalar type name used across tensor4all APIs.
pub type AnyScalar = Scalar;

impl Scalar {
    fn wrap_native(native: NativeTensor) -> Result<Self> {
        if native.dims().is_empty() {
            Ok(Self { native })
        } else {
            Err(anyhow!(
                "Scalar requires a rank-0 tensor, got dims {:?}",
                native.dims()
            ))
        }
    }

    fn value(&self) -> ScalarValue {
        scalar_value_from_native(&self.native)
    }

    pub(crate) fn from_native(value: NativeTensor) -> Result<Self> {
        Self::wrap_native(value)
    }

    pub(crate) fn as_native(&self) -> &NativeTensor {
        &self.native
    }

    /// Creates a scalar from any supported public tensor element type.
    pub fn from_value<T: TensorElement>(value: T) -> Self {
        let native = T::scalar_native_tensor(value)
            .unwrap_or_else(|e| panic!("failed to build scalar native tensor: {e}"));
        Self { native }
    }

    /// Creates a real scalar from an `f64` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::from_real(2.5);
    /// assert!((s.real() - 2.5).abs() < 1e-10);
    /// assert!(s.is_real());
    /// ```
    pub fn from_real(x: f64) -> Self {
        Self::from_value(x)
    }

    /// Creates a complex scalar from real and imaginary parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::from_complex(1.0, -1.0);
    /// assert!((s.real() - 1.0).abs() < 1e-10);
    /// assert!((s.imag() - (-1.0)).abs() < 1e-10);
    /// assert!(s.is_complex());
    /// ```
    pub fn from_complex(re: f64, im: f64) -> Self {
        Self::from_value(Complex64::new(re, im))
    }

    /// Backward-compatible constructor for a real scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::new_real(42.0);
    /// assert!((s.real() - 42.0).abs() < 1e-10);
    /// ```
    pub fn new_real(x: f64) -> Self {
        Self::from_real(x)
    }

    /// Backward-compatible constructor for a complex scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::new_complex(3.0, 4.0);
    /// assert!((s.abs() - 5.0).abs() < 1e-10); // |3 + 4i| = 5
    /// ```
    pub fn new_complex(re: f64, im: f64) -> Self {
        Self::from_complex(re, im)
    }

    /// Returns the detached primal value as a scalar.
    pub fn primal(&self) -> Self {
        Self::wrap_native(self.native.detach())
            .unwrap_or_else(|e| panic!("Scalar::primal returned a non-scalar tensor: {e}"))
    }

    /// Returns whether the scalar participates in reverse-mode AD.
    pub fn requires_grad(&self) -> bool {
        self.native.requires_grad()
    }

    /// Enables or disables reverse-mode gradient tracking.
    pub fn set_requires_grad(&mut self, enabled: bool) -> Result<()> {
        let placeholder = rank0_real_tensor(0.0);
        let native = std::mem::replace(&mut self.native, placeholder);
        self.native = native.with_requires_grad(enabled);
        Ok(())
    }

    /// Returns accumulated reverse-mode gradient when available.
    pub fn grad(&self) -> Option<Self> {
        self.native.grad().ok().flatten().map(|native| {
            Self::wrap_native(native)
                .unwrap_or_else(|e| panic!("Scalar::grad returned a non-scalar tensor: {e}"))
        })
    }

    /// Clears accumulated reverse-mode gradients.
    pub fn zero_grad(&self) -> Result<()> {
        self.native
            .zero_grad()
            .map_err(|e| anyhow!("Scalar::zero_grad failed: {e}"))
    }

    /// Accumulates reverse-mode gradients into `inputs`.
    pub fn backward(&self, grad_output: Option<&Self>, inputs: &[&Self]) -> Result<()> {
        let _ = inputs;
        with_default_runtime("backward", || match grad_output {
            Some(seed) => self
                .native
                .backward_with_seed(seed.as_native())
                .map_err(|e| anyhow!("Scalar::backward failed: {e}")),
            None => self
                .native
                .backward()
                .map_err(|e| anyhow!("Scalar::backward failed: {e}")),
        })
    }

    /// Returns the real part while intentionally dropping AD metadata.
    pub fn real(&self) -> f64 {
        self.value().real()
    }

    /// Returns the imaginary part while intentionally dropping AD metadata.
    pub fn imag(&self) -> f64 {
        self.value().imag()
    }

    /// Returns the absolute value while intentionally dropping AD metadata.
    pub fn abs(&self) -> f64 {
        self.value().abs()
    }

    /// Returns true when the scalar is complex.
    pub fn is_complex(&self) -> bool {
        self.value().is_complex()
    }

    /// Returns true when the scalar is real.
    pub fn is_real(&self) -> bool {
        !self.is_complex()
    }

    /// Returns true when the scalar is zero.
    pub fn is_zero(&self) -> bool {
        self.value().is_zero()
    }

    /// Returns the underlying value as `f64` when real.
    pub fn as_f64(&self) -> Option<f64> {
        match self.value() {
            ScalarValue::F32(value) => Some(value as f64),
            ScalarValue::F64(value) => Some(value),
            ScalarValue::C32(_) | ScalarValue::C64(_) => None,
        }
    }

    /// Returns the underlying value as `Complex64` when complex.
    pub fn as_c64(&self) -> Option<Complex64> {
        match self.value() {
            ScalarValue::F32(_) | ScalarValue::F64(_) => None,
            ScalarValue::C32(value) => Some(Complex64::new(value.re as f64, value.im as f64)),
            ScalarValue::C64(value) => Some(value),
        }
    }

    /// Returns the complex conjugate.
    pub fn conj(&self) -> Self {
        match self.value() {
            ScalarValue::F32(value) => Self::from_value(value),
            ScalarValue::F64(value) => Self::from_value(value),
            ScalarValue::C32(value) => Self::from_value(value.conj()),
            ScalarValue::C64(value) => Self::from_value(value.conj()),
        }
    }

    /// Returns the real part as a scalar, preserving scalar semantics.
    pub fn real_part(&self) -> Self {
        Self::from_real(self.real())
    }

    /// Returns the imaginary part as a scalar, preserving scalar semantics.
    pub fn imag_part(&self) -> Self {
        Self::from_real(self.imag())
    }

    /// Compose a complex scalar from real-valued parts.
    pub fn compose_complex(real: Self, imag: Self) -> Result<Self> {
        if !real.is_real() || !imag.is_real() {
            return Err(anyhow!(
                "compose_complex requires real-valued inputs, got real={:?}, imag={:?}",
                real.native.scalar_type(),
                imag.native.scalar_type()
            ));
        }
        Ok(Self::from_complex(real.real(), imag.real()))
    }

    /// Square root, preserving AD metadata.
    pub fn sqrt(&self) -> Self {
        if self.is_complex() || self.real() < 0.0 {
            let value = self.value().into_complex().sqrt();
            if value.im == 0.0 {
                Self::from_real(value.re)
            } else {
                Self::from_value(value)
            }
        } else {
            Self::from_real(self.real().sqrt())
        }
    }

    /// Real exponent power, preserving AD metadata.
    pub fn powf(&self, exponent: f64) -> Self {
        let needs_complex_promotion =
            self.is_complex() || (self.real() < 0.0 && exponent.fract() != 0.0);
        if needs_complex_promotion {
            let value = self.value().into_complex().powf(exponent);
            if value.im == 0.0 {
                Self::from_real(value.re)
            } else {
                Self::from_value(value)
            }
        } else {
            Self::from_real(self.real().powf(exponent))
        }
    }

    /// Integer exponent power, preserving AD metadata.
    pub fn powi(&self, exponent: i32) -> Self {
        self.powf(exponent as f64)
    }
}

impl SumFromStorage for Scalar {
    fn sum_from_storage(storage: &Storage) -> Self {
        match scalar_value_from_storage(storage) {
            ScalarValue::F32(value) => Self::from_value(value),
            ScalarValue::F64(value) => Self::from_value(value),
            ScalarValue::C32(value) => Self::from_value(value),
            ScalarValue::C64(value) => Self::from_value(value),
        }
    }
}

impl From<f32> for Scalar {
    fn from(value: f32) -> Self {
        Self::from_value(value)
    }
}

impl From<f64> for Scalar {
    fn from(value: f64) -> Self {
        Self::from_value(value)
    }
}

impl From<Complex32> for Scalar {
    fn from(value: Complex32) -> Self {
        Self::from_value(value)
    }
}

impl From<Complex64> for Scalar {
    fn from(value: Complex64) -> Self {
        Self::from_value(value)
    }
}

impl TryFrom<Scalar> for f64 {
    type Error = &'static str;

    fn try_from(value: Scalar) -> std::result::Result<Self, Self::Error> {
        match value.value() {
            ScalarValue::F32(real) => Ok(real as f64),
            ScalarValue::F64(real) => Ok(real),
            ScalarValue::C32(_) | ScalarValue::C64(_) => {
                Err("cannot convert complex scalar to f64")
            }
        }
    }
}

impl From<Scalar> for Complex64 {
    fn from(value: Scalar) -> Self {
        value.value().into_complex()
    }
}

impl Add for Scalar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self.value(), rhs.value()) {
            (ScalarValue::F32(lhs), ScalarValue::F32(rhs)) => Self::from_value(lhs + rhs),
            (lhs, rhs) if lhs.is_complex() || rhs.is_complex() => {
                Self::from_value(lhs.into_complex() + rhs.into_complex())
            }
            (lhs, rhs) => Self::from_real(lhs.real() + rhs.real()),
        }
    }
}

impl Sub for Scalar {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Mul for Scalar {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self.value(), rhs.value()) {
            (ScalarValue::F32(lhs), ScalarValue::F32(rhs)) => Self::from_value(lhs * rhs),
            (lhs, rhs) if lhs.is_complex() || rhs.is_complex() => {
                Self::from_value(lhs.into_complex() * rhs.into_complex())
            }
            (lhs, rhs) => Self::from_real(lhs.real() * rhs.real()),
        }
    }
}

impl Div for Scalar {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        match (self.value(), rhs.value()) {
            (ScalarValue::F32(lhs), ScalarValue::F32(rhs)) => Self::from_value(lhs / rhs),
            (lhs, rhs) if lhs.is_complex() || rhs.is_complex() => {
                Self::from_value(lhs.into_complex() / rhs.into_complex())
            }
            (lhs, rhs) => Self::from_real(lhs.real() / rhs.real()),
        }
    }
}

impl Neg for Scalar {
    type Output = Self;

    fn neg(self) -> Self::Output {
        scalar_tensor_result("neg", neg_native(&self.native))
    }
}

impl Mul<Scalar> for f64 {
    type Output = Scalar;

    fn mul(self, rhs: Scalar) -> Self::Output {
        Scalar::from_real(self) * rhs
    }
}

impl Mul<Scalar> for Complex64 {
    type Output = Scalar;

    fn mul(self, rhs: Scalar) -> Self::Output {
        Scalar::from(self) * rhs
    }
}

impl Div<Scalar> for Complex64 {
    type Output = Scalar;

    fn div(self, rhs: Scalar) -> Self::Output {
        Scalar::from(self) / rhs
    }
}

impl Default for Scalar {
    fn default() -> Self {
        Self::zero()
    }
}

impl Zero for Scalar {
    fn zero() -> Self {
        Self::from_real(0.0)
    }

    fn is_zero(&self) -> bool {
        Scalar::is_zero(self)
    }
}

impl One for Scalar {
    fn one() -> Self {
        Self::from_real(1.0)
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.native.scalar_type() == other.native.scalar_type() && self.value() == other.value()
    }
}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self.value(), other.value()) {
            (ScalarValue::F32(lhs), ScalarValue::F32(rhs)) => lhs.partial_cmp(&rhs),
            (ScalarValue::F32(lhs), ScalarValue::F64(rhs)) => (lhs as f64).partial_cmp(&rhs),
            (ScalarValue::F64(lhs), ScalarValue::F32(rhs)) => lhs.partial_cmp(&(rhs as f64)),
            (ScalarValue::F64(lhs), ScalarValue::F64(rhs)) => lhs.partial_cmp(&rhs),
            _ => None,
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value() {
            ScalarValue::F32(value) => value.fmt(f),
            ScalarValue::F64(value) => value.fmt(f),
            ScalarValue::C32(value) => value.fmt(f),
            ScalarValue::C64(value) => value.fmt(f),
        }
    }
}

impl fmt::Debug for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scalar")
            .field("scalar_type", &self.native.scalar_type())
            .field("value", &self.value())
            .finish()
    }
}

impl Clone for Scalar {
    fn clone(&self) -> Self {
        self.primal()
    }
}

#[cfg(test)]
mod tests;
