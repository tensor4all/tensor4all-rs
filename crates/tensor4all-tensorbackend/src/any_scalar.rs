use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use anyhow::{anyhow, Result};
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tenferro::{AdMode, ScalarType, ScalarValue as TfScalarValue, Tensor as NativeTensor};

use crate::storage::{Storage, SumFromStorage};
use crate::tensor_element::TensorElement;
use crate::{tangent_native_tensor, tenferro_bridge::with_default_runtime};

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
    match native
        .try_scalar_value()
        .unwrap_or_else(|e| panic!("failed to read scalar tensor value: {e}"))
    {
        TfScalarValue::F32(value) => ScalarValue::F32(value),
        TfScalarValue::F64(value) => ScalarValue::F64(value),
        TfScalarValue::C32(value) => ScalarValue::C32(value),
        TfScalarValue::C64(value) => ScalarValue::C64(value),
    }
}

fn scalar_tensor_result(op: &'static str, native: Result<NativeTensor>) -> Scalar {
    match native {
        Ok(native) => Scalar::wrap_native(native)
            .unwrap_or_else(|e| panic!("Scalar::{op} returned a non-scalar tensor: {e}")),
        Err(err) => panic!("Scalar::{op} failed: {err}"),
    }
}

fn scalar_native_op(
    op: &'static str,
    f: impl FnOnce() -> tenferro::Result<NativeTensor>,
) -> Result<NativeTensor> {
    with_default_runtime(op, || f().map_err(|e| anyhow!("{e}")))
}

fn neg_native(native: &NativeTensor) -> Result<NativeTensor> {
    scalar_native_op("neg", || native.scale(&rank0_real_tensor(-1.0)))
}

fn promote_scalar_native(native: &NativeTensor, target: ScalarType) -> Result<NativeTensor> {
    scalar_native_op("to_scalar_type", || native.to_scalar_type(target))
}

/// Dynamic scalar used across tensor4all backends.
///
/// This is a tensor4all-owned rank-0 wrapper over tenferro's dynamic tensor.
#[derive(Clone)]
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
    pub fn from_real(x: f64) -> Self {
        Self::from_value(x)
    }

    /// Creates a complex scalar from real and imaginary parts.
    pub fn from_complex(re: f64, im: f64) -> Self {
        Self::from_value(Complex64::new(re, im))
    }

    /// Backward-compatible constructor for a real scalar.
    pub fn new_real(x: f64) -> Self {
        Self::from_real(x)
    }

    /// Backward-compatible constructor for a complex scalar.
    pub fn new_complex(re: f64, im: f64) -> Self {
        Self::from_complex(re, im)
    }

    /// Returns the upstream AD mode.
    pub fn mode(&self) -> AdMode {
        self.native.mode()
    }

    /// Returns the detached primal value as a scalar.
    pub fn primal(&self) -> Self {
        Self::wrap_native(self.native.detach())
            .unwrap_or_else(|e| panic!("Scalar::primal returned a non-scalar tensor: {e}"))
    }

    /// Returns the detached forward tangent when present.
    pub fn tangent(&self) -> Option<Self> {
        tangent_native_tensor(&self.native).and_then(|native| Self::from_native(native).ok())
    }

    /// Returns whether the scalar participates in reverse-mode AD.
    pub fn requires_grad(&self) -> bool {
        self.native.requires_grad()
    }

    /// Enables or disables reverse-mode gradient tracking.
    pub fn set_requires_grad(&mut self, enabled: bool) -> Result<()> {
        self.native
            .set_requires_grad(enabled)
            .map_err(|e| anyhow!("Scalar::set_requires_grad failed: {e}"))
    }

    /// Returns accumulated reverse-mode gradient when available.
    pub fn grad(&self) -> Option<Self> {
        self.native.grad().map(|native| {
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
        let grad_output_native = grad_output.map(|value| value.as_native());
        let input_native: Vec<&NativeTensor> =
            inputs.iter().map(|value| value.as_native()).collect();
        with_default_runtime("backward", || {
            self.native
                .backward(
                    grad_output_native,
                    &input_native,
                    tenferro::BackwardOptions::default(),
                )
                .map_err(|e| anyhow!("Scalar::backward failed: {e}"))
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
        Self::wrap_native(self.native.conj())
            .unwrap_or_else(|e| panic!("Scalar::conj returned a non-scalar tensor: {e}"))
    }

    /// Returns the real part as a scalar, preserving scalar semantics.
    pub fn real_part(&self) -> Self {
        scalar_tensor_result(
            "real_part",
            scalar_native_op("real_part", || self.native.real_part()),
        )
    }

    /// Returns the imaginary part as a scalar, preserving scalar semantics.
    pub fn imag_part(&self) -> Self {
        scalar_tensor_result(
            "imag_part",
            scalar_native_op("imag_part", || self.native.imag_part()),
        )
    }

    /// Compose a complex scalar from real-valued parts.
    pub fn compose_complex(real: Self, imag: Self) -> Result<Self> {
        let native = scalar_native_op("compose_complex", || {
            NativeTensor::compose_complex(real.native, imag.native)
        })
        .map_err(|e| anyhow!("Scalar::compose_complex failed: {e}"))?;
        Self::wrap_native(native)
    }

    /// Square root, preserving AD metadata.
    pub fn sqrt(&self) -> Self {
        let native = if self.is_complex() || self.real() < 0.0 {
            let promoted = promote_scalar_native(&self.native, ScalarType::C64);
            promoted.and_then(|value| scalar_native_op("sqrt", || value.sqrt()))
        } else {
            scalar_native_op("sqrt", || self.native.sqrt())
        };
        scalar_tensor_result("sqrt", native)
    }

    /// Real exponent power, preserving AD metadata.
    pub fn powf(&self, exponent: f64) -> Self {
        let needs_complex_promotion =
            self.is_complex() || (self.real() < 0.0 && exponent.fract() != 0.0);
        let native = if needs_complex_promotion {
            let promoted = promote_scalar_native(&self.native, ScalarType::C64);
            promoted.and_then(|value| {
                scalar_native_op("powf", || value.pow(&rank0_real_tensor(exponent)))
            })
        } else {
            scalar_native_op("powf", || self.native.pow(&rank0_real_tensor(exponent)))
        };
        scalar_tensor_result("powf", native)
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
        scalar_tensor_result(
            "add",
            scalar_native_op("add", || self.native.add(&rhs.native)),
        )
    }
}

impl Sub for Scalar {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let neg_rhs = neg_native(&rhs.native)
            .unwrap_or_else(|err| panic!("Scalar::sub failed while negating rhs: {err}"));
        scalar_tensor_result("sub", scalar_native_op("sub", || self.native.add(&neg_rhs)))
    }
}

impl Mul for Scalar {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        scalar_tensor_result(
            "mul",
            scalar_native_op("mul", || self.native.scale(&rhs.native)),
        )
    }
}

impl Div for Scalar {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        scalar_tensor_result(
            "div",
            scalar_native_op("div", || self.native.div_scalar(&rhs.native)),
        )
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
        self.mode() == other.mode()
            && self.native.scalar_type() == other.native.scalar_type()
            && self.value() == other.value()
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
            .field("mode", &self.mode())
            .field("scalar_type", &self.native.scalar_type())
            .field("value", &self.value())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_from_value_supports_all_supported_element_types() {
        let f32_scalar = Scalar::from_value(1.25_f32);
        let f64_scalar = Scalar::from_value(-2.5_f64);
        let c32_scalar = Scalar::from_value(Complex32::new(3.0, -0.5));
        let c64_scalar = Scalar::from_value(Complex64::new(-1.0, 2.0));

        assert_eq!(f32_scalar.real(), 1.25);
        assert_eq!(f64_scalar.real(), -2.5);
        assert_eq!(c32_scalar.real(), 3.0);
        assert_eq!(c32_scalar.imag(), -0.5);
        assert_eq!(Complex64::from(c64_scalar), Complex64::new(-1.0, 2.0));
    }

    #[test]
    fn any_scalar_sum_from_real_storage_stays_real() {
        let dense = Storage::from_dense_f64_col_major(vec![1.0, -2.5], &[2]).unwrap();
        let diag = Storage::from_diag_f64_col_major(vec![3.0, 4.5], 2).unwrap();

        let dense_sum = AnyScalar::sum_from_storage(&dense);
        let diag_sum = AnyScalar::sum_from_storage(&diag);

        assert!(dense_sum.is_real());
        assert_eq!(dense_sum.real(), -1.5);
        assert!(diag_sum.is_real());
        assert_eq!(diag_sum.real(), 7.5);
    }

    #[test]
    fn any_scalar_sum_from_complex_storage_stays_complex() {
        let dense = Storage::from_dense_c64_col_major(
            vec![Complex64::new(1.0, -1.0), Complex64::new(-0.5, 2.0)],
            &[2],
        )
        .unwrap();

        let sum = AnyScalar::sum_from_storage(&dense);
        let sum_c64: Complex64 = sum.into();
        assert_eq!(sum_c64, Complex64::new(0.5, 1.0));
    }

    #[test]
    fn scalar_arithmetic_uses_runtime_bridge() {
        let sum = AnyScalar::from_real(1.5) + AnyScalar::from_real(2.0);
        let diff = AnyScalar::from_complex(3.0, -1.0) - AnyScalar::from_real(1.0);
        let prod = AnyScalar::from_real(2.0) * AnyScalar::from_complex(0.0, 1.0);

        assert_eq!(sum.as_f64(), Some(3.5));
        assert_eq!(Complex64::from(diff), Complex64::new(2.0, -1.0));
        assert_eq!(Complex64::from(prod), Complex64::new(0.0, 2.0));
    }
}
