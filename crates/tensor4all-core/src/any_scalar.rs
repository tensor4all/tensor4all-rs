use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use anyhow::{anyhow, Result};
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tenferro::DType;
use tensor4all_tensorbackend::AnyScalar as BackendScalar;

use crate::defaults::tensordynlen::TensorDynLen;
use crate::TensorElement;
use tensor4all_tensorbackend::{Storage, SumFromStorage};

#[derive(Clone, Copy, Debug, PartialEq)]
enum ScalarValue {
    F32(f32),
    F64(f64),
    I64(i64),
    C32(Complex32),
    C64(Complex64),
}

impl ScalarValue {
    fn real(self) -> f64 {
        match self {
            Self::F32(value) => value as f64,
            Self::F64(value) => value,
            Self::I64(value) => value as f64,
            Self::C32(value) => value.re as f64,
            Self::C64(value) => value.re,
        }
    }

    fn imag(self) -> f64 {
        match self {
            Self::F32(_) | Self::F64(_) | Self::I64(_) => 0.0,
            Self::C32(value) => value.im as f64,
            Self::C64(value) => value.im,
        }
    }

    fn abs(self) -> f64 {
        match self {
            Self::F32(value) => value.abs() as f64,
            Self::F64(value) => value.abs(),
            Self::I64(value) => value.abs() as f64,
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
            Self::I64(value) => value == 0,
            Self::C32(value) => value == Complex32::new(0.0, 0.0),
            Self::C64(value) => value == Complex64::new(0.0, 0.0),
        }
    }

    fn into_complex(self) -> Complex64 {
        match self {
            Self::F32(value) => Complex64::new(value as f64, 0.0),
            Self::F64(value) => Complex64::new(value, 0.0),
            Self::I64(value) => Complex64::new(value as f64, 0.0),
            Self::C32(value) => Complex64::new(value.re as f64, value.im as f64),
            Self::C64(value) => value,
        }
    }
}

/// Dynamic scalar compatibility wrapper for tensor4all-core.
///
/// This owns a rank-0 [`TensorDynLen`] so that scalar values can participate in
/// the same eager autodiff graph as tensors while preserving the existing
/// dynamic scalar API shape.
#[derive(Clone)]
pub struct AnyScalar {
    tensor: TensorDynLen,
}

impl AnyScalar {
    fn wrap_tensor(tensor: TensorDynLen) -> Result<Self> {
        let dims = tensor.dims();
        anyhow::ensure!(
            dims.is_empty(),
            "AnyScalar requires a rank-0 tensor, got dims {:?}",
            dims
        );
        Ok(Self { tensor })
    }

    fn from_tensor_result(tensor: Result<TensorDynLen>, op: &'static str) -> Self {
        Self::wrap_tensor(
            tensor
                .unwrap_or_else(|e| panic!("AnyScalar::{op} returned invalid scalar tensor: {e}")),
        )
        .unwrap_or_else(|e| panic!("AnyScalar::{op} returned non-scalar tensor: {e}"))
    }

    fn from_eager_binary<E>(
        lhs: &Self,
        rhs: &Self,
        op: &'static str,
        f: impl FnOnce(
            &tenferro::EagerTensor<tenferro::CpuBackend>,
            &tenferro::EagerTensor<tenferro::CpuBackend>,
        ) -> std::result::Result<tenferro::EagerTensor<tenferro::CpuBackend>, E>,
    ) -> Self
    where
        E: fmt::Display,
    {
        let result = f(lhs.tensor.as_inner(), rhs.tensor.as_inner())
            .unwrap_or_else(|e| panic!("AnyScalar::{op} failed: {e}"));
        Self::from_tensor_result(TensorDynLen::from_inner(vec![], result), op)
    }

    fn from_eager_unary<E>(
        input: &Self,
        op: &'static str,
        f: impl FnOnce(
            &tenferro::EagerTensor<tenferro::CpuBackend>,
        ) -> std::result::Result<tenferro::EagerTensor<tenferro::CpuBackend>, E>,
    ) -> Self
    where
        E: fmt::Display,
    {
        let result =
            f(input.tensor.as_inner()).unwrap_or_else(|e| panic!("AnyScalar::{op} failed: {e}"));
        Self::from_tensor_result(TensorDynLen::from_inner(vec![], result), op)
    }

    fn value(&self) -> ScalarValue {
        match self.tensor.as_native().dtype() {
            DType::F32 => ScalarValue::F32(
                self.tensor
                    .to_vec::<f32>()
                    .unwrap_or_else(|e| panic!("failed to read f32 scalar value: {e}"))[0],
            ),
            DType::F64 => ScalarValue::F64(
                self.tensor
                    .to_vec::<f64>()
                    .unwrap_or_else(|e| panic!("failed to read f64 scalar value: {e}"))[0],
            ),
            DType::I64 => ScalarValue::I64(
                self.tensor
                    .as_native()
                    .as_slice::<i64>()
                    .and_then(|values| values.first().copied())
                    .unwrap_or_else(|| panic!("failed to read i64 scalar value")),
            ),
            DType::C32 => ScalarValue::C32(
                self.tensor
                    .to_vec::<Complex32>()
                    .unwrap_or_else(|e| panic!("failed to read c32 scalar value: {e}"))[0],
            ),
            DType::C64 => ScalarValue::C64(
                self.tensor
                    .to_vec::<Complex64>()
                    .unwrap_or_else(|e| panic!("failed to read c64 scalar value: {e}"))[0],
            ),
        }
    }

    fn from_backend_scalar(value: BackendScalar) -> Self {
        if let Some(value) = value.as_c64() {
            Self::from_value(value)
        } else {
            Self::from_value(value.real())
        }
    }

    pub(crate) fn from_tensor_unchecked(tensor: TensorDynLen) -> Self {
        Self::wrap_tensor(tensor).unwrap_or_else(|e| panic!("AnyScalar tensor wrapper failed: {e}"))
    }

    pub(crate) fn as_tensor(&self) -> &TensorDynLen {
        &self.tensor
    }

    /// Creates an `AnyScalar` from a tensor element.
    ///
    /// Use this when you already have a scalar value that implements
    /// [`TensorElement`] and want to lift it into the dynamic scalar wrapper.
    ///
    /// # Arguments
    ///
    /// * `value` - The scalar value to wrap.
    ///
    /// # Returns
    ///
    /// A rank-0 `AnyScalar` containing `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::from_value(3.0f64);
    /// assert_eq!(scalar.real(), 3.0);
    /// assert!(scalar.is_real());
    /// ```
    pub fn from_value<T: TensorElement>(value: T) -> Self {
        Self::from_tensor_result(TensorDynLen::scalar(value), "from_value")
    }

    /// Creates a real-valued `AnyScalar`.
    ///
    /// This is a convenience wrapper around [`AnyScalar::from_value`].
    ///
    /// # Arguments
    ///
    /// * `x` - The real scalar value to wrap.
    ///
    /// # Returns
    ///
    /// A rank-0 `AnyScalar` with real dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::from_real(1.25);
    /// assert_eq!(scalar.as_f64(), Some(1.25));
    /// assert!(scalar.is_real());
    /// ```
    pub fn from_real(x: f64) -> Self {
        Self::from_value(x)
    }

    /// Creates a complex-valued `AnyScalar`.
    ///
    /// This is a convenience wrapper around [`AnyScalar::from_value`].
    ///
    /// # Arguments
    ///
    /// * `re` - The real part of the complex value.
    /// * `im` - The imaginary part of the complex value.
    ///
    /// # Returns
    ///
    /// A rank-0 `AnyScalar` containing the requested complex number.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::from_complex(1.0, -2.0);
    /// assert_eq!(scalar.as_c64().map(|z| (z.re, z.im)), Some((1.0, -2.0)));
    /// assert!(scalar.is_complex());
    /// ```
    pub fn from_complex(re: f64, im: f64) -> Self {
        Self::from_value(Complex64::new(re, im))
    }

    /// Creates a real-valued `AnyScalar`.
    ///
    /// This is an alias for [`AnyScalar::from_real`].
    ///
    /// # Arguments
    ///
    /// * `x` - The real scalar value to wrap.
    ///
    /// # Returns
    ///
    /// A rank-0 `AnyScalar` with real dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_real(2.5);
    /// assert_eq!(scalar.real(), 2.5);
    /// assert!(scalar.is_real());
    /// ```
    pub fn new_real(x: f64) -> Self {
        Self::from_real(x)
    }

    /// Creates a complex-valued `AnyScalar`.
    ///
    /// This is an alias for [`AnyScalar::from_complex`].
    ///
    /// # Arguments
    ///
    /// * `re` - The real part of the complex value.
    /// * `im` - The imaginary part of the complex value.
    ///
    /// # Returns
    ///
    /// A rank-0 `AnyScalar` containing the requested complex number.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(2.0, 3.0);
    /// assert_eq!(scalar.as_c64().map(|z| (z.re, z.im)), Some((2.0, 3.0)));
    /// assert!(scalar.is_complex());
    /// ```
    pub fn new_complex(re: f64, im: f64) -> Self {
        Self::from_complex(re, im)
    }

    /// Returns the detached primal value of this scalar.
    ///
    /// This is an alias for [`AnyScalar::detach`].
    ///
    /// # Returns
    ///
    /// A scalar with the same value and no gradient tracking.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let primal = AnyScalar::new_real(5.0).enable_grad().primal();
    /// assert_eq!(primal.real(), 5.0);
    /// assert!(!primal.tracks_grad());
    /// ```
    pub fn primal(&self) -> Self {
        self.detach()
    }

    /// Enables gradient tracking for this scalar.
    ///
    /// # Returns
    ///
    /// A new scalar that shares the same value but participates in autodiff.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_real(2.0).enable_grad();
    /// assert!(scalar.tracks_grad());
    /// ```
    pub fn enable_grad(self) -> Self {
        Self::from_tensor_unchecked(self.tensor.enable_grad())
    }

    /// Returns whether this scalar tracks gradients.
    ///
    /// # Returns
    ///
    /// `true` when the scalar participates in autodiff and can accumulate a
    /// gradient, otherwise `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_real(1.0);
    /// assert!(!scalar.tracks_grad());
    /// ```
    pub fn tracks_grad(&self) -> bool {
        self.tensor.tracks_grad()
    }

    /// Returns the stored gradient, if any.
    ///
    /// # Returns
    ///
    /// `Ok(Some(_))` when a gradient is available, `Ok(None)` when no gradient
    /// has been recorded, or an error if the backend cannot read it.
    ///
    /// # Errors
    ///
    /// Propagates autodiff or tensor access failures from the underlying
    /// tensor runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let x = AnyScalar::new_real(2.0).enable_grad();
    /// let y = &x * &x;
    /// y.backward().unwrap();
    ///
    /// let grad = x.grad().unwrap().unwrap();
    /// assert_eq!(grad.real(), 4.0);
    /// ```
    pub fn grad(&self) -> Result<Option<Self>> {
        self.tensor
            .grad()
            .map(|maybe_grad| maybe_grad.map(Self::from_tensor_unchecked))
    }

    /// Clears the stored gradient for this scalar.
    ///
    /// # Returns
    ///
    /// `Ok(())` when the gradient buffer was cleared successfully.
    ///
    /// # Errors
    ///
    /// Propagates tensor runtime failures from the underlying autodiff state.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let x = AnyScalar::new_real(2.0).enable_grad();
    /// let y = &x * &x;
    /// y.backward().unwrap();
    /// assert!(x.grad().unwrap().is_some());
    ///
    /// x.clear_grad().unwrap();
    /// assert!(x.grad().unwrap().is_none());
    /// ```
    pub fn clear_grad(&self) -> Result<()> {
        self.tensor.clear_grad()
    }

    /// Runs reverse-mode autodiff starting from this scalar.
    ///
    /// # Returns
    ///
    /// `Ok(())` when gradients were accumulated successfully.
    ///
    /// # Errors
    ///
    /// Propagates failures from the underlying tensor autodiff engine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let x = AnyScalar::new_real(2.0).enable_grad();
    /// let y = &x * &x;
    /// y.backward().unwrap();
    ///
    /// let grad = x.grad().unwrap().unwrap();
    /// assert_eq!(grad.real(), 4.0);
    /// ```
    pub fn backward(&self) -> Result<()> {
        self.tensor.backward()
    }

    /// Returns a detached copy of this scalar.
    ///
    /// # Returns
    ///
    /// A scalar with the same value but without gradient tracking.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let detached = AnyScalar::new_real(7.0).enable_grad().detach();
    /// assert_eq!(detached.real(), 7.0);
    /// assert!(!detached.tracks_grad());
    /// ```
    pub fn detach(&self) -> Self {
        Self::from_tensor_unchecked(self.tensor.detach())
    }

    /// Returns the real part of this scalar.
    ///
    /// # Returns
    ///
    /// The real component as an `f64`, regardless of the underlying storage
    /// type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(3.0, -4.0);
    /// assert_eq!(scalar.real(), 3.0);
    /// ```
    pub fn real(&self) -> f64 {
        self.value().real()
    }

    /// Returns the imaginary part of this scalar.
    ///
    /// # Returns
    ///
    /// The imaginary component as an `f64`. Real-valued scalars return `0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(3.0, -4.0);
    /// assert_eq!(scalar.imag(), -4.0);
    /// ```
    pub fn imag(&self) -> f64 {
        self.value().imag()
    }

    /// Returns the magnitude of this scalar.
    ///
    /// # Returns
    ///
    /// The absolute value for real scalars or the complex norm for complex
    /// scalars.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(3.0, -4.0);
    /// assert_eq!(scalar.abs(), 5.0);
    /// ```
    pub fn abs(&self) -> f64 {
        self.value().abs()
    }

    /// Returns whether this scalar is complex-valued.
    ///
    /// # Returns
    ///
    /// `true` for complex dtypes and `false` for real or integer dtypes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// assert!(AnyScalar::new_complex(1.0, 2.0).is_complex());
    /// assert!(!AnyScalar::new_real(1.0).is_complex());
    /// ```
    pub fn is_complex(&self) -> bool {
        self.value().is_complex()
    }

    /// Returns whether this scalar is real-valued.
    ///
    /// # Returns
    ///
    /// `true` when the scalar is not complex-valued.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// assert!(AnyScalar::new_real(1.0).is_real());
    /// assert!(!AnyScalar::new_complex(1.0, 2.0).is_real());
    /// ```
    pub fn is_real(&self) -> bool {
        !self.is_complex()
    }

    /// Returns whether this scalar is exactly zero.
    ///
    /// # Returns
    ///
    /// `true` for exact zeros and `false` for any nonzero value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// assert!(AnyScalar::new_real(0.0).is_zero());
    /// assert!(!AnyScalar::new_complex(0.0, 1.0).is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        self.value().is_zero()
    }

    /// Returns this scalar as an `f64` when it is real-valued.
    ///
    /// # Returns
    ///
    /// `Some(value)` for real and integer scalars, or `None` for complex
    /// scalars.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// assert_eq!(AnyScalar::new_real(2.5).as_f64(), Some(2.5));
    /// assert_eq!(AnyScalar::new_complex(2.5, 1.0).as_f64(), None);
    /// ```
    pub fn as_f64(&self) -> Option<f64> {
        match self.value() {
            ScalarValue::F32(value) => Some(value as f64),
            ScalarValue::F64(value) => Some(value),
            ScalarValue::I64(value) => Some(value as f64),
            ScalarValue::C32(_) | ScalarValue::C64(_) => None,
        }
    }

    /// Returns this scalar as a `Complex64` when it is complex-valued.
    ///
    /// # Returns
    ///
    /// `Some(value)` for complex scalars or `None` for real and integer
    /// scalars.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(2.5, 1.0);
    /// assert_eq!(scalar.as_c64().map(|z| (z.re, z.im)), Some((2.5, 1.0)));
    /// assert_eq!(AnyScalar::new_real(2.5).as_c64(), None);
    /// ```
    pub fn as_c64(&self) -> Option<Complex64> {
        match self.value() {
            ScalarValue::F32(_) | ScalarValue::F64(_) | ScalarValue::I64(_) => None,
            ScalarValue::C32(value) => Some(Complex64::new(value.re as f64, value.im as f64)),
            ScalarValue::C64(value) => Some(value),
        }
    }

    /// Returns the complex conjugate of this scalar.
    ///
    /// # Returns
    ///
    /// The conjugated scalar. Real-valued inputs are returned unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(3.0, -4.0).conj();
    /// assert_eq!(scalar.as_c64().map(|z| (z.re, z.im)), Some((3.0, 4.0)));
    /// ```
    pub fn conj(&self) -> Self {
        Self::from_eager_unary(self, "conj", |tensor| tensor.conj())
    }

    /// Returns the real part as a real-valued scalar.
    ///
    /// # Returns
    ///
    /// A real-valued scalar containing the real component of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(3.0, -4.0).real_part();
    /// assert_eq!(scalar.real(), 3.0);
    /// assert!(scalar.is_real());
    /// ```
    pub fn real_part(&self) -> Self {
        Self::from_real(self.real())
    }

    /// Returns the imaginary part as a real-valued scalar.
    ///
    /// # Returns
    ///
    /// A real-valued scalar containing the imaginary component of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(3.0, -4.0).imag_part();
    /// assert_eq!(scalar.real(), -4.0);
    /// assert!(scalar.is_real());
    /// ```
    pub fn imag_part(&self) -> Self {
        Self::from_real(self.imag())
    }

    /// Combines two real-valued scalars into a complex scalar.
    ///
    /// # Arguments
    ///
    /// * `real` - The real component.
    /// * `imag` - The imaginary component.
    ///
    /// # Returns
    ///
    /// A complex `AnyScalar` whose real and imaginary parts come from the
    /// inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if either input is not real-valued.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::compose_complex(
    ///     AnyScalar::new_real(3.0),
    ///     AnyScalar::new_real(-4.0),
    /// )
    /// .unwrap();
    /// assert_eq!(scalar.as_c64().map(|z| (z.re, z.im)), Some((3.0, -4.0)));
    /// ```
    pub fn compose_complex(real: Self, imag: Self) -> Result<Self> {
        if !real.is_real() || !imag.is_real() {
            return Err(anyhow!(
                "compose_complex requires real-valued inputs, got real={:?}, imag={:?}",
                real.tensor.as_native().dtype(),
                imag.tensor.as_native().dtype()
            ));
        }
        let imag_term = &imag * &Self::new_complex(0.0, 1.0);
        Ok(&real + &imag_term)
    }

    /// Returns the square root of this scalar.
    ///
    /// # Returns
    ///
    /// The principal square root. Negative real inputs and complex inputs use
    /// complex arithmetic.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_real(9.0).sqrt();
    /// assert_eq!(scalar.real(), 3.0);
    /// assert!(scalar.is_real());
    /// ```
    pub fn sqrt(&self) -> Self {
        if self.is_complex() || self.real() < 0.0 {
            Self::from_backend_scalar(self.to_backend_scalar().sqrt())
        } else {
            Self::from_eager_unary(self, "sqrt", |tensor| tensor.sqrt())
        }
    }

    /// Raises this scalar to a floating-point power.
    ///
    /// # Arguments
    ///
    /// * `exponent` - The exponent to apply.
    ///
    /// # Returns
    ///
    /// The value of `self^exponent`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_real(2.0).powf(3.0);
    /// assert_eq!(scalar.real(), 8.0);
    /// ```
    pub fn powf(&self, exponent: f64) -> Self {
        Self::from_backend_scalar(self.to_backend_scalar().powf(exponent))
    }

    /// Raises this scalar to an integer power.
    ///
    /// # Arguments
    ///
    /// * `exponent` - The integer exponent to apply. Negative exponents return
    ///   the reciprocal power.
    ///
    /// # Returns
    ///
    /// The value of `self^exponent`. Zero exponents return `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// assert_eq!(AnyScalar::new_real(2.0).powi(3).real(), 8.0);
    /// assert_eq!(AnyScalar::new_real(2.0).powi(-1).real(), 0.5);
    /// ```
    pub fn powi(&self, exponent: i32) -> Self {
        if exponent == 0 {
            return Self::one();
        }

        let mut base = self.clone();
        let mut power = exponent.unsigned_abs();
        let mut acc = Self::one();

        while power > 0 {
            if power % 2 == 1 {
                acc = &acc * &base;
            }
            power /= 2;
            if power > 0 {
                base = &base * &base;
            }
        }

        if exponent < 0 {
            Self::one() / acc
        } else {
            acc
        }
    }

    pub(crate) fn to_backend_scalar(&self) -> BackendScalar {
        match self.value() {
            ScalarValue::F32(value) => BackendScalar::from_value(value),
            ScalarValue::F64(value) => BackendScalar::from_value(value),
            ScalarValue::I64(value) => BackendScalar::from_value(value as f64),
            ScalarValue::C32(value) => BackendScalar::from_value(value),
            ScalarValue::C64(value) => BackendScalar::from_value(value),
        }
    }
}

impl SumFromStorage for AnyScalar {
    fn sum_from_storage(storage: &Storage) -> Self {
        Self::from_backend_scalar(BackendScalar::sum_from_storage(storage))
    }
}

impl From<f32> for AnyScalar {
    fn from(value: f32) -> Self {
        Self::from_value(value)
    }
}

impl From<f64> for AnyScalar {
    fn from(value: f64) -> Self {
        Self::from_value(value)
    }
}

impl From<Complex32> for AnyScalar {
    fn from(value: Complex32) -> Self {
        Self::from_value(value)
    }
}

impl From<Complex64> for AnyScalar {
    fn from(value: Complex64) -> Self {
        Self::from_value(value)
    }
}

impl TryFrom<AnyScalar> for f64 {
    type Error = &'static str;

    fn try_from(value: AnyScalar) -> std::result::Result<Self, Self::Error> {
        value.as_f64().ok_or("cannot convert complex scalar to f64")
    }
}

impl From<AnyScalar> for Complex64 {
    fn from(value: AnyScalar) -> Self {
        value.value().into_complex()
    }
}

impl Add<&AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn add(self, rhs: &AnyScalar) -> Self::Output {
        AnyScalar::from_eager_binary(self, rhs, "add", |lhs, rhs| lhs.add(rhs))
    }
}

impl Add<AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn add(self, rhs: AnyScalar) -> Self::Output {
        Add::add(&self, &rhs)
    }
}

impl Add<AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn add(self, rhs: AnyScalar) -> Self::Output {
        Add::add(self, &rhs)
    }
}

impl Add<&AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn add(self, rhs: &AnyScalar) -> Self::Output {
        Add::add(&self, rhs)
    }
}

impl Sub<&AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn sub(self, rhs: &AnyScalar) -> Self::Output {
        Add::add(self, &Neg::neg(rhs))
    }
}

impl Sub<AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn sub(self, rhs: AnyScalar) -> Self::Output {
        Sub::sub(&self, &rhs)
    }
}

impl Sub<AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn sub(self, rhs: AnyScalar) -> Self::Output {
        Sub::sub(self, &rhs)
    }
}

impl Sub<&AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn sub(self, rhs: &AnyScalar) -> Self::Output {
        Sub::sub(&self, rhs)
    }
}

impl Mul<&AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn mul(self, rhs: &AnyScalar) -> Self::Output {
        AnyScalar::from_eager_binary(self, rhs, "mul", |lhs, rhs| lhs.mul(rhs))
    }
}

impl Mul<AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn mul(self, rhs: AnyScalar) -> Self::Output {
        Mul::mul(&self, &rhs)
    }
}

impl Mul<AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn mul(self, rhs: AnyScalar) -> Self::Output {
        Mul::mul(self, &rhs)
    }
}

impl Mul<&AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn mul(self, rhs: &AnyScalar) -> Self::Output {
        Mul::mul(&self, rhs)
    }
}

impl Div<&AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn div(self, rhs: &AnyScalar) -> Self::Output {
        if self.tensor.as_native().dtype() == rhs.tensor.as_native().dtype() {
            AnyScalar::from_eager_binary(self, rhs, "div", |lhs, rhs| lhs.div(rhs))
        } else {
            AnyScalar::from_backend_scalar(self.to_backend_scalar() / rhs.to_backend_scalar())
        }
    }
}

impl Div<AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn div(self, rhs: AnyScalar) -> Self::Output {
        Div::div(&self, &rhs)
    }
}

impl Div<AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn div(self, rhs: AnyScalar) -> Self::Output {
        Div::div(self, &rhs)
    }
}

impl Div<&AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn div(self, rhs: &AnyScalar) -> Self::Output {
        Div::div(&self, rhs)
    }
}

impl Neg for &AnyScalar {
    type Output = AnyScalar;

    fn neg(self) -> Self::Output {
        AnyScalar::from_eager_unary(self, "neg", |tensor| tensor.neg())
    }
}

impl Neg for AnyScalar {
    type Output = AnyScalar;

    fn neg(self) -> Self::Output {
        Neg::neg(&self)
    }
}

impl Mul<AnyScalar> for f64 {
    type Output = AnyScalar;

    fn mul(self, rhs: AnyScalar) -> Self::Output {
        AnyScalar::from_real(self) * rhs
    }
}

impl Mul<AnyScalar> for Complex64 {
    type Output = AnyScalar;

    fn mul(self, rhs: AnyScalar) -> Self::Output {
        AnyScalar::from(self) * rhs
    }
}

impl Div<AnyScalar> for Complex64 {
    type Output = AnyScalar;

    fn div(self, rhs: AnyScalar) -> Self::Output {
        AnyScalar::from(self) / rhs
    }
}

impl Default for AnyScalar {
    fn default() -> Self {
        Self::zero()
    }
}

impl Zero for AnyScalar {
    fn zero() -> Self {
        Self::from_real(0.0)
    }

    fn is_zero(&self) -> bool {
        AnyScalar::is_zero(self)
    }
}

impl One for AnyScalar {
    fn one() -> Self {
        Self::from_real(1.0)
    }
}

impl PartialEq for AnyScalar {
    fn eq(&self, other: &Self) -> bool {
        self.tensor.as_native().dtype() == other.tensor.as_native().dtype()
            && self.value() == other.value()
    }
}

impl PartialOrd for AnyScalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self.value(), other.value()) {
            (ScalarValue::F32(lhs), ScalarValue::F32(rhs)) => lhs.partial_cmp(&rhs),
            (ScalarValue::F32(lhs), ScalarValue::F64(rhs)) => (lhs as f64).partial_cmp(&rhs),
            (ScalarValue::F32(lhs), ScalarValue::I64(rhs)) => {
                (lhs as f64).partial_cmp(&(rhs as f64))
            }
            (ScalarValue::F64(lhs), ScalarValue::F32(rhs)) => lhs.partial_cmp(&(rhs as f64)),
            (ScalarValue::F64(lhs), ScalarValue::F64(rhs)) => lhs.partial_cmp(&rhs),
            (ScalarValue::F64(lhs), ScalarValue::I64(rhs)) => lhs.partial_cmp(&(rhs as f64)),
            (ScalarValue::I64(lhs), ScalarValue::F32(rhs)) => {
                (lhs as f64).partial_cmp(&(rhs as f64))
            }
            (ScalarValue::I64(lhs), ScalarValue::F64(rhs)) => (lhs as f64).partial_cmp(&rhs),
            (ScalarValue::I64(lhs), ScalarValue::I64(rhs)) => lhs.partial_cmp(&rhs),
            _ => None,
        }
    }
}

impl fmt::Display for AnyScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value() {
            ScalarValue::F32(value) => value.fmt(f),
            ScalarValue::F64(value) => value.fmt(f),
            ScalarValue::I64(value) => value.fmt(f),
            ScalarValue::C32(value) => value.fmt(f),
            ScalarValue::C64(value) => value.fmt(f),
        }
    }
}

impl fmt::Debug for AnyScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyScalar")
            .field("dtype", &self.tensor.as_native().dtype())
            .field("value", &self.value())
            .field("tracks_grad", &self.tracks_grad())
            .finish()
    }
}
