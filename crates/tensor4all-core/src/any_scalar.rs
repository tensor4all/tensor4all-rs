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

#[allow(missing_docs)]
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

    pub fn from_value<T: TensorElement>(value: T) -> Self {
        Self::from_tensor_result(TensorDynLen::scalar(value), "from_value")
    }

    pub fn from_real(x: f64) -> Self {
        Self::from_value(x)
    }

    pub fn from_complex(re: f64, im: f64) -> Self {
        Self::from_value(Complex64::new(re, im))
    }

    pub fn new_real(x: f64) -> Self {
        Self::from_real(x)
    }

    pub fn new_complex(re: f64, im: f64) -> Self {
        Self::from_complex(re, im)
    }

    pub fn primal(&self) -> Self {
        self.detach()
    }

    pub fn enable_grad(self) -> Self {
        Self::from_tensor_unchecked(self.tensor.enable_grad())
    }

    pub fn tracks_grad(&self) -> bool {
        self.tensor.tracks_grad()
    }

    pub fn grad(&self) -> Result<Option<Self>> {
        self.tensor
            .grad()
            .map(|maybe_grad| maybe_grad.map(Self::from_tensor_unchecked))
    }

    pub fn clear_grad(&self) -> Result<()> {
        self.tensor.clear_grad()
    }

    pub fn backward(&self) -> Result<()> {
        self.tensor.backward()
    }

    pub fn detach(&self) -> Self {
        Self::from_tensor_unchecked(self.tensor.detach())
    }

    pub fn real(&self) -> f64 {
        self.value().real()
    }

    pub fn imag(&self) -> f64 {
        self.value().imag()
    }

    pub fn abs(&self) -> f64 {
        self.value().abs()
    }

    pub fn is_complex(&self) -> bool {
        self.value().is_complex()
    }

    pub fn is_real(&self) -> bool {
        !self.is_complex()
    }

    pub fn is_zero(&self) -> bool {
        self.value().is_zero()
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self.value() {
            ScalarValue::F32(value) => Some(value as f64),
            ScalarValue::F64(value) => Some(value),
            ScalarValue::I64(value) => Some(value as f64),
            ScalarValue::C32(_) | ScalarValue::C64(_) => None,
        }
    }

    pub fn as_c64(&self) -> Option<Complex64> {
        match self.value() {
            ScalarValue::F32(_) | ScalarValue::F64(_) | ScalarValue::I64(_) => None,
            ScalarValue::C32(value) => Some(Complex64::new(value.re as f64, value.im as f64)),
            ScalarValue::C64(value) => Some(value),
        }
    }

    pub fn conj(&self) -> Self {
        Self::from_eager_unary(self, "conj", |tensor| tensor.conj())
    }

    pub fn real_part(&self) -> Self {
        Self::from_real(self.real())
    }

    pub fn imag_part(&self) -> Self {
        Self::from_real(self.imag())
    }

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

    pub fn sqrt(&self) -> Self {
        if self.is_complex() || self.real() < 0.0 {
            Self::from_backend_scalar(self.to_backend_scalar().sqrt())
        } else {
            Self::from_eager_unary(self, "sqrt", |tensor| tensor.sqrt())
        }
    }

    pub fn powf(&self, exponent: f64) -> Self {
        Self::from_backend_scalar(self.to_backend_scalar().powf(exponent))
    }

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
