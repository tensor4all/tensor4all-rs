use num_complex::{Complex64, ComplexFloat};
use num_traits::{One, Zero};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[cfg(feature = "backend-libtorch")]
use tch::{Kind, Tensor};

use crate::storage::{Storage, SumFromStorage};

/// Dynamic scalar value (for dynamic element type tensors).
///
/// Supports both real (`f64`) and complex (`Complex64`) scalar values.
/// When the `backend-libtorch` feature is enabled, also supports PyTorch
/// tensor scalars that preserve autograd computation graphs.
///
/// # Autograd Support
///
/// The `TorchF64` and `TorchC64` variants hold 0-dimensional PyTorch tensors
/// that can track gradients. Use these variants when you need automatic
/// differentiation through scalar operations.
#[derive(Debug)]
pub enum AnyScalar {
    /// Real number (f64)
    F64(f64),
    /// Complex number (Complex64)
    C64(Complex64),
    /// PyTorch real scalar (0-dimensional tensor, preserves autograd)
    #[cfg(feature = "backend-libtorch")]
    TorchF64(Tensor),
    /// PyTorch complex scalar (0-dimensional tensor, preserves autograd)
    #[cfg(feature = "backend-libtorch")]
    TorchC64(Tensor),
}

// Manual Clone implementation because tch::Tensor uses shallow_clone
impl Clone for AnyScalar {
    fn clone(&self) -> Self {
        match self {
            AnyScalar::F64(x) => AnyScalar::F64(*x),
            AnyScalar::C64(z) => AnyScalar::C64(*z),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchF64(t) => AnyScalar::TorchF64(t.shallow_clone()),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchC64(t) => AnyScalar::TorchC64(t.shallow_clone()),
        }
    }
}

// Manual PartialEq implementation
impl PartialEq for AnyScalar {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AnyScalar::F64(a), AnyScalar::F64(b)) => a == b,
            (AnyScalar::C64(a), AnyScalar::C64(b)) => a == b,
            #[cfg(feature = "backend-libtorch")]
            (AnyScalar::TorchF64(a), AnyScalar::TorchF64(b)) => {
                a.double_value(&[]) == b.double_value(&[])
            }
            #[cfg(feature = "backend-libtorch")]
            (AnyScalar::TorchC64(a), AnyScalar::TorchC64(b)) => {
                let a_re = a.real().double_value(&[]);
                let a_im = a.imag().double_value(&[]);
                let b_re = b.real().double_value(&[]);
                let b_im = b.imag().double_value(&[]);
                a_re == b_re && a_im == b_im
            }
            _ => false,
        }
    }
}

impl SumFromStorage for AnyScalar {
    fn sum_from_storage(storage: &Storage) -> Self {
        match storage {
            Storage::DenseF64(_) | Storage::DiagF64(_) => {
                AnyScalar::F64(f64::sum_from_storage(storage))
            }
            Storage::DenseC64(_) | Storage::DiagC64(_) => {
                AnyScalar::C64(Complex64::sum_from_storage(storage))
            }
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchF64(ts) => {
                // Return Torch scalar to preserve autograd
                AnyScalar::TorchF64(ts.tensor().sum(Kind::Double))
            }
            #[cfg(feature = "backend-libtorch")]
            Storage::TorchC64(ts) => {
                // Return Torch scalar to preserve autograd
                AnyScalar::TorchC64(ts.tensor().sum(Kind::ComplexDouble))
            }
        }
    }
}

impl AnyScalar {
    /// Create a real scalar value.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    /// let s = AnyScalar::new_real(3.5);
    /// ```
    pub fn new_real(x: f64) -> Self {
        x.into()
    }

    /// Create a complex scalar value from real and imaginary parts.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    /// let s = AnyScalar::new_complex(1.0, 2.0);  // 1 + 2i
    /// ```
    pub fn new_complex(re: f64, im: f64) -> Self {
        Complex64::new(re, im).into()
    }

    /// Check if this scalar is complex.
    pub fn is_complex(&self) -> bool {
        match self {
            AnyScalar::C64(_) => true,
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchC64(_) => true,
            _ => false,
        }
    }

    /// Check if this scalar is a Torch tensor (preserves autograd).
    #[cfg(feature = "backend-libtorch")]
    pub fn is_torch(&self) -> bool {
        matches!(self, AnyScalar::TorchF64(_) | AnyScalar::TorchC64(_))
    }

    /// Get the real part of the scalar.
    ///
    /// Note: For Torch scalars, this extracts the value and breaks the autograd graph.
    pub fn real(&self) -> f64 {
        match self {
            AnyScalar::F64(x) => *x,
            AnyScalar::C64(z) => z.re,
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchF64(t) => t.double_value(&[]),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchC64(t) => t.real().double_value(&[]),
        }
    }

    /// Get the absolute value (magnitude).
    ///
    /// Note: For Torch scalars, this extracts the value and breaks the autograd graph.
    pub fn abs(&self) -> f64 {
        match self {
            AnyScalar::F64(x) => x.abs(),
            AnyScalar::C64(z) => z.abs(),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchF64(t) => t.abs().double_value(&[]),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchC64(t) => t.abs().double_value(&[]),
        }
    }

    /// Compute square root.
    ///
    /// For negative real numbers, returns a complex number with the principal value.
    /// For Torch scalars, preserves the autograd graph.
    pub fn sqrt(&self) -> Self {
        match self {
            AnyScalar::F64(x) => {
                if *x >= 0.0 {
                    AnyScalar::F64(x.sqrt())
                } else {
                    AnyScalar::C64(Complex64::new(*x, 0.0).sqrt())
                }
            }
            AnyScalar::C64(z) => AnyScalar::C64(z.sqrt()),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchF64(t) => AnyScalar::TorchF64(t.sqrt()),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchC64(t) => AnyScalar::TorchC64(t.sqrt()),
        }
    }

    /// Raise to a floating-point power.
    ///
    /// For negative real numbers, returns a complex number with the principal value.
    /// For Torch scalars, preserves the autograd graph.
    pub fn powf(&self, exp: f64) -> Self {
        match self {
            AnyScalar::F64(x) => {
                if *x >= 0.0 {
                    AnyScalar::F64(x.powf(exp))
                } else {
                    AnyScalar::C64(Complex64::new(*x, 0.0).powf(exp))
                }
            }
            AnyScalar::C64(z) => AnyScalar::C64(z.powf(exp)),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchF64(t) => AnyScalar::TorchF64(t.pow_tensor_scalar(exp)),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchC64(t) => AnyScalar::TorchC64(t.pow_tensor_scalar(exp)),
        }
    }

    /// Raise to an integer power.
    ///
    /// For Torch scalars, preserves the autograd graph.
    pub fn powi(&self, exp: i32) -> Self {
        match self {
            AnyScalar::F64(x) => AnyScalar::F64(x.powi(exp)),
            AnyScalar::C64(z) => AnyScalar::C64(z.powi(exp)),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchF64(t) => AnyScalar::TorchF64(t.pow_tensor_scalar(exp as f64)),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchC64(t) => AnyScalar::TorchC64(t.pow_tensor_scalar(exp as f64)),
        }
    }

    /// Check if this scalar is zero.
    ///
    /// This method is provided directly on `AnyScalar` so that downstream crates
    /// don't need to import the `num_traits::Zero` trait.
    ///
    /// Note: For Torch scalars, this extracts the value and breaks the autograd graph.
    pub fn is_zero(&self) -> bool {
        match self {
            AnyScalar::F64(x) => *x == 0.0,
            AnyScalar::C64(z) => z.re == 0.0 && z.im == 0.0,
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchF64(t) => t.double_value(&[]) == 0.0,
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchC64(t) => {
                t.real().double_value(&[]) == 0.0 && t.imag().double_value(&[]) == 0.0
            }
        }
    }
}

// Autograd methods for AnyScalar
#[cfg(feature = "backend-libtorch")]
impl AnyScalar {
    /// Check if this scalar requires gradient computation.
    ///
    /// Returns `false` for non-Torch scalars (F64, C64).
    pub fn requires_grad(&self) -> bool {
        match self {
            AnyScalar::TorchF64(t) | AnyScalar::TorchC64(t) => t.requires_grad(),
            _ => false,
        }
    }

    /// Set whether this scalar requires gradient computation.
    ///
    /// Returns an error for non-Torch scalars.
    pub fn set_requires_grad(&mut self, requires_grad: bool) -> anyhow::Result<()> {
        match self {
            AnyScalar::TorchF64(t) => {
                *t = t.set_requires_grad(requires_grad);
                Ok(())
            }
            AnyScalar::TorchC64(t) => {
                *t = t.set_requires_grad(requires_grad);
                Ok(())
            }
            _ => anyhow::bail!("Cannot set requires_grad on non-Torch scalar"),
        }
    }

    /// Get the gradient of this scalar, if it exists.
    ///
    /// Returns `None` for non-Torch scalars or if no gradient has been computed.
    pub fn grad(&self) -> Option<AnyScalar> {
        match self {
            AnyScalar::TorchF64(t) => {
                let g = t.grad();
                if g.defined() {
                    Some(AnyScalar::TorchF64(g))
                } else {
                    None
                }
            }
            AnyScalar::TorchC64(t) => {
                let g = t.grad();
                if g.defined() {
                    Some(AnyScalar::TorchC64(g))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Compute gradients by backpropagating from this scalar.
    ///
    /// Returns an error for non-Torch scalars.
    pub fn backward(&self) -> anyhow::Result<()> {
        match self {
            AnyScalar::TorchF64(t) | AnyScalar::TorchC64(t) => {
                t.backward();
                Ok(())
            }
            _ => anyhow::bail!("Cannot call backward on non-Torch scalar"),
        }
    }

    /// Detach this scalar from the computation graph.
    ///
    /// Returns a new scalar that shares data but doesn't track gradients.
    /// For non-Torch scalars, returns a clone.
    pub fn detach(&self) -> Self {
        match self {
            AnyScalar::TorchF64(t) => AnyScalar::TorchF64(t.detach()),
            AnyScalar::TorchC64(t) => AnyScalar::TorchC64(t.detach()),
            other => other.clone(),
        }
    }

    /// Get the underlying Torch tensor, if this is a Torch scalar.
    pub fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            AnyScalar::TorchF64(t) | AnyScalar::TorchC64(t) => Some(t),
            _ => None,
        }
    }

    /// Convert this scalar to a Torch scalar (f64).
    ///
    /// If already a Torch scalar, returns a clone.
    /// Otherwise, creates a new 0-dimensional tensor.
    pub fn to_torch_f64(&self) -> Self {
        match self {
            AnyScalar::F64(x) => AnyScalar::TorchF64(Tensor::from(*x)),
            AnyScalar::C64(z) => AnyScalar::TorchF64(Tensor::from(z.re)),
            AnyScalar::TorchF64(t) => AnyScalar::TorchF64(t.shallow_clone()),
            AnyScalar::TorchC64(t) => AnyScalar::TorchF64(t.real()),
        }
    }

    /// Convert to f64 value, extracting from tensor if needed.
    ///
    /// Note: This breaks the autograd graph.
    pub fn to_f64(&self) -> f64 {
        self.real()
    }
}

// Helper to convert AnyScalar to Tensor for mixed operations
#[cfg(feature = "backend-libtorch")]
fn to_tensor(s: &AnyScalar) -> Tensor {
    match s {
        AnyScalar::F64(x) => Tensor::from(*x),
        AnyScalar::C64(z) => {
            let real = Tensor::from(z.re);
            let imag = Tensor::from(z.im);
            Tensor::complex(&real, &imag)
        }
        AnyScalar::TorchF64(t) => t.shallow_clone(),
        AnyScalar::TorchC64(t) => t.shallow_clone(),
    }
}

#[cfg(feature = "backend-libtorch")]
fn is_any_complex(a: &AnyScalar, b: &AnyScalar) -> bool {
    matches!(a, AnyScalar::C64(_) | AnyScalar::TorchC64(_))
        || matches!(b, AnyScalar::C64(_) | AnyScalar::TorchC64(_))
}

#[cfg(feature = "backend-libtorch")]
fn is_any_torch(a: &AnyScalar, b: &AnyScalar) -> bool {
    matches!(a, AnyScalar::TorchF64(_) | AnyScalar::TorchC64(_))
        || matches!(b, AnyScalar::TorchF64(_) | AnyScalar::TorchC64(_))
}

// 四則演算の実装
impl Add for AnyScalar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(feature = "backend-libtorch")]
        {
            if is_any_torch(&self, &rhs) {
                let result = to_tensor(&self) + to_tensor(&rhs);
                return if is_any_complex(&self, &rhs) {
                    AnyScalar::TorchC64(result)
                } else {
                    AnyScalar::TorchF64(result)
                };
            }
        }
        match (self, rhs) {
            (AnyScalar::F64(a), AnyScalar::F64(b)) => AnyScalar::F64(a + b),
            (AnyScalar::F64(a), AnyScalar::C64(b)) => AnyScalar::C64(Complex64::new(a, 0.0) + b),
            (AnyScalar::C64(a), AnyScalar::F64(b)) => AnyScalar::C64(a + Complex64::new(b, 0.0)),
            (AnyScalar::C64(a), AnyScalar::C64(b)) => AnyScalar::C64(a + b),
            #[cfg(feature = "backend-libtorch")]
            _ => unreachable!(), // Torch cases handled above
        }
    }
}

impl Sub for AnyScalar {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(feature = "backend-libtorch")]
        {
            if is_any_torch(&self, &rhs) {
                let result = to_tensor(&self) - to_tensor(&rhs);
                return if is_any_complex(&self, &rhs) {
                    AnyScalar::TorchC64(result)
                } else {
                    AnyScalar::TorchF64(result)
                };
            }
        }
        match (self, rhs) {
            (AnyScalar::F64(a), AnyScalar::F64(b)) => AnyScalar::F64(a - b),
            (AnyScalar::F64(a), AnyScalar::C64(b)) => AnyScalar::C64(Complex64::new(a, 0.0) - b),
            (AnyScalar::C64(a), AnyScalar::F64(b)) => AnyScalar::C64(a - Complex64::new(b, 0.0)),
            (AnyScalar::C64(a), AnyScalar::C64(b)) => AnyScalar::C64(a - b),
            #[cfg(feature = "backend-libtorch")]
            _ => unreachable!(),
        }
    }
}

impl Mul for AnyScalar {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(feature = "backend-libtorch")]
        {
            if is_any_torch(&self, &rhs) {
                let result = to_tensor(&self) * to_tensor(&rhs);
                return if is_any_complex(&self, &rhs) {
                    AnyScalar::TorchC64(result)
                } else {
                    AnyScalar::TorchF64(result)
                };
            }
        }
        match (self, rhs) {
            (AnyScalar::F64(a), AnyScalar::F64(b)) => AnyScalar::F64(a * b),
            (AnyScalar::F64(a), AnyScalar::C64(b)) => AnyScalar::C64(Complex64::new(a, 0.0) * b),
            (AnyScalar::C64(a), AnyScalar::F64(b)) => AnyScalar::C64(a * Complex64::new(b, 0.0)),
            (AnyScalar::C64(a), AnyScalar::C64(b)) => AnyScalar::C64(a * b),
            #[cfg(feature = "backend-libtorch")]
            _ => unreachable!(),
        }
    }
}

impl Div for AnyScalar {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        #[cfg(feature = "backend-libtorch")]
        {
            if is_any_torch(&self, &rhs) {
                let result = to_tensor(&self) / to_tensor(&rhs);
                return if is_any_complex(&self, &rhs) {
                    AnyScalar::TorchC64(result)
                } else {
                    AnyScalar::TorchF64(result)
                };
            }
        }
        match (self, rhs) {
            (AnyScalar::F64(a), AnyScalar::F64(b)) => AnyScalar::F64(a / b),
            (AnyScalar::F64(a), AnyScalar::C64(b)) => AnyScalar::C64(Complex64::new(a, 0.0) / b),
            (AnyScalar::C64(a), AnyScalar::F64(b)) => AnyScalar::C64(a / Complex64::new(b, 0.0)),
            (AnyScalar::C64(a), AnyScalar::C64(b)) => AnyScalar::C64(a / b),
            #[cfg(feature = "backend-libtorch")]
            _ => unreachable!(),
        }
    }
}

impl Neg for AnyScalar {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            AnyScalar::F64(x) => AnyScalar::F64(-x),
            AnyScalar::C64(z) => AnyScalar::C64(-z),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchF64(t) => AnyScalar::TorchF64(t.neg()),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchC64(t) => AnyScalar::TorchC64(t.neg()),
        }
    }
}

// 型変換
impl From<f64> for AnyScalar {
    fn from(x: f64) -> Self {
        AnyScalar::F64(x)
    }
}

impl From<Complex64> for AnyScalar {
    fn from(z: Complex64) -> Self {
        AnyScalar::C64(z)
    }
}

impl std::convert::TryFrom<AnyScalar> for f64 {
    type Error = &'static str;

    fn try_from(value: AnyScalar) -> Result<Self, Self::Error> {
        match value {
            AnyScalar::F64(x) => Ok(x),
            AnyScalar::C64(_) => Err("Cannot convert complex number to f64"),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchF64(t) => Ok(t.double_value(&[])),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchC64(_) => Err("Cannot convert complex tensor to f64"),
        }
    }
}

impl From<AnyScalar> for Complex64 {
    fn from(value: AnyScalar) -> Self {
        match value {
            AnyScalar::F64(x) => Complex64::new(x, 0.0),
            AnyScalar::C64(z) => z,
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchF64(t) => Complex64::new(t.double_value(&[]), 0.0),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchC64(t) => {
                Complex64::new(t.real().double_value(&[]), t.imag().double_value(&[]))
            }
        }
    }
}

// 標準トレイト
impl Default for AnyScalar {
    fn default() -> Self {
        AnyScalar::F64(0.0)
    }
}

impl Zero for AnyScalar {
    fn zero() -> Self {
        AnyScalar::F64(0.0)
    }

    fn is_zero(&self) -> bool {
        match self {
            AnyScalar::F64(x) => x.is_zero(),
            AnyScalar::C64(z) => z.is_zero(),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchF64(t) => t.double_value(&[]) == 0.0,
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchC64(t) => {
                t.real().double_value(&[]) == 0.0 && t.imag().double_value(&[]) == 0.0
            }
        }
    }
}

impl One for AnyScalar {
    fn one() -> Self {
        AnyScalar::F64(1.0)
    }
}

impl PartialOrd for AnyScalar {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (AnyScalar::F64(a), AnyScalar::F64(b)) => a.partial_cmp(b),
            #[cfg(feature = "backend-libtorch")]
            (AnyScalar::TorchF64(a), AnyScalar::TorchF64(b)) => {
                a.double_value(&[]).partial_cmp(&b.double_value(&[]))
            }
            #[cfg(feature = "backend-libtorch")]
            (AnyScalar::F64(a), AnyScalar::TorchF64(b)) => a.partial_cmp(&b.double_value(&[])),
            #[cfg(feature = "backend-libtorch")]
            (AnyScalar::TorchF64(a), AnyScalar::F64(b)) => a.double_value(&[]).partial_cmp(b),
            _ => None, // Complex numbers are not ordered
        }
    }
}

impl fmt::Display for AnyScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnyScalar::F64(x) => write!(f, "{}", x),
            AnyScalar::C64(z) => write!(f, "{}", z),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchF64(t) => write!(f, "{}", t.double_value(&[])),
            #[cfg(feature = "backend-libtorch")]
            AnyScalar::TorchC64(t) => {
                let re = t.real().double_value(&[]);
                let im = t.imag().double_value(&[]);
                write!(f, "{}+{}i", re, im)
            }
        }
    }
}
