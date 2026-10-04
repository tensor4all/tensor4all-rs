//! Scalar bounds and binary scaling. Scaling components individually avoids
//! constructing an overflowing reciprocal factor for subnormal inputs.

use num_complex::{Complex32, Complex64};
use std::{fmt::Debug, hash::Hash};

mod sealed {
    pub trait Scale: Copy {
        fn magnitude(self) -> f64;
        fn min_nonzero_component_magnitude(self) -> Option<f64>;
        fn scale_binary(self, exponent: i64) -> Self;
        fn loses_component_to(self, scaled: Self) -> bool;
        fn product_underflows(self, other: Self) -> bool;
        fn component_product_underflows(a: f64, b: f64) -> bool;
    }
}

/// A supported real or complex floating-point scalar: f32, f64, Complex32 or
/// Complex64. The trait is sealed because binary scaling is part of the contract.
///
/// # Examples
/// ```
/// use tensor4all_treersi::TreeRsiScalar;
/// fn square<T: TreeRsiScalar>(x: T) -> T { x * x }
/// assert_eq!(square(3.0_f32), 9.0);
/// ```
pub trait TreeRsiScalar:
    sealed::Scale
    + tensor4all_core::Scalar
    + tensor4all_core::MatrixLuciScalar
    + tensor4all_tensorbackend::TensorElement
    + tensor4all_tensorbackend::BlasMul
{
}
impl<T> TreeRsiScalar for T where
    T: sealed::Scale
        + tensor4all_core::Scalar
        + tensor4all_core::MatrixLuciScalar
        + tensor4all_tensorbackend::TensorElement
        + tensor4all_tensorbackend::BlasMul
{
}

/// An ordered node label. Ordering fixes traversal and random-draw order.
///
/// # Examples
/// ```
/// use tensor4all_treersi::TreeRsiNode;
/// fn sorted<V: TreeRsiNode>(mut nodes: Vec<V>) -> Vec<V> { nodes.sort(); nodes }
/// assert_eq!(sorted(vec![2usize, 0, 1]), vec![0, 1, 2]);
/// ```
pub trait TreeRsiNode: Clone + Debug + Eq + Hash + Ord + Send + Sync + 'static {}
impl<V: Clone + Debug + Eq + Hash + Ord + Send + Sync + 'static> TreeRsiNode for V {}

fn scale_real(mut x: f64, mut exponent: i64) -> f64 {
    if x == 0.0 {
        return x;
    }
    // No finite nonzero f64 can survive these exponents. Avoid an unbounded
    // loop when a long tree has accumulated a large symbolic exponent.
    if exponent > 2098 {
        return x.signum() * f64::INFINITY;
    }
    if exponent < -2098 {
        return x.signum() * 0.0;
    }
    while exponent > 512 {
        x *= 2.0_f64.powi(512);
        exponent -= 512;
    }
    while exponent < -512 {
        x *= 2.0_f64.powi(-512);
        exponent += 512;
    }
    x * 2.0_f64.powi(exponent as i32)
}

macro_rules! real {
    ($t:ty) => {
        impl sealed::Scale for $t {
            fn magnitude(self) -> f64 {
                (self as f64).abs()
            }
            fn min_nonzero_component_magnitude(self) -> Option<f64> {
                let value = (self as f64).abs();
                (value != 0.0).then_some(value)
            }
            fn scale_binary(self, exponent: i64) -> Self {
                scale_real(self as f64, exponent) as Self
            }
            fn loses_component_to(self, scaled: Self) -> bool {
                self != 0.0 && scaled == 0.0
            }
            fn product_underflows(self, other: Self) -> bool {
                self != 0.0 && other != 0.0 && self * other == 0.0
            }
            fn component_product_underflows(a: f64, b: f64) -> bool {
                a != 0.0 && b != 0.0 && (a as $t) * (b as $t) == 0.0
            }
        }
    };
}
macro_rules! complex {
    ($t:ty, $r:ty) => {
        impl sealed::Scale for $t {
            // Component max avoids squaring overflow and is sufficient for scaling.
            fn magnitude(self) -> f64 {
                if !self.re.is_finite() || !self.im.is_finite() {
                    return f64::INFINITY;
                }
                (self.re as f64).abs().max((self.im as f64).abs())
            }
            fn min_nonzero_component_magnitude(self) -> Option<f64> {
                let re = (self.re as f64).abs();
                let im = (self.im as f64).abs();
                match (re != 0.0, im != 0.0) {
                    (true, true) => Some(re.min(im)),
                    (true, false) => Some(re),
                    (false, true) => Some(im),
                    (false, false) => None,
                }
            }
            fn loses_component_to(self, scaled: Self) -> bool {
                (self.re != 0.0 && scaled.re == 0.0) || (self.im != 0.0 && scaled.im == 0.0)
            }
            fn product_underflows(self, other: Self) -> bool {
                (self.re != 0.0 && other.re != 0.0 && self.re * other.re == 0.0)
                    || (self.im != 0.0 && other.im != 0.0 && self.im * other.im == 0.0)
                    || (self.re != 0.0 && other.im != 0.0 && self.re * other.im == 0.0)
                    || (self.im != 0.0 && other.re != 0.0 && self.im * other.re == 0.0)
            }
            fn component_product_underflows(a: f64, b: f64) -> bool {
                a != 0.0 && b != 0.0 && (a as $r) * (b as $r) == 0.0
            }
            fn scale_binary(self, exponent: i64) -> Self {
                Self::new(
                    scale_real(self.re as f64, exponent) as $r,
                    scale_real(self.im as f64, exponent) as $r,
                )
            }
        }
    };
}
real!(f32);
real!(f64);
complex!(Complex32, f32);
complex!(Complex64, f64);

pub(crate) fn magnitude<T: TreeRsiScalar>(x: T) -> f64 {
    x.magnitude()
}
pub(crate) fn scale<T: TreeRsiScalar>(x: T, exponent: i64) -> T {
    x.scale_binary(exponent)
}

pub(crate) fn loses_component<T: TreeRsiScalar>(original: T, scaled: T) -> bool {
    original.loses_component_to(scaled)
}

pub(crate) fn min_nonzero_component_magnitude<T: TreeRsiScalar>(value: T) -> Option<f64> {
    value.min_nonzero_component_magnitude()
}

pub(crate) fn product_underflows<T: TreeRsiScalar>(a: T, b: T) -> bool {
    a.product_underflows(b)
}

pub(crate) fn component_product_underflows<T: TreeRsiScalar>(a: f64, b: f64) -> bool {
    T::component_product_underflows(a, b)
}
