use num_complex::Complex64;
use rand_distr::{Distribution, StandardNormal};

/// Scalar types supported by Alternating Cross Interpolation.
///
/// Use this trait bound when writing generic helpers around ACI elementwise
/// operations. It combines the tensor-train scalar requirements, matrix-CI
/// scalar requirements, and deterministic random initialization support needed
/// by the sweep algorithm.
///
/// Related types: [`AciOptions`](crate::AciOptions) configures ACI runs for an
/// `AciScalar`, and [`AciResult`](crate::AciResult) returns a tensor train with
/// the same scalar type.
///
/// # Examples
///
/// ```
/// use tensor4all_aci::AciScalar;
///
/// fn accepts_aci_scalar<T: AciScalar>(value: T) -> T {
///     value
/// }
///
/// assert_eq!(accepts_aci_scalar(3.0_f64), 3.0);
/// ```
pub trait AciScalar:
    tensor4all_simplett::TTScalar + tensor4all_tcicore::MatrixLuciScalar + Copy
{
    #[doc(hidden)]
    fn sample_standard_normal(rng: &mut rand_chacha::ChaCha8Rng) -> Self;
}

impl AciScalar for f64 {
    fn sample_standard_normal(rng: &mut rand_chacha::ChaCha8Rng) -> Self {
        StandardNormal.sample(rng)
    }
}

impl AciScalar for Complex64 {
    fn sample_standard_normal(rng: &mut rand_chacha::ChaCha8Rng) -> Self {
        let re = StandardNormal.sample(rng);
        let im = StandardNormal.sample(rng);
        Complex64::new(re, im)
    }
}
