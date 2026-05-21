use num_complex::Complex64;
use rand_distr::{Distribution, StandardNormal};

pub(crate) trait AciRandomScalar: tensor4all_simplett::TTScalar {
    fn sample_standard_normal(rng: &mut rand_chacha::ChaCha8Rng) -> Self;
}

pub(crate) trait AciScalar:
    tensor4all_simplett::TTScalar + tensor4all_tcicore::MatrixLuciScalar + AciRandomScalar + Copy
{
}

impl<T> AciScalar for T where
    T: tensor4all_simplett::TTScalar
        + tensor4all_tcicore::MatrixLuciScalar
        + AciRandomScalar
        + Copy
{
}

impl AciRandomScalar for f64 {
    fn sample_standard_normal(rng: &mut rand_chacha::ChaCha8Rng) -> Self {
        StandardNormal.sample(rng)
    }
}

impl AciRandomScalar for Complex64 {
    fn sample_standard_normal(rng: &mut rand_chacha::ChaCha8Rng) -> Self {
        let re = StandardNormal.sample(rng);
        let im = StandardNormal.sample(rng);
        Complex64::new(re, im)
    }
}
