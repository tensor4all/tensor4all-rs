//! Single-precision CI reconstruction and dtype regressions for #854.

use num_complex::Complex32;
use tensor4all_core::{
    factorize, factorize_full_rank, Canonical, DynIndex, FactorizeAlg, FactorizeOptions, IdxTensor,
    Scalar, TensorContractionLike,
};
use tensor4all_tensorbackend::TensorElement;

trait TestScalar: Scalar + TensorElement {
    fn value(re: f32, im: f32) -> Self;
}
impl TestScalar for f32 {
    fn value(re: f32, _: f32) -> Self {
        re
    }
}
impl TestScalar for Complex32 {
    fn value(re: f32, im: f32) -> Self {
        Self::new(re, im)
    }
}

fn assert_ci<T: TestScalar>() {
    for zero in [false, true] {
        let i = DynIndex::new_dyn(2);
        let j = DynIndex::new_dyn(3);
        let values: Vec<T> = (1..=6)
            .map(|p| {
                if zero {
                    T::zero()
                } else {
                    T::value(p as f32, 0.07 * (p * p) as f32)
                }
            })
            .collect();
        let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], values.clone()).unwrap();
        for canonical in [Canonical::Left, Canonical::Right] {
            for full_rank in [false, true] {
                let mut options = FactorizeOptions::ci();
                options.canonical = canonical;
                let factors = if full_rank {
                    factorize_full_rank(
                        &tensor,
                        std::slice::from_ref(&i),
                        FactorizeAlg::CI,
                        canonical,
                    )
                } else {
                    factorize(&tensor, std::slice::from_ref(&i), &options)
                }
                .unwrap();
                // Decoding as T rejects accidental widening or wrong dtype.
                factors.left.to_vec::<T>().unwrap();
                factors.right.to_vec::<T>().unwrap();
                let recovered = factors
                    .left
                    .contract_pair(&factors.right)
                    .unwrap()
                    .permute_indices(&[i.clone(), j.clone()])
                    .unwrap()
                    .to_vec::<T>()
                    .unwrap();
                let error = recovered
                    .iter()
                    .zip(&values)
                    .map(|(&a, &b)| (a - b).abs_val())
                    .fold(0.0_f64, f64::max);
                assert!(error < 3e-6, "CI reconstruction error {error:e}, zero={zero}, full={full_rank}, canonical={canonical:?}");
            }
        }
    }
}

#[test]
fn ci_f32_preserves_dtype_and_reconstructs_both_orientations() {
    assert_ci::<f32>();
}

#[test]
fn ci_c32_preserves_dtype_and_reconstructs_both_orientations() {
    assert_ci::<Complex32>();
}
