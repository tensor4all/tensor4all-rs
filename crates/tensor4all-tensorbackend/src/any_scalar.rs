use num_complex::Complex64;
use tenferro_dyadtensor::DynAdScalar;

use crate::storage::{Storage, SumFromStorage};

/// Dynamic scalar used across tensor4all backends.
///
/// This is an alias to tenferro's dynamic AD scalar:
/// `tenferro_dyadtensor::DynAdScalar`.
pub type AnyScalar = DynAdScalar;

impl SumFromStorage for AnyScalar {
    fn sum_from_storage(storage: &Storage) -> Self {
        match storage {
            Storage::DenseF64(_) | Storage::DiagF64(_) => {
                AnyScalar::new_real(f64::sum_from_storage(storage))
            }
            Storage::DenseC64(_) | Storage::DiagC64(_) => {
                let z = Complex64::sum_from_storage(storage);
                AnyScalar::new_complex(z.re, z.im)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{DenseStorageC64, DenseStorageF64, DiagStorageF64};

    #[test]
    fn any_scalar_sum_from_real_storage_stays_real() {
        let dense = Storage::DenseF64(DenseStorageF64::from_vec_with_shape(vec![1.0, -2.5], &[2]));
        let diag = Storage::DiagF64(DiagStorageF64::from_vec(vec![3.0, 4.5]));

        let dense_sum = AnyScalar::sum_from_storage(&dense);
        let diag_sum = AnyScalar::sum_from_storage(&diag);

        assert!(dense_sum.is_real());
        assert_eq!(dense_sum.real(), -1.5);
        assert!(diag_sum.is_real());
        assert_eq!(diag_sum.real(), 7.5);
    }

    #[test]
    fn any_scalar_sum_from_complex_storage_stays_complex() {
        let dense = Storage::DenseC64(DenseStorageC64::from_vec_with_shape(
            vec![Complex64::new(1.0, -1.0), Complex64::new(-0.5, 2.0)],
            &[2],
        ));

        let sum = AnyScalar::sum_from_storage(&dense);
        let sum_c64: Complex64 = sum.into();
        assert_eq!(sum_c64, Complex64::new(0.5, 1.0));
    }
}
