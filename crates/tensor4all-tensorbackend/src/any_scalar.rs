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
