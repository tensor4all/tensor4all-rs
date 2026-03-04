use num_complex::Complex64;
use tenferro_dyadtensor::DynAdValue;

use crate::storage::{Storage, SumFromStorage};

/// Dynamic scalar used across tensor4all backends.
///
/// This is an alias to tenferro's dynamic AD scalar:
/// `tenferro_dyadtensor::DynAdValue`.
pub type AnyScalar = DynAdValue;

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
