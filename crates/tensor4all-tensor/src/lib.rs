pub mod storage;
pub mod tensor;

pub use storage::{AnyScalar, DenseStorageFactory, Storage, SumFromStorage, make_mut_storage};
pub use tensor::{TensorDynLen, TensorStaticLen, compute_permutation_from_indices};

