//! Result types returned by Alternating Cross Interpolation.

use tensor4all_simplett::{TTScalar, TensorTrain};

/// Output of an Alternating Cross Interpolation run.
///
/// The result contains the approximating tensor train plus convergence metadata
/// collected during the run. Use [`tensor_train`](Self::tensor_train) for
/// subsequent tensor-train operations, and inspect [`ranks`](Self::ranks) and
/// [`errors`](Self::errors) to diagnose truncation or convergence behavior.
///
/// Related types: [`AciOptions`](crate::AciOptions) configures the run that
/// produces this value; [`TensorTrain`] stores the approximating tensor.
///
/// # Examples
///
/// ```
/// use tensor4all_aci::AciResult;
/// use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};
///
/// let tensor_train = TensorTrain::<f64>::constant(&[2, 3], 4.0);
/// let result = AciResult {
///     tensor_train,
///     ranks: vec![1],
///     errors: vec![0.0],
/// };
///
/// assert_eq!(result.tensor_train.site_dims(), vec![2, 3]);
/// assert_eq!(result.ranks, vec![1]);
/// assert_eq!(result.errors, vec![0.0]);
/// assert!((result.tensor_train.evaluate(&[1, 2]).unwrap() - 4.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct AciResult<T: TTScalar> {
    /// Tensor-train approximation produced by ACI.
    pub tensor_train: TensorTrain<T>,

    /// Maximum or sweep-level bond dimensions recorded during the run.
    ///
    /// Implementations should store these in iteration order so callers can
    /// compare rank growth against [`AciOptions::max_bond_dim`](crate::AciOptions::max_bond_dim).
    pub ranks: Vec<usize>,

    /// Residual or convergence estimates recorded during the run.
    ///
    /// Values are stored in iteration order and use the same scaling convention
    /// as [`AciOptions::tolerance`](crate::AciOptions::tolerance).
    pub errors: Vec<f64>,
}
