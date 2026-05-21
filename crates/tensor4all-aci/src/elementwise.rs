//! Public elementwise Alternating Cross Interpolation sweep APIs.

use crate::scalar::AciScalar;
use crate::validation::{validate_inputs, validate_options};
use crate::{AciOptions, AciResult, ElementwiseBatch, ElementwiseProblem, Result};
use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};

/// Runs batched elementwise ACI over tensor-train inputs.
///
/// This function approximates the pointwise application of `op` to `inputs`.
/// The callback receives batches in column-major input/point layout through
/// [`ElementwiseBatch`] and writes one output value per interpolation point.
/// Forward and backward sweeps alternate until the configured iteration limit
/// is reached or the conservative convergence rule accepts the current ranks
/// and maximum pivot error.
///
/// For single-site tensor trains there are no bonds to sweep, so the returned
/// [`AciResult`] contains the initialized tensor train and empty `ranks` and
/// `errors` histories.
///
/// # Arguments
///
/// * `op` - Batched callback. For each point, write the elementwise output to
///   the corresponding entry of the output slice. Return
///   [`AciError::Operator`](crate::AciError::Operator) or another ACI error to
///   stop the sweep.
/// * `inputs` - Non-empty tensor trains with the same length and site
///   dimensions. Core dimensions must be positive.
/// * `options` - Sweep limits, tolerance, maximum bond dimension, random seed,
///   and optional initial guess. When in doubt, start with
///   [`AciOptions::default`].
///
/// # Returns
///
/// Returns an [`AciResult`] containing the approximating tensor train plus one
/// rank and error sample after each completed sweep.
///
/// # Errors
///
/// Returns [`AciError`](crate::AciError) when options or inputs are invalid,
/// initial guess construction fails, the operator callback fails, or a local
/// matrix-CI update fails.
///
/// # Panics
///
/// This function does not intentionally panic. A panic from the caller-provided
/// `op` callback is not caught.
///
/// # Examples
///
/// ```
/// use tensor4all_aci::{elementwise_batched, AciOptions, ElementwiseBatch};
/// use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};
///
/// let a = TensorTrain::<f64>::constant(&[2, 2], 2.0);
/// let b = TensorTrain::<f64>::constant(&[2, 2], 3.0);
/// let result = elementwise_batched(
///     |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
///         for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
///             *value = batch.get(0, point)? * batch.get(1, point)?;
///         }
///         Ok(())
///     },
///     &[a, b],
///     &AciOptions::default(),
/// )
/// .unwrap();
///
/// assert!((result.tensor_train.evaluate(&[0, 0]).unwrap() - 6.0).abs() < 1e-12);
/// assert!((result.tensor_train.evaluate(&[1, 1]).unwrap() - 6.0).abs() < 1e-12);
/// assert_eq!(result.ranks.len(), result.errors.len());
/// ```
pub fn elementwise_batched<T, F>(
    mut op: F,
    inputs: &[TensorTrain<T>],
    options: &AciOptions<T>,
) -> Result<AciResult<T>>
where
    T: AciScalar,
    F: for<'batch> FnMut(ElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    validate_options(options)?;
    validate_inputs(inputs)?;

    let mut problem = ElementwiseProblem::new(inputs.to_vec(), options.clone())?;
    if problem.len() == 1 {
        return Ok(AciResult {
            tensor_train: problem.solution,
            ranks: Vec::new(),
            errors: Vec::new(),
        });
    }

    let mut ranks = Vec::new();
    let mut errors = Vec::new();

    for iteration in 0..options.max_iters {
        let forward = iteration % 2 == 0;
        if forward {
            for bond in 0..problem.len() - 1 {
                problem.local_update(bond, true, options, &mut op)?;
            }
        } else {
            for bond in (0..problem.len() - 1).rev() {
                problem.local_update(bond, false, options, &mut op)?;
            }
        }

        let max_error = problem.pivot_errors.iter().copied().fold(0.0_f64, f64::max);
        ranks.push(problem.solution.rank());
        errors.push(max_error);

        if iteration + 1 >= options.min_iters
            && max_error <= options.tolerance
            && ranks_are_stable(&ranks, options.min_iters)
        {
            break;
        }
    }

    Ok(AciResult {
        tensor_train: problem.solution,
        ranks,
        errors,
    })
}

/// Runs scalar elementwise ACI over tensor-train inputs.
///
/// This convenience wrapper evaluates `op` once per interpolation point. The
/// callback receives one value from each input tensor train in input order and
/// returns the corresponding output value. Use [`elementwise_batched`] when the
/// operator can evaluate many points more efficiently at once or can fail.
///
/// # Arguments
///
/// * `op` - Scalar callback applied to one interpolation point. The input slice
///   has length `inputs.len()`.
/// * `inputs` - Non-empty tensor trains with the same length and site
///   dimensions. Core dimensions must be positive.
/// * `options` - Sweep limits, tolerance, maximum bond dimension, random seed,
///   and optional initial guess.
///
/// # Returns
///
/// Returns an [`AciResult`] containing the approximating tensor train plus
/// sweep-by-sweep rank and error histories.
///
/// # Errors
///
/// Returns [`AciError`](crate::AciError) when options or inputs are invalid,
/// batch extraction fails, initial guess construction fails, or a local
/// matrix-CI update fails.
///
/// # Panics
///
/// This function does not intentionally panic. A panic from the caller-provided
/// `op` callback is not caught.
///
/// # Examples
///
/// ```
/// use tensor4all_aci::{elementwise, AciOptions};
/// use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};
///
/// let a = TensorTrain::<f64>::constant(&[2, 2], 2.0);
/// let b = TensorTrain::<f64>::constant(&[2, 2], 5.0);
/// let result = elementwise(
///     |values| values[0] + values[1],
///     &[a, b],
///     &AciOptions::default(),
/// )
/// .unwrap();
///
/// assert!((result.tensor_train.evaluate(&[0, 0]).unwrap() - 7.0).abs() < 1e-12);
/// assert!((result.tensor_train.evaluate(&[1, 1]).unwrap() - 7.0).abs() < 1e-12);
/// assert_eq!(result.ranks.len(), result.errors.len());
/// ```
pub fn elementwise<T, F>(
    mut op: F,
    inputs: &[TensorTrain<T>],
    options: &AciOptions<T>,
) -> Result<AciResult<T>>
where
    T: AciScalar,
    F: FnMut(&[T]) -> T,
{
    let mut scratch = Vec::new();
    elementwise_batched(
        |batch, output| {
            scratch.clear();
            scratch.reserve(batch.n_inputs());
            for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
                scratch.clear();
                for input in 0..batch.n_inputs() {
                    scratch.push(batch.get(input, point)?);
                }
                *value = op(&scratch);
            }
            Ok(())
        },
        inputs,
        options,
    )
}

fn ranks_are_stable(ranks: &[usize], min_iters: usize) -> bool {
    if ranks.is_empty() || min_iters == 0 {
        return min_iters == 0;
    }

    if ranks.len() > min_iters {
        let baseline_index = ranks.len() - min_iters - 1;
        let baseline = ranks[baseline_index];
        ranks[baseline_index + 1..]
            .iter()
            .all(|&rank| rank <= baseline)
    } else {
        let baseline = ranks[0];
        ranks.iter().all(|&rank| rank <= baseline)
    }
}
