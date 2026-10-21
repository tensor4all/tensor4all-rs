//! Contraction operations for tensor trains
//!
//! This module provides various ways to combine tensor trains:
//! - `inner_product`: Inner product (returns scalar)

use crate::compression::CompressionMethod;
use crate::einsum_helper::{
    einsum_tensors, tensor_to_col_major_vec, typed_tensor_from_col_major_slice, EinsumScalar,
};
use crate::error::{Result, SimpleTensorTrainError};
use crate::tensortrain::SimpleTensorTrain;
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::Tensor3Ops;
use tensor4all_core::Scalar;
use tensor4all_tensorbackend::Matrix;

fn contraction_helper_error(context: &str, err: impl std::fmt::Display) -> SimpleTensorTrainError {
    SimpleTensorTrainError::InvalidOperation {
        message: format!("{context}: {err}"),
    }
}

fn inner_product_read_error(err: impl std::fmt::Display) -> SimpleTensorTrainError {
    contraction_helper_error("Failed to read inner_product result", err)
}

/// Options for MPO-MPO contraction with on-the-fly compression.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::ContractionOptions;
///
/// let opts = ContractionOptions::default();
/// assert!((opts.tolerance - 1e-12).abs() < 1e-15);
/// assert_eq!(opts.max_bond_dim, None);
/// ```
#[derive(Debug, Clone)]
pub struct ContractionOptions {
    /// Relative truncation tolerance during contraction.
    pub tolerance: f64,
    /// Hard upper bound on bond dimension. `None` (default) means no limit.
    pub max_bond_dim: Option<usize>,
    /// Decomposition method for intermediate compression.
    pub method: CompressionMethod,
}

impl Default for ContractionOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_bond_dim: None,
            method: CompressionMethod::LU,
        }
    }
}

impl<T: TTScalar + Scalar + Default + EinsumScalar> SimpleTensorTrain<T> {
    /// Inner product (inner_product product) of two tensor trains.
    ///
    /// Computes `sum_i self[i] * other[i]` by contracting the site tensors
    /// from left to right. Both tensor trains must have the same length and
    /// matching site dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`SimpleTensorTrainError::InvalidOperation`] when the two trains
    /// differ in length or site dimensions, or when an intermediate environment
    /// contraction cannot be read.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain};
    ///
    /// let a = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    /// let b = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
    ///
    /// // inner_product = sum_ij a[i,j]*b[i,j] = 1*2 * 2*3 = 12
    /// let d = a.inner_product(&b).unwrap();
    /// assert!((d - 12.0).abs() < 1e-10);
    /// ```
    pub fn inner_product(&self, other: &Self) -> Result<T> {
        if self.len() != other.len() {
            return Err(SimpleTensorTrainError::InvalidOperation {
                message: format!(
                    "Cannot compute inner_product product of tensor trains with different lengths: {} vs {}",
                    self.len(),
                    other.len()
                ),
            });
        }

        if self.is_empty() {
            return Ok(T::zero());
        }

        let n = self.len();

        // Start with contraction of first site
        // result[ra, rb] = sum_s a[0, s, ra] * b[0, s, rb]
        let a0 = self.site_tensor(0);
        let b0 = other.site_tensor(0);

        if a0.site_dim() != b0.site_dim() {
            return Err(SimpleTensorTrainError::InvalidOperation {
                message: format!(
                    "Site dimensions mismatch at site 0: {} vs {}",
                    a0.site_dim(),
                    b0.site_dim()
                ),
            });
        }

        let mut result = Matrix::from_col_major_vec(
            a0.right_dim(),
            b0.right_dim(),
            tensor_to_col_major_vec(
                &einsum_tensors("asr,ast->rt", &[a0.as_inner(), b0.as_inner()]).map_err(|err| {
                    contraction_helper_error("Failed to contract first inner_product site", err)
                })?,
            )
            .map_err(inner_product_read_error)?,
        );

        // Contract through remaining sites
        for i in 1..n {
            let a = self.site_tensor(i);
            let b = other.site_tensor(i);

            if a.site_dim() != b.site_dim() {
                return Err(SimpleTensorTrainError::InvalidOperation {
                    message: format!(
                        "Site dimensions mismatch at site {}: {} vs {}",
                        i,
                        a.site_dim(),
                        b.site_dim()
                    ),
                });
            }

            let result_tf = typed_tensor_from_col_major_slice(
                result.as_col_major_slice(),
                &[result.nrows(), result.ncols()],
            )
            .map_err(|err| {
                contraction_helper_error("Failed to prepare intermediate inner_product env", err)
            })?;

            result = Matrix::from_col_major_vec(
                a.right_dim(),
                b.right_dim(),
                tensor_to_col_major_vec(
                    &einsum_tensors("ij,isk,jsl->kl", &[&result_tf, a.as_inner(), b.as_inner()])
                        .map_err(|err| {
                            contraction_helper_error(
                                &format!("Failed to contract inner_product site {i}"),
                                err,
                            )
                        })?,
                )
                .map_err(inner_product_read_error)?,
            );
        }

        // Final result should be 1x1
        if result.nrows() != 1 || result.ncols() != 1 {
            return Err(SimpleTensorTrainError::InvalidOperation {
                message: format!(
                    "Final contraction resulted in a {:?} environment, expected [1, 1]",
                    (result.nrows(), result.ncols())
                ),
            });
        }
        Ok(result[[0, 0]])
    }

    /// Compute the inner product with another train on a caller-owned explicit `Session`.
    ///
    /// The same site-by-site computation as [`SimpleTensorTrain::inner_product`], with every contraction
    /// running on the caller's session: the operands are read through borrowed views instead of being
    /// duplicated, and the running environment stays a concrete tensor between sites instead of being
    /// converted into a [`Matrix`] and back at every site. The method performs no session entry of its own,
    /// so a caller can hold one session across a wider region.
    ///
    /// The result is this crate's bilinear product, with no conjugation of either operand, exactly as the
    /// compatibility entry point computes it.
    ///
    /// # Errors
    ///
    /// Returns [`SimpleTensorTrainError::InvalidOperation`] when the trains have different lengths, when a
    /// site dimension differs at site 0 or at a later site, when the session rejects a contraction, when the
    /// final environment cannot be read back, or when it is not the 1x1 environment the train invariants
    /// imply - a core replaced through `site_tensors_mut` can break that.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain};
    /// use tensor4all_tensorbackend::CpuExecutionContext;
    /// use tensor4all_tensorbackend::CpuBackend;
    ///
    /// let a = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    /// let b = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
    /// let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?);
    /// let value = context.with_concrete_session(|session| {
    ///     a.inner_product_with_session(&b, session)
    /// })??;
    /// assert!((value - 12.0).abs() < 1e-10);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn inner_product_with_session(
        &self,
        other: &Self,
        session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    ) -> Result<T> {
        if self.len() != other.len() {
            return Err(SimpleTensorTrainError::InvalidOperation {
                message: format!(
                    "Cannot compute inner_product product of tensor trains with different lengths: {} vs {}",
                    self.len(),
                    other.len()
                ),
            });
        }

        if self.is_empty() {
            return Ok(T::zero());
        }

        let n = self.len();

        let a0 = self.site_tensor(0);
        let b0 = other.site_tensor(0);
        if a0.site_dim() != b0.site_dim() {
            return Err(SimpleTensorTrainError::InvalidOperation {
                message: format!(
                    "Site dimensions mismatch at site 0: {} vs {}",
                    a0.site_dim(),
                    b0.site_dim()
                ),
            });
        }

        // result[ra, rb] = sum_s a[0, s, ra] * b[0, s, rb]
        let mut environment = session
            .einsum_reads(
                &[T::tensor_read(a0.as_inner()), T::tensor_read(b0.as_inner())],
                &[&[0, 1, 2], &[0, 1, 3]],
                &[2, 3],
            )
            .map_err(|err| {
                contraction_helper_error("Failed to contract first inner_product site", err)
            })?;

        for i in 1..n {
            let a = self.site_tensor(i);
            let b = other.site_tensor(i);

            if a.site_dim() != b.site_dim() {
                return Err(SimpleTensorTrainError::InvalidOperation {
                    message: format!(
                        "Site dimensions mismatch at site {}: {} vs {}",
                        i,
                        a.site_dim(),
                        b.site_dim()
                    ),
                });
            }

            // The environment stays a concrete tensor between sites; only the contraction runs on the
            // session, and the environment read borrows it for the duration of that one call.
            environment = session
                .einsum_reads(
                    &[
                        tenferro_tensor::TensorRead::from_tensor(&environment),
                        T::tensor_read(a.as_inner()),
                        T::tensor_read(b.as_inner()),
                    ],
                    // The environment carries (a's left bond, b's left bond); each site tensor joins on its
                    // own left bond and on the shared site axis.
                    &[&[0, 1], &[0, 2, 3], &[1, 2, 4]],
                    &[3, 4],
                )
                .map_err(|err| {
                    contraction_helper_error(
                        &format!("Failed to contract inner_product site {i}"),
                        err,
                    )
                })?;
        }

        // A valid train ends with unit boundary bonds, so the environment is 1x1 and the scalar is its only
        // element. Reading it checked turns a malformed train into a typed error instead of a silent
        // difference from the compatibility route, which selects the same element.
        let values = environment
            .as_slice::<T>()
            .map_err(inner_product_read_error)?;
        match values {
            [value] => Ok(*value),
            other => Err(contraction_helper_error(
                "inner_product_with_session",
                format!(
                    "the final environment must hold one element, got {}",
                    other.len()
                ),
            )),
        }
    }
}

/// Free-function wrapper for [`SimpleTensorTrain::inner_product`].
///
/// # Errors
///
/// Returns [`SimpleTensorTrainError::InvalidOperation`], the same error as
/// [`SimpleTensorTrain::inner_product`].
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain, contraction::inner_product};
///
/// let a = SimpleTensorTrain::<f64>::constant(&[2, 3], 3.0);
/// let b = SimpleTensorTrain::<f64>::constant(&[2, 3], 4.0);
/// let d = inner_product(&a, &b).unwrap();
/// // 3*4 * 2*3 = 72
/// assert!((d - 72.0).abs() < 1e-10);
/// ```
pub fn inner_product<T: TTScalar + Scalar + Default + EinsumScalar>(
    a: &SimpleTensorTrain<T>,
    b: &SimpleTensorTrain<T>,
) -> Result<T> {
    a.inner_product(b)
}

#[cfg(test)]
mod tests;
