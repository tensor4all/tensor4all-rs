//! Batched input views for elementwise ACI operators.

use crate::{AciError, Result};

/// Borrowed column-major input batch for an elementwise operator.
///
/// `ElementwiseBatch` views a flat slice as a matrix with `n_inputs` rows and
/// `n_points` columns. Values use column-major layout: the value for input
/// `input` at interpolation point `point` is stored at
/// `input + n_inputs * point`.
///
/// Related types: [`AciOptions`](crate::AciOptions) configures ACI runs that
/// evaluate operators over batches, while [`AciError`] reports shape and access
/// errors for invalid batches.
///
/// # Examples
///
/// ```
/// use tensor4all_aci::ElementwiseBatch;
///
/// let values = [10.0, 20.0, 11.0, 21.0, 12.0, 22.0];
/// let batch = ElementwiseBatch::new(&values, 2, 3).unwrap();
///
/// assert_eq!(batch.n_inputs(), 2);
/// assert_eq!(batch.n_points(), 3);
/// assert_eq!(batch.get(0, 0).unwrap(), 10.0);
/// assert_eq!(batch.get(1, 0).unwrap(), 20.0);
/// assert_eq!(batch.get(0, 2).unwrap(), 12.0);
/// assert_eq!(batch.get(1, 2).unwrap(), 22.0);
/// assert_eq!(batch.as_col_major_slice(), values.as_slice());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ElementwiseBatch<'a, T> {
    values: &'a [T],
    n_inputs: usize,
    n_points: usize,
}

impl<'a, T> ElementwiseBatch<'a, T> {
    /// Creates a borrowed column-major batch view.
    ///
    /// `values` must contain exactly `n_inputs * n_points` entries in
    /// column-major order. Both `n_inputs` and `n_points` must be nonzero.
    ///
    /// # Arguments
    ///
    /// * `values` - Flat column-major values with inputs varying fastest.
    /// * `n_inputs` - Number of operator inputs per interpolation point.
    /// * `n_points` - Number of interpolation points in the batch.
    ///
    /// # Returns
    ///
    /// Returns an [`ElementwiseBatch`] borrowing `values` when the length and
    /// dimensions are valid.
    ///
    /// # Errors
    ///
    /// Returns [`AciError::EmptyInputs`] when `n_inputs` or `n_points` is zero,
    /// [`AciError::InvalidOptions`] if the product overflows `usize`, and
    /// [`AciError::LengthMismatch`] if `values.len()` does not equal
    /// `n_inputs * n_points`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_aci::ElementwiseBatch;
    ///
    /// let values = [1, 2, 3, 4];
    /// let batch = ElementwiseBatch::new(&values, 2, 2).unwrap();
    /// assert_eq!(batch.n_inputs(), 2);
    /// assert_eq!(batch.n_points(), 2);
    /// assert_eq!(batch.get(1, 1).unwrap(), 4);
    /// ```
    ///
    /// ```
    /// use tensor4all_aci::{AciError, ElementwiseBatch};
    ///
    /// let err = ElementwiseBatch::<f64>::new(&[1.0, 2.0, 3.0], 2, 2).unwrap_err();
    /// assert!(matches!(err, AciError::LengthMismatch { expected: 4, got: 3 }));
    /// assert!(err.to_string().contains("length"));
    /// ```
    pub fn new(values: &'a [T], n_inputs: usize, n_points: usize) -> Result<Self> {
        if n_inputs == 0 || n_points == 0 {
            return Err(AciError::EmptyInputs);
        }

        let expected = n_inputs
            .checked_mul(n_points)
            .ok_or_else(|| AciError::InvalidOptions {
                message: format!(
                    "batch shape overflows usize: n_inputs={n_inputs}, n_points={n_points}"
                ),
            })?;

        if values.len() != expected {
            return Err(AciError::LengthMismatch {
                expected,
                got: values.len(),
            });
        }

        Ok(Self {
            values,
            n_inputs,
            n_points,
        })
    }

    /// Returns the number of operator inputs per interpolation point.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_aci::ElementwiseBatch;
    ///
    /// let values = [0.0, 1.0, 2.0, 3.0];
    /// let batch = ElementwiseBatch::new(&values, 2, 2).unwrap();
    /// assert_eq!(batch.n_inputs(), 2);
    /// ```
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    /// Returns the number of interpolation points in the batch.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_aci::ElementwiseBatch;
    ///
    /// let values = [0.0, 1.0, 2.0, 3.0];
    /// let batch = ElementwiseBatch::new(&values, 2, 2).unwrap();
    /// assert_eq!(batch.n_points(), 2);
    /// ```
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Returns one value using column-major indexing.
    ///
    /// The returned value is `values[input + n_inputs * point]`, so `input`
    /// varies fastest in the flat buffer.
    ///
    /// # Arguments
    ///
    /// * `input` - Zero-based input index, less than [`n_inputs`](Self::n_inputs).
    /// * `point` - Zero-based interpolation point, less than [`n_points`](Self::n_points).
    ///
    /// # Returns
    ///
    /// Returns a copy of the requested value.
    ///
    /// # Errors
    ///
    /// Returns [`AciError::BatchIndexOutOfBounds`] if `input` or `point` is
    /// outside the corresponding batch axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_aci::ElementwiseBatch;
    ///
    /// let values = [10, 20, 11, 21];
    /// let batch = ElementwiseBatch::new(&values, 2, 2).unwrap();
    /// assert_eq!(batch.get(0, 1).unwrap(), 11);
    /// assert_eq!(batch.get(1, 1).unwrap(), 21);
    /// ```
    ///
    /// ```
    /// use tensor4all_aci::{AciError, ElementwiseBatch};
    ///
    /// let values = [10, 20, 11, 21];
    /// let batch = ElementwiseBatch::new(&values, 2, 2).unwrap();
    /// let err = batch.get(2, 0).unwrap_err();
    /// assert!(matches!(
    ///     err,
    ///     AciError::BatchIndexOutOfBounds {
    ///         axis: "input",
    ///         index: 2,
    ///         len: 2
    ///     }
    /// ));
    /// ```
    pub fn get(&self, input: usize, point: usize) -> Result<T>
    where
        T: Copy,
    {
        if input >= self.n_inputs {
            return Err(AciError::BatchIndexOutOfBounds {
                axis: "input",
                index: input,
                len: self.n_inputs,
            });
        }
        if point >= self.n_points {
            return Err(AciError::BatchIndexOutOfBounds {
                axis: "point",
                index: point,
                len: self.n_points,
            });
        }

        Ok(self.values[input + self.n_inputs * point])
    }

    /// Returns the borrowed flat slice in column-major input/point layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_aci::ElementwiseBatch;
    ///
    /// let values = [10.0, 20.0, 11.0, 21.0];
    /// let batch = ElementwiseBatch::new(&values, 2, 2).unwrap();
    /// assert_eq!(batch.as_col_major_slice(), values.as_slice());
    /// ```
    pub fn as_col_major_slice(&self) -> &'a [T] {
        self.values
    }
}
