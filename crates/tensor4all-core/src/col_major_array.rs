//! N-dimensional column-major array types.
//!
//! Column-major layout: the element at multi-index `[i0, i1, i2, ...]` is stored
//! at flat offset `i0 + shape[0] * (i1 + shape[1] * (i2 + ...))`.
//!
//! Three flavors are provided:
//! - [`ColMajorArrayRef`] — borrowed data and shape (read-only)
//! - [`ColMajorArrayMut`] — mutably borrowed data, borrowed shape
//! - [`ColMajorArray`] — fully owned data and shape

/// Errors that can occur when constructing or modifying a column-major array.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum ColMajorArrayError {
    /// The length of the data does not match the product of the shape dimensions.
    #[error("Shape mismatch: shape {shape:?} requires {expected} elements, but got {actual}")]
    ShapeMismatch {
        /// The requested shape.
        shape: Vec<usize>,
        /// Number of elements implied by the shape.
        expected: usize,
        /// Number of elements actually provided.
        actual: usize,
    },

    /// The column length does not match `nrows`.
    #[error("Column length mismatch: expected {expected} elements, but got {actual}")]
    ColumnLengthMismatch {
        /// Expected number of rows.
        expected: usize,
        /// Actual number of elements in the column.
        actual: usize,
    },

    /// A 2D operation was called on an array that is not 2-dimensional.
    #[error("Expected a 2D array, but ndim = {ndim}")]
    Not2D {
        /// The actual number of dimensions.
        ndim: usize,
    },

    /// The product of shape dimensions overflows `usize`.
    #[error("Shape product overflow: shape {shape:?} overflows usize")]
    ShapeOverflow {
        /// The shape that caused the overflow.
        shape: Vec<usize>,
    },

    /// Incrementing the column count would overflow `usize`.
    #[error("Column count overflow")]
    ColumnCountOverflow,
}

// ---------------------------------------------------------------------------
// Helper: compute the total number of elements from a shape
// ---------------------------------------------------------------------------

fn checked_shape_numel(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .copied()
        .try_fold(1usize, |acc, d| acc.checked_mul(d))
}

/// Compute the flat offset for a column-major multi-index, using checked
/// arithmetic. Returns `None` if any index is out of bounds or on overflow.
fn flat_offset(shape: &[usize], index: &[usize]) -> Option<usize> {
    if index.len() != shape.len() {
        return None;
    }
    // Traverse from the last axis to the first:
    //   offset = i_{n-1}
    //   offset = i_{n-2} + shape[n-2] * offset  -- but we build from the back
    // Actually, column-major: offset = i0 + s0*(i1 + s1*(i2 + ...))
    // Evaluate right-to-left (Horner-like):
    let mut offset: usize = 0;
    for (idx, dim) in index.iter().zip(shape.iter()).rev() {
        if *idx >= *dim {
            return None;
        }
        offset = offset.checked_mul(*dim)?.checked_add(*idx)?;
    }
    Some(offset)
}

// ===========================================================================
// ColMajorArrayRef
// ===========================================================================

/// A borrowed, read-only view of an N-dimensional column-major array.
#[derive(Debug, Clone, Copy)]
pub struct ColMajorArrayRef<'a, T> {
    data: &'a [T],
    shape: &'a [usize],
}

impl<'a, T> ColMajorArrayRef<'a, T> {
    /// Create a new borrowed array view.
    ///
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// index mismatch, or a backend failure).
    ///
    pub fn new(data: &'a [T], shape: &'a [usize]) -> Result<Self, ColMajorArrayError> {
        let expected =
            checked_shape_numel(shape).ok_or_else(|| ColMajorArrayError::ShapeOverflow {
                shape: shape.to_vec(),
            })?;
        if data.len() != expected {
            return Err(ColMajorArrayError::ShapeMismatch {
                shape: shape.to_vec(),
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, shape })
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Shape of the array.
    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the array is empty (zero elements).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Flat (contiguous) data slice.
    pub fn data(&self) -> &[T] {
        self.data
    }

    /// Get a reference to the element at the given multi-index, or `None` if
    /// out of bounds.
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        let off = flat_offset(self.shape, index)?;
        self.data.get(off)
    }
}

// ===========================================================================
// ColMajorArrayMut
// ===========================================================================

/// A mutably borrowed view of an N-dimensional column-major array.
#[derive(Debug)]
pub struct ColMajorArrayMut<'a, T> {
    data: &'a mut [T],
    shape: &'a [usize],
}

impl<'a, T> ColMajorArrayMut<'a, T> {
    /// Create a new mutable borrowed array view.
    ///
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// index mismatch, or a backend failure).
    ///
    pub fn new(data: &'a mut [T], shape: &'a [usize]) -> Result<Self, ColMajorArrayError> {
        let expected =
            checked_shape_numel(shape).ok_or_else(|| ColMajorArrayError::ShapeOverflow {
                shape: shape.to_vec(),
            })?;
        if data.len() != expected {
            return Err(ColMajorArrayError::ShapeMismatch {
                shape: shape.to_vec(),
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, shape })
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Shape of the array.
    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the array is empty (zero elements).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Flat (contiguous) data slice (read-only).
    pub fn data(&self) -> &[T] {
        self.data
    }

    /// Flat (contiguous) data slice (mutable).
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    /// Get a reference to the element at the given multi-index, or `None` if
    /// out of bounds.
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        let off = flat_offset(self.shape, index)?;
        self.data.get(off)
    }

    /// Get a mutable reference to the element at the given multi-index, or
    /// `None` if out of bounds.
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        let off = flat_offset(self.shape, index)?;
        self.data.get_mut(off)
    }
}

// ===========================================================================
// ColMajorArray (owned)
// ===========================================================================

/// A fully owned N-dimensional column-major array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColMajorArray<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T> ColMajorArray<T> {
    /// Create a new owned array from data and shape.
    ///
    /// Returns an error if `data.len()` does not equal the product of the
    /// shape dimensions.
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// index mismatch, or a backend failure).
    ///
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, ColMajorArrayError> {
        let expected =
            checked_shape_numel(&shape).ok_or_else(|| ColMajorArrayError::ShapeOverflow {
                shape: shape.clone(),
            })?;
        if data.len() != expected {
            return Err(ColMajorArrayError::ShapeMismatch {
                shape,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, shape })
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Shape of the array.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the array is empty (zero elements).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Flat (contiguous) data slice (read-only).
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Flat (contiguous) data slice (mutable).
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get a reference to the element at the given multi-index, or `None` if
    /// out of bounds.
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        let off = flat_offset(&self.shape, index)?;
        self.data.get(off)
    }

    /// Get a mutable reference to the element at the given multi-index, or
    /// `None` if out of bounds.
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        let off = flat_offset(&self.shape, index)?;
        self.data.get_mut(off)
    }

    /// Consume the array and return the underlying data vector.
    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    /// Borrow as a [`ColMajorArrayRef`].
    pub fn as_ref(&self) -> ColMajorArrayRef<'_, T> {
        ColMajorArrayRef {
            data: &self.data,
            shape: &self.shape,
        }
    }

    /// Borrow as a [`ColMajorArrayMut`].
    pub fn as_mut(&mut self) -> ColMajorArrayMut<'_, T> {
        ColMajorArrayMut {
            data: &mut self.data,
            shape: &self.shape,
        }
    }

    // -- 2D helpers ---------------------------------------------------------

    /// Number of rows, or `None` when the array is not 2D.
    pub fn nrows(&self) -> Option<usize> {
        if self.ndim() == 2 {
            Some(self.shape[0])
        } else {
            None
        }
    }

    /// Number of columns, or `None` when the array is not 2D.
    pub fn ncols(&self) -> Option<usize> {
        if self.ndim() == 2 {
            Some(self.shape[1])
        } else {
            None
        }
    }

    /// Return a slice for column `j` of a 2D array, or `None` if `j` is out
    /// of range or if the array is not 2D.
    pub fn column(&self, j: usize) -> Option<&[T]> {
        if self.ndim() != 2 {
            return None;
        }
        let nrows = self.shape[0];
        if j >= self.shape[1] {
            return None;
        }
        let start = nrows.checked_mul(j)?;
        let end = start.checked_add(nrows)?;
        Some(&self.data[start..end])
    }

    /// Append a column to a 2D array.
    ///
    /// The `col` slice must have length equal to `nrows()`. This extends
    /// the internal data and increments `shape[1]`.
    ///
    /// Returns an error if the array is not 2D or if the column length does
    /// not match `nrows()`.
    /// # Errors
    ///
    /// Returns an error when the column length does not match the row count (a
    /// shape mismatch).
    ///
    pub fn push_column(&mut self, col: &[T]) -> Result<(), ColMajorArrayError>
    where
        T: Clone,
    {
        if self.ndim() != 2 {
            return Err(ColMajorArrayError::Not2D { ndim: self.ndim() });
        }
        let nrows = self.shape[0];
        if col.len() != nrows {
            return Err(ColMajorArrayError::ColumnLengthMismatch {
                expected: nrows,
                actual: col.len(),
            });
        }
        self.data.extend_from_slice(col);
        self.shape[1] = self.shape[1]
            .checked_add(1)
            .ok_or(ColMajorArrayError::ColumnCountOverflow)?;
        Ok(())
    }
}

// -- Factories (require trait bounds on T) ----------------------------------

impl<T: Clone> ColMajorArray<T> {
    /// Create an array filled with a given value.
    ///
    /// Returns an error if the product of shape dimensions overflows `usize`.
    /// # Errors
    ///
    /// Returns an error when the dimensions are invalid (a shape mismatch) or the
    /// fill fails.
    ///
    pub fn filled(shape: Vec<usize>, value: T) -> Result<Self, ColMajorArrayError> {
        let n = checked_shape_numel(&shape).ok_or_else(|| ColMajorArrayError::ShapeOverflow {
            shape: shape.clone(),
        })?;
        Ok(Self {
            data: vec![value; n],
            shape,
        })
    }
}

impl<T: Default + Clone> ColMajorArray<T> {
    /// Create an array filled with [`Default::default()`] (e.g., zeros for
    /// numeric types).
    ///
    /// Returns an error if the product of shape dimensions overflows `usize`.
    /// # Errors
    ///
    /// Returns an error when the dimensions are invalid (a shape mismatch).
    ///
    pub fn zeros(shape: Vec<usize>) -> Result<Self, ColMajorArrayError> {
        let n = checked_shape_numel(&shape).ok_or_else(|| ColMajorArrayError::ShapeOverflow {
            shape: shape.clone(),
        })?;
        Ok(Self {
            data: vec![T::default(); n],
            shape,
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests;
