//! Fixed-rank tensor type backed by `tenferro_tensor::TypedTensor<T>`.
//!
//! This wrapper preserves a compile-time rank while delegating storage and
//! indexing to tenferro's typed dense tensor implementation.

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use tenferro_tensor::{TensorScalar, TypedTensor as TfTensor};
use tensor4all_tensorbackend::invariant_violated;

use crate::error::{Result, SimpleTensorTrainError};

/// Rank-N host-backed tensor backed by `tenferro_tensor::TypedTensor<T>`.
#[derive(Debug)]
pub struct Tensor<T: TensorScalar, const N: usize> {
    inner: TfTensor<T>,
    dims: [usize; N],
}

/// Iterator over tensor elements in column-major order.
pub struct TensorIter<'a, T: TensorScalar, const N: usize> {
    tensor: &'a TfTensor<T>,
    dims: [usize; N],
    next: usize,
    len: usize,
}

/// Mutable iterator over tensor elements in column-major order.
pub struct TensorIterMut<'a, T: TensorScalar, const N: usize> {
    tensor: *mut TfTensor<T>,
    dims: [usize; N],
    next: usize,
    len: usize,
    _marker: PhantomData<&'a mut TfTensor<T>>,
}

/// 2D tensor (matrix).
pub type Tensor2<T> = Tensor<T, 2>;

/// 3D tensor.
pub type Tensor3<T> = Tensor<T, 3>;

/// 4D tensor.
pub type Tensor4<T> = Tensor<T, 4>;

fn col_major_index_from_linear<const N: usize>(mut linear: usize, dims: &[usize; N]) -> [usize; N] {
    let mut idx = [0usize; N];
    for axis in 0..N {
        let dim = dims[axis];
        if dim == 0 {
            return idx;
        }
        idx[axis] = linear % dim;
        linear /= dim;
    }
    idx
}

fn col_major_data_to_tensor<T: TensorScalar, const N: usize>(
    dims: [usize; N],
    data: Vec<T>,
) -> Result<Tensor<T, N>> {
    let inner = TfTensor::from_vec_col_major(dims.to_vec(), data).map_err(|error| {
        SimpleTensorTrainError::InvalidOperation {
            message: format!("invalid tensor shape or data: {error}"),
        }
    })?;
    Ok(Tensor::from_tenferro_unchecked(inner))
}

impl<T: TensorScalar, const N: usize> Clone for Tensor<T, N> {
    fn clone(&self) -> Self {
        // INVARIANT: constructors validate host accessibility, and mutable inner
        // access documents that callers must preserve this wrapper invariant.
        Self {
            inner: self.inner.duplicate().unwrap_or_else(|error| {
                invariant_violated("host-backed tensor duplication failed", error)
            }),
            dims: self.dims,
        }
    }
}

impl<T: TensorScalar + PartialEq, const N: usize> PartialEq for Tensor<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.dims() == other.dims() && self.iter().eq(other.iter())
    }
}

impl<T: TensorScalar + Eq, const N: usize> Eq for Tensor<T, N> {}

impl<'a, T: TensorScalar, const N: usize> Iterator for TensorIter<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.len {
            return None;
        }

        let idx = col_major_index_from_linear(self.next, &self.dims);
        self.next += 1;
        Some(
            self.tensor
                .get(&idx[..])
                .unwrap_or_else(|error| invariant_violated("tensor index failed", error)),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.next;
        (remaining, Some(remaining))
    }
}

impl<'a, T: TensorScalar, const N: usize> ExactSizeIterator for TensorIter<'a, T, N> {}

impl<'a, T: TensorScalar, const N: usize> Iterator for TensorIterMut<'a, T, N> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.len {
            return None;
        }

        let idx = col_major_index_from_linear(self.next, &self.dims);
        self.next += 1;

        // Safety: each logical index is visited at most once, so returned
        // mutable references never alias each other.
        let tensor = unsafe { &mut *self.tensor };
        let elem = tensor
            .get_mut(&idx[..])
            .unwrap_or_else(|error| invariant_violated("tensor index failed", error));
        let ptr = elem as *mut T;
        Some(unsafe { &mut *ptr })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.next;
        (remaining, Some(remaining))
    }
}

impl<'a, T: TensorScalar, const N: usize> ExactSizeIterator for TensorIterMut<'a, T, N> {}

impl<T: TensorScalar, const N: usize> Tensor<T, N> {
    pub(crate) fn from_tenferro_unchecked(tensor: TfTensor<T>) -> Self {
        debug_assert_eq!(
            tensor.shape().len(),
            N,
            "tensor rank mismatch: expected rank {N}, got {}",
            tensor.shape().len()
        );
        // INVARIANT: crate-private callers construct or receive CPU host tensors.
        debug_assert!(
            tensor.host_data().is_ok(),
            "simplett tensors must remain host-backed"
        );
        let mut dims = [0usize; N];
        for (dim, value) in dims.iter_mut().zip(tensor.shape().iter().copied()) {
            *dim = value;
        }
        Self {
            inner: tensor,
            dims,
        }
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.inner.n_elements()
    }

    /// Whether the tensor is empty (zero elements).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dimension along `axis`.
    pub fn dim(&self, axis: usize) -> usize {
        self.dims()[axis]
    }

    /// All dimensions.
    pub fn dims(&self) -> &[usize; N] {
        &self.dims
    }

    /// Export the host-backed tensor as a column-major flat vector.
    ///
    /// # Panics
    ///
    /// Panics if crate-internal code has violated the wrapper's host-backed
    /// storage invariant.
    pub fn to_col_major_vec(&self) -> Vec<T> {
        // INVARIANT: public and crate-private constructors validate that a
        // simplett tensor is host-backed, and `as_inner_mut` requires callers
        // to preserve that invariant.
        self.inner
            .host_data()
            .unwrap_or_else(|error| {
                invariant_violated("host-backed tensor data is unavailable", error)
            })
            .to_vec()
    }

    /// Iterate over all elements in column-major order.
    pub fn iter(&self) -> TensorIter<'_, T, N> {
        TensorIter {
            tensor: &self.inner,
            dims: *self.dims(),
            next: 0,
            len: self.len(),
        }
    }

    /// Iterate mutably over all elements in column-major order.
    pub fn iter_mut(&mut self) -> TensorIterMut<'_, T, N> {
        TensorIterMut {
            tensor: &mut self.inner,
            dims: *self.dims(),
            next: 0,
            len: self.len(),
            _marker: PhantomData,
        }
    }

    /// Borrow the wrapped tenferro tensor.
    pub fn as_inner(&self) -> &TfTensor<T> {
        &self.inner
    }

    /// Consume this wrapper and return the inner tenferro tensor.
    pub fn into_inner(self) -> TfTensor<T> {
        self.inner
    }

    /// Try to wrap an existing tenferro tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor rank does not match `N` or its data is
    /// not host-accessible.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::tensor::{Tensor2, Tensor3};
    /// use tenferro_tensor::TypedTensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rank_2 = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// let tensor = Tensor2::try_from_tenferro(rank_2)?;
    /// assert_eq!(tensor.dims(), &[2, 2]);
    ///
    /// let rank_2 = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// assert!(Tensor3::try_from_tenferro(rank_2).is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_from_tenferro(tensor: TfTensor<T>) -> Result<Self> {
        Self::from_tenferro(tensor)
    }

    /// Wrap an existing tenferro tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor rank does not match `N` or its data is
    /// not host-accessible.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::tensor::{Tensor2, Tensor3};
    /// use tenferro_tensor::TypedTensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rank_2 = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// let tensor = Tensor2::from_tenferro(rank_2)?;
    /// assert_eq!(tensor.dims(), &[2, 2]);
    ///
    /// let rank_2 = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// assert!(Tensor3::from_tenferro(rank_2).is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_tenferro(tensor: TfTensor<T>) -> Result<Self> {
        if tensor.shape().len() != N {
            return Err(SimpleTensorTrainError::InvalidOperation {
                message: format!(
                    "tensor rank mismatch: expected rank {N}, got {}",
                    tensor.shape().len()
                ),
            });
        }
        tensor
            .host_data()
            .map_err(|error| SimpleTensorTrainError::InvalidOperation {
                message: format!("simplett tensors must be host-backed: {error}"),
            })?;
        let mut dims = [0usize; N];
        dims.copy_from_slice(tensor.shape());
        Ok(Self {
            inner: tensor,
            dims,
        })
    }

    /// Create a tensor by applying `f` to each multi-index in column-major order.
    ///
    /// # Errors
    ///
    /// Returns [`SimpleTensorTrainError::InvalidOperation`] when the shape product
    /// overflows, allocation cannot reserve the required capacity, or tenferro
    /// rejects the generated shape and data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::tensor::Tensor2;
    ///
    /// let tensor = Tensor2::from_fn([2, 2], |[i, j]| (i + 2 * j) as f64)?;
    /// assert_eq!(tensor.to_col_major_vec(), vec![0.0, 1.0, 2.0, 3.0]);
    /// # Ok::<(), tensor4all_simplett::SimpleTensorTrainError>(())
    /// ```
    pub fn from_fn(dims: [usize; N], mut f: impl FnMut([usize; N]) -> T) -> Result<Self> {
        let total = dims
            .iter()
            .try_fold(1usize, |total, &dim| total.checked_mul(dim))
            .ok_or_else(|| SimpleTensorTrainError::InvalidOperation {
                message: format!("tensor shape product overflows usize: {dims:?}"),
            })?;
        let mut data = Vec::new();
        data.try_reserve_exact(total).map_err(|error| {
            SimpleTensorTrainError::InvalidOperation {
                message: format!("cannot allocate tensor shape {dims:?}: {error}"),
            }
        })?;

        for linear in 0..total {
            data.push(f(col_major_index_from_linear(linear, &dims)));
        }

        col_major_data_to_tensor(dims, data)
    }
}

impl<T: TensorScalar, const N: usize> Tensor<T, N> {
    /// Create a tensor filled with `value`.
    ///
    /// # Errors
    ///
    /// Returns [`SimpleTensorTrainError::InvalidOperation`] when the shape product
    /// overflows, allocation cannot reserve the required capacity, or tenferro
    /// rejects the generated shape and data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::tensor::Tensor2;
    ///
    /// let tensor = Tensor2::from_elem([2, 3], 4.0_f64)?;
    /// assert_eq!(tensor.to_col_major_vec(), vec![4.0; 6]);
    /// # Ok::<(), tensor4all_simplett::SimpleTensorTrainError>(())
    /// ```
    pub fn from_elem(dims: [usize; N], value: T) -> Result<Self> {
        let total = dims
            .iter()
            .try_fold(1usize, |total, &dim| total.checked_mul(dim))
            .ok_or_else(|| SimpleTensorTrainError::InvalidOperation {
                message: format!("tensor shape product overflows usize: {dims:?}"),
            })?;
        let mut data = Vec::new();
        data.try_reserve_exact(total).map_err(|error| {
            SimpleTensorTrainError::InvalidOperation {
                message: format!("cannot allocate tensor shape {dims:?}: {error}"),
            }
        })?;
        data.resize(total, value);
        col_major_data_to_tensor(dims, data)
    }
}

impl<T: TensorScalar, const N: usize> Index<[usize; N]> for Tensor<T, N> {
    type Output = T;

    /// # Panics
    ///
    /// Panics if any element of `idx` is out of bounds, following the standard
    /// [`Index`] contract of `Vec` and other collections.
    fn index(&self, idx: [usize; N]) -> &T {
        self.inner
            .get(&idx[..])
            .unwrap_or_else(|error| invariant_violated("tensor index failed", error))
    }
}

impl<T: TensorScalar, const N: usize> IndexMut<[usize; N]> for Tensor<T, N> {
    /// # Panics
    ///
    /// Panics if any element of `idx` is out of bounds, following the standard
    /// [`IndexMut`] contract of `Vec` and other collections.
    fn index_mut(&mut self, idx: [usize; N]) -> &mut T {
        self.inner
            .get_mut(&idx[..])
            .unwrap_or_else(|error| invariant_violated("tensor index failed", error))
    }
}

#[cfg(test)]
mod tests;
