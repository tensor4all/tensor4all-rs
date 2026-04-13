//! Fixed-rank tensor type backed by `tenferro_tensor::TypedTensor<T>`.
//!
//! This wrapper preserves a compile-time rank while delegating storage and
//! indexing to tenferro's typed dense tensor implementation.

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use tenferro_tensor::{TensorScalar, TypedTensor as TfTensor};

use crate::einsum_helper::{tensor_to_row_major_vec, typed_tensor_from_row_major_slice};

/// Rank-N tensor backed by `tenferro_tensor::TypedTensor<T>`.
#[derive(Debug)]
pub struct Tensor<T: TensorScalar, const N: usize>(TfTensor<T>);

/// Iterator over tensor elements in row-major order.
pub struct TensorIter<'a, T: TensorScalar, const N: usize> {
    tensor: &'a TfTensor<T>,
    dims: [usize; N],
    next: usize,
    len: usize,
}

/// Mutable iterator over tensor elements in row-major order.
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

fn row_major_index_from_linear<const N: usize>(mut linear: usize, dims: &[usize; N]) -> [usize; N] {
    let mut idx = [0usize; N];
    for axis in (0..N).rev() {
        let dim = dims[axis];
        if dim == 0 {
            return idx;
        }
        idx[axis] = linear % dim;
        linear /= dim;
    }
    idx
}

fn row_major_data_to_tensor<T: TensorScalar, const N: usize>(
    dims: [usize; N],
    data: Vec<T>,
) -> Tensor<T, N> {
    let inner = typed_tensor_from_row_major_slice(&data, &dims);
    Tensor::from_tenferro(inner)
}

impl<T: TensorScalar, const N: usize> Clone for Tensor<T, N> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
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

        let idx = row_major_index_from_linear(self.next, &self.dims);
        self.next += 1;
        Some(self.tensor.get(&idx[..]))
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

        let idx = row_major_index_from_linear(self.next, &self.dims);
        self.next += 1;

        // Safety: each logical index is visited at most once, so returned
        // mutable references never alias each other.
        let tensor = unsafe { &mut *self.tensor };
        let elem = tensor.get_mut(&idx[..]);
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
    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.0.n_elements()
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
        self.0.shape.as_slice().try_into().unwrap_or_else(|_| {
            panic!(
                "tensor rank mismatch: expected rank {N}, got {}",
                self.0.shape.len()
            )
        })
    }

    /// Export the tensor as a row-major flat vector.
    pub fn to_row_major_vec(&self) -> Vec<T> {
        tensor_to_row_major_vec(&self.0)
    }

    /// Iterate over all elements in row-major order.
    pub fn iter(&self) -> TensorIter<'_, T, N> {
        TensorIter {
            tensor: &self.0,
            dims: *self.dims(),
            next: 0,
            len: self.len(),
        }
    }

    /// Iterate mutably over all elements in row-major order.
    pub fn iter_mut(&mut self) -> TensorIterMut<'_, T, N> {
        TensorIterMut {
            tensor: &mut self.0,
            dims: *self.dims(),
            next: 0,
            len: self.len(),
            _marker: PhantomData,
        }
    }

    /// Borrow the wrapped tenferro tensor.
    pub fn as_inner(&self) -> &TfTensor<T> {
        &self.0
    }

    /// Mutably borrow the wrapped tenferro tensor.
    pub fn as_inner_mut(&mut self) -> &mut TfTensor<T> {
        &mut self.0
    }

    /// Consume this wrapper and return the inner tenferro tensor.
    pub fn into_inner(self) -> TfTensor<T> {
        self.0
    }

    /// Wrap an existing tenferro tensor, panicking on rank mismatch.
    pub fn from_tenferro(tensor: TfTensor<T>) -> Self {
        if tensor.shape.len() != N {
            panic!(
                "tensor rank mismatch: expected rank {N}, got {}",
                tensor.shape.len()
            );
        }
        Self(tensor)
    }

    /// Create a tensor by applying `f` to each multi-index (row-major order).
    pub fn from_fn(dims: [usize; N], mut f: impl FnMut([usize; N]) -> T) -> Self {
        let total: usize = dims.iter().product();
        let mut data = Vec::with_capacity(total);

        for linear in 0..total {
            data.push(f(row_major_index_from_linear(linear, &dims)));
        }

        row_major_data_to_tensor(dims, data)
    }
}

impl<T: TensorScalar, const N: usize> Tensor<T, N> {
    /// Create a tensor filled with `value`.
    pub fn from_elem(dims: [usize; N], value: T) -> Self {
        let total: usize = dims.iter().product();
        row_major_data_to_tensor(dims, vec![value; total])
    }
}

impl<T: TensorScalar, const N: usize> Index<[usize; N]> for Tensor<T, N> {
    type Output = T;

    fn index(&self, idx: [usize; N]) -> &T {
        self.0.get(&idx[..])
    }
}

impl<T: TensorScalar, const N: usize> IndexMut<[usize; N]> for Tensor<T, N> {
    fn index_mut(&mut self, idx: [usize; N]) -> &mut T {
        self.0.get_mut(&idx[..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::einsum_helper::{tensor_to_row_major_vec, typed_tensor_from_row_major_slice};

    #[test]
    fn test_tensor2_from_elem() {
        let t: Tensor2<f64> = Tensor2::from_elem([3, 4], 0.0);
        assert_eq!(t.len(), 12);
        assert_eq!(t.dim(0), 3);
        assert_eq!(t.dim(1), 4);
        assert_eq!(t[[0, 0]], 0.0);
    }

    #[test]
    fn test_tensor2_indexing() {
        let mut t: Tensor2<f64> = Tensor2::from_elem([2, 3], 0.0);
        t[[0, 0]] = 1.0;
        t[[0, 1]] = 2.0;
        t[[0, 2]] = 3.0;
        t[[1, 0]] = 4.0;
        t[[1, 1]] = 5.0;
        t[[1, 2]] = 6.0;
        assert_eq!(t[[0, 0]], 1.0);
        assert_eq!(t[[1, 2]], 6.0);
        assert_eq!(t.to_row_major_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor3_from_fn() {
        let t: Tensor3<f64> =
            Tensor3::from_fn([2, 3, 4], |[i, j, k]| (i * 100 + j * 10 + k) as f64);
        assert_eq!(t[[0, 0, 0]], 0.0);
        assert_eq!(t[[1, 2, 3]], 123.0);
        assert_eq!(t[[0, 1, 2]], 12.0);
        assert_eq!(t.dim(0), 2);
        assert_eq!(t.dim(1), 3);
        assert_eq!(t.dim(2), 4);
        assert_eq!(t.len(), 24);
    }

    #[test]
    fn test_tensor4_from_fn() {
        let t: Tensor4<f64> = Tensor4::from_fn([2, 3, 4, 5], |[i, j, k, l]| {
            (i * 1000 + j * 100 + k * 10 + l) as f64
        });
        assert_eq!(t[[1, 2, 3, 4]], 1234.0);
        assert_eq!(t.dim(0), 2);
        assert_eq!(t.dim(1), 3);
        assert_eq!(t.dim(2), 4);
        assert_eq!(t.dim(3), 5);
        assert_eq!(t.len(), 120);
    }

    #[test]
    fn test_iter() {
        let t: Tensor2<f64> = Tensor2::from_fn([2, 3], |[i, j]| (i * 3 + j) as f64);
        let collected: Vec<f64> = t.iter().copied().collect();
        assert_eq!(collected, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_dims() {
        let t: Tensor3<f64> = Tensor3::from_elem([2, 3, 4], 1.0);
        assert_eq!(t.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_from_tenferro_roundtrip() {
        let inner = typed_tensor_from_row_major_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tensor = Tensor2::from_tenferro(inner);

        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(
            tensor.to_row_major_vec(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        let inner_again = tensor.clone().into_inner();
        assert_eq!(
            tensor_to_row_major_vec(&inner_again),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    #[should_panic(expected = "rank")]
    fn test_from_tenferro_rank_mismatch_panics() {
        let inner = typed_tensor_from_row_major_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let _ = Tensor3::from_tenferro(inner);
    }

    #[test]
    fn test_clone_allows_index_mut() {
        let original: Tensor2<f64> = Tensor2::from_fn([2, 2], |[i, j]| (i * 2 + j) as f64);
        let mut cloned = original.clone();
        cloned[[1, 0]] = 99.0;

        assert_eq!(original[[1, 0]], 2.0);
        assert_eq!(cloned[[1, 0]], 99.0);
    }
}
