//! Core types for tensor train operations

use mdarray::DTensor;

/// Local index type (index within a single tensor site)
pub type LocalIndex = usize;

/// Multi-index type (indices across all sites)
pub type MultiIndex = Vec<LocalIndex>;

/// A 3D tensor represented using mdarray
/// Shape is (left_dim, site_dim, right_dim)
pub type Tensor3<T> = DTensor<T, 3>;

/// Helper functions for Tensor3 operations
pub trait Tensor3Ops<T: Clone + Default> {
    /// Get the left (bond) dimension
    fn left_dim(&self) -> usize;

    /// Get the site (physical) dimension
    fn site_dim(&self) -> usize;

    /// Get the right (bond) dimension
    fn right_dim(&self) -> usize;

    /// Get element at (left, site, right)
    fn get3(&self, l: usize, s: usize, r: usize) -> &T;

    /// Get mutable element at (left, site, right)
    fn get3_mut(&mut self, l: usize, s: usize, r: usize) -> &mut T;

    /// Set element at (left, site, right)
    fn set3(&mut self, l: usize, s: usize, r: usize, value: T);

    /// Get a slice for fixed site index: returns (left_dim, right_dim) matrix as flat Vec
    fn slice_site(&self, s: usize) -> Vec<T>;

    /// Reshape this tensor to a matrix (left_dim * site_dim, right_dim)
    fn as_left_matrix(&self) -> (Vec<T>, usize, usize);

    /// Reshape this tensor to a matrix (left_dim, site_dim * right_dim)
    fn as_right_matrix(&self) -> (Vec<T>, usize, usize);
}

impl<T: Clone + Default> Tensor3Ops<T> for Tensor3<T> {
    fn left_dim(&self) -> usize {
        self.dim(0)
    }

    fn site_dim(&self) -> usize {
        self.dim(1)
    }

    fn right_dim(&self) -> usize {
        self.dim(2)
    }

    fn get3(&self, l: usize, s: usize, r: usize) -> &T {
        &self[[l, s, r]]
    }

    fn get3_mut(&mut self, l: usize, s: usize, r: usize) -> &mut T {
        &mut self[[l, s, r]]
    }

    fn set3(&mut self, l: usize, s: usize, r: usize, value: T) {
        self[[l, s, r]] = value;
    }

    fn slice_site(&self, s: usize) -> Vec<T> {
        let left_dim = self.left_dim();
        let right_dim = self.right_dim();
        let mut result = Vec::with_capacity(left_dim * right_dim);
        for l in 0..left_dim {
            for r in 0..right_dim {
                result.push(self[[l, s, r]].clone());
            }
        }
        result
    }

    fn as_left_matrix(&self) -> (Vec<T>, usize, usize) {
        let left_dim = self.left_dim();
        let site_dim = self.site_dim();
        let right_dim = self.right_dim();
        let rows = left_dim * site_dim;
        let cols = right_dim;
        let mut result = Vec::with_capacity(rows * cols);
        for l in 0..left_dim {
            for s in 0..site_dim {
                for r in 0..right_dim {
                    result.push(self[[l, s, r]].clone());
                }
            }
        }
        (result, rows, cols)
    }

    fn as_right_matrix(&self) -> (Vec<T>, usize, usize) {
        let left_dim = self.left_dim();
        let site_dim = self.site_dim();
        let right_dim = self.right_dim();
        let rows = left_dim;
        let cols = site_dim * right_dim;
        let mut result = Vec::with_capacity(rows * cols);
        for l in 0..left_dim {
            for s in 0..site_dim {
                for r in 0..right_dim {
                    result.push(self[[l, s, r]].clone());
                }
            }
        }
        (result, rows, cols)
    }
}

/// Create a zero-filled Tensor3
pub fn tensor3_zeros<T: Clone + Default>(left_dim: usize, site_dim: usize, right_dim: usize) -> Tensor3<T> {
    Tensor3::from_elem([left_dim, site_dim, right_dim], T::default())
}

/// Create a Tensor3 from flat data (row-major order)
pub fn tensor3_from_data<T: Clone>(data: Vec<T>, left_dim: usize, site_dim: usize, right_dim: usize) -> Tensor3<T> {
    assert_eq!(data.len(), left_dim * site_dim * right_dim);
    Tensor3::from_fn([left_dim, site_dim, right_dim], |idx| {
        let l = idx[0];
        let s = idx[1];
        let r = idx[2];
        data[(l * site_dim + s) * right_dim + r].clone()
    })
}
